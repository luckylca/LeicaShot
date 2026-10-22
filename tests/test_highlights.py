from __future__ import annotations

import numpy as np
import pytest

from laika import highlights
from laika.highlights import (
    aperture_kernel,
    composite_highlights,
    extract_highlights,
    recover_clipped_highlights_inplace,
    scatter_highlights,
    soften_highlights_inplace,
)


def test_highlight_split_keeps_neutral_and_diffuse_bright_surfaces() -> None:
    neutral = np.full((64, 80, 3), 96, dtype=np.uint8)
    split = extract_highlights(neutral)
    np.testing.assert_array_equal(split.base_rgb, neutral)
    assert not np.any(split.highlight_linear)

    diffuse = np.zeros((256, 256, 3), dtype=np.uint8)
    diffuse[90:166, 80:176] = 245
    split = extract_highlights(diffuse)
    np.testing.assert_array_equal(split.base_rgb, diffuse)
    assert not np.any(split.highlight_linear)


def test_highlight_split_extracts_a_compact_point_source() -> None:
    rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    rgb[126:130, 126:130] = (255, 220, 160)

    split = extract_highlights(rgb, row_chunk_height=31)

    assert float(split.highlight_linear.sum()) > 0.0
    assert int(split.base_rgb[127, 127, 0]) < 255
    assert int(split.base_rgb[0, 0, 0]) == 0
    assert np.isfinite(split.highlight_linear).all()
    assert float(split.highlight_linear.min()) >= 0.0


def test_aperture_kernel_is_symmetric_normalized_and_aperture_shaped() -> None:
    round_kernel = aperture_kernel(7, 0.95)
    polygon_kernel = aperture_kernel(7, 2.8)
    assert round_kernel.shape == polygon_kernel.shape == (17, 17)
    assert float(round_kernel.sum()) == pytest.approx(1.0, abs=2e-7)
    assert float(polygon_kernel.sum()) == pytest.approx(1.0, abs=2e-7)
    np.testing.assert_array_equal(round_kernel, round_kernel[::-1, ::-1])
    assert not np.array_equal(round_kernel, polygon_kernel)


@pytest.mark.parametrize("radius,aperture", [(7, 2.8), (63, 5.6)])
def test_opposite_field_nodes_reuse_bitwise_identical_pupil_kernels(
    monkeypatch: pytest.MonkeyPatch, radius: int, aperture: float
) -> None:
    shapes = [
        highlights._optical_shape(x, y, 23, 19)
        for y, *_ in highlights._field_axis_nodes(19)
        for x, *_ in highlights._field_axis_nodes(23)
    ]
    reference = [
        highlights._optical_pupil_kernel(radius, aperture, *shape) for shape in shapes
    ]
    original = highlights._optical_pupil_kernel
    calls = 0

    def counted_kernel(
        kernel_radius: int,
        kernel_aperture: float,
        ux: float,
        uy: float,
        radial_ratio: float,
    ) -> np.ndarray:
        nonlocal calls
        calls += 1
        return original(kernel_radius, kernel_aperture, ux, uy, radial_ratio)

    monkeypatch.setattr(highlights, "_optical_pupil_kernel", counted_kernel)
    actual = highlights._field_pupil_kernels(shapes, radius, aperture)

    assert calls == 13
    assert actual[0] is actual[-1]
    for cached, direct in zip(actual, reference, strict=True):
        np.testing.assert_array_equal(cached, direct)


def test_fractional_constant_depth_interpolates_adjacent_full_psfs() -> None:
    high = np.zeros((61, 61, 3), dtype=np.float32)
    high[30, 30] = (0.8, 0.5, 0.2)
    coc = np.full((61, 61), 1.25, dtype=np.float32)

    scattered = scatter_highlights(high, coc, 2.8, radius_scale=3.0, row_chunk_height=7)
    expected = np.zeros_like(high)
    expected[30 - 4 : 30 + 5, 30 - 4 : 30 + 5] += (
        high[30, 30][None, None, :]
        * np.float32(0.25)
        * aperture_kernel(3, 2.8)[::-1, ::-1, None]
    )
    expected[30 - 5 : 30 + 6, 30 - 5 : 30 + 6] += (
        high[30, 30][None, None, :]
        * np.float32(0.75)
        * aperture_kernel(4, 2.8)[::-1, ::-1, None]
    )
    np.testing.assert_allclose(scattered, expected, rtol=0, atol=2e-6)


def test_depth_layer_scatter_conserves_constant_light_and_does_not_mutate() -> None:
    rng = np.random.default_rng(12)
    high = rng.random((21, 29, 3), dtype=np.float32) * np.float32(0.4)
    original = high.copy()
    coc = np.linspace(-3.2, 4.1, 21 * 29, dtype=np.float32).reshape(21, 29)
    output = scatter_highlights(high, coc, 4.0, radius_scale=1.0, row_chunk_height=5)
    np.testing.assert_array_equal(high, original)
    assert output.shape == high.shape
    assert output.dtype == np.float32
    assert np.isfinite(output).all()
    assert float(output.min()) >= 0.0


def test_field_optics_preserves_constant_radiance_through_signed_depth_layers() -> None:
    rng = np.random.default_rng(71)
    high = np.full((19, 23, 3), 0.37, dtype=np.float32)
    coc = rng.uniform(-2.5, 2.5, size=(19, 23)).astype(np.float32)

    output = scatter_highlights(
        high,
        np.ascontiguousarray(coc),
        2.8,
        radius_scale=1.0,
        field_optics=True,
        row_chunk_height=5,
    )

    np.testing.assert_allclose(output, high, rtol=0, atol=4e-6)


@pytest.mark.parametrize("has_highlight", [False, True])
def test_field_optics_skips_rgb_filter_only_for_empty_layer_radiance(
    monkeypatch: pytest.MonkeyPatch, has_highlight: bool
) -> None:
    high = np.zeros((9, 11, 3), dtype=np.float32)
    if has_highlight:
        high[4, 5] = (0.8, 0.5, 0.2)
    else:
        high.fill(np.float32(-0.0))
    coc = np.zeros((9, 11), dtype=np.float32)
    original_filter = highlights.cv2.filter2D
    filtered_dimensions: list[int] = []

    def counted_filter(source, *args, **kwargs):
        filtered_dimensions.append(source.ndim)
        return original_filter(source, *args, **kwargs)

    monkeypatch.setattr(highlights.cv2, "filter2D", counted_filter)

    result = scatter_highlights(
        high,
        coc,
        2.8,
        radius_scale=1.0,
        field_optics=True,
        row_chunk_height=4,
    )

    assert (3 in filtered_dimensions) is has_highlight
    assert np.isfinite(result).all()
    if has_highlight:
        assert float(result.sum()) > 0.0
    else:
        assert not np.any(result)


@pytest.mark.parametrize("row_chunk_height", [1, 7, 64])
def test_chunked_residual_scatter_combination_is_bitwise_legacy_exact(
    row_chunk_height: int,
) -> None:
    rng = np.random.default_rng(1701)
    output = rng.random((13, 17, 3), dtype=np.float32)
    own = rng.random((13, 17, 3), dtype=np.float32)
    coverage = rng.random((13, 17), dtype=np.float32)
    expected = output.copy()
    expected += own * (1.0 - coverage[:, :, None])

    actual = output.copy()
    highlights._combine_own_into_output(actual, own, coverage, row_chunk_height)

    np.testing.assert_array_equal(actual, expected)


def test_field_optics_does_not_allocate_an_unused_full_frame_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    high = np.zeros((9, 11, 3), dtype=np.float32)
    high[4, 5] = (0.8, 0.5, 0.2)
    coc = np.ones((9, 11), dtype=np.float32)
    original_zeros_like = np.zeros_like
    allocated_shapes: list[tuple[int, ...]] = []

    def counted_zeros_like(
        value: np.ndarray, *args: object, **kwargs: object
    ) -> np.ndarray:
        result = original_zeros_like(value, *args, **kwargs)
        if value.shape == high.shape:
            allocated_shapes.append(result.shape)
        return result

    monkeypatch.setattr(highlights.np, "zeros_like", counted_zeros_like)
    result = scatter_highlights(high, coc, 2.8, field_optics=True, row_chunk_height=3)

    assert result.shape == high.shape
    assert allocated_shapes == [high.shape, high.shape]


def test_scatter_and_composite_honor_cancellation_and_zero_light_identity() -> None:
    def cancelled() -> None:
        raise RuntimeError("cancelled")

    high = np.zeros((8, 9, 3), dtype=np.float32)
    coc = np.ones((8, 9), dtype=np.float32)
    with pytest.raises(RuntimeError, match="cancelled"):
        scatter_highlights(high, coc, 2.8, cancel_check=cancelled)

    image = np.arange(8 * 9 * 3, dtype=np.uint8).reshape(8, 9, 3)
    result = composite_highlights(
        image, image, np.zeros(image.shape, dtype=np.float32), row_chunk_height=3
    )
    np.testing.assert_array_equal(result, image)


def test_scatter_progress_is_monotonic_and_cancelable_between_depth_layers() -> None:
    high = np.zeros((19, 23, 3), dtype=np.float32)
    high[9, 11] = 0.7
    coc = np.full((19, 23), 0.4, dtype=np.float32)
    progress: list[float] = []
    result = scatter_highlights(
        high,
        coc,
        2.8,
        radius_scale=1.0,
        progress_callback=progress.append,
    )
    assert np.any(result)
    assert progress == sorted(progress)
    assert progress[-1] == 1.0


def test_clipped_highlight_recovery_only_boosts_clipped_defocused_sources() -> None:
    source = np.zeros((24, 24, 3), dtype=np.uint8)
    source[6:9, 6:9] = 255
    source[15:18, 15:18] = 255
    source[2:5, 15:18] = 200
    coc = np.full((24, 24), 2.0, dtype=np.float32)
    coc[15:18, 15:18] = 0.0
    high = np.zeros((24, 24, 3), dtype=np.float32)
    high[6:9, 6:9] = 0.3
    high[15:18, 15:18] = 0.3
    high[2:5, 15:18] = 0.4
    original = high.copy()

    changed = recover_clipped_highlights_inplace(high, source, coc, row_chunk_height=5)

    assert changed
    assert np.all(high[6:9, 6:9] > original[6:9, 6:9])
    np.testing.assert_array_equal(high[15:18, 15:18], original[15:18, 15:18])
    np.testing.assert_array_equal(high[2:5, 15:18], original[2:5, 15:18])
    assert np.isfinite(high).all()


def test_soft_bokeh_edges_preserves_in_focus_pixels_and_softens_defocused_light() -> (
    None
):
    coc = np.zeros((41, 41), dtype=np.float32)
    coc[:, 20:] = 6.0
    light = np.zeros((41, 41, 3), dtype=np.float32)
    light[20, 21] = (0.8, 0.4, 0.2)
    original = light.copy()

    changed = soften_highlights_inplace(light, coc)

    assert changed
    np.testing.assert_array_equal(light[:, :20], original[:, :20])
    assert np.count_nonzero(light[19:22, 20:24, 0]) > 1
    assert np.isfinite(light).all()


def test_soft_bokeh_edges_returns_unchanged_for_in_focus_plane() -> None:
    coc = np.zeros((9, 11), dtype=np.float32)
    light = np.zeros((9, 11, 3), dtype=np.float32)
    light[4, 5] = (0.3, 0.2, 0.1)
    original = light.copy()

    changed = soften_highlights_inplace(light, coc)

    assert not changed
    np.testing.assert_array_equal(light, original)
