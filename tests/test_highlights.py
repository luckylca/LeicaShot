from __future__ import annotations

import numpy as np
import pytest

from laika import highlights
from laika.highlights import (
    aperture_kernel,
    composite_highlights,
    extract_highlight_radiance,
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


def test_deferred_highlight_radiance_matches_eager_split_exactly() -> None:
    rgb = np.zeros((257, 263, 3), dtype=np.uint8)
    rgb[126:131, 128:134] = (255, 220, 160)
    eager = extract_highlights(rgb, row_chunk_height=31)
    base_only = extract_highlights(
        rgb,
        row_chunk_height=31,
        include_radiance=False,
    )
    progress: list[float] = []

    assert base_only.highlight_linear is None
    np.testing.assert_array_equal(base_only.base_rgb, eager.base_rgb)
    deferred = extract_highlight_radiance(
        rgb,
        base_only.base_rgb,
        row_chunk_height=31,
        progress_callback=progress.append,
    )

    np.testing.assert_array_equal(deferred, eager.highlight_linear)
    assert progress[-1] == 1.0


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


def test_fp32_radiance_workspace_preserves_final_composite_pixels() -> None:
    rng = np.random.default_rng(20261024)
    height, width = 81, 97
    high = np.zeros((height, width, 3), dtype=np.float32)
    for _ in range(90):
        y, x = rng.integers(0, height), rng.integers(0, width)
        high[max(0, y - 2) : y + 3, max(0, x - 2) : x + 3] += rng.random(
            3, dtype=np.float32
        )
    original = high.copy()
    coc = rng.uniform(-2.5, 2.5, (height, width)).astype(np.float32)

    fp64_reference = scatter_highlights(
        high,
        coc,
        2.8,
        radius_scale=3.0,
        field_optics=True,
        row_chunk_height=19,
        _float32_radiance_workspace=False,
    )
    fp32_workspace = scatter_highlights(
        high,
        coc,
        2.8,
        radius_scale=3.0,
        field_optics=True,
        row_chunk_height=19,
    )

    np.testing.assert_array_equal(high, original)
    assert np.isfinite(fp32_workspace).all()
    assert float(np.max(np.abs(fp32_workspace - fp64_reference))) <= 2e-6
    base_rgb = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    rendered_rgb = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    reference_rgb = composite_highlights(
        base_rgb, rendered_rgb, fp64_reference, row_chunk_height=19
    )
    candidate_rgb = composite_highlights(
        base_rgb, rendered_rgb, fp32_workspace, row_chunk_height=19
    )
    np.testing.assert_array_equal(candidate_rgb, reference_rgb)


def test_field_optics_skips_zero_mass_filters_without_changing_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(43)
    height = width = 101
    high = rng.random((height, width, 3), dtype=np.float32) * np.float32(0.8)
    coc = np.empty((height, width), dtype=np.float32)
    coc[:, :50] = np.float32(-4.0)
    coc[:, 50:] = np.float32(4.0)
    levels = highlights._source_levels(coc, 3.0, None)
    original_filter = highlights.cv2.filter2D
    zero_mass_calls: list[int] = []

    def counted_filter(source, *args, **kwargs):
        if source.ndim == 2 and not np.any(source):
            zero_mass_calls.append(1)
        return original_filter(source, *args, **kwargs)

    monkeypatch.setattr(highlights.cv2, "filter2D", counted_filter)
    baseline = highlights._scatter_field_optical(
        high,
        coc,
        2.8,
        3.0,
        32,
        levels,
        None,
        None,
        skip_empty_mass=False,
    )
    baseline_zero_mass_calls = len(zero_mass_calls)
    assert baseline_zero_mass_calls > 0

    zero_mass_calls.clear()
    candidate = highlights._scatter_field_optical(
        high,
        coc,
        2.8,
        3.0,
        32,
        levels,
        None,
        None,
        skip_empty_mass=True,
    )

    np.testing.assert_array_equal(candidate, baseline)
    assert zero_mass_calls == []


def test_field_optical_scatter_chunk_sizes_preserve_composited_pixels() -> None:
    rng = np.random.default_rng(2081)
    high = np.zeros((57, 61, 3), dtype=np.float32)
    high[9, 14] = (0.7, 0.4, 0.2)
    high[43, 48] = (0.2, 0.5, 0.8)
    coc = rng.uniform(-1.5, 1.5, size=(57, 61)).astype(np.float32)
    levels = highlights._source_levels(coc, 3.0, None)
    expected = highlights._scatter_field_optical(
        high,
        coc,
        2.8,
        3.0,
        19,
        levels,
        None,
        None,
        float32_radiance_workspace=False,
    )

    for chunk_height in (128, 256, 512):
        actual = highlights._scatter_field_optical(
            high,
            coc,
            2.8,
            3.0,
            chunk_height,
            levels,
            None,
            None,
            float32_radiance_workspace=False,
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
        base = np.full((57, 61, 3), 128, dtype=np.uint8)
        np.testing.assert_array_equal(
            highlights.composite_highlights(base, base, actual),
            highlights.composite_highlights(base, base, expected),
        )


def test_fp32_mass_convolution_preserves_quantized_highlight_composite() -> None:
    rng = np.random.default_rng(8842)
    high = np.zeros((73, 81, 3), dtype=np.float32)
    for _ in range(24):
        y = int(rng.integers(0, high.shape[0]))
        x = int(rng.integers(0, high.shape[1]))
        high[y, x] = rng.random(3, dtype=np.float32)
    coc = rng.uniform(-1.0, 1.0, size=high.shape[:2]).astype(np.float32)
    levels = highlights._source_levels(coc, 3.0, None)
    reference = highlights._scatter_field_optical(
        high, coc, 2.8, 3.0, 256, levels, None, None, mass_filter_float32=False
    )
    candidate = highlights._scatter_field_optical(
        high, coc, 2.8, 3.0, 256, levels, None, None, mass_filter_float32=True
    )
    default = highlights._scatter_field_optical(
        high, coc, 2.8, 3.0, 256, levels, None, None
    )

    np.testing.assert_allclose(candidate, reference, rtol=0.0, atol=1e-6)
    np.testing.assert_array_equal(default, reference)
    base = rng.integers(0, 256, size=(*high.shape[:2], 3), dtype=np.uint8)
    np.testing.assert_array_equal(
        highlights.composite_highlights(base, base, candidate),
        highlights.composite_highlights(base, base, reference),
    )


def test_inplace_field_weight_multiplication_matches_allocating_expression() -> None:
    rng = np.random.default_rng(1809)
    weights = rng.random((31, 37), dtype=np.float64)
    mass = rng.random((31, 37), dtype=np.float64)
    radiance = rng.random((31, 37, 3), dtype=np.float64)
    expected_mass = mass * weights
    expected_radiance = radiance * weights[:, :, None]

    actual_mass = mass.copy()
    actual_radiance = radiance.copy()
    np.multiply(actual_mass, weights, out=actual_mass)
    np.multiply(actual_radiance, weights[:, :, None], out=actual_radiance)

    np.testing.assert_array_equal(actual_mass, expected_mass)
    np.testing.assert_array_equal(actual_radiance, expected_radiance)


def test_fft32_outer_mass_filter_preserves_quantized_highlight_composite() -> None:
    rng = np.random.default_rng(314159)
    high = np.zeros((73, 81, 3), dtype=np.float32)
    for _ in range(24):
        y = int(rng.integers(0, high.shape[0]))
        x = int(rng.integers(0, high.shape[1]))
        high[y, x] = rng.random(3, dtype=np.float32)
    coc = rng.uniform(-4.0, 4.0, size=high.shape[:2]).astype(np.float32)
    levels = highlights._source_levels(coc, 3.0, None)
    reference = highlights._scatter_field_optical(
        high, coc, 2.8, 3.0, 256, levels, None, None
    )
    candidate = highlights._scatter_field_optical(
        high,
        coc,
        2.8,
        3.0,
        256,
        levels,
        None,
        None,
        mass_filter_fft32_outer=True,
    )

    np.testing.assert_allclose(candidate, reference, rtol=0.0, atol=2e-7)
    base = rng.integers(0, 256, size=(*high.shape[:2], 3), dtype=np.uint8)
    np.testing.assert_array_equal(
        highlights.composite_highlights(base, base, candidate),
        highlights.composite_highlights(base, base, reference),
    )


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


def test_sparse_mass_filter_matches_full_convolution() -> None:
    rng = np.random.default_rng(20261024)
    for height, width, radius, box in (
        (64, 96, 2, (0, 4, 0, 8)),
        (85, 123, 8, (30, 40, 30, 45)),
        (93, 111, 13, (74, 92, 99, 111)),
        (99, 105, 0, (17, 24, 56, 69)),
    ):
        weights = np.zeros((height, width), dtype=np.float32)
        y0, y1, x0, x1 = box
        weights[y0:y1, x0:x1] = rng.random((y1 - y0, x1 - x0), dtype=np.float32)
        kernel = highlights.aperture_kernel(radius, 2.8).astype(np.float64)
        half = kernel.shape[0] // 2
        expected = highlights.cv2.filter2D(
            weights.astype(np.float64),
            highlights.cv2.CV_64F,
            kernel,
            borderType=highlights.cv2.BORDER_REFLECT_101,
        )[half : height - half or None, half : width - half or None]
        cropped = highlights._sparse_mass_filter(
            weights,
            kernel,
            height - 2 * half,
            width - 2 * half,
        )
        assert cropped is not None
        tile, offset_y, offset_x = cropped
        result = np.zeros_like(expected)
        result[
            offset_y : offset_y + tile.shape[0], offset_x : offset_x + tile.shape[1]
        ] = tile
        np.testing.assert_allclose(result, expected, rtol=0.0, atol=5e-12)


def test_sparse_mass_scatter_preserves_quantized_pixels() -> None:
    rng = np.random.default_rng(19271)
    h, w = 80, 160
    high = np.zeros((h, w, 3), dtype=np.float32)
    high[9:13, 10:14] = rng.random((4, 4, 3), dtype=np.float32)
    high[51:55, 135:139] = rng.random((4, 4, 3), dtype=np.float32)
    coc = np.full((h, w), np.float32(-3.5), dtype=np.float32)
    coc[:, 75:] = np.float32(4.0)
    coc[30:36, 15:20] = np.float32(1.3)
    for chunk in (32, 128):
        ref = scatter_highlights(
            high.copy(), coc, 2.8, row_chunk_height=chunk, field_optics=True
        )
        opt = scatter_highlights(
            high.copy(),
            coc,
            2.8,
            row_chunk_height=chunk,
            field_optics=True,
            _sparse_mass_roi=True,
        )
        np.testing.assert_allclose(opt, ref, rtol=0.0, atol=1e-6)
        base = np.full((h, w, 3), 100, dtype=np.uint8)
        np.testing.assert_array_equal(
            highlights.composite_highlights(base, base, opt),
            highlights.composite_highlights(base, base, ref),
        )


def test_sparse_mass_scatter_matches_dense_and_border_cases() -> None:
    rng = np.random.default_rng(10082026)
    for pattern in ("uniform", "two_half_planes", "random", "border"):
        h, w = 91, 137
        high = np.zeros((h, w, 3), dtype=np.float32)
        high[0:2, 0:2] = rng.random((2, 2, 3), dtype=np.float32)
        high[-2:, -2:] = rng.random((2, 2, 3), dtype=np.float32)
        if pattern == "uniform":
            coc = np.full((h, w), 0.2, dtype=np.float32)
        elif pattern == "two_half_planes":
            coc = np.full((h, w), -3, dtype=np.float32)
            coc[:, w // 2 :] = 2.4
        elif pattern == "border":
            coc = np.full((h, w), -4.1, dtype=np.float32)
            coc[:4, :4] = 2.5
            coc[-3:, -6:] = 4.1
        else:
            coc = rng.uniform(-4.4, 3.9, (h, w)).astype(np.float32)
        baseline = scatter_highlights(
            high.copy(),
            coc,
            2.8,
            field_optics=True,
            row_chunk_height=25,
            _sparse_mass_roi=False,
        )
        result = scatter_highlights(
            high.copy(),
            coc,
            2.8,
            field_optics=True,
            row_chunk_height=25,
            _sparse_mass_roi=True,
        )
        np.testing.assert_allclose(result, baseline, atol=1e-6, rtol=0.0)
        bgr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        np.testing.assert_array_equal(
            highlights.composite_highlights(bgr, bgr, result),
            highlights.composite_highlights(bgr, bgr, baseline),
        )


def test_sparse_radiance_roi_preserves_quantized_scatter() -> None:
    rng = np.random.default_rng(7206)
    h, w = 98, 157
    for pattern in ("sparse", "dense", "border"):
        high = np.zeros((h, w, 3), dtype=np.float32)
        if pattern == "dense":
            high[:] = rng.random((h, w, 3), dtype=np.float32) * 0.04
        elif pattern == "border":
            high[0:2, 0:3] = 0.8
            high[-2:, -4:] = 0.2
        else:
            high[23:25, 56:59] = rng.random((2, 3, 3), dtype=np.float32)
            high[83:87, 130:134] = 0.8
        coc = rng.uniform(-4.2, 3.6, (h, w)).astype(np.float32)
        baseline = scatter_highlights(
            high.copy(),
            coc,
            2.8,
            row_chunk_height=31,
            field_optics=True,
            _sparse_radiance_roi=False,
        )
        compact = scatter_highlights(
            high.copy(),
            coc,
            2.8,
            row_chunk_height=31,
            field_optics=True,
            _sparse_radiance_roi=True,
        )
        np.testing.assert_allclose(compact, baseline, rtol=0.0, atol=2e-6)
        bgr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        np.testing.assert_array_equal(
            highlights.composite_highlights(bgr, bgr, compact),
            highlights.composite_highlights(bgr, bgr, baseline),
        )


def test_highlight_output_support_guides_without_pixel_changes() -> None:
    rng = np.random.default_rng(998716)
    height, width = 129, 241
    for pattern in ("empty", "small", "edges", "many", "dense"):
        high = np.zeros((height, width, 3), dtype=np.float32)
        if pattern == "small":
            high[44:46, 28:30] = (0.8, 0.1, 0.6)
            high[80, 209] = (0.0, 0.7, 0.0)
        elif pattern == "edges":
            high[0, 0] = (0.5, 0.5, 0.4)
            high[-1, -1] = (0.1, 0.7, 0.2)
            high[0, -1] = (0.1, 0.2, 0.8)
        elif pattern == "many":
            for y, x in zip(
                rng.integers(0, height, 30),
                rng.integers(0, width, 30),
                strict=True,
            ):
                high[y, x] = rng.random(3, dtype=np.float32)
        elif pattern == "dense":
            high = rng.random((height, width, 3), dtype=np.float32) * 0.25
        coc = rng.uniform(-5.0, 3.3, (height, width)).astype(np.float32)
        for chunk in (25, 128):
            baseline = scatter_highlights(
                high,
                coc,
                2.8,
                field_optics=True,
                row_chunk_height=chunk,
                _output_support_roi=False,
            )
            guided = scatter_highlights(
                high,
                coc,
                2.8,
                field_optics=True,
                row_chunk_height=chunk,
                _output_support_roi=True,
            )
            np.testing.assert_allclose(guided, baseline, rtol=0, atol=3e-6)
            base = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
            np.testing.assert_array_equal(
                highlights.composite_highlights(base, base, guided),
                highlights.composite_highlights(base, base, baseline),
            )


def test_highlight_output_spans_contain_all_possible_light() -> None:
    high = np.zeros((93, 171, 3), dtype=np.float32)
    high[0, 0, :] = 1.0
    high[46:49, 87:90, :] = 0.5
    high[-1, -1, :] = 0.8
    radius = 7
    chunks = 25
    spans = highlights._highlight_output_spans(high, radius, chunks)
    import cv2

    footprint = cv2.dilate(
        np.any(high != 0, axis=2).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_RECT, (2 * radius + 1, 2 * radius + 1)),
    )
    for y0, intervals in zip(range(0, high.shape[0], chunks), spans, strict=True):
        union = np.zeros(high.shape[1], dtype=np.uint8)
        for x0, x1 in intervals:
            union[x0:x1] = 1
        for row in footprint[y0 : y0 + chunks]:
            assert np.all(union[row > 0] == 1)


@pytest.mark.parametrize("pattern", ["zero", "points", "border", "dense"])
def test_coc_interval_pruning_exact_on_depth_strata(pattern: str) -> None:
    rng = np.random.default_rng(20261025)
    h, w = 37, 61
    high = np.zeros((h, w, 3), dtype=np.float32)
    if pattern == "points":
        high[15, 8] = (0.9, 0.2, 0.4)
        high[23, 47] = (0.3, 0.7, 0.2)
    elif pattern == "border":
        high[0, 0] = (0.1, 0.4, 0.6)
        high[-1, -1] = (0.9, 0.8, 0.1)
    elif pattern == "dense":
        high = rng.random((h, w, 3), dtype=np.float32) * 0.2
    coc = np.full((h, w), -5.0, dtype=np.float32)
    coc[:, w // 2 :] = 6.0
    coc[10:15] = rng.uniform(-3.0, 3.0, (5, w)).astype(np.float32)
    kwargs = {
        "radius_scale": 3.0,
        "field_optics": True,
        "row_chunk_height": 7,
    }
    control = scatter_highlights(high, coc, 2.8, _coc_interval_pruning=False, **kwargs)
    optimized = scatter_highlights(high, coc, 2.8, _coc_interval_pruning=True, **kwargs)
    default = scatter_highlights(high, coc, 2.8, **kwargs)
    np.testing.assert_array_equal(default, optimized)
    np.testing.assert_array_equal(optimized, control)


@pytest.mark.parametrize("chunk", [1, 7, 19, 64])
def test_coc_interval_pruning_reflection_and_float_boundaries(
    chunk: int,
) -> None:
    rng = np.random.default_rng(20261025 + chunk)
    h, w = 29, 43
    high = np.zeros((h, w, 3), dtype=np.float32)
    high[0, 0] = (0.8, 0.2, 0.4)
    high[-1, -1] = (0.7, 0.3, 0.9)
    high[12, 21] = (0.6, 0.8, 0.4)
    coc = rng.uniform(-6, 6, size=(h, w)).astype(np.float32)
    # Deliberately exercise values on layer support boundaries.
    coc[0, :] = np.float32(1 / 3)
    coc[-1, :] = np.float32(-2 / 3)
    coc[12, :] = np.float32(17 / 3)
    coc = np.ascontiguousarray(coc)
    kwargs = {
        "radius_scale": 3.0,
        "field_optics": True,
        "row_chunk_height": chunk,
    }
    original = scatter_highlights(high, coc, 2.8, _coc_interval_pruning=False, **kwargs)
    optimized = scatter_highlights(high, coc, 2.8, _coc_interval_pruning=True, **kwargs)
    np.testing.assert_array_equal(optimized, original)


def test_coc_interval_pruning_does_not_change_full_support_legacy_mode() -> None:
    high = np.zeros((15, 19, 3), dtype=np.float32)
    high[2:4, 5:7] = (0.8, 0.3, 0.6)
    coc = np.full((15, 19), -0.5, dtype=np.float32)
    kwargs = {"field_optics": True, "_output_support_roi": False}
    a = scatter_highlights(high, coc, 2.8, **kwargs)
    b = scatter_highlights(high, coc, 2.8, _coc_interval_pruning=True, **kwargs)
    np.testing.assert_array_equal(b, a)


def test_coc_pruning_prescan_remains_cancellable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    high = np.zeros((22, 25, 3), dtype=np.float32)
    high[7, 8] = (0.3, 0.6, 0.9)
    coc = np.zeros((22, 25), dtype=np.float32)
    post_support = False
    original = highlights._highlight_output_spans

    def prepared(*args, **kwargs):
        nonlocal post_support
        result = original(*args, **kwargs)
        post_support = True
        return result

    def cancelled() -> None:
        if post_support:
            raise RuntimeError("cancelled during CoC prescan")

    monkeypatch.setattr(highlights, "_highlight_output_spans", prepared)
    with pytest.raises(RuntimeError, match="during CoC prescan"):
        scatter_highlights(
            high,
            coc,
            2.8,
            field_optics=True,
            cancel_check=cancelled,
            _coc_interval_pruning=True,
        )


@pytest.mark.parametrize("variant", ["sparse", "two_islands", "dense", "edges"])
@pytest.mark.parametrize("chunk_height", [1, 9, 32])
def test_compact_optical_band_exact_vs_full_width(
    variant: str, chunk_height: int
) -> None:
    h, w = 27, 123
    rng = np.random.default_rng(2760)
    highlight = np.zeros((h, w, 3), dtype=np.float32)
    if variant == "sparse":
        highlight[7, 6] = (0.8, 0.6, 0.2)
    elif variant == "two_islands":
        highlight[8, 6] = (0.8, 0.6, 0.2)
        highlight[22, 105] = (0.1, 0.9, 0.4)
    elif variant == "dense":
        highlight = rng.random((h, w, 3), dtype=np.float32) / 4
    else:
        highlight[0, 0] = (0.7, 0.2, 0.3)
        highlight[-1, -1] = (0.5, 0.9, 0.4)
    coc = np.zeros((h, w), dtype=np.float32)
    coc[:, :29] = -2
    coc[:, 60:] = 3
    coc[9:15] = rng.uniform(-2.6, 2.6, (6, w)).astype(np.float32)
    kwargs = {
        "field_optics": True,
        "radius_scale": 2.0,
        "row_chunk_height": chunk_height,
    }
    original = scatter_highlights(
        highlight, coc, 2.8, _compact_band_output=False, **kwargs
    )
    compact = scatter_highlights(
        highlight, coc, 2.8, _compact_band_output=True, **kwargs
    )
    np.testing.assert_array_equal(compact, original)


@pytest.mark.parametrize("variant", ["empty", "single", "border", "dense", "vertical"])
def test_softening_zero_radiance_line_skip_exact(variant: str) -> None:
    rng = np.random.default_rng(42)
    light = np.zeros((47, 73, 3), dtype=np.float32)
    if variant == "single":
        light[13, 18] = (0.5, 0.8, 0.1)
    elif variant == "border":
        light[0, 0] = (0.5, 0.8, 0.1)
        light[-1, -1] = (0.7, 0.4, 0.6)
    elif variant == "dense":
        light = rng.random(light.shape, dtype=np.float32)
    elif variant == "vertical":
        light[:, 51] = rng.random((47, 3), dtype=np.float32)
    coc = np.full((47, 73), 5.0, dtype=np.float32)
    coc[::3, :] = 0.0
    original = light.copy()
    candidate = light.copy()
    orig_changed = soften_highlights_inplace(original, coc, _skip_zero_lines=False)
    new_changed = soften_highlights_inplace(candidate, coc, _skip_zero_lines=True)
    assert orig_changed == new_changed
    np.testing.assert_array_equal(candidate, original)


@pytest.mark.parametrize(
    "variant", ["empty", "one", "two", "dense", "edge", "subnormal", "masked"]
)
@pytest.mark.parametrize("shape", [(23, 47), (81, 103)])
def test_softening_support_roi_exact_vs_full_line(
    variant: str, shape: tuple[int, int]
) -> None:
    rng = np.random.default_rng(13241)
    h, w = shape
    light = np.zeros((h, w, 3), dtype=np.float32)
    if variant == "one":
        light[h // 2, w // 3] = (0.3, 0.7, 0.8)
    elif variant == "two":
        light[h // 4, w // 4] = (0.2, 0.6, 0.9)
        light[3 * h // 4, 3 * w // 4] = (0.9, 0.1, 0.2)
    elif variant == "dense":
        light = rng.random((h, w, 3), dtype=np.float32) * np.float32(0.9)
    elif variant == "edge":
        light[0, 0] = (0.9, 0.4, 0.1)
        light[-1, -1] = (0.5, 0.2, 0.9)
    elif variant == "subnormal":
        light[h // 3, w // 3] = np.nextafter(np.float32(0.0), np.float32(1.0))
    elif variant == "masked":
        light[h // 2, w // 2] = (0.9, 0.3, 0.2)
    coc = rng.uniform(-11.2, 11.2, (h, w)).astype(np.float32)
    if variant == "masked":
        coc[h // 2, w // 2] = 0.0
    baseline = light.copy()
    candidate = light.copy()
    old = soften_highlights_inplace(baseline, coc, _radiance_support_range=False)
    new = soften_highlights_inplace(candidate, coc, _radiance_support_range=True)
    assert old is new
    np.testing.assert_array_equal(candidate, baseline)
