from __future__ import annotations

import numpy as np
import pytest

from laika.depth import (
    _preprocess_rgb,
    depth_analysis_size,
    depth_input_geometry,
    depth_network_size,
    resize_bilinear_align_corners,
)


def test_depth_network_size_matches_lower_bound_aspect_ratio_rule() -> None:
    assert depth_network_size(4000, 3000) == (686, 518)
    assert depth_network_size(3000, 4000) == (518, 686)
    assert depth_network_size(3000, 3000) == (518, 518)
    assert depth_network_size(40_000, 1_000) == (20_720, 518)


def test_reference_analysis_size_caps_long_edge_and_preserves_aspect_ratio() -> None:
    assert depth_analysis_size(3840, 2160) == (1220, 686)
    assert depth_analysis_size(4000, 3000) == (1220, 915)
    assert depth_analysis_size(1000, 750) == (1000, 750)
    assert depth_analysis_size(1, 1) == (1, 1)


def test_depth_input_geometry_pads_analysis_edges_without_resizing_output() -> None:
    reference = depth_input_geometry(3840, 2160)
    assert (reference.analysis_width, reference.analysis_height) == (1220, 686)
    assert (reference.network_width, reference.network_height) == (1232, 686)

    upstream = depth_input_geometry(3840, 2160, input_size=518)
    assert (upstream.analysis_width, upstream.analysis_height) == (924, 518)
    assert (upstream.network_width, upstream.network_height) == (924, 518)

    with pytest.raises(ValueError, match="either input_size"):
        depth_input_geometry(3840, 2160, input_size=518, max_analysis_edge=1220)


@pytest.mark.parametrize(
    ("source", "shape", "expected"),
    [
        (
            np.asarray([[0.0, 2.0], [4.0, 6.0]], dtype=np.float32),
            (3, 3),
            np.asarray(
                [[0.0, 1.0, 2.0], [2.0, 3.0, 4.0], [4.0, 5.0, 6.0]],
                dtype=np.float32,
            ),
        ),
        (
            np.asarray([[2.0, 6.0]], dtype=np.float32),
            (3, 1),
            np.asarray([[2.0], [2.0], [2.0]], dtype=np.float32),
        ),
    ],
)
def test_align_corners_resize(source, shape, expected) -> None:
    actual = resize_bilinear_align_corners(source, *shape, max_block_pixels=2)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
    assert actual.dtype == np.float32
    assert actual.flags.c_contiguous


def test_depth_network_size_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="positive"):
        depth_network_size(0, 1000)
    with pytest.raises(ValueError, match="divisible by 14"):
        depth_network_size(1000, 1000, input_size=519)


def test_preprocess_rgb_normalizes_and_orders_channels() -> None:
    import cv2

    rgb = np.zeros((518, 518, 3), dtype=np.uint8)
    rgb[:, :, 0] = 255
    rgb[:, :, 1] = 128
    tensor, geometry = _preprocess_rgb(
        rgb, 518, cv2, np, 4_000_000, max_analysis_edge=None
    )
    expected = (
        np.asarray([1.0, 128.0 / 255.0, 0.0], dtype=np.float32)
        - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    ) / np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    assert (geometry.network_width, geometry.network_height) == (518, 518)
    assert tensor.shape == (1, 3, 518, 518)
    np.testing.assert_allclose(tensor[0, :, 0, 0], expected, rtol=0, atol=1e-6)
    assert tensor.flags.c_contiguous


def test_reference_preprocess_replicates_right_and_bottom_padding() -> None:
    import cv2

    rgb = np.zeros((19, 23, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.arange(23, dtype=np.uint8)[None, :]
    rgb[:, :, 1] = np.arange(19, dtype=np.uint8)[:, None]
    tensor, geometry = _preprocess_rgb(
        rgb, None, cv2, np, 10_000, max_analysis_edge=1220
    )

    assert (geometry.analysis_width, geometry.analysis_height) == (23, 19)
    assert (geometry.network_width, geometry.network_height) == (28, 28)
    assert np.all(tensor[0, :, :19, 23:] == tensor[0, :, :19, 22:23])
    assert np.all(tensor[0, :, 19:, :] == tensor[0, :, 18:19, :])


def test_reference_preprocess_uses_pillow_bicubic_before_normalization() -> None:
    import cv2
    from PIL import Image

    rgb = np.random.default_rng(44).integers(0, 256, (600, 800, 3), dtype=np.uint8)
    tensor, geometry = _preprocess_rgb(
        rgb, None, cv2, np, 500_000, max_analysis_edge=600
    )
    resized = np.asarray(
        Image.fromarray(rgb).resize((600, 450), resample=Image.Resampling.BICUBIC),
        dtype=np.uint8,
    )
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    expected = np.empty((3, 462, 602), dtype=np.float32)
    for channel in range(3):
        plane = resized[:, :, channel].astype(np.float32)
        np.divide(plane, np.float32(255.0), out=plane)
        plane -= mean[channel]
        plane /= std[channel]
        expected[channel, :450, :600] = plane
        expected[channel, 450:, :600] = plane[449:450, :]
        expected[channel, :450, 600:] = plane[:, 599:600]
        expected[channel, 450:, 600:] = plane[449:450, 599:600]

    assert (geometry.analysis_width, geometry.analysis_height) == (600, 450)
    assert (geometry.network_width, geometry.network_height) == (602, 462)
    np.testing.assert_array_equal(tensor[0], expected)


def test_preprocess_rgb_rejects_non_rgb_input() -> None:
    import cv2

    with pytest.raises(ValueError, match="uint8 array"):
        _preprocess_rgb(np.zeros((16, 16), dtype=np.uint8), 518, cv2, np, 4_000_000)


def test_preprocess_rgb_rejects_excessive_network_size_before_resize() -> None:
    import cv2

    rgb = np.zeros((1, 8000, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="network input exceeds"):
        _preprocess_rgb(rgb, 518, cv2, np, 1_000_000)


def test_align_corners_resize_rejects_invalid_arrays() -> None:
    with pytest.raises(ValueError, match="2-D floating-point"):
        resize_bilinear_align_corners(np.zeros((2, 2), dtype=np.uint8), 4, 4)
    with pytest.raises(ValueError, match="positive"):
        resize_bilinear_align_corners(np.zeros((2, 2), dtype=np.float32), 0, 4)
