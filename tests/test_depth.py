from __future__ import annotations

import numpy as np
import pytest

from laika.depth import (
    _preprocess_rgb,
    depth_network_size,
    resize_bilinear_align_corners,
)


def test_depth_network_size_matches_lower_bound_aspect_ratio_rule() -> None:
    assert depth_network_size(4000, 3000) == (686, 518)
    assert depth_network_size(3000, 4000) == (518, 686)
    assert depth_network_size(3000, 3000) == (518, 518)
    assert depth_network_size(40_000, 1_000) == (20_720, 518)


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
    tensor, width, height = _preprocess_rgb(rgb, 518, cv2, np, 4_000_000)
    expected = (
        np.asarray([1.0, 128.0 / 255.0, 0.0], dtype=np.float32)
        - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    ) / np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    assert (width, height) == (518, 518)
    assert tensor.shape == (1, 3, 518, 518)
    np.testing.assert_allclose(tensor[0, :, 0, 0], expected, rtol=0, atol=1e-6)
    assert tensor.flags.c_contiguous


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
