from __future__ import annotations

import cv2
import numpy as np
import pytest

from laika.color import fuse_lab_detail


def reference_blur_difference(
    original: np.ndarray, predicted: np.ndarray
) -> np.ndarray:
    source_lab = cv2.cvtColor(original, cv2.COLOR_RGB2LAB)
    predicted_lab = cv2.cvtColor(predicted, cv2.COLOR_RGB2LAB)
    fused = source_lab.astype(np.float32)
    for channel, sigma in enumerate((25.0, 20.0, 20.0)):
        source_low = cv2.GaussianBlur(
            source_lab[:, :, channel].astype(np.float32), (0, 0), sigmaX=sigma
        )
        predicted_low = cv2.GaussianBlur(
            predicted_lab[:, :, channel].astype(np.float32), (0, 0), sigmaX=sigma
        )
        fused[:, :, channel] = np.clip(
            predicted_low + source_lab[:, :, channel] - source_low, 0, 255
        )
    return cv2.cvtColor(fused.astype(np.uint8), cv2.COLOR_LAB2RGB)


def synthetic_photos(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(92)
    source = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    coarse = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    predicted = cv2.GaussianBlur(coarse, (0, 0), sigmaX=18)
    return source, predicted


@pytest.mark.parametrize("row_chunk_height", [48, 128, 512])
def test_streamed_row_halos_match_full_frame_filter(row_chunk_height: int) -> None:
    original, predicted = synthetic_photos(213, 527)

    whole = fuse_lab_detail(
        original,
        predicted,
        row_chunk_height=original.shape[0],
    )
    streamed = fuse_lab_detail(
        original,
        predicted,
        row_chunk_height=row_chunk_height,
    )

    np.testing.assert_array_equal(streamed, whole)


def test_lab_fusion_tracks_two_blur_reference_and_preserves_inputs() -> None:
    original, predicted = synthetic_photos(257, 263)
    original_before = original.copy()
    predicted_before = predicted.copy()

    actual = fuse_lab_detail(original, predicted, row_chunk_height=80)
    expected = reference_blur_difference(original, predicted)
    error = np.abs(actual.astype(np.int16) - expected.astype(np.int16))

    assert error.max() <= 12
    assert error.mean() < 0.1
    assert np.count_nonzero(error > 2) / error.size < 0.0005
    np.testing.assert_array_equal(original, original_before)
    np.testing.assert_array_equal(predicted, predicted_before)


def test_equal_color_input_round_trips_through_reference_lab_codes() -> None:
    original, _ = synthetic_photos(121, 251)
    expected = cv2.cvtColor(
        cv2.cvtColor(original, cv2.COLOR_RGB2LAB), cv2.COLOR_LAB2RGB
    )

    actual = fuse_lab_detail(original, original, row_chunk_height=64)

    np.testing.assert_array_equal(actual, expected)


def test_lab_fusion_checks_shapes_layout_and_filter_parameters() -> None:
    original = np.zeros((20, 30, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="equal-size"):
        fuse_lab_detail(original, np.zeros((20, 29, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="row_chunk_height"):
        fuse_lab_detail(original, original, row_chunk_height=0)
    with pytest.raises(ValueError, match="sigmas"):
        fuse_lab_detail(original, original, sigma_ab=float("inf"))
    with pytest.raises(ValueError, match="C-contiguous"):
        fuse_lab_detail(original[:, ::2], original[:, ::2])


def test_fusion_observes_cancellation_between_row_bands() -> None:
    original, predicted = synthetic_photos(32, 300)
    checks = 0

    def cancel_after_one_band() -> None:
        nonlocal checks
        checks += 1
        if checks == 2:
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        fuse_lab_detail(
            original,
            predicted,
            row_chunk_height=64,
            cancel_check=cancel_after_one_band,
        )
    assert checks == 2
