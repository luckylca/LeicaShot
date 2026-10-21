"""Native RGB/Lab detail fusion used after an external color predictor."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any


def _gaussian_radius(sigma: float) -> int:
    kernel_size = int(round(sigma * 8.0 + 1.0)) | 1
    return (kernel_size - 1) // 2


def fuse_lab_detail(
    original_rgb: Any,
    predicted_rgb: Any,
    *,
    sigma_l: float = 25.0,
    sigma_ab: float = 20.0,
    row_chunk_height: int = 512,
    cancel_check: Callable[[], None] | None = None,
) -> Any:
    """Fuse predicted color with original high-frequency detail in Lab space.

    Inputs are equal-size, C-contiguous HWC RGB uint8 arrays. Per Lab channel,
    the output is ``original + Gaussian(predicted - original)`` with OpenCV's
    8-bit sRGB Lab conversion, a reflect-101 Gaussian border, clipping, and
    truncation to uint8. Row chunks include the exact Gaussian halo so the
    result matches a full-frame filter while bounding temporary Lab/float data.
    The returned RGB array is newly allocated; inputs are unchanged.
    """
    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Lab fusion requires NumPy and OpenCV") from exc

    original = np.asarray(original_rgb)
    predicted = np.asarray(predicted_rgb)
    if (
        original.dtype != np.uint8
        or predicted.dtype != np.uint8
        or original.ndim != 3
        or predicted.shape != original.shape
        or original.shape[2] != 3
        or original.shape[0] < 1
        or original.shape[1] < 1
        or not original.flags.c_contiguous
        or not predicted.flags.c_contiguous
    ):
        raise ValueError(
            "original_rgb and predicted_rgb must be equal-size C-contiguous "
            "HWC uint8 RGB arrays"
        )
    if row_chunk_height < 1:
        raise ValueError("row_chunk_height must be positive")
    if (
        not math.isfinite(sigma_l)
        or not math.isfinite(sigma_ab)
        or sigma_l <= 0
        or sigma_ab <= 0
    ):
        raise ValueError("Gaussian sigmas must be finite and positive")

    height, width = original.shape[:2]
    sigmas = (sigma_l, sigma_ab, sigma_ab)
    halo = max(_gaussian_radius(sigma) for sigma in sigmas)
    fused_lab = np.empty((height, width, 3), dtype=np.uint8)
    for row_start in range(0, height, row_chunk_height):
        if cancel_check is not None:
            cancel_check()
        row_stop = min(height, row_start + row_chunk_height)
        source_start = max(0, row_start - halo)
        source_stop = min(height, row_stop + halo)
        source_lab = cv2.cvtColor(original[source_start:source_stop], cv2.COLOR_RGB2LAB)
        predicted_lab = cv2.cvtColor(
            predicted[source_start:source_stop], cv2.COLOR_RGB2LAB
        )
        center_start = row_start - source_start
        center_stop = center_start + row_stop - row_start
        scratch = np.empty((source_stop - source_start, width), dtype=np.float32)
        for channel, sigma in enumerate(sigmas):
            np.subtract(
                predicted_lab[:, :, channel],
                source_lab[:, :, channel],
                out=scratch,
                dtype=np.float32,
            )
            cv2.GaussianBlur(
                scratch,
                (0, 0),
                sigmaX=sigma,
                dst=scratch,
                borderType=cv2.BORDER_REFLECT_101,
            )
            np.add(scratch, source_lab[:, :, channel], out=scratch)
            np.clip(scratch, 0, 255, out=scratch)
            fused_lab[row_start:row_stop, :, channel] = scratch[
                center_start:center_stop
            ].astype(np.uint8)
        del source_lab, predicted_lab, scratch

    if cancel_check is not None:
        cancel_check()
    return cv2.cvtColor(fused_lab, cv2.COLOR_LAB2RGB)


__all__ = ["fuse_lab_detail"]
