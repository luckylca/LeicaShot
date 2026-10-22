"""SAM + ViTMatte subject masks and low-memory protected-background compositing."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from laika.matting import (
    MatteTileConfig,
    ViTMatteOnnx,
    prepare_vitmatte_input,
    refine_matte_tiled,
)
from laika.subject import (
    SAM_ANALYSIS_LONG_EDGE,
    Sam2TinyOnnx,
    normalized_points_to_sam,
    prepare_sam_input,
    select_subject_mask,
)
from laika.subject import build_subject_trimap as build_trimap


@dataclass(frozen=True)
class SubjectAlphaResult:
    """One union alpha and measured model/tile work for the selected points."""

    alpha: Any
    stage_seconds: dict[str, float]
    point_scores: tuple[float, ...]
    cache_hits: int


def prepare_subject_analysis_rgb(rgb: Any) -> Any:
    """Resize the already EXIF-oriented image to the reference 2048 long edge."""
    import cv2
    import numpy as np

    image = np.asarray(rgb)
    if (
        image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or min(image.shape[:2]) < 1
        or not image.flags.c_contiguous
    ):
        raise ValueError("rgb must be a contiguous HWC uint8 RGB image")
    height, width = image.shape[:2]
    long_edge = max(height, width)
    if long_edge <= SAM_ANALYSIS_LONG_EDGE:
        return image
    scale = SAM_ANALYSIS_LONG_EDGE / long_edge
    output_width = max(1, int(round(width * scale)))
    output_height = max(1, int(round(height * scale)))
    return np.ascontiguousarray(
        cv2.resize(
            image,
            (output_width, output_height),
            interpolation=cv2.INTER_CUBIC,
        )
    )


def segment_subject_points(
    analysis_rgb: Any,
    points: Sequence[tuple[float, float]],
    model: Sam2TinyOnnx,
    *,
    cancel_check: Callable[[], None] | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> tuple[list[Any], tuple[float, ...], dict[str, float], int]:
    """Run SAM once per positive point while reusing the image encoder features."""
    import numpy as np

    image = np.asarray(analysis_rgb)
    selected_points = tuple(dict.fromkeys(tuple(point) for point in points))
    if not selected_points:
        raise ValueError("subject protection requires at least one positive point")
    height, width = image.shape[:2]
    prepared = prepare_sam_input(image)
    masks: list[Any] = []
    scores: list[float] = []
    encoder_seconds = 0.0
    decoder_seconds = 0.0
    cache_hits = 0
    for index, point in enumerate(selected_points):
        if cancel_check is not None:
            cancel_check()
        coords = normalized_points_to_sam([point])
        prediction = model.predict(
            prepared,
            coords,
            np.ones(1, dtype=np.int32),
            cancel_check=cancel_check,
        )
        selected = select_subject_mask(
            prediction.mask_logits,
            prediction.scores,
            point,
            width,
            height,
        )
        # Retain binary masks as one byte per pixel between model stages. Their
        # values are 0/1; converting to ViTMatte float input never rescales them.
        masks.append(np.ascontiguousarray(selected.mask > 0.5, dtype=np.uint8))
        scores.append(selected.score)
        encoder_seconds += prediction.encoder_seconds
        decoder_seconds += prediction.decoder_seconds
        cache_hits += int(prediction.cache_hit)
        if progress_callback is not None:
            progress_callback((index + 1) / len(selected_points))
    return (
        masks,
        tuple(scores),
        {"encoder": encoder_seconds, "decoder": decoder_seconds},
        cache_hits,
    )


def refine_subject_masks(
    analysis_rgb: Any,
    masks: Sequence[Any],
    points: Sequence[tuple[float, float]],
    model: ViTMatteOnnx,
    *,
    tile_size: int = 1024,
    cancel_check: Callable[[], None] | None = None,
    progress_callback: Callable[[float], None] | None = None,
    cache_hits: int = 0,
) -> SubjectAlphaResult:
    """Refine each SAM selection independently, then max-union its alpha."""
    import numpy as np

    image = np.asarray(analysis_rgb)
    height, width = image.shape[:2]
    if len(masks) != len(points) or not masks:
        raise ValueError("each positive subject point must have one SAM mask")
    if tile_size not in (512, 768, 1024):
        raise ValueError("matte tile size must be 512, 768, or 1024")
    config = MatteTileConfig(tile_size=tile_size)
    alpha_union = np.zeros((height, width), dtype=np.float32)
    timings = {"trimap": 0.0, "matte_input": 0.0, "matte_inference": 0.0}
    point_scores: list[float] = []
    for index, (mask_u8, point) in enumerate(zip(masks, points, strict=True)):
        if cancel_check is not None:
            cancel_check()
        mask = np.ascontiguousarray(mask_u8, dtype=np.float32)
        if (
            not np.isfinite(mask).all()
            or float(mask.min()) < 0.0
            or float(mask.max()) > 1.0
        ):
            raise ValueError("SAM mask must be finite and lie in [0, 1]")
        stage_started = time.perf_counter()
        trimap = build_trimap(mask)
        timings["trimap"] += time.perf_counter() - stage_started
        del mask

        stage_started = time.perf_counter()
        pixel_values = prepare_vitmatte_input(image, trimap)
        timings["matte_input"] += time.perf_counter() - stage_started
        del trimap

        def infer_tile(tile: Any) -> Any:
            if cancel_check is not None:
                cancel_check()
            started = time.perf_counter()
            prediction = model.infer(tile)
            timings["matte_inference"] += time.perf_counter() - started
            if cancel_check is not None:
                cancel_check()
            return prediction

        padded_alpha = refine_matte_tiled(
            pixel_values,
            infer_tile,
            config=config,
            cancel_check=cancel_check,
        )
        alpha = np.ascontiguousarray(padded_alpha[0, 0, :height, :width])
        point_x = min(int(point[0] * width), width - 1)
        point_y = min(int(point[1] * height), height - 1)
        if float(alpha[point_y, point_x]) < 0.5:
            raise ValueError("refined subject matte does not cover its positive point")
        np.maximum(alpha_union, alpha, out=alpha_union)
        point_scores.append(float(alpha[point_y, point_x]))
        del pixel_values, padded_alpha, alpha
        if progress_callback is not None:
            progress_callback((index + 1) / len(masks))
    if cancel_check is not None:
        cancel_check()
    return SubjectAlphaResult(
        alpha=alpha_union,
        stage_seconds=timings,
        point_scores=tuple(point_scores),
        cache_hits=cache_hits,
    )


def _pool2(image: Any) -> Any:
    import numpy as np

    height, width = image.shape
    padded = np.pad(image, ((0, height % 2), (0, width % 2)))
    return padded.reshape(padded.shape[0] // 2, 2, padded.shape[1] // 2, 2).sum(
        axis=(1, 3), dtype=np.float32
    )


def _upsample_half(image: Any, height: int, width: int) -> Any:
    import numpy as np

    source_height, source_width = image.shape
    source_x = np.clip(
        (np.arange(width, dtype=np.float32) + np.float32(0.5)) * np.float32(0.5)
        - np.float32(0.5),
        0,
        source_width - 1,
    )
    source_y = np.clip(
        (np.arange(height, dtype=np.float32) + np.float32(0.5)) * np.float32(0.5)
        - np.float32(0.5),
        0,
        source_height - 1,
    )
    x0 = np.floor(source_x).astype(np.intp)
    y0 = np.floor(source_y).astype(np.intp)
    x1 = np.minimum(x0 + 1, source_width - 1)
    y1 = np.minimum(y0 + 1, source_height - 1)
    wx = (source_x - x0).astype(np.float32)
    wy = (source_y - y0).astype(np.float32)
    upper = image[y0[:, None], x0[None, :]] * (1 - wx)[None, :]
    upper += image[y0[:, None], x1[None, :]] * wx[None, :]
    lower = image[y1[:, None], x0[None, :]] * (1 - wx)[None, :]
    lower += image[y1[:, None], x1[None, :]] * wx[None, :]
    upper *= (1 - wy)[:, None]
    lower *= wy[:, None]
    upper += lower
    return upper


def _pull_push_channel(frame: Any, weights: Sequence[Any], cancel_check: Any) -> None:
    import numpy as np

    values: list[Any] = []
    for level in range(1, len(weights)):
        if cancel_check is not None:
            cancel_check()
        previous_weight = np.minimum(weights[level - 1], np.float32(1.0))
        previous_value = frame if level == 1 else values[-1]
        summed = _pool2(previous_weight * previous_value)
        current = np.zeros_like(weights[level])
        np.divide(summed, weights[level], out=current, where=weights[level] > 0)
        values.append(current)

    for level in range(len(values) - 2, 0, -1):
        if cancel_check is not None:
            cancel_check()
        current = values[level - 1]
        weight = np.minimum(weights[level], np.float32(1.0))
        upsampled = _upsample_half(values[level], *current.shape)
        missing = weight < 1.0
        current[missing] = (
            weight[missing] * current[missing]
            + (1.0 - weight[missing]) * upsampled[missing]
        )
        del upsampled

    fine = _upsample_half(values[0], *frame.shape)
    weight = np.minimum(weights[0], np.float32(1.0))
    missing = weight < 1.0
    frame[missing] = (
        weight[missing] * frame[missing] + (1.0 - weight[missing]) * fine[missing]
    )


def fill_subject_background_inplace(
    image_float: Any,
    coc: Any,
    alpha: Any,
    *,
    cancel_check: Callable[[], None] | None = None,
) -> bool:
    """Pull-push fill protected pixels before ARNet; preserves known background."""
    import numpy as np

    image = np.asarray(image_float)
    defocus = np.asarray(coc)
    matte = np.asarray(alpha)
    if (
        image.dtype != np.float32
        or image.ndim != 3
        or image.shape[2] != 3
        or not image.flags.c_contiguous
        or defocus.dtype != np.float32
        or defocus.shape != image.shape[:2]
        or not defocus.flags.c_contiguous
        or matte.dtype != np.float32
        or matte.shape != image.shape[:2]
        or not matte.flags.c_contiguous
    ):
        raise ValueError(
            "image, CoC, and subject alpha must be matching float32 arrays"
        )
    if not np.isfinite(matte).all() or float(matte.min()) < 0 or float(matte.max()) > 1:
        raise ValueError("subject alpha must be finite and lie in [0, 1]")
    if cancel_check is not None:
        cancel_check()
    weight0 = np.zeros_like(matte)
    soft = (matte > 0) & (matte < np.float32(0.02))
    weight0[matte <= 0] = 1.0
    weight0[soft] = 1.0 - matte[soft] / np.float32(0.02)
    if not bool(np.any(weight0 < 1.0)) or not bool(np.any(weight0 > 0.0)):
        return False
    weights = [weight0]
    while weights[-1].shape != (1, 1):
        if cancel_check is not None:
            cancel_check()
        weights.append(_pool2(np.minimum(weights[-1], np.float32(1.0))))
    for channel in range(4):
        plane = defocus if channel == 3 else image[:, :, channel]
        _pull_push_channel(plane, weights, cancel_check)
    if cancel_check is not None:
        cancel_check()
    return True


def composite_protected_subject(
    source_rgb: Any,
    blurred_rgb: Any,
    alpha: Any,
    *,
    row_chunk_height: int = 128,
    cancel_check: Callable[[], None] | None = None,
) -> Any:
    """Composite the sharp subject over the clean background and decontaminate edges.

    Mixed matte pixels use bounded local background estimates. The correction is
    enabled only when nearby known-background samples are sufficiently smooth and
    the implied premultiplied foreground stays within the matte's feasible range.
    Opaque subject and known background endpoints remain exact.
    """
    import cv2
    import numpy as np

    source = np.asarray(source_rgb)
    blurred = np.asarray(blurred_rgb)
    matte = np.asarray(alpha)
    if (
        source.dtype != np.uint8
        or source.ndim != 3
        or source.shape[2] != 3
        or blurred.dtype != np.uint8
        or blurred.shape != source.shape
        or matte.dtype != np.float32
        or matte.shape != source.shape[:2]
        or not matte.flags.c_contiguous
    ):
        raise ValueError("RGB frames and subject alpha must have matching H/W")
    if row_chunk_height < 1:
        raise ValueError("row_chunk_height must be positive")
    if not np.isfinite(matte).all() or float(matte.min()) < 0 or float(matte.max()) > 1:
        raise ValueError("subject alpha must be finite and lie in [0, 1]")
    height, width = matte.shape
    radius = min(32, max(5, int(np.ceil(max(width, height) / 1220.0 * 5))))
    kernel = (radius * 2 + 1, radius * 2 + 1)
    result = np.empty_like(source)

    def smooth(value: Any) -> Any:
        clipped = np.clip(value, 0.0, 1.0)
        return clipped * clipped * (3.0 - 2.0 * clipped)

    for top in range(0, height, row_chunk_height):
        if cancel_check is not None:
            cancel_check()
        end = min(height, top + row_chunk_height)
        y0, y1 = max(0, top - radius), min(height, end + radius)
        local_alpha = matte[y0:y1]
        known = (local_alpha <= 0.0).astype(np.float32)
        density = cv2.boxFilter(
            known,
            ddepth=cv2.CV_32F,
            ksize=kernel,
            normalize=False,
            borderType=cv2.BORDER_CONSTANT,
        )
        target = slice(top - y0, end - y0)
        alpha_rows = matte[top:end]
        for channel in range(3):
            sharp_local = source[y0:y1, :, channel].astype(np.float32)
            blur_local = blurred[y0:y1, :, channel].astype(np.float32)
            sharp_sum = cv2.boxFilter(
                sharp_local * known,
                ddepth=cv2.CV_32F,
                ksize=kernel,
                normalize=False,
                borderType=cv2.BORDER_CONSTANT,
            )[target]
            blur_sum = cv2.boxFilter(
                blur_local * known,
                ddepth=cv2.CV_32F,
                ksize=kernel,
                normalize=False,
                borderType=cv2.BORDER_CONSTANT,
            )[target]
            square_sum = cv2.boxFilter(
                sharp_local * sharp_local * known,
                ddepth=cv2.CV_32F,
                ksize=kernel,
                normalize=False,
                borderType=cv2.BORDER_CONSTANT,
            )[target]
            count = density[target]
            mean_sharp = np.zeros_like(sharp_sum)
            mean_blur = np.zeros_like(blur_sum)
            np.divide(sharp_sum, count, out=mean_sharp, where=count > 0)
            np.divide(blur_sum, count, out=mean_blur, where=count > 0)
            variance = np.maximum(
                0.0, square_sum / np.maximum(count, 1.0) - mean_sharp**2
            )
            sigma = np.sqrt(variance)
            source_rows = source[top:end, :, channel].astype(np.float32)
            blurred_rows = blurred[top:end, :, channel].astype(np.float32)
            alpha = alpha_rows
            baseline = source_rows * alpha + blurred_rows * (1.0 - alpha)
            premultiplied = source_rows - (1.0 - alpha) * mean_sharp
            valid = (
                (alpha > 0.0)
                & (alpha < 1.0)
                & (count >= 4.0)
                & (premultiplied >= -3.0)
                & (premultiplied <= alpha * 255.0 + 3.0)
            )
            confidence = np.clip(count / 4.0, 0.0, 1.0) * smooth((32.0 - sigma) / 20.0)
            confidence *= smooth(alpha / 0.05)
            proposed = source_rows + (1.0 - alpha) * (mean_blur - mean_sharp)
            corrected = baseline + confidence * (proposed - baseline)
            corrected[~valid] = baseline[~valid]
            result[top:end, :, channel] = np.rint(np.clip(corrected, 0, 255)).astype(
                np.uint8
            )
        del density, known
    if cancel_check is not None:
        cancel_check()
    return result


__all__ = [
    "SubjectAlphaResult",
    "composite_protected_subject",
    "fill_subject_background_inplace",
    "prepare_subject_analysis_rgb",
    "refine_subject_masks",
    "segment_subject_points",
]
