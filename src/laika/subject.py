"""SAM 2.1 Tiny ONNX inference with bounded, reusable image embeddings.

The input-preparation helper matches Laika's inspected reference contract: orient
the source image first, resize it to a square with separable triangle antialiasing,
then apply RGB ImageNet normalization. Prompt coordinates are normalized against
the oriented source image before mapping into SAM's fixed 1024-pixel frame.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

SAM_IMAGE_SIZE = 1024
SAM_ANALYSIS_LONG_EDGE = 2048
SAM_MASK_SIZE = 256
SAM_FEATURE_OUTPUTS = (
    "image_features_0",
    "image_features_1",
    "image_embeddings",
)
SAM_DECODER_INPUTS = (
    *SAM_FEATURE_OUTPUTS,
    "point_coords",
    "point_labels",
)
SAM_FEATURE_SHAPES = (
    (1, 32, 256, 256),
    (1, 64, 128, 128),
    (1, 256, 64, 64),
)
_SAM_MEAN = (0.485, 0.456, 0.406)
_SAM_STD = (0.229, 0.224, 0.225)
_RESAMPLE_WORKING_BYTES = 4 * 1024 * 1024


def _resample_plan(
    input_size: int,
    output_size: int,
    start: int,
    stop: int,
) -> tuple[Any, Any]:
    """Build a bounded triangle-antialias plan for an output-coordinate slice."""
    import numpy as np

    ratio = input_size / output_size
    support = max(1.0, ratio)
    positions = np.arange(start, stop, dtype=np.float64)
    centers = (positions + 0.5) * ratio - 0.5
    lows = np.maximum(0, np.ceil(centers - support).astype(np.int64))
    highs = np.minimum(input_size - 1, np.floor(centers + support).astype(np.int64))
    tap_count = int(np.max(highs - lows + 1))
    indices = np.zeros((len(positions), tap_count), dtype=np.int64)
    weights = np.zeros((len(positions), tap_count), dtype=np.float64)
    for row, (center, low, high) in enumerate(zip(centers, lows, highs, strict=True)):
        taps = np.arange(low, high + 1, dtype=np.int64)
        tap_weights = np.maximum(0.0, 1.0 - np.abs(taps - center) / support)
        tap_weights /= np.sum(tap_weights)
        indices[row, : len(taps)] = taps
        weights[row, : len(taps)] = tap_weights
    return indices, weights


def _resample_rgb_axis(
    image: Any,
    output_size: int,
    *,
    axis: int,
    working_bytes: int = _RESAMPLE_WORKING_BYTES,
) -> Any:
    """Resize HWC float RGB with the reference's separable triangle filter."""
    import numpy as np

    if axis == 1:
        input_size = image.shape[1]
        ratio = input_size / output_size
        max_taps = max(1, int(np.ceil(2.0 * max(1.0, ratio))) + 1)
        pixels_per_chunk = max(
            1, working_bytes // (max_taps * 3 * np.dtype(np.float64).itemsize)
        )
        output = np.empty((image.shape[0], output_size, 3), dtype=np.float32)
        for start in range(0, output_size, pixels_per_chunk):
            stop = min(output_size, start + pixels_per_chunk)
            indices, weights = _resample_plan(input_size, output_size, start, stop)
            for y in range(image.shape[0]):
                samples = image[y, indices, :]
                if samples.dtype == np.uint8:
                    samples = samples.astype(np.float32) / np.float32(255.0)
                samples = samples.astype(np.float64, copy=False)
                output[y, start:stop, :] = np.einsum(
                    "ot,otc->oc", weights, samples, optimize=False
                )
        return output

    if axis == 0:
        input_size = image.shape[0]
        ratio = input_size / output_size
        max_taps = max(1, int(np.ceil(2.0 * max(1.0, ratio))) + 1)
        pixels_per_chunk = max(
            1, working_bytes // (max_taps * 3 * np.dtype(np.float64).itemsize)
        )
        output = np.empty((output_size, image.shape[1], 3), dtype=np.float32)
        for y in range(output_size):
            indices, weights = _resample_plan(input_size, output_size, y, y + 1)
            row_indices = indices[0]
            row_weights = weights[0]
            active = row_weights > 0
            row_indices = row_indices[active]
            row_weights = row_weights[active]
            for start in range(0, image.shape[1], pixels_per_chunk):
                stop = min(image.shape[1], start + pixels_per_chunk)
                samples = image[row_indices, start:stop, :].astype(
                    np.float64, copy=False
                )
                output[y, start:stop, :] = np.einsum(
                    "t,twc->wc", row_weights, samples, optimize=False
                )
        return output

    raise ValueError("axis must be 0 (vertical) or 1 (horizontal)")


def _prepare_sam_rgb(rgb_image: Any, output_size: int) -> Any:
    """Resize HWC RGB and return a normalized NCHW model tensor."""
    import numpy as np

    if not isinstance(rgb_image, np.ndarray):
        raise ValueError("rgb_image must be a NumPy RGB array")
    if (
        rgb_image.ndim != 3
        or rgb_image.shape[2] != 3
        or rgb_image.shape[0] < 1
        or rgb_image.shape[1] < 1
    ):
        raise ValueError("rgb_image must have shape [height, width, 3]")
    if max(rgb_image.shape[:2]) > SAM_ANALYSIS_LONG_EDGE:
        raise ValueError(
            "rgb_image must be the subject-analysis raster with a long edge "
            "at most 2048"
        )
    if not rgb_image.flags.c_contiguous:
        raise ValueError("rgb_image must be C-contiguous to avoid a hidden copy")
    if rgb_image.dtype not in (np.uint8, np.float32):
        raise ValueError("rgb_image must use uint8 or float32 RGB values")
    if rgb_image.dtype == np.float32 and (
        not np.isfinite(rgb_image).all()
        or float(np.min(rgb_image)) < 0.0
        or float(np.max(rgb_image)) > 1.0
    ):
        raise ValueError("float32 rgb_image values must be finite and in [0, 1]")

    horizontal = _resample_rgb_axis(rgb_image, output_size, axis=1)
    resized = _resample_rgb_axis(horizontal, output_size, axis=0)
    tensor = np.ascontiguousarray(resized.transpose(2, 0, 1)[None, ...])
    for channel, (mean, std) in enumerate(zip(_SAM_MEAN, _SAM_STD, strict=True)):
        np.subtract(tensor[0, channel], np.float32(mean), out=tensor[0, channel])
        np.divide(tensor[0, channel], np.float32(std), out=tensor[0, channel])
    return tensor


def prepare_sam_input(rgb_image: Any) -> Any:
    """Prepare an oriented RGB image for SAM as contiguous float32 NCHW.

    ``rgb_image`` is the subject-analysis HWC RGB raster (long edge at most 2048),
    in `uint8` or `float32` [0, 1]. Apply EXIF orientation and any explicit
    subject-analysis resize before calling this function. The separable triangle filter
    performs antialiasing during both downsampling and upsampling and follows the
    reference's width-to-1024 pass followed by its height-to-1024 pass.
    """
    return _prepare_sam_rgb(rgb_image, SAM_IMAGE_SIZE)


def normalized_points_to_sam(point_coords: Any) -> Any:
    """Map oriented-image normalized ``(x, y)`` prompts into SAM's pixel frame."""
    import numpy as np

    coords = np.asarray(point_coords, dtype=np.float32)
    if coords.ndim == 2:
        coords = coords[None, ...]
    if coords.ndim != 3 or coords.shape[0] != 1 or coords.shape[2] != 2:
        raise ValueError("point_coords must have shape [P, 2] or [1, P, 2]")
    if coords.shape[1] < 1:
        raise ValueError("point_coords must contain at least one point")
    if not np.isfinite(coords).all():
        raise ValueError(
            "normalized prompt coordinates must be finite and lie in [0, 1)"
        )
    if np.any(coords < 0.0) or np.any(coords >= 1.0):
        raise ValueError("normalized prompt coordinates must lie in [0, 1)")
    return np.ascontiguousarray(coords * np.float32(SAM_IMAGE_SIZE))


class SubjectInferenceError(RuntimeError):
    """The SAM 2.1 ONNX graphs or their inputs violate the adapter contract."""


@dataclass(frozen=True)
class SubjectPrediction:
    """Low-resolution SAM mask logits and quality scores for one prompt set."""

    mask_logits: Any
    scores: Any
    encoder_seconds: float
    decoder_seconds: float
    cache_hit: bool


@dataclass(frozen=True)
class SelectedSubjectMask:
    """Binary subject mask restored to the analysis raster dimensions."""

    mask: Any
    candidate_index: int
    score: float


def _sample_low_res_mask(mask: Any, point_xy: Any, width: int, height: int) -> Any:
    """Match the reference's half-pixel bilinear sampling at the clicked pixel."""
    import numpy as np

    point_x = min(int(float(point_xy[0]) * width), width - 1)
    point_y = min(int(float(point_xy[1]) * height), height - 1)
    source_x = (np.float32(point_x) + np.float32(0.5)) * np.float32(
        mask.shape[1]
    ) / np.float32(width) - np.float32(0.5)
    source_y = (np.float32(point_y) + np.float32(0.5)) * np.float32(
        mask.shape[0]
    ) / np.float32(height) - np.float32(0.5)
    ix, iy = int(np.floor(source_x)), int(np.floor(source_y))
    fx, fy = source_x - ix, source_y - iy
    value = np.float32(0.0)
    for dy in range(2):
        y = min(max(iy + dy, 0), mask.shape[0] - 1)
        wy = np.float32(1.0) - fy if dy == 0 else fy
        for dx in range(2):
            x = min(max(ix + dx, 0), mask.shape[1] - 1)
            wx = np.float32(1.0) - fx if dx == 0 else fx
            value += mask[y, x] * wy * wx
    return value


def _resize_mask_bilinear(mask: Any, width: int, height: int) -> Any:
    """Resize one mask using half-pixel bilinear sampling in bounded row bands."""
    import numpy as np

    input_height, input_width = mask.shape
    output = np.empty((height, width), dtype=np.float32)
    source_x = (np.arange(width, dtype=np.float32) + np.float32(0.5)) * np.float32(
        input_width
    ) / np.float32(width) - np.float32(0.5)
    ix = np.floor(source_x).astype(np.int64)
    fx = source_x - ix.astype(np.float32)
    x0 = np.clip(ix, 0, input_width - 1)
    x1 = np.clip(ix + 1, 0, input_width - 1)
    wx0, wx1 = np.float32(1.0) - fx, fx
    chunk_height = max(1, min(64, height))

    for start in range(0, height, chunk_height):
        stop = min(height, start + chunk_height)
        source_y = (
            np.arange(start, stop, dtype=np.float32) + np.float32(0.5)
        ) * np.float32(input_height) / np.float32(height) - np.float32(0.5)
        iy = np.floor(source_y).astype(np.int64)
        fy = source_y - iy.astype(np.float32)
        y0 = np.clip(iy, 0, input_height - 1)
        y1 = np.clip(iy + 1, 0, input_height - 1)
        wy0, wy1 = (np.float32(1.0) - fy)[:, None], fy[:, None]

        values = mask[y0[:, None], x0[None, :]] * wy0
        values *= wx0[None, :]
        values += mask[y0[:, None], x1[None, :]] * wy0 * wx1[None, :]
        values += mask[y1[:, None], x0[None, :]] * wy1 * wx0[None, :]
        values += mask[y1[:, None], x1[None, :]] * wy1 * wx1[None, :]
        output[start:stop] = values
    return output


def select_subject_mask(
    mask_logits: Any,
    scores: Any,
    point_xy: Any,
    width: int,
    height: int,
) -> SelectedSubjectMask:
    """Choose the highest-score SAM candidate covering a positive prompt.

    The candidate's low-resolution logits are bilinearly sampled at the clicked
    pixel, then resized to the subject-analysis raster and thresholded at zero,
    matching the inspected reference behavior.
    """
    import numpy as np

    logits = np.asarray(mask_logits)
    candidates = np.asarray(scores)
    if logits.shape != (1, 3, SAM_MASK_SIZE, SAM_MASK_SIZE):
        raise ValueError("mask_logits must have shape [1, 3, 256, 256]")
    if candidates.shape not in ((1, 3), (3,)):
        raise ValueError("scores must have shape [3] or [1, 3]")
    if logits.dtype != np.float32 or candidates.dtype != np.float32:
        raise ValueError("SAM mask logits and scores must be float32")
    if not np.isfinite(logits).all() or not np.isfinite(candidates).all():
        raise ValueError("SAM mask logits and scores must be finite")
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, (int, np.integer))
        or not isinstance(height, (int, np.integer))
        or width < 1
        or height < 1
        or max(width, height) > SAM_ANALYSIS_LONG_EDGE
    ):
        raise ValueError("analysis dimensions must be within the 2048-pixel budget")
    normalized = np.asarray(point_xy, dtype=np.float32)
    if normalized.shape != (2,) or not np.isfinite(normalized).all():
        raise ValueError("point_xy must contain one finite normalized (x, y) point")
    if np.any(normalized < 0.0) or np.any(normalized >= 1.0):
        raise ValueError("point_xy must lie in the normalized range [0, 1)")

    ordered_candidates = np.argsort(-candidates.reshape(3), kind="stable")
    for candidate_index in ordered_candidates:
        low_res = logits[0, candidate_index]
        if _sample_low_res_mask(low_res, normalized, width, height) <= 0.0:
            continue
        resized = _resize_mask_bilinear(low_res, width, height)
        np.greater(resized, np.float32(0.0), out=resized)
        point_x = min(int(float(normalized[0]) * width), width - 1)
        point_y = min(int(float(normalized[1]) * height), height - 1)
        if resized[point_y, point_x] != 1.0:
            raise ValueError("resized SAM mask does not cover the positive prompt")
        return SelectedSubjectMask(
            mask=resized,
            candidate_index=int(candidate_index),
            score=float(candidates.reshape(3)[candidate_index]),
        )
    raise ValueError("no SAM candidate contains the positive prompt")


def subject_trimap_radius(width: int, height: int) -> int:
    """Scale the reference's 5-pixel unknown-band radius to analysis dimensions."""
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, Integral)
        or not isinstance(height, Integral)
        or width < 1
        or height < 1
        or max(width, height) > SAM_ANALYSIS_LONG_EDGE
    ):
        raise ValueError("analysis dimensions must be within the 2048-pixel budget")
    return max(5, round(5.0 * max(width, height) / 1220.0))


def build_subject_trimap(mask: Any, radius: int | None = None) -> Any:
    """Build the reference's 0 / 128 / 255 trimap with an elliptical neighborhood.

    A neighborhood is foreground only when all in-image ellipse pixels are
    foreground, background only when none are foreground, and unknown otherwise.
    Pixels outside the image are excluded from the neighborhood, matching the
    reference's neutral border handling. Row prefix sums keep the work linear in
    image width for each ellipse row instead of scanning a disk per pixel.
    """
    import numpy as np

    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError("mask must be a 2D NumPy array")
    height, width = mask.shape
    if height < 1 or width < 1 or max(width, height) > SAM_ANALYSIS_LONG_EDGE:
        raise ValueError("mask dimensions must be within the 2048-pixel budget")
    if not mask.flags.c_contiguous:
        raise ValueError("mask must be C-contiguous")
    if mask.dtype != np.float32:
        raise ValueError("mask values must use float32")
    if not np.isfinite(mask).all() or np.any(mask < 0.0) or np.any(mask > 1.0):
        raise ValueError("mask values must be finite and in [0, 1]")
    if radius is None:
        radius = subject_trimap_radius(width, height)
    if isinstance(radius, bool) or not isinstance(radius, (int, np.integer)):
        raise ValueError("radius must be a positive integer")
    if radius < 1:
        raise ValueError("radius must be a positive integer")

    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    row_radii = np.rint(
        radius * np.sqrt(np.maximum(0.0, 1.0 - (offsets / radius) ** 2))
    ).astype(np.int64)
    prefix = np.empty((height, width + 1), dtype=np.int32)
    prefix[:, 0] = 0
    np.cumsum(mask > 0.5, axis=1, dtype=np.int32, out=prefix[:, 1:])

    output = np.full(
        (height, width), np.float32(128.0) / np.float32(255.0), dtype=np.float32
    )
    x = np.arange(width, dtype=np.int64)[None, :]
    chunk_height = min(64, height)
    for start in range(0, height, chunk_height):
        stop = min(height, start + chunk_height)
        rows = np.arange(start, stop, dtype=np.int64)
        inside = np.zeros((len(rows), width), dtype=np.int32)
        cells = np.zeros_like(inside)
        for offset, row_radius in zip(offsets.astype(np.int64), row_radii, strict=True):
            source_y = rows + offset
            valid = (source_y >= 0) & (source_y < height)
            source_y = np.clip(source_y, 0, height - 1)
            left = np.maximum(0, x - row_radius)
            right = np.minimum(width, x + row_radius + 1)
            row_count = (
                prefix[source_y[:, None], right] - prefix[source_y[:, None], left]
            )
            row_count *= valid[:, None]
            inside += row_count
            cells += (right - left) * valid[:, None]
        core = inside == cells
        background = inside == 0
        output[start:stop][core] = 1.0
        output[start:stop][background] = 0.0
    return output


class ImageEmbeddingCache:
    """Byte-bounded LRU cache for immutable SAM image encoder outputs."""

    def __init__(self, max_bytes: int = 96 * 1024 * 1024) -> None:
        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        self.max_bytes = max_bytes
        self._entries: OrderedDict[str, tuple[Any, ...]] = OrderedDict()
        self._entry_bytes: dict[str, int] = {}
        self._used_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self._lock = threading.RLock()

    @property
    def used_bytes(self) -> int:
        with self._lock:
            return self._used_bytes

    @property
    def entry_count(self) -> int:
        with self._lock:
            return len(self._entries)

    def get(self, key: str) -> tuple[Any, ...] | None:
        with self._lock:
            value = self._entries.get(key)
            if value is None:
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return value

    def put(self, key: str, features: tuple[Any, ...]) -> None:
        with self._lock:
            if self.max_bytes == 0:
                return
            if key in self._entries:
                self._remove(key)

            immutable_features = tuple(features)
            for feature in immutable_features:
                feature.setflags(write=False)
            entry_bytes = sum(int(feature.nbytes) for feature in immutable_features)
            if entry_bytes > self.max_bytes:
                return

            while self._used_bytes + entry_bytes > self.max_bytes:
                oldest_key = next(iter(self._entries))
                self._remove(oldest_key)
                self.evictions += 1

            self._entries[key] = immutable_features
            self._entry_bytes[key] = entry_bytes
            self._used_bytes += entry_bytes

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._remove(key)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._entry_bytes.clear()
            self._used_bytes = 0

    def _remove(self, key: str) -> None:
        if key in self._entries:
            del self._entries[key]
            self._used_bytes -= self._entry_bytes.pop(key)


def _image_fingerprint(image_tensor: Any) -> str:
    import numpy as np

    if not isinstance(image_tensor, np.ndarray):
        raise ValueError("image_tensor must be a NumPy array")
    if image_tensor.dtype != np.float32 or image_tensor.shape != (
        1,
        3,
        SAM_IMAGE_SIZE,
        SAM_IMAGE_SIZE,
    ):
        raise ValueError("image_tensor must be float32 with shape [1, 3, 1024, 1024]")
    if not image_tensor.flags.c_contiguous:
        raise ValueError("image_tensor must be C-contiguous to avoid a hidden copy")
    if not np.isfinite(image_tensor).all():
        raise ValueError("image_tensor must contain only finite values")
    digest = hashlib.sha256()
    digest.update(str(image_tensor.shape).encode("ascii"))
    digest.update(image_tensor.dtype.str.encode("ascii"))
    digest.update(memoryview(image_tensor).cast("B"))
    return digest.hexdigest()


def _onnx_session(path: str | Path, intra_op_num_threads: int) -> Any:
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise SubjectInferenceError(
            "SAM inference requires the optional onnxruntime dependency"
        ) from exc

    options = ort.SessionOptions()
    options.intra_op_num_threads = intra_op_num_threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


class Sam2TinyOnnx:
    """Run the split SAM 2.1 Tiny image encoder and prompt decoder on CPU.

    The encoder graph must accept ``image`` and return the three fixed SAM 2.1
    feature maps. The decoder graph follows the upstream split graph contract:
    image features plus point prompts in the 1024-pixel model frame. Embeddings
    are keyed by the exact input tensor bytes and held only within this instance.
    Calls on one instance are serialized so concurrent requests cannot cause
    duplicate encoder passes or exceed its configured cache budget.
    """

    def __init__(
        self,
        image_encoder_path: str | Path,
        image_decoder_path: str | Path,
        *,
        cache_max_bytes: int = 96 * 1024 * 1024,
        intra_op_num_threads: int = 1,
        session_factory: Callable[[str | Path, int], Any] | None = None,
    ) -> None:
        if intra_op_num_threads < 1:
            raise ValueError("intra_op_num_threads must be positive")
        factory = session_factory or _onnx_session
        self._encoder = factory(image_encoder_path, intra_op_num_threads)
        self._decoder = factory(image_decoder_path, intra_op_num_threads)
        self._cache = ImageEmbeddingCache(cache_max_bytes)
        self._inference_lock = threading.RLock()
        self._validate_graph_contracts()

    @property
    def cache(self) -> ImageEmbeddingCache:
        return self._cache

    def predict(
        self,
        image_tensor: Any,
        point_coords: Any,
        point_labels: Any,
    ) -> SubjectPrediction:
        """Predict three candidate masks using model-frame ``(x, y)`` prompts.

        Coordinates are pixels in the already-prepared 1024×1024 SAM input.
        Positive labels select foreground, zero labels select background, and
        ``-1`` pads unused point positions. The caller may call this repeatedly
        with the same image tensor; the encoder outputs are then reused.
        """
        with self._inference_lock:
            return self._predict_locked(image_tensor, point_coords, point_labels)

    def _predict_locked(
        self,
        image_tensor: Any,
        point_coords: Any,
        point_labels: Any,
    ) -> SubjectPrediction:
        import numpy as np

        fingerprint = _image_fingerprint(image_tensor)
        coords = np.asarray(point_coords, dtype=np.float32)
        labels = np.asarray(point_labels, dtype=np.int32)
        if coords.ndim == 2:
            coords = coords[None, ...]
        if labels.ndim == 1:
            labels = labels[None, ...]
        if coords.ndim != 3 or coords.shape[0] != 1 or coords.shape[2] != 2:
            raise ValueError("point_coords must have shape [P, 2] or [1, P, 2]")
        if labels.shape != coords.shape[:2] or coords.shape[1] < 1:
            raise ValueError("point_labels must match the point count")
        if (
            not np.isfinite(coords).all()
            or np.any(coords < 0)
            or np.any(coords >= SAM_IMAGE_SIZE)
        ):
            raise ValueError("point coordinates must lie within the 1024×1024 frame")
        if not np.isin(labels, (-1, 0, 1)).all():
            raise ValueError("point labels must be -1, 0, or 1")

        features = self._cache.get(fingerprint)
        cache_hit = features is not None
        if features is None:
            encoder_started = time.perf_counter()
            outputs = self._encoder.run(
                list(SAM_FEATURE_OUTPUTS), {"image": image_tensor}
            )
            features = tuple(outputs)
            self._validate_features(features)
            self._cache.put(fingerprint, features)
            encoder_seconds = time.perf_counter() - encoder_started
        else:
            encoder_seconds = 0.0

        coords = np.ascontiguousarray(coords, dtype=np.float32)
        labels = np.ascontiguousarray(labels, dtype=np.int32)
        decoder_inputs = dict(zip(SAM_FEATURE_OUTPUTS, features, strict=True))
        decoder_inputs.update(
            {
                "point_coords": coords,
                "point_labels": labels,
            }
        )
        decoder_started = time.perf_counter()
        logits, scores = self._decoder.run(
            ["low_res_masks", "iou_predictions"], decoder_inputs
        )
        decoder_seconds = time.perf_counter() - decoder_started
        self._validate_prediction(logits, scores)
        return SubjectPrediction(
            mask_logits=logits,
            scores=scores,
            encoder_seconds=encoder_seconds,
            decoder_seconds=decoder_seconds,
            cache_hit=cache_hit,
        )

    def invalidate(self, image_tensor: Any) -> None:
        with self._inference_lock:
            self._cache.invalidate(_image_fingerprint(image_tensor))

    def close(self) -> None:
        with self._inference_lock:
            self._cache.clear()
            self._encoder = None
            self._decoder = None

    def __enter__(self) -> Sam2TinyOnnx:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _validate_graph_contracts(self) -> None:
        encoder_inputs = {item.name for item in self._encoder.get_inputs()}
        encoder_outputs = {item.name for item in self._encoder.get_outputs()}
        decoder_inputs = {item.name for item in self._decoder.get_inputs()}
        decoder_outputs = {item.name for item in self._decoder.get_outputs()}
        if encoder_inputs != {"image"} or not set(SAM_FEATURE_OUTPUTS).issubset(
            encoder_outputs
        ):
            raise SubjectInferenceError("SAM image encoder graph contract is invalid")
        if not set(SAM_DECODER_INPUTS).issubset(decoder_inputs) or not {
            "low_res_masks",
            "iou_predictions",
        }.issubset(decoder_outputs):
            raise SubjectInferenceError("SAM image decoder graph contract is invalid")

    @staticmethod
    def _validate_features(features: tuple[Any, ...]) -> None:
        import numpy as np

        if len(features) != len(SAM_FEATURE_SHAPES):
            raise SubjectInferenceError("SAM image encoder returned the wrong outputs")
        for feature, expected_shape in zip(features, SAM_FEATURE_SHAPES, strict=True):
            if not isinstance(feature, np.ndarray) or feature.shape != expected_shape:
                raise SubjectInferenceError(
                    f"SAM image feature shape mismatch: expected {expected_shape}"
                )
            if feature.dtype != np.float32 or not np.isfinite(feature).all():
                raise SubjectInferenceError("SAM image features must be finite float32")

    @staticmethod
    def _validate_prediction(logits: Any, scores: Any) -> None:
        import numpy as np

        if not isinstance(logits, np.ndarray) or logits.shape != (1, 3, 256, 256):
            raise SubjectInferenceError(
                "SAM decoder returned invalid low-resolution masks"
            )
        if not isinstance(scores, np.ndarray) or scores.shape != (1, 3):
            raise SubjectInferenceError("SAM decoder returned invalid mask scores")
        if logits.dtype != np.float32 or scores.dtype != np.float32:
            raise SubjectInferenceError("SAM mask logits and scores must be float32")
        if not np.isfinite(logits).all() or not np.isfinite(scores).all():
            raise SubjectInferenceError("SAM decoder returned non-finite outputs")
