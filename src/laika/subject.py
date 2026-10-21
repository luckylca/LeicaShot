"""SAM 2.1 Tiny ONNX inference with bounded, reusable image embeddings.

The adapter consumes the fixed-size, ImageNet-normalized RGB tensor used by the
pinned SAM 2.1 image encoder export. Image resize and orientation remain the caller's
responsibility so point-coordinate transforms stay explicit at the pipeline
boundary.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SAM_IMAGE_SIZE = 1024
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
