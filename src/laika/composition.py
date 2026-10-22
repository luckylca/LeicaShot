"""GAIC crop proposals and explicit crop application for Laika."""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

GAIC_MODEL_ID = "gaic-mobilenetv2-reddim16"
GAIC_SOURCE_CHECKPOINT_SHA256 = (
    "eb1d5eed2710c79d3270add448111b9e6fa2475e246694c0ad995a6efb299e8c"
)
GAIC_SOURCE_CODE_REVISION = "877d6a325943c0a098324dc6d4a975ff4097c3f6"
GAIC_ANALYSIS_LONG_EDGE = 1220
GAIC_INPUTS = ("image", "boxes")
GAIC_OUTPUT = "scores"


class CompositionInferenceError(RuntimeError):
    """A GAIC model or crop scoring request failed validation or inference."""


@dataclass(frozen=True)
class CompositionCrop:
    """Normalized half-open crop coordinates in EXIF-oriented image space."""

    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        coords = (self.left, self.top, self.right, self.bottom)
        if (
            not all(math.isfinite(value) for value in coords)
            or self.left < 0.0
            or self.top < 0.0
            or self.right > 1.0
            or self.bottom > 1.0
            or self.right <= self.left
            or self.bottom <= self.top
        ):
            raise ValueError("crop coordinates must be normalized and nonempty")

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def area(self) -> float:
        return self.width * self.height

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.left, self.top, self.right, self.bottom)


@dataclass(frozen=True)
class CompositionSuggestion:
    crop: CompositionCrop
    score: float
    improvement_over_original: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "crop": asdict(self.crop),
            "score": self.score,
            "improvement_over_original": self.improvement_over_original,
        }


@dataclass(frozen=True)
class CompositionResult:
    width: int
    height: int
    analysis_width: int
    analysis_height: int
    candidate_count: int
    suggestions: tuple[CompositionSuggestion, ...]
    model_sha256: str
    runtime_version: str
    provider: str
    stage_seconds: dict[str, float]
    total_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "analysis_width": self.analysis_width,
            "analysis_height": self.analysis_height,
            "candidate_count": self.candidate_count,
            "suggestions": [item.as_dict() for item in self.suggestions],
            "model_sha256": self.model_sha256,
            "runtime_version": self.runtime_version,
            "provider": self.provider,
            "stage_seconds": self.stage_seconds,
            "total_seconds": self.total_seconds,
        }


def _round_pixel(value: float) -> int:
    """Match Kotlin roundToInt for nonnegative crop edges (half rounds up)."""
    return int(math.floor(value + 0.5))


def crop_pixel_box(
    crop: CompositionCrop, width: int, height: int
) -> tuple[int, int, int, int]:
    if min(width, height) < 1:
        raise ValueError("image dimensions must be positive")
    left = min(width - 1, max(0, _round_pixel(crop.left * width)))
    top = min(height - 1, max(0, _round_pixel(crop.top * height)))
    right = min(width, max(left + 1, _round_pixel(crop.right * width)))
    bottom = min(height, max(top + 1, _round_pixel(crop.bottom * height)))
    return left, top, right, bottom


def snap_crop(crop: CompositionCrop, width: int, height: int) -> CompositionCrop:
    left, top, right, bottom = crop_pixel_box(crop, width, height)
    return CompositionCrop(
        left / width,
        top / height,
        right / width,
        bottom / height,
    )


def crop_iou(first: CompositionCrop, second: CompositionCrop) -> float:
    width = max(0.0, min(first.right, second.right) - max(first.left, second.left))
    height = max(0.0, min(first.bottom, second.bottom) - max(first.top, second.top))
    intersection = width * height
    return intersection / (first.area + second.area - intersection)


def _place_crop(
    crop_width: float, crop_height: float, left: float, top: float
) -> CompositionCrop:
    left = min(max(left, 0.0), 1.0 - crop_width)
    top = min(max(top, 0.0), 1.0 - crop_height)
    return CompositionCrop(left, top, left + crop_width, top + crop_height)


def _fit_crop(width: int, height: int, ratio: float, scale: float) -> CompositionCrop:
    if width < 1 or height < 1 or not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("image dimensions and crop aspect ratio must be positive")
    if not 0.0 < scale <= 1.0:
        raise ValueError("crop scale must be in (0, 1]")
    relative_ratio = ratio * height / width
    crop_width = min(1.0, relative_ratio) * scale
    crop_height = min(1.0, 1.0 / relative_ratio) * scale
    return _place_crop(
        crop_width, crop_height, 0.5 - crop_width / 2, 0.5 - crop_height / 2
    )


def enumerate_candidate_crops(
    width: int,
    height: int,
    *,
    target_long_edge: int | None = None,
) -> tuple[CompositionCrop, ...]:
    """Generate the reference grid-anchor crop proposals in deterministic order."""
    if min(width, height) < 1:
        raise ValueError("image dimensions must be positive")
    if target_long_edge is not None and target_long_edge < 1:
        raise ValueError("target_long_edge must be positive")
    full = CompositionCrop(0.0, 0.0, 1.0, 1.0)
    ratios = tuple(
        dict.fromkeys(
            (
                width / height,
                1.0,
                4.0 / 3.0,
                1.5,
                16.0 / 9.0,
                3.0 / 4.0,
                2.0 / 3.0,
                9.0 / 16.0,
            )
        )
    )
    proposals: dict[CompositionCrop, None] = {full: None}
    for ratio in ratios:
        for scale in (1.0, 0.85, 0.70, 0.55):
            base = _fit_crop(width, height, ratio, scale)
            for x_index in range(5):
                for y_index in range(5):
                    crop = _place_crop(
                        base.width,
                        base.height,
                        (1.0 - base.width) * x_index / 4.0,
                        (1.0 - base.height) * y_index / 4.0,
                    )
                    crop = snap_crop(crop, width, height)
                    if target_long_edge is not None:
                        crop_box = crop_pixel_box(crop, width, height)
                        crop_long_edge = max(
                            crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]
                        )
                        if (
                            max(width, height) >= target_long_edge
                            and crop_long_edge < target_long_edge
                        ):
                            continue
                    proposals[crop] = None
    return tuple(proposals)


def select_composition_suggestions(
    crops: Sequence[CompositionCrop],
    scores: Any,
    *,
    limit: int = 4,
    iou_limit: float = 0.88,
) -> tuple[CompositionSuggestion, ...]:
    import numpy as np

    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    full = CompositionCrop(0.0, 0.0, 1.0, 1.0)
    if not crops or crops[0] != full:
        raise ValueError("the full-image crop must be the first scored candidate")
    if len(crops) != values.size:
        raise ValueError("crop candidate and score counts must match")
    if limit < 1 or not 0.0 < iou_limit <= 1.0:
        raise ValueError("selection limit and IoU threshold must be positive")
    if not np.isfinite(values).all():
        raise CompositionInferenceError("GAIC returned non-finite crop scores")
    selected = [0]
    for index in sorted(range(values.size), key=lambda i: (-float(values[i]), i)):
        if len(selected) >= limit:
            break
        if index == 0 or values[index] <= values[0]:
            continue
        if all(crop_iou(crops[index], crops[prior]) <= iou_limit for prior in selected):
            selected.append(index)
    selected.sort(key=lambda i: (-float(values[i]), i))
    return tuple(
        CompositionSuggestion(
            crop=crops[index],
            score=float(values[index]),
            improvement_over_original=float(values[index] - values[0]),
        )
        for index in selected
    )


def prepare_gaic_analysis_rgb(
    rgb: Any, *, max_long_edge: int = GAIC_ANALYSIS_LONG_EDGE
) -> Any:
    """Create an EXIF-upright analysis raster without upscaling the source."""
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
    if not 1 <= max_long_edge <= GAIC_ANALYSIS_LONG_EDGE:
        raise ValueError(f"max_long_edge must be in [1, {GAIC_ANALYSIS_LONG_EDGE}]")
    height, width = image.shape[:2]
    if max(width, height) <= max_long_edge:
        return image
    scale = max_long_edge / max(width, height)
    out_width = max(1, _round_pixel(width * scale))
    out_height = max(1, _round_pixel(height * scale))
    return np.ascontiguousarray(
        cv2.resize(
            image,
            (out_width, out_height),
            interpolation=cv2.INTER_LINEAR_EXACT,
        )
    )


def prepare_gaic_tensors(rgb: Any, crops: Sequence[CompositionCrop]) -> tuple[Any, Any]:
    """Pack normalized NCHW RGB and image-pixel crop boxes for the GAIC graph."""
    import numpy as np

    image = np.asarray(rgb)
    if (
        image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or not image.flags.c_contiguous
    ):
        raise ValueError("rgb must be a contiguous HWC uint8 RGB image")
    height, width = image.shape[:2]
    if max(width, height) > GAIC_ANALYSIS_LONG_EDGE:
        raise ValueError("GAIC analysis image exceeds the 1220-pixel long-edge limit")
    model_width, model_height = max(32, width), max(32, height)
    tensor = np.empty((1, 3, model_height, model_width), dtype=np.float32)
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
    for channel in range(3):
        plane = tensor[0, channel]
        plane[...] = image[
            np.minimum(np.arange(model_height), height - 1)[:, None],
            np.minimum(np.arange(model_width), width - 1)[None, :],
            channel,
        ]
        np.divide(plane, np.float32(255.0), out=plane)
        np.subtract(plane, mean[channel], out=plane)
        np.divide(plane, std[channel], out=plane)
    boxes = np.empty((len(crops), 4), dtype=np.float32)
    for index, crop in enumerate(crops):
        boxes[index] = (
            crop.left * width,
            crop.top * height,
            crop.right * width,
            crop.bottom * height,
        )
    return tensor, boxes


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class GaicCompositionOnnx:
    """CPU-only GAIC ONNX session with strict checkpoint/export provenance."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        model_sha256: str,
        intra_op_num_threads: int = 1,
        session_factory: Callable[..., Any] | None = None,
    ) -> None:
        if intra_op_num_threads < 1:
            raise ValueError("intra_op_num_threads must be positive")
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise CompositionInferenceError(
                f"GAIC ONNX graph is unavailable: {self.model_path}"
            )
        actual = _file_sha256(self.model_path)
        if actual != model_sha256.lower():
            raise CompositionInferenceError("GAIC ONNX SHA-256 does not match")
        self.model_sha256 = actual
        self.intra_op_num_threads = intra_op_num_threads
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - optional runtime
            raise CompositionInferenceError(
                "GAIC inference requires ONNX Runtime; install the inference extra"
            ) from exc
        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_op_num_threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        factory = session_factory or ort.InferenceSession
        try:
            self._session = factory(
                str(self.model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise CompositionInferenceError(
                f"cannot load GAIC ONNX graph: {exc}"
            ) from exc
        self.runtime_version = ort.__version__
        self.provider = "CPUExecutionProvider"
        self._inference_lock = threading.RLock()
        self._run_options_lock = threading.RLock()
        self._active_run_options: Any | None = None
        self._validate_graph()

    @classmethod
    def from_validation_report(
        cls,
        model_path: str | Path,
        *,
        intra_op_num_threads: int = 1,
        session_factory: Callable[..., Any] | None = None,
    ) -> GaicCompositionOnnx:
        selected_path = Path(model_path)
        report_path = Path(f"{selected_path}.validation.json")
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CompositionInferenceError(
                "GAIC validation report is missing or invalid"
            ) from exc
        if (
            report.get("model_id") != GAIC_MODEL_ID
            or report.get("source_checkpoint_sha256") != GAIC_SOURCE_CHECKPOINT_SHA256
            or report.get("source_code_revision") != GAIC_SOURCE_CODE_REVISION
            or report.get("passed") is not True
            or report.get("dynamic_shapes") is not True
            or report.get("input_names") != list(GAIC_INPUTS)
            or report.get("output_name") != GAIC_OUTPUT
        ):
            raise CompositionInferenceError(
                "GAIC validation report does not match the pinned MobileNetV2 model"
            )
        expected_hash = report.get("onnx_sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise CompositionInferenceError(
                "GAIC validation report has no valid graph hash"
            )
        return cls(
            selected_path,
            model_sha256=expected_hash,
            intra_op_num_threads=intra_op_num_threads,
            session_factory=session_factory,
        )

    def _validate_graph(self) -> None:
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if [node.name for node in inputs] != list(GAIC_INPUTS):
            raise CompositionInferenceError("GAIC inputs must be named image and boxes")
        if len(outputs) != 1 or outputs[0].name != GAIC_OUTPUT:
            raise CompositionInferenceError("GAIC output must be named scores")
        if any(node.type != "tensor(float)" for node in (*inputs, *outputs)):
            raise CompositionInferenceError("GAIC external tensors must be FP32")
        image_shape, boxes_shape, score_shape = (
            inputs[0].shape,
            inputs[1].shape,
            outputs[0].shape,
        )
        if (
            len(image_shape) != 4
            or image_shape[:2] != [1, 3]
            or len(boxes_shape) != 2
            or boxes_shape[1] != 4
            or len(score_shape) != 1
            or isinstance(boxes_shape[0], int)
            or isinstance(score_shape[0], int)
            or any(isinstance(axis, int) for axis in image_shape[2:])
        ):
            raise CompositionInferenceError(
                "GAIC graph must have dynamic image height/width and candidate count"
            )
        if self._session.get_providers() != ["CPUExecutionProvider"]:
            raise CompositionInferenceError("GAIC must use CPUExecutionProvider")

    def score(
        self,
        rgb: Any,
        crops: Sequence[CompositionCrop],
        *,
        cancel_check: Callable[[], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> Any:
        import numpy as np
        import onnxruntime as ort

        if self._session is None:
            raise CompositionInferenceError("GAIC ONNX session is closed")
        if not crops:
            raise ValueError("at least one crop candidate is required")
        if cancel_check is not None:
            cancel_check()
        image_tensor, boxes = prepare_gaic_tensors(rgb, crops)
        if cancel_check is not None:
            cancel_check()
        with self._inference_lock:
            run_options = ort.RunOptions()
            stop_watcher = threading.Event()
            watcher: threading.Thread | None = None
            if cancel_event is not None:

                def watch_cancellation() -> None:
                    while not stop_watcher.wait(0.02):
                        if cancel_event.is_set():
                            run_options.terminate = True
                            return

                watcher = threading.Thread(
                    target=watch_cancellation,
                    name="laika-gaic-cancel-watch",
                    daemon=True,
                )
                watcher.start()
            with self._run_options_lock:
                self._active_run_options = run_options
            try:
                scores = self._session.run(
                    [GAIC_OUTPUT],
                    {"image": image_tensor, "boxes": boxes},
                    run_options=run_options,
                )[0]
            except Exception as exc:
                if cancel_check is not None:
                    cancel_check()
                raise CompositionInferenceError(
                    f"GAIC inference failed: {exc}"
                ) from exc
            finally:
                stop_watcher.set()
                if watcher is not None:
                    watcher.join()
                with self._run_options_lock:
                    self._active_run_options = None
        del image_tensor, boxes
        values = np.asarray(scores, dtype=np.float32).reshape(-1)
        if values.size != len(crops) or not np.isfinite(values).all():
            raise CompositionInferenceError("GAIC returned invalid crop scores")
        if cancel_check is not None:
            cancel_check()
        return values

    def cancel(self) -> None:
        """Request termination of an active ONNX inference run, if supported."""
        with self._run_options_lock:
            if self._active_run_options is not None:
                self._active_run_options.terminate = True

    def close(self) -> None:
        self.cancel()
        with self._inference_lock:
            self._session = None


def analyze_composition_rgb(
    rgb: Any,
    model: GaicCompositionOnnx,
    *,
    target_long_edge: int | None = None,
    cancel_check: Callable[[], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> CompositionResult:
    """Generate deterministic grid-anchor proposals and rank them with GAIC."""
    import numpy as np

    source = np.asarray(rgb)
    if source.ndim != 3 or source.shape[2] != 3 or source.dtype != np.uint8:
        raise ValueError("rgb must be an HWC uint8 RGB image")
    height, width = source.shape[:2]
    if min(width, height) < 1:
        raise ValueError("image dimensions must be positive")
    timings: dict[str, float] = {}
    started = time.perf_counter()
    analysis = prepare_gaic_analysis_rgb(source)
    timings["analysis_resize"] = time.perf_counter() - started
    if cancel_check is not None:
        cancel_check()
    stage_started = time.perf_counter()
    # The reference creates and pixel-snaps candidate boxes in the original,
    # oriented source coordinates; only their normalized boxes are mapped to
    # the smaller analysis tensor for scoring.
    crops = enumerate_candidate_crops(
        width,
        height,
        target_long_edge=target_long_edge,
    )
    timings["candidate_generation"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    scores = model.score(
        analysis,
        crops,
        cancel_check=cancel_check,
        cancel_event=cancel_event,
    )
    timings["model_inference"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    suggestions = select_composition_suggestions(crops, scores)
    timings["selection"] = time.perf_counter() - stage_started
    if not np.isfinite(scores).all():
        raise CompositionInferenceError("GAIC returned non-finite scores")
    return CompositionResult(
        width=width,
        height=height,
        analysis_width=int(analysis.shape[1]),
        analysis_height=int(analysis.shape[0]),
        candidate_count=len(crops),
        suggestions=suggestions,
        model_sha256=model.model_sha256,
        runtime_version=model.runtime_version,
        provider=model.provider,
        stage_seconds=timings,
        total_seconds=time.perf_counter() - started,
    )


def apply_crop_rgb(rgb: Any, crop: CompositionCrop) -> Any:
    import numpy as np

    image = np.asarray(rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("rgb must be an HWC uint8 RGB image")
    left, top, right, bottom = crop_pixel_box(crop, image.shape[1], image.shape[0])
    return np.ascontiguousarray(image[top:bottom, left:right])


__all__ = [
    "CompositionCrop",
    "CompositionInferenceError",
    "CompositionResult",
    "CompositionSuggestion",
    "GAIC_ANALYSIS_LONG_EDGE",
    "GaicCompositionOnnx",
    "analyze_composition_rgb",
    "apply_crop_rgb",
    "crop_iou",
    "crop_pixel_box",
    "enumerate_candidate_crops",
    "prepare_gaic_analysis_rgb",
    "prepare_gaic_tensors",
    "select_composition_suggestions",
    "snap_crop",
]
