"""CPU ONNX Runtime adapter for the BokehMe ARNet defocus renderer."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from math import ceil, floor, isfinite
from pathlib import Path
from typing import Any, Literal

BOKEH_MODEL_FILENAME = "legendary-1.0-arnet.fp16.onnx"
BOKEH_SOURCE_CHECKPOINT_SHA256 = (
    "6c7dba6f32e3afc8e984eb5adfecfea5d13b6f4ebd132a5a890beba384be4752"
)
BOKEH_REFERENCE_ONNX_SHA256 = (
    "a3aa5fa6a0818f36dbfa4be4dd5471d874cea5981357b7aeb28919954e70d05c"
)
BOKEH_INPUTS = ("image", "defocus")
BOKEH_OUTPUT = "output"
BOKEH_STRATEGIES = ("low-memory", "balanced")


class BokehInferenceError(RuntimeError):
    """The local ARNet model or one of its tensor contracts is invalid."""


@dataclass(frozen=True)
class BokehPrediction:
    """One ARNet pass in RGB NCHW float32, with timings and backend metadata."""

    rgb_nchw: Any
    width: int
    height: int
    preprocess_seconds: float
    inference_seconds: float


@dataclass(frozen=True)
class BokehRuntimeInfo:
    """Resolved model and CPU runtime details for diagnostics and benchmarks."""

    model_sha256: str
    runtime_version: str
    provider: str
    strategy: str
    intra_op_num_threads: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_model_hash(model_path: Path) -> str:
    """Read the exporter report and bind it to this exact local ONNX file."""
    report_path = Path(f"{model_path}.validation.json")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BokehInferenceError(
            "BokehMe validation report is missing or invalid; pass an explicit "
            "trusted SHA-256 only when the model was independently verified"
        ) from exc
    if (
        report.get("model_id") != "bokehme-arnet-fp16"
        or report.get("checkpoint_sha256") != BOKEH_SOURCE_CHECKPOINT_SHA256
        or report.get("passed") is not True
        or report.get("dynamic_shapes") is not True
    ):
        raise BokehInferenceError("BokehMe validation report does not match ARNet")
    digest = report.get("onnx_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise BokehInferenceError("BokehMe validation report has no ONNX SHA-256")
    return digest.lower()


def _range_check(values: Any, low: float, high: float, label: str) -> None:
    """Check a contiguous array in bounded slices to avoid a full-size mask."""
    import numpy as np

    flat = values.reshape(-1)
    chunk_size = 1_048_576
    for start in range(0, flat.size, chunk_size):
        part = flat[start : start + chunk_size]
        if not bool(np.isfinite(part).all()):
            raise ValueError(f"{label} must contain only finite values")
        if float(part.min()) < low or float(part.max()) > high:
            raise ValueError(f"{label} values must be in [{low}, {high}]")


class BokehMeARNet:
    """Hash-verified, CPU-only BokehMe ARNet ONNX session.

    RGB inputs are HWC uint8 or float32 in [0, 1]. Defocus is an H×W float32
    map in [-1, 1], already scaled for the requested optical pass. The model
    receives FP32 NCHW tensors and returns FP32 RGB NCHW values in [0, 1].
    """

    def __init__(
        self,
        model_path: str | os.PathLike[str],
        *,
        strategy: Literal["low-memory", "balanced"] = "low-memory",
        intra_op_num_threads: int | None = None,
        expected_sha256: str | None = None,
        session_factory: Any | None = None,
    ) -> None:
        if strategy not in BOKEH_STRATEGIES:
            raise ValueError(f"strategy must be one of {', '.join(BOKEH_STRATEGIES)}")
        if intra_op_num_threads is None:
            intra_op_num_threads = (
                1 if strategy == "low-memory" else min(4, os.cpu_count() or 1)
            )
        if intra_op_num_threads < 1:
            raise ValueError("intra_op_num_threads must be positive")

        self.model_path = Path(model_path)
        self.strategy = strategy
        self.intra_op_num_threads = intra_op_num_threads
        self._session: Any | None = None
        if not self.model_path.is_file():
            raise BokehInferenceError(
                f"BokehMe model is unavailable: {self.model_path}"
            )
        if expected_sha256 is None:
            expected_sha256 = _validated_model_hash(self.model_path)
        actual_sha256 = _sha256(self.model_path)
        if actual_sha256 != expected_sha256:
            raise BokehInferenceError(
                "BokehMe model SHA-256 mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )

        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - optional runtime
            raise BokehInferenceError(
                "BokehMe inference requires ONNX Runtime; install the inference extra"
            ) from exc

        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_op_num_threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.enable_cpu_mem_arena = strategy == "balanced"
        options.enable_mem_pattern = strategy == "balanced"
        options.enable_mem_reuse = True
        factory = session_factory or ort.InferenceSession
        try:
            session = factory(
                str(self.model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise BokehInferenceError(f"cannot load BokehMe ONNX model: {exc}") from exc

        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if [node.name for node in inputs] != list(BOKEH_INPUTS):
            raise BokehInferenceError(
                "BokehMe must expose inputs named 'image' and 'defocus' in order"
            )
        if len(outputs) != 1 or outputs[0].name != BOKEH_OUTPUT:
            raise BokehInferenceError("BokehMe must expose one output named 'output'")
        if any(node.type != "tensor(float)" for node in (*inputs, *outputs)):
            raise BokehInferenceError("BokehMe external tensors must be float32")
        image_shape, defocus_shape, output_shape = (
            inputs[0].shape,
            inputs[1].shape,
            outputs[0].shape,
        )
        if (
            len(image_shape) != 4
            or image_shape[:2] != [1, 3]
            or len(defocus_shape) != 4
            or defocus_shape[:2] != [1, 1]
            or len(output_shape) != 4
            or output_shape[2:] != ["height", "width"]
        ):
            raise BokehInferenceError(
                "BokehMe must expose dynamic NCHW image, defocus, and output tensors"
            )
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise BokehInferenceError("BokehMe must run on CPUExecutionProvider")

        self.model_sha256 = actual_sha256
        self.runtime_version = ort.__version__
        self.provider = "CPUExecutionProvider"
        self._session = session

    @property
    def runtime_info(self) -> BokehRuntimeInfo:
        return BokehRuntimeInfo(
            model_sha256=self.model_sha256,
            runtime_version=self.runtime_version,
            provider=self.provider,
            strategy=self.strategy,
            intra_op_num_threads=self.intra_op_num_threads,
        )

    def infer(self, rgb: Any, defocus: Any) -> BokehPrediction:
        """Run ARNet once and return its owned NCHW output buffer."""
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - optional runtime
            raise BokehInferenceError("BokehMe inference requires NumPy") from exc
        if self._session is None:
            raise BokehInferenceError("BokehMe ONNX session is closed")

        image = np.asarray(rgb)
        if (
            image.ndim != 3
            or image.shape[2] != 3
            or image.shape[0] < 1
            or image.shape[1] < 1
            or image.dtype not in (np.dtype(np.uint8), np.dtype(np.float32))
            or not image.flags.c_contiguous
        ):
            raise ValueError("rgb must be a contiguous HWC uint8 or float32 RGB array")
        height, width = image.shape[:2]
        defocus_map = np.asarray(defocus)
        if (
            defocus_map.dtype != np.float32
            or defocus_map.shape != (height, width)
            or not defocus_map.flags.c_contiguous
        ):
            raise ValueError(
                "defocus must be a contiguous H×W float32 map matching rgb"
            )
        if image.dtype == np.float32:
            _range_check(image, 0.0, 1.0, "rgb")
        _range_check(defocus_map, -1.0, 1.0, "defocus")

        started = time.perf_counter()
        tensor = np.empty((1, 3, height, width), dtype=np.float32)
        for channel in range(3):
            plane = tensor[0, channel]
            plane[...] = image[:, :, channel]
            if image.dtype == np.uint8:
                np.divide(plane, np.float32(255.0), out=plane)
        defocus_tensor = defocus_map.reshape(1, 1, height, width)
        preprocess_seconds = time.perf_counter() - started

        inference_started = time.perf_counter()
        output = self._session.run(
            [BOKEH_OUTPUT],
            {"image": tensor, "defocus": defocus_tensor},
        )[0]
        inference_seconds = time.perf_counter() - inference_started
        if (
            output.dtype != np.float32
            or output.shape != (1, 3, height, width)
            or not output.flags.c_contiguous
        ):
            raise BokehInferenceError("BokehMe returned an invalid output tensor")
        _range_check(output, float("-inf"), float("inf"), "model output")
        np.clip(output, 0.0, 1.0, out=output)
        return BokehPrediction(
            rgb_nchw=output,
            width=width,
            height=height,
            preprocess_seconds=preprocess_seconds,
            inference_seconds=inference_seconds,
        )

    def warmup(self, width: int = 64, height: int = 64) -> float:
        """Run one explicitly requested small warmup pass and return its time."""
        if width < 2 or height < 2:
            raise ValueError("warmup dimensions must be at least two pixels")
        import numpy as np

        started = time.perf_counter()
        self.infer(
            np.zeros((height, width, 3), dtype=np.float32),
            np.zeros((height, width), dtype=np.float32),
        )
        return time.perf_counter() - started

    def close(self) -> None:
        """Release the native ONNX Runtime session."""
        self._session = None

    def __enter__(self) -> BokehMeARNet:
        if self._session is None:
            raise BokehInferenceError("BokehMe ONNX session is closed")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


@dataclass(frozen=True)
class BokehRenderResult:
    """Full-size RGB result from reference-style optical ARNet passes."""

    rgb: Any
    width: int
    height: int
    focus: float
    aperture: float
    scales: tuple[float, ...]
    stage_seconds: dict[str, float]
    total_seconds: float


APERTURES = (0.95, 1.8, 2.0, 2.8, 4.0, 5.6)
FOCAL_MM = 50.0
SENSOR_LONG_MM = 36.0
SENSOR_SHORT_MM = 24.0
VIRTUAL_NEAR_MM = 1200.0
VIRTUAL_FAR_MM = 30000.0


def normalize_depth_map(raw_depth: Any) -> Any:
    """Match the reference 2nd/98th percentile depth normalization."""
    import numpy as np

    depth = np.asarray(raw_depth)
    if (
        depth.ndim != 2
        or depth.dtype != np.float32
        or not depth.flags.c_contiguous
        or min(depth.shape) < 1
    ):
        raise ValueError("depth must be a contiguous 2-D float32 map")
    _range_check(depth, float("-inf"), float("inf"), "depth")
    ordered = depth.reshape(-1).copy()
    ordered.sort()
    low = _percentile_sorted(ordered, 0.02)
    high = _percentile_sorted(ordered, 0.98)
    if high - low < 1e-5:
        raise ValueError("depth map has insufficient range for focus selection")
    normalized = np.empty_like(depth)
    np.subtract(depth, np.float32(low), out=normalized)
    np.divide(normalized, np.float32(high - low), out=normalized)
    np.clip(normalized, 0.0, 1.0, out=normalized)
    return normalized


def _percentile_sorted(values: Any, quantile: float) -> float:

    position = quantile * (values.size - 1)
    lower = floor(position)
    upper = min(lower + 1, values.size - 1)
    fraction = position - lower
    return float(values[lower] * (1.0 - fraction) + values[upper] * fraction)


def focus_at_point(depth: Any, x: float, y: float) -> float:
    """Select a stable median focus value at normalized display coordinates."""
    import numpy as np

    values = np.asarray(depth)
    if (
        values.ndim != 2
        or values.dtype != np.float32
        or not values.flags.c_contiguous
        or not (isfinite(x) and isfinite(y) and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0)
    ):
        raise ValueError("depth and normalized focus coordinates are invalid")
    _range_check(values, 0.0, 1.0, "normalized depth")
    height, width = values.shape
    center_x = int(np.rint(x * (width - 1)))
    center_y = int(np.rint(y * (height - 1)))
    radius = int(np.rint(min(width, height) * 0.006))
    radius = min(8, max(1, radius))
    x0, x1 = max(0, center_x - radius), min(width, center_x + radius + 1)
    y0, y1 = max(0, center_y - radius), min(height, center_y + radius + 1)
    neighborhood = values[y0:y1, x0:x1].reshape(-1).copy()
    neighborhood.sort()
    low = _percentile_sorted(neighborhood, 0.1)
    median = _percentile_sorted(neighborhood, 0.5)
    high = _percentile_sorted(neighborhood, 0.9)
    if (
        high - low > 0.08
        or abs(float(values[center_y, center_x]) - median) > 0.04
        or float(values.max()) - float(values.min()) <= 1e-5
    ):
        raise ValueError("focus point crosses a depth edge or has unreliable depth")
    return float(np.float32(median))


def full_frame_coc(depth: Any, focus: float, aperture: float) -> Any:
    """Compute reference thin-lens circle-of-confusion radii in pixels."""
    import numpy as np

    values = np.asarray(depth)
    if (
        values.ndim != 2
        or values.dtype != np.float32
        or not values.flags.c_contiguous
        or min(values.shape) < 1
    ):
        raise ValueError("depth must be a contiguous 2-D float32 map")
    _range_check(values, 0.0, 1.0, "normalized depth")
    if not isfinite(focus) or not 0.0 <= focus <= 1.0:
        raise ValueError("focus must be in [0, 1]")
    if not isfinite(aperture) or aperture not in APERTURES:
        raise ValueError(f"aperture must be one of {APERTURES}")

    # The reference stores both parameters as Float before doing Double lens math.
    focus = float(np.float32(focus))
    aperture = float(np.float32(aperture))

    height, width = values.shape
    sensor_long = min(
        SENSOR_LONG_MM,
        SENSOR_SHORT_MM * max(width, height) / min(width, height),
    )
    inverse_near = 1.0 / VIRTUAL_NEAR_MM
    inverse_far = 1.0 / VIRTUAL_FAR_MM
    inverse_focus = inverse_far + focus * (inverse_near - inverse_far)
    gain = (
        FOCAL_MM**2
        / aperture
        / (1.0 - FOCAL_MM * inverse_focus)
        * (inverse_near - inverse_far)
        * max(width, height)
        / (2.0 * sensor_long)
    )
    native = np.empty_like(values)
    np.subtract(values, np.float32(focus), out=native)
    np.multiply(native, np.float32(gain), out=native)
    return native


def _smooth(value: Any) -> Any:
    import numpy as np

    clipped = np.clip(value, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _max_abs_rows(values: Any, chunk_height: int = 256) -> float:
    """Measure the largest absolute value without a full-frame temporary."""
    maximum = 0.0
    for row_start in range(0, values.shape[0], chunk_height):
        rows = values[row_start : row_start + chunk_height]
        maximum = max(maximum, float(abs(rows).max(initial=0.0)))
    return maximum


def _resize_nchw_to_rgb(nchw: Any, width: int, height: int, cv2: Any) -> Any:
    import numpy as np

    output = np.empty((height, width, 3), dtype=np.float32)
    for channel in range(3):
        output[:, :, channel] = cv2.resize(
            nchw[0, channel],
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
    return output


def resize_rgb_cubic(rgb: Any, width: int, height: int) -> Any:
    """Resize HWC RGB uint8 with OpenCV's reference-compatible cubic kernel."""
    import cv2
    import numpy as np

    image = np.asarray(rgb)
    if (
        image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or not image.flags.c_contiguous
    ):
        raise ValueError("rgb must be contiguous HWC uint8 RGB")
    if width < 1 or height < 1:
        raise ValueError("resize dimensions must be positive")
    if image.shape[:2] == (height, width):
        return image
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)


def render_bokeh(
    rgb: Any,
    depth: Any,
    focus: float,
    aperture: float,
    model: BokehMeARNet,
    *,
    row_chunk_height: int = 128,
    cancel_check: Callable[[], None] | None = None,
) -> BokehRenderResult:
    """Render a full-size photo using the reference multiscale ARNet schedule.

    This renderer performs the depth-dependent optical plate passes. Highlight
    extraction, optical highlight scatter, and subject-protection compositing
    remain separate later stages of the complete reference pipeline.
    """
    import cv2
    import numpy as np

    image = np.asarray(rgb)
    depth_map = np.asarray(depth)
    if (
        image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or min(image.shape[:2]) < 2
        or not image.flags.c_contiguous
    ):
        raise ValueError("rgb must be contiguous HWC uint8 RGB with sides >= 2")
    height, width = image.shape[:2]
    if (
        depth_map.dtype != np.float32
        or depth_map.shape != (height, width)
        or not depth_map.flags.c_contiguous
    ):
        raise ValueError("depth must be a matching contiguous H×W float32 map")
    _range_check(depth_map, 0.0, 1.0, "normalized depth")
    if row_chunk_height < 1:
        raise ValueError("row_chunk_height must be positive")
    started = time.perf_counter()
    stage_seconds: dict[str, float] = {}
    optics_started = time.perf_counter()
    coc = full_frame_coc(depth_map, focus, aperture)
    stage_seconds["optics"] = time.perf_counter() - optics_started
    max_abs_coc = _max_abs_rows(coc)
    if max_abs_coc < 1e-6:
        return BokehRenderResult(
            rgb=image.copy(),
            width=width,
            height=height,
            focus=focus,
            aperture=aperture,
            scales=(),
            stage_seconds={"optics": time.perf_counter() - started},
            total_seconds=time.perf_counter() - started,
        )

    stage_started = time.perf_counter()
    image_float = image.astype(np.float32)
    np.divide(image_float, np.float32(255.0), out=image_float)
    target_scale = max(1.0, float(np.float32(max_abs_coc / 8.0)))
    stage_seconds["prepare"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    composite: Any | None = None
    scales: list[float] = []
    previous_scale = 1.0
    requested_scale = 1.0

    while True:
        if cancel_check is not None:
            cancel_check()
        requested_scale32 = np.float32(requested_scale)
        pass_width = max(2, ceil(float(np.float32(width) / requested_scale32)))
        pass_height = max(2, ceil(float(np.float32(height) / requested_scale32)))
        actual_scale = float(
            max(
                np.float32(width) / np.float32(pass_width),
                np.float32(height) / np.float32(pass_height),
            )
        )
        if (pass_width, pass_height) == (width, height):
            small_rgb = image_float
        else:
            small_rgb = cv2.resize(
                image_float,
                (pass_width, pass_height),
                interpolation=cv2.INTER_AREA,
            )
        if (pass_width, pass_height) == (width, height):
            small_coc = coc
        else:
            small_coc = cv2.resize(
                coc,
                (pass_width, pass_height),
                interpolation=cv2.INTER_LINEAR,
            )
        defocus = np.empty((pass_height, pass_width), dtype=np.float32)
        np.divide(small_coc, np.float32(actual_scale), out=defocus)
        np.divide(defocus, np.float32(10.0), out=defocus)
        np.clip(defocus, -1.0, 1.0, out=defocus)
        prediction = model.infer(small_rgb, defocus)
        enlarged = _resize_nchw_to_rgb(prediction.rgb_nchw, width, height, cv2)
        del prediction, small_rgb, small_coc, defocus

        if composite is None:
            composite = enlarged
        else:
            for row_start in range(0, height, row_chunk_height):
                row_stop = min(height, row_start + row_chunk_height)
                weight = _smooth(
                    (np.abs(coc[row_start:row_stop]) - np.float32(5.0 * previous_scale))
                    / np.float32(3.0 * previous_scale)
                )[:, :, None]
                current = composite[row_start:row_stop]
                current *= 1.0 - weight
                current += enlarged[row_start:row_stop] * weight
            del enlarged

        scales.append(actual_scale)
        stage_seconds[f"arnet_pass_{len(scales)}"] = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        previous_scale = actual_scale
        if requested_scale >= target_scale:
            break
        requested_scale = float(np.float32(min(requested_scale * 2.0, target_scale)))

    assert composite is not None
    final_rgb = np.empty_like(image)
    for row_start in range(0, height, row_chunk_height):
        if cancel_check is not None:
            cancel_check()
        row_stop = min(height, row_start + row_chunk_height)
        weight = _smooth(np.abs(coc[row_start:row_stop]))[:, :, None]
        mixed = image_float[row_start:row_stop] * (1.0 - weight)
        mixed += composite[row_start:row_stop] * weight
        np.clip(mixed, 0.0, 1.0, out=mixed)
        np.multiply(mixed, 255.0, out=mixed)
        np.rint(mixed, out=mixed)
        final_rgb[row_start:row_stop] = mixed.astype(np.uint8)
    stage_seconds["composite"] = time.perf_counter() - stage_started
    del image_float, composite, coc
    if cancel_check is not None:
        cancel_check()
    return BokehRenderResult(
        rgb=final_rgb,
        width=width,
        height=height,
        focus=focus,
        aperture=aperture,
        scales=tuple(scales),
        stage_seconds=stage_seconds,
        total_seconds=time.perf_counter() - started,
    )
