"""CPU ONNX Runtime adapter for the validated Depth Anything V2 Small graph."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DepthInferenceError(RuntimeError):
    """The depth model or its input/output contract is invalid."""


@dataclass(frozen=True)
class DepthPrediction:
    """A full source-size depth map and measurements for one inference."""

    depth: Any
    source_width: int
    source_height: int
    network_width: int
    network_height: int
    stage_seconds: dict[str, float]
    total_seconds: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _constrain_to_multiple(value: float, multiple: int, minimum: int) -> int:
    rounded = round(value / multiple) * multiple
    if rounded < minimum:
        rounded = math.ceil(value / multiple) * multiple
    return int(rounded)


def depth_network_size(
    source_width: int, source_height: int, input_size: int = 518
) -> tuple[int, int]:
    """Return official lower-bound, aspect-preserving, multiple-of-14 size."""
    if source_width < 1 or source_height < 1:
        raise ValueError("source dimensions must be positive")
    if input_size < 518 or input_size % 14:
        raise ValueError("input_size must be at least 518 and divisible by 14")

    scale = max(input_size / source_height, input_size / source_width)
    network_height = _constrain_to_multiple(
        source_height * scale, multiple=14, minimum=input_size
    )
    network_width = _constrain_to_multiple(
        source_width * scale, multiple=14, minimum=input_size
    )
    return network_width, network_height


def _preprocess_rgb(
    rgb: Any,
    input_size: int,
    cv2: Any,
    np: Any,
    max_network_pixels: int,
) -> tuple[Any, int, int]:
    if max_network_pixels < 1:
        raise ValueError("network pixel budget must be positive")
    image = np.asarray(rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("rgb must be a uint8 array with shape [H, W, 3]")
    source_height, source_width = image.shape[:2]
    if source_height < 1 or source_width < 1:
        raise ValueError("RGB dimensions must be positive")

    network_width, network_height = depth_network_size(
        source_width, source_height, input_size
    )
    if network_width * network_height > max_network_pixels:
        raise ValueError(
            "network input exceeds the configured pixel budget; "
            "increase the explicit limit or reduce the requested input size"
        )
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = np.empty(
        (1, 3, network_height, network_width), dtype=np.float32, order="C"
    )
    same_size = (network_width, network_height) == (source_width, source_height)
    for channel in range(3):
        plane = image[:, :, channel].astype(np.float32)
        np.divide(plane, np.float32(255.0), out=plane)
        if same_size:
            resized = plane
        else:
            resized = cv2.resize(
                plane,
                (network_width, network_height),
                interpolation=cv2.INTER_CUBIC,
            )
            del plane
        resized -= mean[channel]
        resized /= std[channel]
        tensor[0, channel] = resized
    return tensor, network_width, network_height


def resize_bilinear_align_corners(
    image: Any,
    output_height: int,
    output_width: int,
    *,
    max_block_pixels: int = 262_144,
) -> Any:
    """Resize a 2-D float map with PyTorch bilinear ``align_corners=True`` math.

    Work is divided into row blocks so coordinate grids and gathered samples do
    not grow to the size of a full-resolution photo.
    """
    import numpy as np

    values = np.asarray(image)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.floating):
        raise ValueError("depth input must be a 2-D floating-point array")
    if output_height < 1 or output_width < 1 or max_block_pixels < 1:
        raise ValueError("output dimensions and block budget must be positive")
    input_height, input_width = values.shape
    if input_height < 1 or input_width < 1:
        raise ValueError("depth input dimensions must be positive")
    if (input_height, input_width) == (output_height, output_width):
        return np.array(values, dtype=np.float32, order="C", copy=True)

    x_scale = (input_width - 1) / (output_width - 1) if output_width > 1 else 0.0
    y_scale = (input_height - 1) / (output_height - 1) if output_height > 1 else 0.0
    x_coordinates = np.arange(output_width, dtype=np.float32) * np.float32(x_scale)
    x0 = np.floor(x_coordinates).astype(np.intp)
    x1 = np.minimum(x0 + 1, input_width - 1)
    weight_x = x_coordinates - x0
    inverse_weight_x = 1.0 - weight_x

    rows_per_block = max(1, max_block_pixels // output_width)
    result = np.empty((output_height, output_width), dtype=np.float32)
    for start in range(0, output_height, rows_per_block):
        stop = min(start + rows_per_block, output_height)
        y_coordinates = np.arange(start, stop, dtype=np.float32) * np.float32(y_scale)
        y0 = np.floor(y_coordinates).astype(np.intp)
        y1 = np.minimum(y0 + 1, input_height - 1)
        weight_y = y_coordinates - y0
        inverse_weight_y = 1.0 - weight_y

        top = values[y0[:, None], x0[None, :]]
        bottom = values[y1[:, None], x0[None, :]]
        top_right = values[y0[:, None], x1[None, :]]
        bottom_right = values[y1[:, None], x1[None, :]]
        top_values = top * inverse_weight_x + top_right * weight_x
        bottom_values = bottom * inverse_weight_x + bottom_right * weight_x
        result[start:stop] = (
            top_values * inverse_weight_y[:, None] + bottom_values * weight_y[:, None]
        )
    return result


class DepthAnythingV2Small:
    """Run a validated dynamic Depth Anything V2 Small ONNX model on CPU.

    The caller owns the RGB input array and returned float32 map. The model
    session belongs to this object and is released by ``close()`` or the
    context manager. This adapter does not retain image features or predictions.
    """

    def __init__(
        self,
        model_path: str | os.PathLike[str],
        *,
        input_size: int = 518,
        intra_op_num_threads: int = 1,
        max_source_pixels: int = 40_000_000,
        max_network_pixels: int = 4_000_000,
    ) -> None:
        if input_size < 518 or input_size % 14:
            raise ValueError("input_size must be at least 518 and divisible by 14")
        if intra_op_num_threads < 1 or max_source_pixels < 1 or max_network_pixels < 1:
            raise ValueError("thread and pixel limits must be positive")
        try:
            import cv2
            import numpy as np
            import onnxruntime as ort
        except ImportError as exc:
            raise DepthInferenceError(
                "depth inference requires the 'inference' extra: NumPy, "
                "OpenCV, and ONNX Runtime"
            ) from exc

        self.model_path = Path(model_path)
        self.input_size = input_size
        self.max_source_pixels = max_source_pixels
        self.max_network_pixels = max_network_pixels
        self._cv2 = cv2
        self._np = np
        self._session: Any | None = None
        self._input_name = "image"
        self._output_name = "depth"
        self._model_input_shape: list[Any] | None = None

        if not self.model_path.is_file():
            raise DepthInferenceError("ONNX model file does not exist")
        report_path = Path(f"{self.model_path}.validation.json")
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DepthInferenceError(
                "validated ONNX export report is missing or invalid"
            ) from exc
        if (
            report.get("model_id") != "depth-anything-v2-small"
            or report.get("dynamic_shapes") is not True
            or report.get("onnx_sha256") != _sha256(self.model_path)
        ):
            raise DepthInferenceError(
                "ONNX model does not match a validated dynamic Small export"
            )

        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_op_num_threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            session = ort.InferenceSession(
                str(self.model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise DepthInferenceError("ONNX Runtime could not load the model") from exc
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise DepthInferenceError(
                "model must have one image input and one depth output"
            )
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name
        self._model_input_shape = list(inputs[0].shape)
        if len(self._model_input_shape) != 4 or self._model_input_shape[:2] != [1, 3]:
            raise DepthInferenceError("model input must have shape [1, 3, H, W]")
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise DepthInferenceError("depth model must run on CPUExecutionProvider")
        self._session = session

    def infer_rgb(self, rgb: Any) -> DepthPrediction:
        """Estimate depth for an H×W×3 uint8 RGB image without resizing output."""
        if self._session is None:
            raise DepthInferenceError("depth session is closed")
        np = self._np
        image = np.asarray(rgb)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("rgb must be a uint8 array with shape [H, W, 3]")
        source_height, source_width = image.shape[:2]
        if source_height < 1 or source_width < 1:
            raise ValueError("RGB dimensions must be positive")
        if source_height * source_width > self.max_source_pixels:
            raise ValueError("RGB image exceeds the configured pixel limit")

        started = time.perf_counter()
        stage_seconds: dict[str, float] = {}
        stage_started = time.perf_counter()
        tensor, network_width, network_height = _preprocess_rgb(
            image,
            self.input_size,
            self._cv2,
            np,
            self.max_network_pixels,
        )
        if self._model_input_shape is not None:
            for axis, actual in ((2, network_height), (3, network_width)):
                declared = self._model_input_shape[axis]
                if isinstance(declared, int) and declared != actual:
                    raise DepthInferenceError(
                        "ONNX graph has static spatial dimensions; "
                        f"axis {axis} is {declared}, requested {actual}"
                    )

        stage_seconds["preprocess"] = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        raw_depth = self._session.run([self._output_name], {self._input_name: tensor})[
            0
        ]
        stage_seconds["inference"] = time.perf_counter() - stage_started
        if (
            raw_depth.shape != (1, network_height, network_width)
            or not np.isfinite(raw_depth).all()
        ):
            raise DepthInferenceError(
                "model returned invalid depth dimensions or values"
            )

        stage_started = time.perf_counter()
        depth = resize_bilinear_align_corners(raw_depth[0], source_height, source_width)
        del raw_depth, tensor
        stage_seconds["restore_size"] = time.perf_counter() - stage_started
        return DepthPrediction(
            depth=depth,
            source_width=source_width,
            source_height=source_height,
            network_width=network_width,
            network_height=network_height,
            stage_seconds=stage_seconds,
            total_seconds=time.perf_counter() - started,
        )

    def close(self) -> None:
        """Drop the ONNX session and its native runtime allocations."""
        self._session = None

    def __enter__(self) -> DepthAnythingV2Small:
        if self._session is None:
            raise DepthInferenceError("depth session is closed")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
