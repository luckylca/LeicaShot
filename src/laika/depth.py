"""CPU ONNX Runtime adapter for the validated Depth Anything V2 Small graph."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Callable
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
    analysis_width: int
    analysis_height: int
    network_width: int
    network_height: int
    stage_seconds: dict[str, float]
    total_seconds: float


@dataclass(frozen=True)
class DepthInputGeometry:
    """Analysis raster and padded tensor dimensions for one source image."""

    analysis_width: int
    analysis_height: int
    network_width: int
    network_height: int


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


def depth_analysis_size(
    source_width: int, source_height: int, max_analysis_edge: int = 1220
) -> tuple[int, int]:
    """Cap the longer analysis edge without enlarging small source images."""
    if source_width < 1 or source_height < 1 or max_analysis_edge < 1:
        raise ValueError("source dimensions and maximum analysis edge must be positive")
    source_edge = max(source_width, source_height)
    if source_edge <= max_analysis_edge:
        return source_width, source_height

    def fit(value: int) -> int:
        return max(1, (value * max_analysis_edge + source_edge // 2) // source_edge)

    return fit(source_width), fit(source_height)


def depth_input_geometry(
    source_width: int,
    source_height: int,
    *,
    input_size: int | None = None,
    max_analysis_edge: int | None = None,
) -> DepthInputGeometry:
    """Return analysis and network dimensions for one explicit sizing policy.

    ``input_size`` selects the upstream model's lower-bound resize. Otherwise,
    the source-reference max-edge analysis policy is used (1220 by default),
    with right/bottom replication padding to the patch multiple of 14.
    """
    if input_size is not None and max_analysis_edge is not None:
        raise ValueError("choose either input_size or max_analysis_edge")
    if input_size is not None:
        network_width, network_height = depth_network_size(
            source_width, source_height, input_size
        )
        return DepthInputGeometry(
            analysis_width=network_width,
            analysis_height=network_height,
            network_width=network_width,
            network_height=network_height,
        )

    if max_analysis_edge is None:
        max_analysis_edge = 1220
    analysis_width, analysis_height = depth_analysis_size(
        source_width, source_height, max_analysis_edge
    )
    network_width = ((analysis_width + 13) // 14) * 14
    network_height = ((analysis_height + 13) // 14) * 14
    return DepthInputGeometry(
        analysis_width=analysis_width,
        analysis_height=analysis_height,
        network_width=network_width,
        network_height=network_height,
    )


def _preprocess_rgb(
    rgb: Any,
    input_size: int | None,
    cv2: Any,
    np: Any,
    max_network_pixels: int,
    *,
    max_analysis_edge: int | None = None,
) -> tuple[Any, DepthInputGeometry]:
    if max_network_pixels < 1:
        raise ValueError("network pixel budget must be positive")
    image = np.asarray(rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("rgb must be a uint8 array with shape [H, W, 3]")
    source_height, source_width = image.shape[:2]
    if source_height < 1 or source_width < 1:
        raise ValueError("RGB dimensions must be positive")

    geometry = depth_input_geometry(
        source_width,
        source_height,
        input_size=input_size,
        max_analysis_edge=max_analysis_edge,
    )
    analysis_width = geometry.analysis_width
    analysis_height = geometry.analysis_height
    network_width = geometry.network_width
    network_height = geometry.network_height
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
    lower_bound_profile = input_size is not None
    if lower_bound_profile:
        resized_rgb = image
    elif (analysis_width, analysis_height) == (source_width, source_height):
        resized_rgb = image
    else:
        from PIL import Image

        source_pil = Image.fromarray(image, mode="RGB")
        try:
            resized_pil = source_pil.resize(
                (analysis_width, analysis_height),
                resample=Image.Resampling.BICUBIC,
            )
        finally:
            source_pil.close()
        try:
            resized_rgb = np.array(resized_pil, dtype=np.uint8, copy=True)
        finally:
            resized_pil.close()

    for channel in range(3):
        plane = resized_rgb[:, :, channel].astype(np.float32)
        np.divide(plane, np.float32(255.0), out=plane)
        if lower_bound_profile and (network_width, network_height) != (
            source_width,
            source_height,
        ):
            resized = cv2.resize(
                plane,
                (network_width, network_height),
                interpolation=cv2.INTER_CUBIC,
            )
            del plane
        else:
            resized = plane
        resized -= mean[channel]
        resized /= std[channel]
        tensor[0, channel, :analysis_height, :analysis_width] = resized
        if network_height > analysis_height:
            tensor[0, channel, analysis_height:, :analysis_width] = tensor[
                0, channel, analysis_height - 1 : analysis_height, :analysis_width
            ]
        if network_width > analysis_width:
            tensor[0, channel, :network_height, analysis_width:] = tensor[
                0, channel, :network_height, analysis_width - 1 : analysis_width
            ]
        del resized
    if resized_rgb is not image:
        del resized_rgb
    return tensor, geometry


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
        input_size: int | None = None,
        max_analysis_edge: int | None = None,
        intra_op_num_threads: int = 1,
        max_source_pixels: int = 40_000_000,
        max_network_pixels: int = 4_000_000,
    ) -> None:
        if input_size is not None and max_analysis_edge is not None:
            raise ValueError("choose either input_size or max_analysis_edge")
        if input_size is None and max_analysis_edge is None:
            max_analysis_edge = 1220
        if input_size is not None and (input_size < 518 or input_size % 14):
            raise ValueError("input_size must be at least 518 and divisible by 14")
        if max_analysis_edge is not None and max_analysis_edge < 1:
            raise ValueError("max_analysis_edge must be positive")
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
        self.max_analysis_edge = max_analysis_edge
        self.max_source_pixels = max_source_pixels
        self.max_network_pixels = max_network_pixels
        self.model_sha256 = (
            _sha256(self.model_path) if self.model_path.is_file() else ""
        )
        self.runtime_version = ort.__version__
        self.provider = "CPUExecutionProvider"
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
            or report.get("onnx_sha256") != self.model_sha256
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

    def infer_rgb(
        self,
        rgb: Any,
        *,
        cancel_check: Callable[[], None] | None = None,
    ) -> DepthPrediction:
        """Estimate depth for an H×W×3 uint8 RGB image without resizing output."""
        if self._session is None:
            raise DepthInferenceError("depth session is closed")
        if cancel_check is not None:
            cancel_check()
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
        tensor, geometry = _preprocess_rgb(
            image,
            self.input_size,
            self._cv2,
            np,
            self.max_network_pixels,
            max_analysis_edge=self.max_analysis_edge,
        )
        network_width = geometry.network_width
        network_height = geometry.network_height
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
        if cancel_check is not None:
            cancel_check()
        stage_seconds["inference"] = time.perf_counter() - stage_started
        if (
            raw_depth.shape != (1, network_height, network_width)
            or not np.isfinite(raw_depth).all()
        ):
            raise DepthInferenceError(
                "model returned invalid depth dimensions or values"
            )

        stage_started = time.perf_counter()
        analysis_depth = raw_depth[
            0, : geometry.analysis_height, : geometry.analysis_width
        ]
        depth = resize_bilinear_align_corners(
            analysis_depth, source_height, source_width
        )
        del raw_depth, tensor
        stage_seconds["restore_size"] = time.perf_counter() - stage_started
        return DepthPrediction(
            depth=depth,
            source_width=source_width,
            source_height=source_height,
            analysis_width=geometry.analysis_width,
            analysis_height=geometry.analysis_height,
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
