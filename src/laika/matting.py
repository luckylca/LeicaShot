"""Memory-bounded ViTMatte ONNX inference and overlap-safe alpha tiling."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VITMATTE_CHECKPOINT_SHA256 = (
    "bda9289db1bb6762d978b42d1c62ae3f34daf7497171a347a1d09657efd788cb"
)


class MatteInferenceError(RuntimeError):
    """The optional matting runtime or graph does not satisfy its contract."""


@dataclass(frozen=True)
class MatteTile:
    """A crop in the padded analysis-image coordinate system."""

    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class MatteTileConfig:
    """Tile budget for ViTMatte's 32-pixel patch-aligned input."""

    tile_size: int = 1024
    feather: int | None = None
    max_step: int | None = None
    single_pass_max_pixels: int = 1248 * 928

    def __post_init__(self) -> None:
        if self.tile_size < 64 or self.tile_size % 32:
            raise ValueError("tile_size must be a multiple of 32 and at least 64")
        if self.single_pass_max_pixels < 0:
            raise ValueError("single_pass_max_pixels cannot be negative")
        feather = self.feather
        if feather is None:
            feather = max(1, self.tile_size * 3 // 32)
            object.__setattr__(self, "feather", feather)
        step = self.max_step
        if step is None:
            step = max(32, self.tile_size * 3 // 4)
            step = step // 32 * 32
            object.__setattr__(self, "max_step", step)
        if feather <= 0 or feather * 2 > self.tile_size:
            raise ValueError("feather must be positive and at most half the tile")
        if step < 32 or step % 32 or step > self.tile_size:
            raise ValueError("max_step must be a 32-aligned value within tile_size")
        if self.tile_size > 32 and self.tile_size - step < 2 * feather:
            raise ValueError("tile overlap must be at least twice the feather width")


def prepare_vitmatte_input(
    rgb: Any, trimap: Any, *, max_analysis_long_edge: int = 2048
) -> Any:
    """Pack RGB and trimap as padded ViTMatte NCHW float32 input.

    RGB is HWC uint8 sRGB or HWC float32 sRGB in ``[0, 1]``. The trimap must
    be float32 ``[H, W]`` in ``[0, 1]``. Spatial padding is placed on the right
    and bottom to a multiple of 32; padded tensor values are zero.
    """
    import numpy as np

    image = np.asarray(rgb)
    matte_hint = np.asarray(trimap)
    if max_analysis_long_edge < 1:
        raise ValueError("max_analysis_long_edge must be positive")
    if (
        image.ndim != 3
        or image.shape[2] != 3
        or image.shape[0] <= 0
        or image.shape[1] <= 0
        or image.dtype not in (np.dtype(np.uint8), np.dtype(np.float32))
    ):
        raise ValueError("rgb must be HWC uint8 or float32 with three channels")
    if max(image.shape[:2]) > max_analysis_long_edge:
        raise ValueError("rgb exceeds the configured matte analysis long edge")
    if matte_hint.dtype != np.float32 or matte_hint.shape != image.shape[:2]:
        raise ValueError("trimap must be float32 with the same H/W as rgb")
    if (
        not np.isfinite(matte_hint).all()
        or np.any(matte_hint < 0)
        or np.any(matte_hint > 1)
    ):
        raise ValueError("trimap must contain finite values in [0, 1]")
    if image.dtype == np.float32:
        for channel in range(3):
            plane = image[:, :, channel]
            if not np.isfinite(plane).all() or np.any(plane < 0) or np.any(plane > 1):
                raise ValueError("float32 rgb values must be finite and lie in [0, 1]")

    height, width = image.shape[:2]
    padded_height = (height + 31) // 32 * 32
    padded_width = (width + 31) // 32 * 32
    result = np.zeros((1, 4, padded_height, padded_width), dtype=np.float32)
    for channel in range(3):
        plane = result[0, channel, :height, :width]
        if image.dtype == np.uint8:
            plane[:] = image[:, :, channel]
            plane *= np.float32(2.0 / 255.0)
        else:
            plane[:] = image[:, :, channel]
            plane *= np.float32(2.0)
        plane -= np.float32(1.0)
    result[0, 3, :height, :width] = matte_hint
    return result


def _tile_starts(length: int, tile: int, max_step: int) -> tuple[int, ...]:
    if length <= 0 or length % 32:
        raise ValueError("matte dimensions must be positive multiples of 32")
    tile = min(tile, length)
    if tile % 32:
        raise ValueError("effective tile dimensions must be multiples of 32")
    if tile == length:
        return (0,)
    last = length - tile
    starts = list(range(0, last + 1, max_step))
    if starts[-1] != last:
        starts.append(last)
    return tuple(starts)


def plan_matte_tiles(
    pixel_values: Any,
    config: MatteTileConfig | None = None,
) -> tuple[MatteTile, ...]:
    """Plan tiles that intersect unknown trimap pixels.

    ``pixel_values`` follows the ViTMatte contract: contiguous float32
    ``[1, 4, H, W]`` with RGB in ``[-1, 1]`` and trimap in the fourth plane.
    Its H/W must already be padded to multiples of 32. The model is not run on
    known-only regions; all unknown pixels are covered by every tile that
    geometrically overlaps them.
    """
    import numpy as np

    selected = config or MatteTileConfig()
    values = np.asarray(pixel_values)
    if (
        values.dtype != np.float32
        or values.ndim != 4
        or values.shape[0] != 1
        or values.shape[1] != 4
        or not values.flags.c_contiguous
    ):
        raise ValueError("pixel_values must be contiguous float32 [1, 4, H, W]")
    _, _, height, width = values.shape
    if width <= 0 or height <= 0 or width % 32 or height % 32:
        raise ValueError("matte dimensions must be positive multiples of 32")
    for plane in values[0]:
        if not np.isfinite(plane).all():
            raise ValueError("pixel_values must contain only finite values")
    trimap = values[0, 3]
    if np.any(trimap < 0) or np.any(trimap > 1):
        raise ValueError("trimap values must lie in [0, 1]")

    if width * height <= selected.single_pass_max_pixels:
        if np.any((trimap > 0) & (trimap < 1)):
            return (MatteTile(0, 0, width, height),)
        return ()

    tile_width = min(selected.tile_size, width)
    tile_height = min(selected.tile_size, height)
    x_starts = _tile_starts(width, tile_width, int(selected.max_step))
    y_starts = _tile_starts(height, tile_height, int(selected.max_step))
    result: list[MatteTile] = []
    for y in y_starts:
        for x in x_starts:
            region = trimap[y : y + tile_height, x : x + tile_width]
            if np.any((region > 0) & (region < 1)):
                result.append(MatteTile(x, y, tile_width, tile_height))
    return tuple(result)


def _checked_alpha(result: Any, width: int, height: int) -> Any:
    import numpy as np

    alpha = np.asarray(result)
    expected = (1, 1, height, width)
    if alpha.dtype != np.float32 or alpha.shape != expected:
        raise MatteInferenceError(
            f"ViTMatte output must be float32 {expected}, got "
            f"{alpha.dtype} {alpha.shape}"
        )
    if not np.isfinite(alpha).all():
        raise MatteInferenceError("ViTMatte output contains non-finite values")
    return alpha[0, 0]


def _axis_feather(length: int, start: int, tile: int, image_length: int, feather: int):
    import numpy as np

    weights = np.ones(length, dtype=np.float32)
    if start > 0:
        ramp = (np.arange(length, dtype=np.float32) + 0.5) / feather
        np.minimum(weights, ramp, out=weights)
    if start + tile < image_length:
        ramp = (np.arange(length - 1, -1, -1, dtype=np.float32) + 0.5) / feather
        np.minimum(weights, ramp, out=weights)
    return weights


def refine_matte_tiled(
    pixel_values: Any,
    infer: Callable[[Any], Any],
    *,
    config: MatteTileConfig | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> Any:
    """Run a model callback by tile and return a full-resolution alpha tensor.

    The callback receives contiguous float32 ``[1, 4, tile_h, tile_w]`` input
    and must return ``[1, 1, tile_h, tile_w]``. Blending uses normalized
    linear edge weights. Known trimap endpoints are restored exactly. Full
    image output accumulators are float32; frames within the configured
    reference single-pass area run whole, while larger frames keep model
    input/output bounded to one tile at a time.
    """
    import numpy as np

    values = np.asarray(pixel_values)
    tiles = plan_matte_tiles(values, config)
    height, width = values.shape[2:]
    trimap = values[0, 3]
    if not tiles:
        if cancel_check is not None:
            cancel_check()
        alpha = trimap.copy()
        alpha[trimap <= 0] = 0
        alpha[trimap >= 1] = 1
        return alpha[None, None]

    if len(tiles) == 1 and tiles[0].width == width and tiles[0].height == height:
        if cancel_check is not None:
            cancel_check()
        tile_alpha = _checked_alpha(infer(values), width, height).copy()
        if cancel_check is not None:
            cancel_check()
        tile_alpha[trimap <= 0] = 0
        tile_alpha[trimap >= 1] = 1
        np.clip(tile_alpha, 0, 1, out=tile_alpha)
        return tile_alpha[None, None]

    selected = config or MatteTileConfig()
    feather = int(selected.feather)
    accumulation = np.zeros((height, width), dtype=np.float32)
    weights = np.zeros_like(accumulation)
    for tile in tiles:
        if cancel_check is not None:
            cancel_check()
        region = values[
            :, :, tile.y : tile.y + tile.height, tile.x : tile.x + tile.width
        ]
        tile_input = np.ascontiguousarray(region)
        tile_alpha = _checked_alpha(infer(tile_input), tile.width, tile.height).copy()
        if cancel_check is not None:
            cancel_check()
        np.clip(tile_alpha, 0, 1, out=tile_alpha)
        horizontal = _axis_feather(tile.width, tile.x, tile.width, width, feather)
        vertical = _axis_feather(tile.height, tile.y, tile.height, height, feather)
        tile_weights = vertical[:, None] * horizontal[None, :]
        np.multiply(tile_alpha, tile_weights, out=tile_alpha)
        target = np.s_[tile.y : tile.y + tile.height, tile.x : tile.x + tile.width]
        accumulation[target] += tile_alpha
        weights[target] += tile_weights

    unknown = (trimap > 0) & (trimap < 1)
    if np.any(unknown & (weights <= 0)):
        raise MatteInferenceError("tile plan left unknown trimap pixels uncovered")
    if cancel_check is not None:
        cancel_check()
    covered = weights > 0
    np.divide(accumulation, weights, out=accumulation, where=covered)
    accumulation[trimap <= 0] = 0
    accumulation[trimap >= 1] = 1
    np.clip(accumulation, 0, 1, out=accumulation)
    return accumulation[None, None]


def _create_onnx_session(model_path: str | Path, intra_op_num_threads: int) -> Any:
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise MatteInferenceError(
            "ViTMatte ONNX inference requires the optional inference dependencies"
        ) from exc
    options = ort.SessionOptions()
    options.intra_op_num_threads = intra_op_num_threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def _sha256_file(path: Path) -> str:
    with path.open("rb") as model_file:
        return hashlib.file_digest(model_file, "sha256").hexdigest()


class ViTMatteOnnx:
    """CPU ONNX adapter for a caller-supplied ViTMatte graph.

    The class never downloads or distributes weights. Callers must separately
    establish the model artifact's rights and supply a SHA-256-verified,
    dynamic-spatial ONNX export whose names match ``pixel_values``/``alpha``.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        model_sha256: str,
        intra_op_num_threads: int = 1,
        session_factory: Callable[[str | Path, int], Any] | None = None,
    ) -> None:
        if intra_op_num_threads < 1:
            raise ValueError("intra_op_num_threads must be positive")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", model_sha256):
            raise ValueError("model_sha256 must contain 64 hexadecimal characters")
        selected_path = Path(model_path)
        if not selected_path.is_file():
            raise MatteInferenceError("ViTMatte ONNX model file does not exist")
        if _sha256_file(selected_path) != model_sha256.lower():
            raise MatteInferenceError("ViTMatte ONNX SHA-256 does not match")
        self.model_sha256 = model_sha256.lower()
        try:
            from importlib.metadata import version

            self.runtime_version = version("onnxruntime")
        except Exception:  # metadata is optional for injected test sessions
            self.runtime_version = "unknown"
        self.provider = "CPUExecutionProvider"
        factory = session_factory or _create_onnx_session
        self._session = factory(selected_path, intra_op_num_threads)
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if {item.name for item in inputs} != {"pixel_values"} or not any(
            item.name == "alpha" for item in outputs
        ):
            self.close()
            raise MatteInferenceError(
                "ONNX graph must accept pixel_values and expose alpha"
            )
        input_shape = getattr(inputs[0], "shape", None)
        if input_shape is not None and (
            len(input_shape) != 4
            or tuple(input_shape[0:2]) != (1, 4)
            or any(
                dimension is not None and not isinstance(dimension, str)
                for dimension in input_shape[2:]
            )
        ):
            self.close()
            raise MatteInferenceError(
                "ONNX graph must have dynamic height and width for tile inference"
            )

    @classmethod
    def from_validation_report(
        cls,
        model_path: str | Path,
        *,
        intra_op_num_threads: int = 1,
        session_factory: Callable[[str | Path, int], Any] | None = None,
    ) -> ViTMatteOnnx:
        """Load a graph only when its pinned-checkpoint export passed validation."""
        selected_path = Path(model_path)
        report_path = Path(f"{selected_path}.validation.json")
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MatteInferenceError(
                "ViTMatte validation report is missing or invalid"
            ) from exc
        if (
            report.get("model_id") != "vitmatte-small-composition-1k"
            or report.get("checkpoint_sha256") != VITMATTE_CHECKPOINT_SHA256
            or report.get("passed") is not True
            or report.get("dynamic_shapes") is not True
            or report.get("input_name") != "pixel_values"
            or report.get("output_name") != "alpha"
        ):
            raise MatteInferenceError(
                "ViTMatte validation report does not match the pinned Small model"
            )
        expected_sha256 = report.get("onnx_sha256")
        if not isinstance(expected_sha256, str) or not re.fullmatch(
            r"[0-9a-fA-F]{64}", expected_sha256
        ):
            raise MatteInferenceError("ViTMatte report has no valid ONNX SHA-256")
        return cls(
            selected_path,
            model_sha256=expected_sha256,
            intra_op_num_threads=intra_op_num_threads,
            session_factory=session_factory,
        )

    def infer(self, pixel_values: Any) -> Any:
        import numpy as np

        if self._session is None:
            raise MatteInferenceError("ViTMatte session has been closed")
        values = np.asarray(pixel_values)
        if (
            values.dtype != np.float32
            or values.ndim != 4
            or values.shape[0:2] != (1, 4)
            or values.shape[2] <= 0
            or values.shape[3] <= 0
            or values.shape[2] % 32
            or values.shape[3] % 32
            or not values.flags.c_contiguous
            or not all(np.isfinite(plane).all() for plane in values[0])
        ):
            raise ValueError(
                "pixel_values must be contiguous finite float32 [1, 4, H, W] "
                "with H/W divisible by 32"
            )
        outputs = self._session.run(["alpha"], {"pixel_values": values})
        return _checked_alpha(outputs[0], values.shape[3], values.shape[2])[None, None]

    def close(self) -> None:
        """Drop the native ONNX session so its caller can release the model stage."""
        self._session = None

    def __enter__(self) -> ViTMatteOnnx:
        if self._session is None:
            raise MatteInferenceError("ViTMatte session has been closed")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = [
    "MatteInferenceError",
    "MatteTile",
    "MatteTileConfig",
    "ViTMatteOnnx",
    "plan_matte_tiles",
    "prepare_vitmatte_input",
    "refine_matte_tiled",
]
