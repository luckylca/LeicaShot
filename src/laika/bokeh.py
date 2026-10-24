"""CPU ONNX Runtime adapter for the BokehMe ARNet defocus renderer."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from math import ceil, floor, isfinite, log2
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
BOKEH_LOW_MEMORY_TILE_EDGE = 256
BOKEH_BALANCED_TILE_EDGE = 512
BOKEH_TILE_EDGE = BOKEH_BALANCED_TILE_EDGE
BOKEH_TILE_HALO = 16
BOKEH_FULL_INFERENCE_PIXEL_LIMIT = 1_000_000


@dataclass(frozen=True)
class BokehTileProgress:
    """One full-context ARNet tile entering or leaving inference."""

    phase: Literal["started", "complete"]
    index: int
    count: int
    frame_width: int
    frame_height: int
    x: int
    y: int
    width: int
    height: int


class BokehInferenceError(RuntimeError):
    """The local ARNet model or one of its tensor contracts is invalid."""


@dataclass(frozen=True)
class BokehPrediction:
    """One ARNet pass in RGB NCHW float32, with timings and backend metadata."""

    rgb_nchw: Any | None
    width: int
    height: int
    preprocess_seconds: float
    inference_seconds: float
    tiled: bool = False
    tile_count: int = 1


@dataclass(frozen=True)
class BokehRuntimeInfo:
    """Resolved model and CPU runtime details for diagnostics and benchmarks."""

    model_sha256: str
    runtime_version: str
    provider: str
    strategy: str
    intra_op_num_threads: int
    inference_tile_edge: int
    full_inference_pixel_limit: int
    cpu_arena_reuse: bool
    arena_shrink_every_tiles: int

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
        inference_tile_edge: int | None = None,
        full_inference_pixel_limit: int = BOKEH_FULL_INFERENCE_PIXEL_LIMIT,
        expected_sha256: str | None = None,
        session_factory: Any | None = None,
        split_graph_dir: str | os.PathLike[str] | None = None,
        split_graph_edge: int = 256,
        reuse_cpu_arena: bool | None = None,
        arena_shrink_every_tiles: int | None = None,
    ) -> None:
        if strategy not in BOKEH_STRATEGIES:
            raise ValueError(f"strategy must be one of {', '.join(BOKEH_STRATEGIES)}")
        if inference_tile_edge is None:
            inference_tile_edge = (
                BOKEH_LOW_MEMORY_TILE_EDGE
                if strategy == "low-memory"
                else BOKEH_BALANCED_TILE_EDGE
            )
        if intra_op_num_threads is None:
            intra_op_num_threads = (
                1 if strategy == "low-memory" else min(4, os.cpu_count() or 1)
            )
        if intra_op_num_threads < 1:
            raise ValueError("intra_op_num_threads must be positive")
        if inference_tile_edge < 64 or inference_tile_edge % 2:
            raise ValueError("inference_tile_edge must be an even value of at least 64")
        if full_inference_pixel_limit < 1:
            raise ValueError("full_inference_pixel_limit must be positive")

        if split_graph_edge < 64 or split_graph_edge % 2:
            raise ValueError("split_graph_edge must be even and at least 64")
        # Reusing the CPU allocator across hundreds of equal-sized convolution
        # tiles avoids repeatedly allocating large feature maps. Keep the
        # dynamic-shape memory pattern disabled for low-memory workers.
        # A per-process opt-out preserves the original low-RSS strategy for
        # tightly constrained or multi-worker deployments.
        if reuse_cpu_arena is None:
            arena_setting = os.environ.get("LAIKA_ARNET_CPU_ARENA", "0").strip().lower()
            if arena_setting not in {
                "1",
                "0",
                "true",
                "false",
                "yes",
                "no",
                "on",
                "off",
            }:
                raise ValueError("LAIKA_ARNET_CPU_ARENA must be 1/0 or true/false")
            reuse_cpu_arena = arena_setting in {"1", "true", "yes", "on"}
        elif not isinstance(reuse_cpu_arena, bool):
            raise TypeError("reuse_cpu_arena must be a bool or None")
        self.reuse_cpu_arena = strategy == "balanced" or reuse_cpu_arena
        if arena_shrink_every_tiles is None:
            raw_interval = os.environ.get("LAIKA_ARNET_ARENA_SHRINK_TILES", "0")
            try:
                arena_shrink_every_tiles = int(raw_interval)
            except ValueError as exc:
                raise ValueError(
                    "LAIKA_ARNET_ARENA_SHRINK_TILES must be an integer"
                ) from exc
        if (
            not isinstance(arena_shrink_every_tiles, int)
            or not 0 <= arena_shrink_every_tiles <= 4096
        ):
            raise ValueError("arena_shrink_every_tiles must be between 0 and 4096")
        self.arena_shrink_every_tiles = (
            arena_shrink_every_tiles if self.reuse_cpu_arena else 0
        )
        self._arena_tile_calls = 0
        self._split_graph_edge = split_graph_edge
        self._split_sessions: tuple[Any, Any, Any] | None = None
        self._split_graph_paths: tuple[Path, Path, Path] | None = None
        self.model_path = Path(model_path)
        self.strategy = strategy
        self.intra_op_num_threads = intra_op_num_threads
        self.inference_tile_edge = inference_tile_edge
        self.full_inference_pixel_limit = full_inference_pixel_limit
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
        options.enable_cpu_mem_arena = self.reuse_cpu_arena
        options.enable_mem_pattern = strategy == "balanced"
        options.enable_mem_reuse = True
        self._ort = ort
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

        if split_graph_dir is not None:
            if strategy != "low-memory":
                raise ValueError("split graphs require the low-memory strategy")
            from .arnet_split import verify_split_graph_dir

            try:
                self._split_graph_paths = verify_split_graph_dir(
                    Path(split_graph_dir), actual_sha256
                )
            except (OSError, ValueError) as exc:
                raise BokehInferenceError(
                    f"invalid ARNet split graph files: {exc}"
                ) from exc
        self._split_session_options = options
        self._split_session_factory = factory
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
            inference_tile_edge=self.inference_tile_edge,
            full_inference_pixel_limit=self.full_inference_pixel_limit,
            cpu_arena_reuse=self.reuse_cpu_arena,
            arena_shrink_every_tiles=self.arena_shrink_every_tiles,
        )

    def infer(
        self,
        rgb: Any,
        defocus: Any,
        *,
        tile_callback: Callable[[BokehTileProgress], None] | None = None,
    ) -> BokehPrediction:
        """Run ARNet once and return its owned NCHW output buffer."""
        image, defocus_map = self._validated_inputs(rgb, defocus)
        height, width = image.shape[:2]
        if self.will_split(width, height):
            return self._infer_split(image, defocus_map, tile_callback=tile_callback)
        if self.will_tile(width, height):
            return self._infer_tiled(image, defocus_map, tile_callback=tile_callback)
        return self._infer_one(image, defocus_map)

    def will_tile(self, width: int, height: int) -> bool:
        """Return whether this strategy uses the validated even-frame tile path."""
        return (
            self.strategy == "low-memory"
            and width * height > self.full_inference_pixel_limit
            and height % 2 == 0
            and width % 2 == 0
        )

    def will_split(self, width: int, height: int) -> bool:
        """Use global-resize / tiled-convolution graphs for large odd frames."""
        return (
            self.strategy == "low-memory"
            and self._split_graph_paths is not None
            and width * height > self.full_inference_pixel_limit
            and (height % 2 != 0 or width % 2 != 0)
        )

    def _infer_split(
        self,
        image: Any,
        defocus_map: Any,
        *,
        tile_callback: Callable[[BokehTileProgress], None] | None = None,
    ) -> BokehPrediction:
        """Preserve global ONNX resize coordinates; tile only local conv layers.

        The eight 3x3 spatial convolutions have half-resolution receptive
        radius eight; eight source pixels of overlap are sufficient.
        The extracted prefix/suffix retain the original global align-corners
        resizes, which a naively tiled full ARNet model cannot reproduce.
        """
        import numpy as np

        if self._split_graph_paths is None:
            raise BokehInferenceError("ARNet split graphs have not been configured")
        if self._split_sessions is None:
            self._split_sessions = tuple(
                self._split_session_factory(
                    str(path),
                    sess_options=self._split_session_options,
                    providers=["CPUExecutionProvider"],
                )
                for path in self._split_graph_paths
            )
        prefix, core, suffix = self._split_sessions
        height, width = image.shape[:2]
        started = time.perf_counter()
        tensor = np.empty((1, 3, height, width), dtype=np.float32)
        for channel in range(3):
            tensor[0, channel] = image[:, :, channel]
            if image.dtype == np.uint8:
                np.divide(tensor[0, channel], np.float32(255.0), out=tensor[0, channel])
        input_defocus = defocus_map.reshape((1, 1, height, width))
        preprocessing = time.perf_counter() - started
        run_start = time.perf_counter()
        feature = prefix.run(
            ["/Concat_5_output_0"], {"image": tensor, "defocus": input_defocus}
        )[0]
        _, _, feature_height, feature_width = feature.shape
        upsampled = np.empty(
            (1, 4, feature_height * 2, feature_width * 2), dtype=np.float16
        )
        edge = self._split_graph_edge
        halo = 8
        tile_count = ceil(feature_height / edge) * ceil(feature_width / edge)
        index = 0
        for y0 in range(0, feature_height, edge):
            y1 = min(feature_height, y0 + edge)
            ya, yb = max(0, y0 - halo), min(feature_height, y1 + halo)
            for x0 in range(0, feature_width, edge):
                x1 = min(feature_width, x0 + edge)
                xa, xb = max(0, x0 - halo), min(feature_width, x1 + halo)
                index += 1
                if tile_callback is not None:
                    event = dict(
                        index=index,
                        count=tile_count,
                        frame_width=width,
                        frame_height=height,
                        x=2 * x0,
                        y=2 * y0,
                        width=min(width - 2 * x0, 2 * (x1 - x0)),
                        height=min(height - 2 * y0, 2 * (y1 - y0)),
                    )
                    tile_callback(BokehTileProgress(phase="started", **event))
                tile = np.ascontiguousarray(feature[:, :, ya:yb, xa:xb])
                part = core.run(
                    ["/upsample/DepthToSpace_output_0"],
                    {"/Concat_5_output_0": tile},
                )[0]
                upsampled[:, :, 2 * y0 : 2 * y1, 2 * x0 : 2 * x1] = part[
                    :, :, 2 * (y0 - ya) : 2 * (y1 - ya), 2 * (x0 - xa) : 2 * (x1 - xa)
                ]
                del tile, part
                if tile_callback is not None:
                    tile_callback(BokehTileProgress(phase="complete", **event))
        del feature
        output = suffix.run(
            ["output"],
            {"image": tensor, "/upsample/DepthToSpace_output_0": upsampled},
        )[0]
        del upsampled, tensor
        np.clip(output, np.float32(0.0), np.float32(1.0), out=output)
        seconds = time.perf_counter() - run_start
        return BokehPrediction(
            rgb_nchw=output,
            width=width,
            height=height,
            preprocess_seconds=preprocessing,
            inference_seconds=seconds,
            tiled=True,
            tile_count=tile_count,
        )

    def infer_into_rgb(
        self,
        rgb: Any,
        defocus: Any,
        output: Any,
        *,
        tile_callback: Callable[[BokehTileProgress], None] | None = None,
    ) -> BokehPrediction:
        """Infer directly into a same-size HWC float32 destination.

        The tiled path writes each core into the caller-owned output as it is
        produced. This avoids assembling a second full-resolution NCHW tensor
        before the renderer converts it to HWC.
        """
        import numpy as np

        image, defocus_map = self._validated_inputs(rgb, defocus)
        height, width = image.shape[:2]
        destination = np.asarray(output)
        if (
            destination.dtype != np.float32
            or destination.shape != (height, width, 3)
            or not destination.flags.c_contiguous
            or not destination.flags.writeable
        ):
            raise ValueError(
                "output must be a writable contiguous H×W×3 float32 RGB array"
            )

        if self.will_split(width, height):
            prediction = self._infer_split(
                image, defocus_map, tile_callback=tile_callback
            )
            for channel in range(3):
                np.copyto(destination[:, :, channel], prediction.rgb_nchw[0, channel])
            return BokehPrediction(
                rgb_nchw=None,
                width=width,
                height=height,
                preprocess_seconds=prediction.preprocess_seconds,
                inference_seconds=prediction.inference_seconds,
                tiled=True,
                tile_count=prediction.tile_count,
            )
        if self.will_tile(width, height):
            return self._infer_tiled_into_rgb(
                image,
                defocus_map,
                destination,
                tile_callback=tile_callback,
            )

        prediction = self._infer_one(image, defocus_map)
        for channel in range(3):
            np.copyto(destination[:, :, channel], prediction.rgb_nchw[0, channel])
        return BokehPrediction(
            rgb_nchw=None,
            width=width,
            height=height,
            preprocess_seconds=prediction.preprocess_seconds,
            inference_seconds=prediction.inference_seconds,
        )

    def _validated_inputs(self, rgb: Any, defocus: Any) -> tuple[Any, Any]:
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
        return image, defocus_map

    def _infer_one(self, image: Any, defocus_map: Any) -> BokehPrediction:
        """Run one native ARNet graph invocation on the supplied image region."""
        import numpy as np

        height, width = image.shape[:2]
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
        self._arena_tile_calls += 1
        if (
            self.arena_shrink_every_tiles
            and self._arena_tile_calls % self.arena_shrink_every_tiles == 0
        ):
            # The next tiles still use the same ONNX session and fixed
            # model; tell ORT to give back retained arena regions after
            # this inference. No change to convolution calculations.
            run_options = self._ort.RunOptions()
            run_options.add_run_config_entry(
                "memory.enable_memory_arena_shrinkage", "cpu:0"
            )
            output = self._session.run(
                [BOKEH_OUTPUT],
                {"image": tensor, "defocus": defocus_tensor},
                run_options=run_options,
            )[0]
        else:
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

    def _infer_tiled(
        self,
        image: Any,
        defocus_map: Any,
        *,
        tile_callback: Callable[[BokehTileProgress], None] | None = None,
    ) -> BokehPrediction:
        """Run exact convolutional tiles with full receptive-field overlap.

        The pinned ARNet applies an even-aligned 2×2 space-to-depth transform,
        then eight 3×3 convolutions at half resolution. A 16-pixel halo on
        every side therefore covers the complete input-space receptive radius.
        Tile origins and extents remain even so each space-to-depth phase is
        identical to the full-frame graph. Only the core region is copied into
        the output tensor.
        """
        import numpy as np

        height, width = image.shape[:2]
        output = np.empty((1, 3, height, width), dtype=np.float32)
        tile_index = 0
        preprocess_seconds = 0.0
        inference_seconds = 0.0
        edge = self.inference_tile_edge
        halo = BOKEH_TILE_HALO
        tile_count = ceil(height / edge) * ceil(width / edge)

        for core_y0 in range(0, height, edge):
            core_y1 = min(height, core_y0 + edge)
            tile_y0 = max(0, core_y0 - halo)
            tile_y1 = min(height, core_y1 + halo)
            for core_x0 in range(0, width, edge):
                core_x1 = min(width, core_x0 + edge)
                tile_x0 = max(0, core_x0 - halo)
                tile_x1 = min(width, core_x1 + halo)
                tile_index += 1
                if tile_callback is not None:
                    tile_callback(
                        BokehTileProgress(
                            phase="started",
                            index=tile_index,
                            count=tile_count,
                            frame_width=width,
                            frame_height=height,
                            x=core_x0,
                            y=core_y0,
                            width=core_x1 - core_x0,
                            height=core_y1 - core_y0,
                        )
                    )
                prediction = self._infer_one(
                    image[tile_y0:tile_y1, tile_x0:tile_x1],
                    defocus_map[tile_y0:tile_y1, tile_x0:tile_x1],
                )
                local_y0 = core_y0 - tile_y0
                local_y1 = local_y0 + core_y1 - core_y0
                local_x0 = core_x0 - tile_x0
                local_x1 = local_x0 + core_x1 - core_x0
                output[0, :, core_y0:core_y1, core_x0:core_x1] = prediction.rgb_nchw[
                    0, :, local_y0:local_y1, local_x0:local_x1
                ]
                preprocess_seconds += prediction.preprocess_seconds
                inference_seconds += prediction.inference_seconds
                del prediction
                if tile_callback is not None:
                    tile_callback(
                        BokehTileProgress(
                            phase="complete",
                            index=tile_index,
                            count=tile_count,
                            frame_width=width,
                            frame_height=height,
                            x=core_x0,
                            y=core_y0,
                            width=core_x1 - core_x0,
                            height=core_y1 - core_y0,
                        )
                    )

        return BokehPrediction(
            rgb_nchw=output,
            width=width,
            height=height,
            preprocess_seconds=preprocess_seconds,
            inference_seconds=inference_seconds,
            tiled=True,
            tile_count=tile_count,
        )

    def _infer_tiled_into_rgb(
        self,
        image: Any,
        defocus_map: Any,
        output: Any,
        *,
        tile_callback: Callable[[BokehTileProgress], None] | None = None,
    ) -> BokehPrediction:
        """Write even-aligned ARNet tile cores into an HWC destination."""
        height, width = image.shape[:2]
        tile_index = 0
        preprocess_seconds = 0.0
        inference_seconds = 0.0
        edge = self.inference_tile_edge
        halo = BOKEH_TILE_HALO
        tile_count = ceil(height / edge) * ceil(width / edge)

        for core_y0 in range(0, height, edge):
            core_y1 = min(height, core_y0 + edge)
            tile_y0 = max(0, core_y0 - halo)
            tile_y1 = min(height, core_y1 + halo)
            for core_x0 in range(0, width, edge):
                core_x1 = min(width, core_x0 + edge)
                tile_x0 = max(0, core_x0 - halo)
                tile_x1 = min(width, core_x1 + halo)
                tile_index += 1
                if tile_callback is not None:
                    tile_callback(
                        BokehTileProgress(
                            phase="started",
                            index=tile_index,
                            count=tile_count,
                            frame_width=width,
                            frame_height=height,
                            x=core_x0,
                            y=core_y0,
                            width=core_x1 - core_x0,
                            height=core_y1 - core_y0,
                        )
                    )
                prediction = self._infer_one(
                    image[tile_y0:tile_y1, tile_x0:tile_x1],
                    defocus_map[tile_y0:tile_y1, tile_x0:tile_x1],
                )
                local_y0 = core_y0 - tile_y0
                local_y1 = local_y0 + core_y1 - core_y0
                local_x0 = core_x0 - tile_x0
                local_x1 = local_x0 + core_x1 - core_x0
                for channel in range(3):
                    output[core_y0:core_y1, core_x0:core_x1, channel] = (
                        prediction.rgb_nchw[
                            0, channel, local_y0:local_y1, local_x0:local_x1
                        ]
                    )
                preprocess_seconds += prediction.preprocess_seconds
                inference_seconds += prediction.inference_seconds
                del prediction
                if tile_callback is not None:
                    tile_callback(
                        BokehTileProgress(
                            phase="complete",
                            index=tile_index,
                            count=tile_count,
                            frame_width=width,
                            frame_height=height,
                            x=core_x0,
                            y=core_y0,
                            width=core_x1 - core_x0,
                            height=core_y1 - core_y0,
                        )
                    )

        return BokehPrediction(
            rgb_nchw=None,
            width=width,
            height=height,
            preprocess_seconds=preprocess_seconds,
            inference_seconds=inference_seconds,
            tiled=True,
            tile_count=tile_count,
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
        self._split_sessions = None

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


def _mask_coc_inplace(coc: Any, alpha: Any, row_chunk_height: int) -> None:
    """Apply subject protection with bounded complement storage."""
    import numpy as np

    height = coc.shape[0]
    for row_start in range(0, height, row_chunk_height):
        row_stop = min(height, row_start + row_chunk_height)
        alpha_rows = alpha[row_start:row_stop]
        complement = np.empty_like(alpha_rows)
        np.subtract(np.float32(1.0), alpha_rows, out=complement)
        np.multiply(coc[row_start:row_stop], complement, out=coc[row_start:row_stop])


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


def _blend_resized_nchw_inplace(
    composite: Any,
    nchw: Any,
    plate_coc: Any,
    previous_scale: float,
    row_chunk_height: int,
    cv2: Any,
    channel_buffer: Any,
) -> None:
    """Upscale and blend one NCHW channel at a time into a full-size frame.

    This keeps OpenCV's exact full-frame linear resize while replacing the
    additional full-size HWC float buffer with one reusable H×W channel.
    """
    import numpy as np

    height, width = composite.shape[:2]
    for channel in range(3):
        cv2.resize(
            nchw[0, channel],
            (width, height),
            dst=channel_buffer,
            interpolation=cv2.INTER_LINEAR,
        )
        for row_start in range(0, height, row_chunk_height):
            row_stop = min(height, row_start + row_chunk_height)
            weight = _smooth(
                (
                    np.abs(plate_coc[row_start:row_stop])
                    - np.float32(5.0 * previous_scale)
                )
                / np.float32(3.0 * previous_scale)
            )[:, :, None]
            weight_2d = weight[:, :, 0]
            current = composite[row_start:row_stop, :, channel]
            current *= 1.0 - weight_2d
            current += channel_buffer[row_start:row_stop] * weight_2d


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
    subject_alpha: Any | None = None,
    row_chunk_height: int = 256,
    cancel_check: Callable[[], None] | None = None,
    pass_callback: Callable[[int, int, float], None] | None = None,
    tile_progress_callback: Callable[[int, int, BokehTileProgress], None] | None = None,
    stage_callback: Callable[[str, float], None] | None = None,
) -> BokehRenderResult:
    """Render a full-size photo through ARNet and depth-layered highlight scatter."""
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
    alpha: Any | None = None
    if subject_alpha is not None:
        alpha = np.asarray(subject_alpha)
        if (
            alpha.dtype != np.float32
            or alpha.shape != (height, width)
            or not alpha.flags.c_contiguous
            or not np.isfinite(alpha).all()
            or float(alpha.min()) < 0.0
            or float(alpha.max()) > 1.0
        ):
            raise ValueError("subject_alpha must be matching contiguous float32 [0, 1]")
    if row_chunk_height < 1:
        raise ValueError("row_chunk_height must be positive")
    started = time.perf_counter()
    stage_seconds: dict[str, float] = {}
    optics_started = time.perf_counter()
    coc = full_frame_coc(depth_map, focus, aperture)
    if alpha is not None:
        _mask_coc_inplace(coc, alpha, row_chunk_height)
    scatter_coc = coc
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

    from laika.highlights import (
        composite_highlights,
        extract_highlight_radiance,
        extract_highlights,
        recover_clipped_highlights_inplace,
        scatter_highlights,
        soften_highlights_inplace,
    )

    defer_highlight_radiance = getattr(model, "strategy", "low-memory") == "low-memory"
    if stage_callback is not None:
        stage_callback("highlight-extraction", 0.0)
    stage_started = time.perf_counter()
    split = extract_highlights(
        image,
        cancel_check=cancel_check,
        row_chunk_height=row_chunk_height,
        include_radiance=not defer_highlight_radiance,
    )
    stage_seconds["highlight_extraction"] = time.perf_counter() - stage_started
    base_rgb = split.base_rgb
    highlight_linear: Any | None = split.highlight_linear
    # Keep the owned arrays directly, not through the split holder. In
    # low-memory mode that holder only owns base_rgb; in balanced mode it
    # would otherwise keep a second reference to the full float32 radiance.
    del split
    if stage_callback is not None:
        stage_callback("highlight-extraction", 1.0)

    stage_started = time.perf_counter()
    image_float = base_rgb.astype(np.float32)
    np.divide(image_float, np.float32(255.0), out=image_float)
    plate_coc = coc.copy() if alpha is not None else coc
    if alpha is not None:
        from laika.protection import fill_subject_background_inplace

        fill_started = time.perf_counter()
        fill_subject_background_inplace(
            image_float,
            plate_coc,
            alpha,
            cancel_check=cancel_check,
        )
        # Pull-push interpolation can accumulate a few float32 ULPs beyond
        # the normalized sRGB interval. ARNet's input contract is strict, so
        # project the filled plate back into the representable RGB range in
        # place before any resized inference pass.
        np.clip(image_float, np.float32(0.0), np.float32(1.0), out=image_float)
        stage_seconds["subject_background_plate"] = time.perf_counter() - fill_started
    plate_max_abs_coc = _max_abs_rows(plate_coc)
    target_scale = max(1.0, float(np.float32(plate_max_abs_coc / 8.0)))
    total_passes = ceil(log2(target_scale)) + 1
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
            # Area downsampling can round normalized float32 sRGB just beyond
            # the ARNet input interval. Keep each resized pass in contract
            # without allocating a second image-sized buffer.
            np.clip(small_rgb, np.float32(0.0), np.float32(1.0), out=small_rgb)
        if (pass_width, pass_height) == (width, height):
            small_coc = plate_coc
        else:
            small_coc = cv2.resize(
                plate_coc,
                (pass_width, pass_height),
                interpolation=cv2.INTER_LINEAR,
            )
        defocus = np.empty((pass_height, pass_width), dtype=np.float32)
        np.divide(small_coc, np.float32(actual_scale), out=defocus)
        np.divide(defocus, np.float32(10.0), out=defocus)
        np.clip(defocus, -1.0, 1.0, out=defocus)
        infer_into_rgb = getattr(model, "infer_into_rgb", None)
        will_tile = getattr(model, "will_tile", None)
        will_split = getattr(model, "will_split", None)
        direct_tiled_output = (
            (pass_width, pass_height) == (width, height)
            and callable(infer_into_rgb)
            and (
                (callable(will_tile) and will_tile(width, height))
                or (callable(will_split) and will_split(width, height))
            )
        )
        stream_blend = (
            getattr(model, "strategy", None) == "low-memory"
            and composite is not None
            and not direct_tiled_output
        )
        supports_tiling = (
            callable(will_tile) and will_tile(pass_width, pass_height)
        ) or (callable(will_split) and will_split(pass_width, pass_height))
        model_tile_callback = None
        if supports_tiling and (
            tile_progress_callback is not None or cancel_check is not None
        ):

            def on_tile(tile: BokehTileProgress) -> None:
                if cancel_check is not None:
                    cancel_check()
                if tile_progress_callback is not None:
                    tile_progress_callback(len(scales) + 1, total_passes, tile)

            model_tile_callback = on_tile
        if direct_tiled_output:
            enlarged = np.empty((height, width, 3), dtype=np.float32)
            if model_tile_callback is None:
                prediction = infer_into_rgb(small_rgb, defocus, enlarged)
            else:
                prediction = infer_into_rgb(
                    small_rgb,
                    defocus,
                    enlarged,
                    tile_callback=model_tile_callback,
                )
        else:
            if model_tile_callback is None:
                prediction = model.infer(small_rgb, defocus)
            else:
                prediction = model.infer(
                    small_rgb,
                    defocus,
                    tile_callback=model_tile_callback,
                )
            if stream_blend:
                channel_buffer = np.empty((height, width), dtype=np.float32)
                _blend_resized_nchw_inplace(
                    composite,
                    prediction.rgb_nchw,
                    plate_coc,
                    previous_scale,
                    row_chunk_height,
                    cv2,
                    channel_buffer,
                )
                del channel_buffer
                enlarged = None
            else:
                enlarged = _resize_nchw_to_rgb(prediction.rgb_nchw, width, height, cv2)
        del prediction, small_rgb, small_coc, defocus

        if composite is None:
            composite = enlarged
            enlarged = None
        elif not stream_blend:
            for row_start in range(0, height, row_chunk_height):
                row_stop = min(height, row_start + row_chunk_height)
                weight = _smooth(
                    (
                        np.abs(plate_coc[row_start:row_stop])
                        - np.float32(5.0 * previous_scale)
                    )
                    / np.float32(3.0 * previous_scale)
                )[:, :, None]
                current = composite[row_start:row_stop]
                current *= 1.0 - weight
                current += enlarged[row_start:row_stop] * weight
            enlarged = None

        scales.append(actual_scale)
        if pass_callback is not None:
            pass_callback(len(scales), total_passes, actual_scale)
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
        weight = _smooth(np.abs(plate_coc[row_start:row_stop]))[:, :, None]
        mixed = image_float[row_start:row_stop] * (1.0 - weight)
        mixed += composite[row_start:row_stop] * weight
        np.clip(mixed, 0.0, 1.0, out=mixed)
        np.multiply(mixed, 255.0, out=mixed)
        np.rint(mixed, out=mixed)
        final_rgb[row_start:row_stop] = mixed.astype(np.uint8)
    stage_seconds["composite"] = time.perf_counter() - stage_started
    # ARNet's float32 working frames are no longer used once final_rgb is ready.
    # Release them before highlight recovery/scatter, which has its own large
    # full-resolution buffers and is the next peak-memory stage.
    release_started = time.perf_counter()
    del image_float, composite, enlarged, plate_coc
    stage_seconds["release_arnet_buffers"] = time.perf_counter() - release_started
    if highlight_linear is None:
        if stage_callback is not None:
            stage_callback("highlight-radiance", 0.0)
        stage_started = time.perf_counter()

        def on_radiance_progress(fraction: float) -> None:
            if stage_callback is not None:
                stage_callback("highlight-radiance", fraction)

        highlight_linear = extract_highlight_radiance(
            image,
            base_rgb,
            cancel_check=cancel_check,
            progress_callback=(
                on_radiance_progress if stage_callback is not None else None
            ),
            row_chunk_height=row_chunk_height,
        )
        stage_seconds["highlight_radiance"] = time.perf_counter() - stage_started
    else:
        # Balanced already built this buffer alongside the base plate.
        stage_seconds["highlight_radiance"] = 0.0
    if bool(np.any(highlight_linear)):
        if stage_callback is not None:
            stage_callback("highlight-recovery", 0.0)
        stage_started = time.perf_counter()
        recover_clipped_highlights_inplace(
            highlight_linear,
            image,
            scatter_coc,
            row_chunk_height=row_chunk_height,
            cancel_check=cancel_check,
        )
        stage_seconds["highlight_recovery"] = time.perf_counter() - stage_started
        if stage_callback is not None:
            stage_callback("highlight-recovery", 1.0)
        stage_started = time.perf_counter()

        def on_scatter_progress(fraction: float) -> None:
            if stage_callback is not None:
                stage_callback("highlight-scatter", fraction)

        scattered = scatter_highlights(
            highlight_linear,
            scatter_coc,
            aperture,
            radius_scale=3.0,
            field_optics=True,
            row_chunk_height=row_chunk_height,
            cancel_check=cancel_check,
            progress_callback=on_scatter_progress,
        )
        # Scatter has consumed the source radiance and returned an independent
        # output. Release the full-resolution HWC float32 input before the
        # following softening and composite stages allocate their work buffers.
        del highlight_linear
        highlight_linear = None
        stage_seconds["highlight_scatter"] = time.perf_counter() - stage_started
        if stage_callback is not None:
            stage_callback("highlight-softening", 0.0)
        stage_started = time.perf_counter()
        soften_highlights_inplace(
            scattered,
            scatter_coc,
            cancel_check=cancel_check,
        )
        stage_seconds["highlight_softening"] = time.perf_counter() - stage_started
        if stage_callback is not None:
            stage_callback("highlight-softening", 1.0)
        stage_started = time.perf_counter()
        final_rgb = composite_highlights(
            base_rgb,
            final_rgb,
            scattered,
            row_chunk_height=row_chunk_height,
            cancel_check=cancel_check,
        )
        stage_seconds["highlight_composite"] = time.perf_counter() - stage_started
        del scattered
        if stage_callback is not None:
            stage_callback("highlight-composite", 1.0)
    else:
        stage_seconds["highlight_scatter"] = 0.0
        stage_seconds["highlight_recovery"] = 0.0
        stage_seconds["highlight_softening"] = 0.0
        stage_seconds["highlight_composite"] = 0.0
    if alpha is not None:
        from laika.protection import composite_protected_subject

        composite_started = time.perf_counter()
        final_rgb = composite_protected_subject(
            image,
            final_rgb,
            alpha,
            row_chunk_height=row_chunk_height,
            cancel_check=cancel_check,
        )
        stage_seconds["subject_composite"] = time.perf_counter() - composite_started
    del coc, base_rgb, highlight_linear
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
