"""Coordinated, memory-bounded processing for Laika image edits."""

from __future__ import annotations

import math
import os
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image

from laika.bokeh import (
    APERTURES,
    BokehMeARNet,
    focus_at_point,
    normalize_depth_map,
    render_bokeh,
    resize_rgb_cubic,
)
from laika.depth import DepthAnythingV2Small
from laika.engine import (
    ImageInputError,
    ImageLimitError,
    ImageLimits,
)
from laika.m9 import M9ColorEngine

ProcessingMode = Literal["color", "bokeh", "color+bokeh"]
ProcessingStrategy = Literal["low-memory", "balanced"]
ProgressCallback = Callable[["ProcessingProgress"], None]
MAX_SUBJECT_POINTS = 16


class ProcessingError(RuntimeError):
    """A coordinated processing request could not be completed."""


class ProcessingCancelled(ProcessingError):
    """A caller cancelled a request at a safe processing boundary."""


@dataclass(frozen=True)
class ProcessingOptions:
    """User-visible operation and resource settings for one image request."""

    mode: ProcessingMode = "color+bokeh"
    strategy: ProcessingStrategy = "low-memory"
    focus_x: float = 0.5
    focus_y: float = 0.5
    aperture: float = 2.8
    output_size: tuple[int, int] | None = None
    max_upload_bytes: int = 50 * 1024 * 1024
    max_pixels: int = 40_000_000
    max_network_pixels: int = 4_000_000
    m9_row_chunk_height: int = 512
    bokeh_row_chunk_height: int = 128
    subject_points: tuple[tuple[float, float], ...] = ()
    matte_tile_size: int = 1024

    def __post_init__(self) -> None:
        if self.mode not in ("color", "bokeh", "color+bokeh"):
            raise ValueError("mode must be color, bokeh, or color+bokeh")
        if self.strategy not in ("low-memory", "balanced"):
            raise ValueError("strategy must be low-memory or balanced")
        if not (0.0 <= self.focus_x <= 1.0 and 0.0 <= self.focus_y <= 1.0):
            raise ValueError("focus coordinates must be in [0, 1]")
        if self.aperture not in APERTURES:
            raise ValueError(f"aperture must be one of {APERTURES}")
        if self.output_size is not None:
            if len(self.output_size) != 2 or min(self.output_size) < 2:
                raise ValueError("output_size must be a (width, height) pair >= 2")
        if min(self.max_upload_bytes, self.max_pixels, self.max_network_pixels) < 1:
            raise ValueError(
                "upload, source-pixel, and network-pixel limits must be positive"
            )
        if min(self.m9_row_chunk_height, self.bokeh_row_chunk_height) < 1:
            raise ValueError("row chunk heights must be positive")
        if self.matte_tile_size not in (512, 768, 1024):
            raise ValueError("matte_tile_size must be 512, 768, or 1024")
        if len(self.subject_points) > MAX_SUBJECT_POINTS:
            raise ValueError(f"at most {MAX_SUBJECT_POINTS} subject points are allowed")
        for point in self.subject_points:
            if (
                len(point) != 2
                or not all(math.isfinite(value) for value in point)
                or not all(0.0 <= value < 1.0 for value in point)
            ):
                raise ValueError("subject points must be normalized (x, y) in [0, 1)")


@dataclass(frozen=True)
class ProcessingProgress:
    """A coarse-grained progress event suitable for SDK and HTTP callbacks."""

    stage: str
    fraction: float
    elapsed_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProcessingResult:
    """Output metadata, model provenance, and stage-level timings."""

    mode: ProcessingMode
    strategy: ProcessingStrategy
    output_path: str
    width: int
    height: int
    output_bytes: int
    models: dict[str, dict[str, Any]]
    stage_seconds: dict[str, float]
    total_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _image_icc_profile(path: Path) -> bytes | None:
    with Image.open(path) as image:
        return image.info.get("icc_profile")


def _save_depth_map(path: Path, depth: Any) -> None:
    import numpy as np

    with path.open("wb") as stream:
        np.save(stream, depth, allow_pickle=False)
        stream.flush()


def _save_png(path: Path, rgb: Any, icc_profile: bytes | None) -> int:
    import numpy as np

    output = np.asarray(rgb)
    if output.dtype != np.uint8 or output.ndim != 3 or output.shape[2] != 3:
        raise ValueError("processing output must be an HWC uint8 RGB array")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w+b") as stream:
            with Image.fromarray(output, mode="RGB") as image:
                options = {"icc_profile": icc_profile} if icc_profile else {}
                image.save(stream, format="PNG", **options)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return path.stat().st_size


class LaikaProcessingEngine:
    """Run M9, Depth Anything V2, and ARNet as one ordered Python request.

    ``low-memory`` creates and closes one model session at a time. ``balanced``
    retains sessions between serialized calls until ``close()``. The color
    stage writes an oriented, full-resolution temporary PNG; the depth and
    bokeh stages share one decoded RGB array from that intermediate image.
    Temporary files and image arrays are owned by the active call and released
    on success, cancellation, or failure.
    """

    def __init__(
        self,
        *,
        m9_model_path: str | os.PathLike[str] | None = None,
        depth_model_path: str | os.PathLike[str] | None = None,
        bokeh_model_path: str | os.PathLike[str] | None = None,
        sam_model_dir: str | os.PathLike[str] | None = None,
        vitmatte_model_path: str | os.PathLike[str] | None = None,
        strategy: ProcessingStrategy = "low-memory",
        threads: int | None = None,
        max_upload_bytes: int = 50 * 1024 * 1024,
        max_pixels: int = 40_000_000,
        max_network_pixels: int = 4_000_000,
        temp_root: str | os.PathLike[str] | None = None,
        m9_factory: Callable[..., Any] | None = None,
        depth_factory: Callable[..., Any] | None = None,
        bokeh_factory: Callable[..., Any] | None = None,
        sam_factory: Callable[..., Any] | None = None,
        matte_factory: Callable[..., Any] | None = None,
    ) -> None:
        if strategy not in ("low-memory", "balanced"):
            raise ValueError("strategy must be low-memory or balanced")
        if threads is not None and threads < 1:
            raise ValueError("threads must be positive")
        if max_network_pixels < 1:
            raise ValueError("max_network_pixels must be positive")
        self.m9_model_path = Path(m9_model_path) if m9_model_path else None
        self.depth_model_path = Path(depth_model_path) if depth_model_path else None
        self.bokeh_model_path = Path(bokeh_model_path) if bokeh_model_path else None
        self.sam_model_dir = Path(sam_model_dir) if sam_model_dir else None
        self.vitmatte_model_path = (
            Path(vitmatte_model_path) if vitmatte_model_path else None
        )
        self.strategy = strategy
        self.threads = threads or (1 if strategy == "low-memory" else 4)
        self.default_limits = ImageLimits(max_upload_bytes, max_pixels)
        self.max_network_pixels = max_network_pixels
        self.temp_root = Path(temp_root) if temp_root is not None else None
        self._m9_factory = m9_factory or M9ColorEngine
        self._depth_factory = depth_factory or DepthAnythingV2Small
        self._bokeh_factory = bokeh_factory or BokehMeARNet
        if sam_factory is None:
            from laika.subject import Sam2TinyOnnx

            sam_factory = Sam2TinyOnnx.from_validation_report
        if matte_factory is None:
            from laika.matting import ViTMatteOnnx

            matte_factory = ViTMatteOnnx.from_validation_report
        self._sam_factory = sam_factory
        self._matte_factory = matte_factory
        self._m9_engine: Any | None = None
        self._depth_engine: Any | None = None
        self._bokeh_engine: Any | None = None
        self._sam_engine: Any | None = None
        self._matte_engine: Any | None = None
        self._lock = threading.RLock()

    def process_file(
        self,
        input_path: str | os.PathLike[str],
        output_path: str | os.PathLike[str],
        *,
        options: ProcessingOptions | None = None,
        progress_callback: ProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
    ) -> ProcessingResult:
        """Process one source photo and atomically publish its PNG result."""
        selected = options or ProcessingOptions(
            strategy=self.strategy,
            max_upload_bytes=self.default_limits.max_upload_bytes,
            max_pixels=self.default_limits.max_pixels,
            max_network_pixels=self.max_network_pixels,
        )
        if selected.strategy != self.strategy:
            raise ValueError("request strategy must match the engine strategy")
        source = Path(input_path)
        destination = Path(output_path)
        if not source.is_file():
            raise ImageInputError("input path is not a regular file")
        if source.resolve() == destination.resolve():
            raise ImageInputError("input and output paths must be different")
        if destination.suffix.lower() != ".png":
            raise ImageInputError("unified processing output must use a .png filename")
        uses_color = selected.mode in ("color", "color+bokeh")
        uses_bokeh = selected.mode in ("bokeh", "color+bokeh")
        if selected.subject_points and not uses_bokeh:
            raise ProcessingError("subject protection requires a bokeh mode")
        if selected.subject_points and (
            self.sam_model_dir is None or self.vitmatte_model_path is None
        ):
            raise ProcessingError(
                "subject protection requires configured SAM and ViTMatte model paths"
            )
        if uses_color and self.m9_model_path is None:
            raise ProcessingError("this mode requires a configured M9 model path")
        if uses_bokeh and (
            self.depth_model_path is None or self.bokeh_model_path is None
        ):
            raise ProcessingError(
                "bokeh modes require configured Depth Anything and ARNet model paths"
            )

        with self._lock:
            return self._process_locked(
                source,
                destination,
                selected,
                progress_callback,
                cancel_event,
            )

    def _process_locked(
        self,
        source: Path,
        destination: Path,
        options: ProcessingOptions,
        progress_callback: ProgressCallback | None,
        cancel_event: threading.Event | None,
    ) -> ProcessingResult:
        from laika.cli import _read_rgb

        started = time.perf_counter()
        stages: dict[str, float] = {}
        models: dict[str, dict[str, Any]] = {}

        def check_cancelled() -> None:
            if cancel_event is not None and cancel_event.is_set():
                raise ProcessingCancelled("image processing was cancelled")

        def notify(stage: str, fraction: float) -> None:
            if stage != "complete":
                check_cancelled()
            if progress_callback is not None:
                progress_callback(
                    ProcessingProgress(
                        stage=stage,
                        fraction=fraction,
                        elapsed_seconds=time.perf_counter() - started,
                    )
                )
            if stage != "complete":
                check_cancelled()

        check_cancelled()
        use_color = options.mode in ("color", "color+bokeh")
        use_bokeh = options.mode in ("bokeh", "color+bokeh")

        if options.mode == "color" and options.output_size is None:
            notify("color", 0.05)
            color_result = self._run_color(
                source, destination, options, check_cancelled
            )
            stages.update(
                {
                    f"color.{key}": value
                    for key, value in color_result.stage_seconds.items()
                }
            )
            models["m9"] = {
                "sha256": color_result.model_sha256,
                "runtime": color_result.runtime_version,
                "provider": color_result.runtime_provider,
            }
            notify("complete", 1.0)
            return ProcessingResult(
                mode=options.mode,
                strategy=options.strategy,
                output_path=str(destination),
                width=color_result.width,
                height=color_result.height,
                output_bytes=destination.stat().st_size,
                models=models,
                stage_seconds=stages,
                total_seconds=time.perf_counter() - started,
            )

        with tempfile.TemporaryDirectory(
            prefix="laika-task-",
            dir=str(self.temp_root) if self.temp_root is not None else None,
        ) as task_dir_name:
            task_dir = Path(task_dir_name)
            frame_path = source
            icc_profile = _image_icc_profile(source)
            source_rgb: Any | None = None
            normalized_depth: Any | None = None
            depth_path = task_dir / "depth.npy"

            # The reference estimates depth from the EXIF-oriented original,
            # before M9 color changes. Close Depth ORT before loading M9.
            if use_bokeh:
                notify("depth", 0.05)
                stage_started = time.perf_counter()
                source_rgb = _read_rgb(
                    source,
                    max_bytes=options.max_upload_bytes,
                    max_pixels=options.max_pixels,
                )
                stages["decode_depth_input"] = time.perf_counter() - stage_started
                model_started = time.perf_counter()
                depth_reused = (
                    options.strategy == "balanced" and self._depth_engine is not None
                )
                depth_engine, depth_owned = self._acquire_depth(options)
                stages["depth.model_load"] = (
                    0.0 if depth_reused else time.perf_counter() - model_started
                )
                try:
                    depth_prediction = depth_engine.infer_rgb(
                        source_rgb, cancel_check=check_cancelled
                    )
                    models["depth"] = {
                        "model": self.depth_model_path.name,
                        "sha256": depth_engine.model_sha256,
                        "runtime": depth_engine.runtime_version,
                        "provider": depth_engine.provider,
                    }
                finally:
                    if depth_owned:
                        depth_engine.close()
                stages.update(
                    {
                        f"depth.{key}": value
                        for key, value in depth_prediction.stage_seconds.items()
                    }
                )
                if use_color:
                    stage_started = time.perf_counter()
                    _save_depth_map(depth_path, depth_prediction.depth)
                    stages["depth.spool"] = time.perf_counter() - stage_started
                    del source_rgb, depth_prediction
                    source_rgb = None
                else:
                    stage_started = time.perf_counter()
                    normalized_depth = normalize_depth_map(depth_prediction.depth)
                    stages["depth.normalize"] = time.perf_counter() - stage_started
                    del depth_prediction
                notify("depth", 0.32 if use_color else 0.4)

            if use_color:
                color_path = (
                    task_dir / "m9-color.png"
                    if use_bokeh or options.output_size is not None
                    else destination
                )
                notify("color", 0.35 if use_bokeh else 0.05)
                color_result = self._run_color(
                    source, color_path, options, check_cancelled
                )
                stages.update(
                    {
                        f"color.{key}": value
                        for key, value in color_result.stage_seconds.items()
                    }
                )
                models["m9"] = {
                    "sha256": color_result.model_sha256,
                    "runtime": color_result.runtime_version,
                    "provider": color_result.runtime_provider,
                }
                if use_bokeh or options.output_size is not None:
                    frame_path = color_path
                    icc_profile = _image_icc_profile(color_path)
                notify("color", 0.62 if use_bokeh else 0.4)

            if use_bokeh and use_color:
                import numpy as np

                stage_started = time.perf_counter()
                raw_depth = np.load(depth_path, mmap_mode="r", allow_pickle=False)
                normalized_depth = normalize_depth_map(raw_depth)
                del raw_depth
                stages["depth.normalize"] = time.perf_counter() - stage_started

            if source_rgb is not None:
                rgb = source_rgb
                source_rgb = None
            else:
                stage_started = time.perf_counter()
                rgb = _read_rgb(
                    frame_path,
                    max_bytes=options.max_upload_bytes,
                    max_pixels=options.max_pixels,
                )
                stages["decode_frame"] = time.perf_counter() - stage_started
            source_height, source_width = rgb.shape[:2]
            if options.output_size is not None:
                output_width, output_height = options.output_size
                if output_width > source_width or output_height > source_height:
                    raise ImageLimitError(
                        "output_size cannot upscale the oriented source"
                    )
                rgb = resize_rgb_cubic(rgb, output_width, output_height)
            else:
                output_height, output_width = source_height, source_width

            if use_bokeh:
                assert normalized_depth is not None
                if normalized_depth.shape != (source_height, source_width):
                    from laika.depth import resize_bilinear_align_corners

                    normalized_depth = resize_bilinear_align_corners(
                        normalized_depth, source_height, source_width
                    )
                if options.output_size is not None:
                    from laika.depth import resize_bilinear_align_corners

                    normalized_depth = resize_bilinear_align_corners(
                        normalized_depth, output_height, output_width
                    )
                focus = focus_at_point(
                    normalized_depth, options.focus_x, options.focus_y
                )
                subject_alpha: Any | None = None
                if options.subject_points:
                    subject_alpha, subject_models, subject_stages = (
                        self._run_subject_protection(
                            rgb,
                            options,
                            check_cancelled,
                            notify,
                        )
                    )
                    models.update(subject_models)
                    stages.update(subject_stages)
                notify("bokeh", 0.68)
                model_started = time.perf_counter()
                bokeh_reused = (
                    options.strategy == "balanced" and self._bokeh_engine is not None
                )
                bokeh_engine, bokeh_owned = self._acquire_bokeh(options)
                stages["bokeh.model_load"] = (
                    0.0 if bokeh_reused else time.perf_counter() - model_started
                )
                stage_started = time.perf_counter()

                def on_bokeh_pass(index: int, total: int, _scale: float) -> None:
                    notify("bokeh", 0.69 + 0.12 * index / max(total, 1))

                def on_bokeh_stage(stage: str, fraction: float) -> None:
                    if stage == "highlight-extraction":
                        progress = 0.68 + 0.01 * fraction
                    elif stage == "highlight-recovery":
                        progress = 0.82 + 0.01 * fraction
                    elif stage == "highlight-scatter":
                        progress = 0.83 + 0.08 * fraction
                    elif stage == "highlight-softening":
                        progress = 0.91 + 0.01 * fraction
                    else:
                        progress = 0.93
                    notify("bokeh", progress)

                try:
                    render = render_bokeh(
                        rgb,
                        normalized_depth,
                        focus,
                        options.aperture,
                        bokeh_engine,
                        subject_alpha=subject_alpha,
                        row_chunk_height=options.bokeh_row_chunk_height,
                        cancel_check=check_cancelled,
                        pass_callback=on_bokeh_pass,
                        stage_callback=on_bokeh_stage,
                    )
                finally:
                    if bokeh_owned:
                        bokeh_engine.close()
                stages.update(
                    {
                        f"bokeh.{key}": value
                        for key, value in render.stage_seconds.items()
                    }
                )
                models["bokeh"] = bokeh_engine.runtime_info.as_dict()
                output_rgb = render.rgb
                notify("bokeh", 0.93)
                del normalized_depth, rgb, render
            else:
                output_rgb = rgb

            check_cancelled()
            notify("export", 0.94)
            stage_started = time.perf_counter()
            output_bytes = _save_png(destination, output_rgb, icc_profile)
            stages["export"] = time.perf_counter() - stage_started
            notify("complete", 1.0)

        return ProcessingResult(
            mode=options.mode,
            strategy=options.strategy,
            output_path=str(destination),
            width=output_width,
            height=output_height,
            output_bytes=output_bytes,
            models=models,
            stage_seconds=stages,
            total_seconds=time.perf_counter() - started,
        )

    def _run_color(
        self,
        source: Path,
        destination: Path,
        options: ProcessingOptions,
        cancel_check: Callable[[], None],
    ) -> Any:
        engine, owned = self._acquire_m9(options)
        try:
            return engine.process_file(source, destination, cancel_check=cancel_check)
        finally:
            if owned:
                engine.close()

    def _acquire_m9(self, options: ProcessingOptions) -> tuple[Any, bool]:
        limits = ImageLimits(options.max_upload_bytes, options.max_pixels)
        if options.strategy == "balanced":
            if self._m9_engine is None:
                self._m9_engine = self._m9_factory(
                    self.m9_model_path,
                    strategy="balanced",
                    intra_op_num_threads=self.threads,
                    limits=limits,
                    row_chunk_height=options.m9_row_chunk_height,
                )
            return self._m9_engine, False
        return (
            self._m9_factory(
                self.m9_model_path,
                strategy="low-memory",
                intra_op_num_threads=self.threads,
                limits=limits,
                row_chunk_height=options.m9_row_chunk_height,
            ),
            True,
        )

    def _acquire_depth(self, options: ProcessingOptions) -> tuple[Any, bool]:
        if options.strategy == "balanced":
            if self._depth_engine is None:
                self._depth_engine = self._depth_factory(
                    self.depth_model_path,
                    intra_op_num_threads=self.threads,
                    max_source_pixels=options.max_pixels,
                    max_network_pixels=options.max_network_pixels,
                )
            return self._depth_engine, False
        return (
            self._depth_factory(
                self.depth_model_path,
                intra_op_num_threads=self.threads,
                max_source_pixels=options.max_pixels,
                max_network_pixels=options.max_network_pixels,
            ),
            True,
        )

    def _acquire_bokeh(self, options: ProcessingOptions) -> tuple[Any, bool]:
        if options.strategy == "balanced":
            if self._bokeh_engine is None:
                self._bokeh_engine = self._bokeh_factory(
                    self.bokeh_model_path,
                    strategy="balanced",
                    intra_op_num_threads=self.threads,
                )
            return self._bokeh_engine, False
        return (
            self._bokeh_factory(
                self.bokeh_model_path,
                strategy="low-memory",
                intra_op_num_threads=self.threads,
            ),
            True,
        )

    def _run_subject_protection(
        self,
        rgb: Any,
        options: ProcessingOptions,
        cancel_check: Callable[[], None],
        notify: Callable[[str, float], None],
    ) -> tuple[Any, dict[str, dict[str, Any]], dict[str, float]]:
        """Create a refined full-output alpha while releasing model stages in order."""
        import cv2
        import numpy as np

        from laika.protection import (
            prepare_subject_analysis_rgb,
            refine_subject_masks,
            segment_subject_points,
        )

        points = tuple(dict.fromkeys(options.subject_points))
        analysis_rgb = prepare_subject_analysis_rgb(rgb)
        models: dict[str, dict[str, Any]] = {}
        stages: dict[str, float] = {}

        notify("subject", 0.625)
        started = time.perf_counter()
        sam_reused = options.strategy == "balanced" and self._sam_engine is not None
        if sam_reused:
            sam_engine, sam_owned = self._sam_engine, False
        elif options.strategy == "balanced":
            sam_engine = self._sam_factory(
                self.sam_model_dir,
                intra_op_num_threads=self.threads,
            )
            self._sam_engine, sam_owned = sam_engine, False
        else:
            sam_engine = self._sam_factory(
                self.sam_model_dir,
                intra_op_num_threads=self.threads,
            )
            sam_owned = True
        stages["subject.sam_model_load"] = (
            0.0 if sam_reused else time.perf_counter() - started
        )
        try:
            stage_started = time.perf_counter()
            masks, scores, timings, cache_hits = segment_subject_points(
                analysis_rgb,
                points,
                sam_engine,
                cancel_check=cancel_check,
                progress_callback=lambda fraction: notify(
                    "subject", 0.63 + 0.015 * fraction
                ),
            )
            stages["subject.sam_prepare"] = time.perf_counter() - stage_started
            stages["subject.sam_encoder"] = timings["encoder"]
            stages["subject.sam_decoder"] = timings["decoder"]
            models["sam"] = {
                "model": "sam2.1-hiera-tiny",
                "source_checkpoint_sha256": (
                    "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"
                ),
                "image_encoder_sha256": getattr(
                    sam_engine, "image_encoder_sha256", None
                ),
                "image_decoder_sha256": getattr(
                    sam_engine, "image_decoder_sha256", None
                ),
                "runtime": getattr(sam_engine, "runtime_version", "unknown"),
                "provider": getattr(sam_engine, "provider", "CPUExecutionProvider"),
                "embedding_cache_hits": cache_hits,
                "point_scores": list(scores),
            }
        finally:
            if sam_owned:
                sam_engine.close()
        del sam_engine

        notify("subject", 0.65)
        started = time.perf_counter()
        matte_reused = options.strategy == "balanced" and self._matte_engine is not None
        if matte_reused:
            matte_engine, matte_owned = self._matte_engine, False
        elif options.strategy == "balanced":
            matte_engine = self._matte_factory(
                self.vitmatte_model_path,
                intra_op_num_threads=self.threads,
            )
            self._matte_engine, matte_owned = matte_engine, False
        else:
            matte_engine = self._matte_factory(
                self.vitmatte_model_path,
                intra_op_num_threads=self.threads,
            )
            matte_owned = True
        stages["subject.vitmatte_model_load"] = (
            0.0 if matte_reused else time.perf_counter() - started
        )
        try:
            stage_started = time.perf_counter()
            refined = refine_subject_masks(
                analysis_rgb,
                masks,
                points,
                matte_engine,
                tile_size=options.matte_tile_size,
                cancel_check=cancel_check,
                cache_hits=cache_hits,
                progress_callback=lambda fraction: notify(
                    "subject", 0.65 + 0.015 * fraction
                ),
            )
            stages["subject.vitmatte"] = time.perf_counter() - stage_started
            stages.update(
                {
                    f"subject.{key}": value
                    for key, value in refined.stage_seconds.items()
                }
            )
            models["vitmatte"] = {
                "model": "vitmatte-small-composition-1k",
                "checkpoint_sha256": (
                    "bda9289db1bb6762d978b42d1c62ae3f34daf7497171a347a1d09657efd788cb"
                ),
                "onnx_sha256": getattr(matte_engine, "model_sha256", None),
                "tile_size": options.matte_tile_size,
            }
            alpha = refined.alpha
        finally:
            if matte_owned:
                matte_engine.close()
        del matte_engine, masks, analysis_rgb

        height, width = rgb.shape[:2]
        if alpha.shape != (height, width):
            stage_started = time.perf_counter()
            alpha = np.ascontiguousarray(
                cv2.resize(alpha, (width, height), interpolation=cv2.INTER_LINEAR),
                dtype=np.float32,
            )
            stages["subject.alpha_resize"] = time.perf_counter() - stage_started
        else:
            alpha = np.ascontiguousarray(alpha, dtype=np.float32)
        np.clip(alpha, 0.0, 1.0, out=alpha)
        cancel_check()
        notify("subject", 0.67)
        return alpha, models, stages

    def close(self) -> None:
        """Release cached Balanced sessions. Low Memory sessions are per-call."""
        with self._lock:
            for name in (
                "_m9_engine",
                "_depth_engine",
                "_bokeh_engine",
                "_sam_engine",
                "_matte_engine",
            ):
                engine = getattr(self, name)
                if engine is not None:
                    engine.close()
                    setattr(self, name, None)

    def __enter__(self) -> LaikaProcessingEngine:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = [
    "LaikaProcessingEngine",
    "ProcessingCancelled",
    "ProcessingError",
    "ProcessingMode",
    "ProcessingOptions",
    "ProcessingProgress",
    "ProcessingResult",
]
