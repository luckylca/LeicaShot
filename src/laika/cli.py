"""Command-line interface for the shared Laika image engine."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

from laika.bokeh import (
    APERTURES,
    BokehInferenceError,
    BokehMeARNet,
    focus_at_point,
    normalize_depth_map,
    render_bokeh,
    resize_rgb_cubic,
)
from laika.composition import CompositionCrop, CompositionInferenceError
from laika.depth import (
    DepthAnythingV2Small,
    DepthInferenceError,
    depth_input_geometry,
    resize_bilinear_align_corners,
)
from laika.engine import ImageInputError, ImageLimitError, ImageLimits, process_image
from laika.m9 import M9ColorEngine, M9InferenceError
from laika.processing import (
    LaikaProcessingEngine,
    ProcessingError,
    ProcessingOptions,
)

SUPPORTED_INPUT_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="laika")
    subparsers = parser.add_subparsers(dest="command", required=True)
    process_parser = subparsers.add_parser(
        "process", help="decode, orient, and losslessly export a still image as PNG"
    )
    process_parser.add_argument("input", help="input JPEG, PNG, or WebP")
    process_parser.add_argument("output", help="output .png path")
    depth_parser = subparsers.add_parser(
        "depth", help="estimate a full-size depth map with a validated ONNX model"
    )
    depth_parser.add_argument("input", help="input JPEG, PNG, or WebP")
    depth_parser.add_argument("output", help="output .npy float32 depth map")
    depth_parser.add_argument(
        "--model", required=True, help="validated dynamic ONNX path"
    )
    depth_size = depth_parser.add_mutually_exclusive_group()
    depth_size.add_argument(
        "--analysis-max-edge",
        type=int,
        help="reference analysis long-edge cap (default: 1220, no upscale)",
    )
    depth_size.add_argument(
        "--input-size",
        type=int,
        help="use the upstream lower-bound policy with this minimum side",
    )
    depth_parser.add_argument("--threads", type=int, default=1)
    depth_parser.add_argument("--max-upload-bytes", type=int, default=50 * 1024 * 1024)
    depth_parser.add_argument("--max-pixels", type=int, default=40_000_000)
    depth_parser.add_argument("--max-network-pixels", type=int, default=4_000_000)
    color_parser = subparsers.add_parser(
        "color", help="run local M9 ONNX color prediction and Lab detail fusion"
    )
    color_parser.add_argument("input", help="input JPEG, PNG, or WebP")
    color_parser.add_argument("output", help="output .png, .jpg, or .jpeg path")
    color_parser.add_argument(
        "--model", required=True, help="local M9 ONNX path (SHA-256 is verified)"
    )
    color_parser.add_argument(
        "--strategy",
        choices=("low-memory", "balanced"),
        default="low-memory",
        help="Low Memory releases the session before full-resolution fusion",
    )
    color_parser.add_argument(
        "--threads", type=int, default=None, help="ONNX Runtime intra-op threads"
    )
    color_parser.add_argument(
        "--row-chunk-height",
        type=int,
        default=512,
        help="exact Lab fusion row band height; default: 512",
    )
    color_parser.add_argument("--max-upload-bytes", type=int, default=50 * 1024 * 1024)
    color_parser.add_argument("--max-pixels", type=int, default=40_000_000)
    bokeh_parser = subparsers.add_parser(
        "bokeh",
        help="render a depth-guided BokehMe ARNet background plate",
    )
    bokeh_parser.add_argument("input", help="input JPEG, PNG, or WebP")
    bokeh_parser.add_argument("depth", help="raw float32 depth map from 'laika depth'")
    bokeh_parser.add_argument("output", help="output .png path")
    bokeh_parser.add_argument("--model", required=True, help="local ARNet ONNX path")
    bokeh_parser.add_argument(
        "--model-sha256",
        help="explicit trusted SHA-256; otherwise require an export validation report",
    )
    bokeh_parser.add_argument("--focus-x", type=float, default=0.5)
    bokeh_parser.add_argument("--focus-y", type=float, default=0.5)
    bokeh_parser.add_argument("--aperture", type=float, choices=APERTURES, default=2.8)
    bokeh_parser.add_argument(
        "--strategy",
        choices=("low-memory", "balanced"),
        default="low-memory",
    )
    bokeh_parser.add_argument("--threads", type=int, default=None)
    bokeh_parser.add_argument("--row-chunk-height", type=int, default=256)
    bokeh_parser.add_argument(
        "--width",
        type=int,
        help="explicit output width; requires --height and never upscales",
    )
    bokeh_parser.add_argument(
        "--height",
        type=int,
        help="explicit output height; requires --width and never upscales",
    )
    bokeh_parser.add_argument("--max-upload-bytes", type=int, default=50 * 1024 * 1024)
    bokeh_parser.add_argument("--max-pixels", type=int, default=40_000_000)
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="run coordinated M9 color, depth, and ARNet processing",
    )
    pipeline_parser.add_argument("input", help="input JPEG, PNG, or WebP")
    pipeline_parser.add_argument("output", help="full-size output .png path")
    pipeline_parser.add_argument(
        "--mode",
        choices=("color", "bokeh", "color+bokeh"),
        default="color+bokeh",
    )
    pipeline_parser.add_argument("--m9-model", help="local M9 ONNX model path")
    pipeline_parser.add_argument("--depth-model", help="validated depth ONNX path")
    pipeline_parser.add_argument("--bokeh-model", help="validated ARNet ONNX path")
    pipeline_parser.add_argument(
        "--sam-model-dir", help="local validated split SAM ONNX directory"
    )
    pipeline_parser.add_argument(
        "--vitmatte-model", help="local validated ViTMatte ONNX path"
    )
    pipeline_parser.add_argument(
        "--subject-point",
        action="append",
        default=[],
        metavar="X,Y",
        help="normalized foreground point; repeat for multiple subjects",
    )
    pipeline_parser.add_argument(
        "--matte-tile-size", type=int, choices=(512, 768, 1024), default=1024
    )
    pipeline_parser.add_argument(
        "--strategy", choices=("low-memory", "balanced"), default="low-memory"
    )
    pipeline_parser.add_argument("--threads", type=int, default=None)
    pipeline_parser.add_argument("--focus-x", type=float, default=0.5)
    pipeline_parser.add_argument("--focus-y", type=float, default=0.5)
    pipeline_parser.add_argument(
        "--aperture", type=float, choices=APERTURES, default=2.8
    )
    pipeline_parser.add_argument("--width", type=int)
    pipeline_parser.add_argument("--height", type=int)
    pipeline_parser.add_argument("--m9-row-chunk-height", type=int, default=512)
    pipeline_parser.add_argument("--bokeh-row-chunk-height", type=int, default=256)
    pipeline_parser.add_argument(
        "--crop",
        type=_parse_composition_crop,
        metavar="LEFT,TOP,RIGHT,BOTTOM",
        help="apply a normalized crop after AI rendering without rerunning GAIC",
    )
    pipeline_parser.add_argument(
        "--max-upload-bytes", type=int, default=50 * 1024 * 1024
    )
    pipeline_parser.add_argument("--max-pixels", type=int, default=40_000_000)
    pipeline_parser.add_argument("--max-network-pixels", type=int, default=4_000_000)
    pipeline_parser.add_argument("--progress", action="store_true")
    composition_parser = subparsers.add_parser(
        "composition", help="suggest GAIC smart crops without modifying the photo"
    )
    composition_parser.add_argument("input", help="input JPEG, PNG, or WebP")
    composition_parser.add_argument(
        "--model", required=True, help="validated GAIC ONNX path"
    )
    composition_parser.add_argument("--target-long-edge", type=int)
    composition_parser.add_argument(
        "--strategy", choices=("low-memory", "balanced"), default="low-memory"
    )
    composition_parser.add_argument("--threads", type=int, default=None)
    composition_parser.add_argument(
        "--max-upload-bytes", type=int, default=50 * 1024 * 1024
    )
    composition_parser.add_argument("--max-pixels", type=int, default=40_000_000)
    crop_parser = subparsers.add_parser(
        "crop", help="apply a saved normalized crop without loading an AI model"
    )
    crop_parser.add_argument("input", help="input JPEG, PNG, or WebP")
    crop_parser.add_argument("output", help="output .png path")
    crop_parser.add_argument(
        "--crop",
        required=True,
        type=_parse_composition_crop,
        metavar="LEFT,TOP,RIGHT,BOTTOM",
        help="normalized half-open crop coordinates in [0, 1]",
    )
    crop_parser.add_argument("--max-upload-bytes", type=int, default=50 * 1024 * 1024)
    crop_parser.add_argument("--max-pixels", type=int, default=40_000_000)
    return parser


def _parse_composition_crop(value: str) -> CompositionCrop:
    try:
        parts = tuple(float(item) for item in value.split(","))
        if len(parts) != 4:
            raise ValueError
        return CompositionCrop(*parts)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "crop must contain four normalized coordinates: LEFT,TOP,RIGHT,BOTTOM"
        ) from exc


def _read_rgb(path: Path, *, max_bytes: int, max_pixels: int) -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise DepthInferenceError(
            "depth CLI requires NumPy; install the 'inference' extra"
        ) from exc

    if not path.is_file():
        raise ImageInputError("input path is not a regular file")
    if max_bytes <= 0 or max_pixels <= 0:
        raise ValueError("input limits must be positive")
    try:
        source = path.open("rb")
    except OSError as exc:
        raise ImageInputError("cannot open input file") from exc

    with source:
        before = os.fstat(source.fileno())
        if before.st_size > max_bytes:
            raise ImageLimitError("input file exceeds the configured byte limit")
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        try:
            opened = Image.open(source)
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
            raise ImageInputError("input is not a valid supported image") from exc

        with opened:
            if opened.format not in SUPPORTED_INPUT_FORMATS:
                raise ImageInputError(
                    f"unsupported decoded image format: {opened.format or 'unknown'}"
                )
            width, height = opened.size
            if width < 1 or height < 1:
                raise ImageInputError("image dimensions must be positive")
            if width * height > max_pixels:
                raise ImageLimitError(
                    "image exceeds the configured decoded-pixel limit"
                )
            if getattr(opened, "is_animated", False):
                raise ImageInputError("animated images are not supported")
            try:
                opened.load()
            except (OSError, Image.DecompressionBombError) as exc:
                raise ImageInputError("image pixels could not be decoded") from exc

            after = os.fstat(source.fileno())
            after_identity = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            )
            if after_identity != identity:
                raise ImageInputError("input changed while it was being decoded")

            oriented = ImageOps.exif_transpose(opened)
            try:
                rgb_image = oriented.convert("RGB")
                try:
                    return np.array(rgb_image, dtype=np.uint8, copy=True)
                finally:
                    rgb_image.close()
            finally:
                if oriented is not opened:
                    oriented.close()


def _save_depth(path: Path, depth: Any) -> None:
    try:
        import numpy as np
    except ImportError as exc:
        raise DepthInferenceError(
            "depth CLI requires NumPy; install the 'inference' extra"
        ) from exc

    if path.suffix.lower() != ".npy":
        raise ValueError("depth output path must end in .npy")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.save(stream, depth, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _read_depth_map(path: Path, *, max_pixels: int) -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional runtime
        raise BokehInferenceError(
            "BokehMe inference requires NumPy; install the inference extra"
        ) from exc
    if not path.is_file():
        raise ImageInputError("depth path is not a regular file")
    try:
        depth = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ImageInputError(
            "depth file is not a valid non-pickled .npy array"
        ) from exc
    if (
        not isinstance(depth, np.ndarray)
        or depth.dtype != np.float32
        or depth.ndim != 2
        or min(depth.shape) < 1
        or depth.size > max_pixels
    ):
        raise ImageLimitError("depth must be a float32 map within the pixel limit")
    return depth


def _save_rgb_png(path: Path, rgb: Any) -> int:
    import numpy as np

    output = np.asarray(rgb)
    if output.dtype != np.uint8 or output.ndim != 3 or output.shape[2] != 3:
        raise ValueError("render output must be an HWC uint8 RGB array")
    if path.suffix.lower() != ".png":
        raise ValueError("Bokeh output path must end in .png")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            with Image.fromarray(output, mode="RGB") as image:
                image.save(stream, format="PNG")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return path.stat().st_size


def _run_bokeh(args: argparse.Namespace) -> int:

    input_path = Path(args.input)
    depth_path = Path(args.depth)
    output_path = Path(args.output)
    if args.output in (args.input, args.depth) or any(
        output_path.exists() and output_path.resolve() == p.resolve()
        for p in (input_path, depth_path)
    ):
        raise ImageInputError("input, depth, and output paths must be different")
    if (args.width is None) != (args.height is None):
        raise ValueError("--width and --height must be supplied together")
    if args.width is not None and (args.width < 2 or args.height < 2):
        raise ValueError("explicit output dimensions must be at least two pixels")
    if args.max_pixels < 1 or args.row_chunk_height < 1:
        raise ValueError("pixel and row-chunk limits must be positive")

    rgb = _read_rgb(
        input_path,
        max_bytes=args.max_upload_bytes,
        max_pixels=args.max_pixels,
    )
    source_height, source_width = rgb.shape[:2]
    raw_depth = _read_depth_map(depth_path, max_pixels=args.max_pixels)
    started = time.perf_counter()
    normalized_depth = normalize_depth_map(raw_depth)
    del raw_depth
    if normalized_depth.shape != (source_height, source_width):
        normalized_depth = resize_bilinear_align_corners(
            normalized_depth, source_height, source_width
        )

    if args.width is not None:
        if args.width > source_width or args.height > source_height:
            raise ValueError("explicit output dimensions cannot upscale the source")
        rgb = resize_rgb_cubic(rgb, args.width, args.height)
        normalized_depth = resize_bilinear_align_corners(
            normalized_depth, args.height, args.width
        )
    focus = focus_at_point(normalized_depth, args.focus_x, args.focus_y)
    with BokehMeARNet(
        args.model,
        strategy=args.strategy,
        intra_op_num_threads=args.threads,
        expected_sha256=args.model_sha256,
        split_graph_dir=os.environ.get("LAIKA_ARNET_SPLIT_DIR") or None,
    ) as model:
        result = render_bokeh(
            rgb,
            normalized_depth,
            focus,
            args.aperture,
            model,
            row_chunk_height=args.row_chunk_height,
        )
        output_bytes = _save_rgb_png(output_path, result.rgb)
        model_info = model.runtime_info.as_dict()
    print(
        json.dumps(
            {
                "output": str(output_path),
                "output_bytes": output_bytes,
                "width": result.width,
                "height": result.height,
                "focus": result.focus,
                "aperture": result.aperture,
                "scales": result.scales,
                "model": model_info,
                "stage_seconds": result.stage_seconds,
                "total_seconds": time.perf_counter() - started,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _run_depth(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    output_path = Path(args.output)
    if args.output == args.input or (
        output_path.exists() and output_path.resolve() == input_path.resolve()
    ):
        raise ImageInputError("input and output paths must be different")
    rgb = _read_rgb(
        input_path,
        max_bytes=args.max_upload_bytes,
        max_pixels=args.max_pixels,
    )
    geometry = depth_input_geometry(
        rgb.shape[1],
        rgb.shape[0],
        input_size=args.input_size,
        max_analysis_edge=args.analysis_max_edge,
    )
    if args.max_network_pixels < 1:
        raise ValueError("network pixel limit must be positive")
    if geometry.network_width * geometry.network_height > args.max_network_pixels:
        raise ImageLimitError(
            "network input exceeds the configured pixel limit; increase the "
            "explicit limit or reduce the requested input size"
        )
    with DepthAnythingV2Small(
        args.model,
        input_size=args.input_size,
        max_analysis_edge=args.analysis_max_edge,
        intra_op_num_threads=args.threads,
        max_source_pixels=args.max_pixels,
        max_network_pixels=args.max_network_pixels,
    ) as depth_engine:
        prediction = depth_engine.infer_rgb(rgb)
    _save_depth(output_path, prediction.depth)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "source_width": prediction.source_width,
                "source_height": prediction.source_height,
                "analysis_width": prediction.analysis_width,
                "analysis_height": prediction.analysis_height,
                "network_width": prediction.network_width,
                "network_height": prediction.network_height,
                "depth_dtype": str(prediction.depth.dtype),
                "stage_seconds": prediction.stage_seconds,
                "total_seconds": prediction.total_seconds,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _run_pipeline(args: argparse.Namespace) -> int:
    if (args.width is None) != (args.height is None):
        raise ValueError("--width and --height must be supplied together")
    output_size = (args.width, args.height) if args.width is not None else None
    subject_points = []
    for item in args.subject_point:
        try:
            x_text, y_text = item.split(",", maxsplit=1)
            subject_points.append((float(x_text), float(y_text)))
        except (ValueError, AttributeError) as exc:
            raise ValueError(
                "--subject-point must use normalized X,Y coordinates"
            ) from exc
    options = ProcessingOptions(
        mode=args.mode,
        strategy=args.strategy,
        focus_x=args.focus_x,
        focus_y=args.focus_y,
        aperture=args.aperture,
        output_size=output_size,
        max_upload_bytes=args.max_upload_bytes,
        max_pixels=args.max_pixels,
        max_network_pixels=args.max_network_pixels,
        m9_row_chunk_height=args.m9_row_chunk_height,
        bokeh_row_chunk_height=args.bokeh_row_chunk_height,
        subject_points=tuple(subject_points),
        matte_tile_size=args.matte_tile_size,
        crop=args.crop,
    )
    with LaikaProcessingEngine(
        m9_model_path=args.m9_model,
        depth_model_path=args.depth_model,
        bokeh_model_path=args.bokeh_model,
        sam_model_dir=args.sam_model_dir,
        vitmatte_model_path=args.vitmatte_model,
        strategy=args.strategy,
        threads=args.threads,
        max_upload_bytes=args.max_upload_bytes,
        max_pixels=args.max_pixels,
        max_network_pixels=args.max_network_pixels,
    ) as engine:
        progress_callback = None
        if args.progress:

            def report_progress(event: Any) -> None:
                print(
                    json.dumps(event.as_dict(), sort_keys=True),
                    file=sys.stderr,
                    flush=True,
                )

            progress_callback = report_progress
        result = engine.process_file(
            args.input,
            args.output,
            options=options,
            progress_callback=progress_callback,
        )
    print(json.dumps(result.as_dict(), ensure_ascii=False, sort_keys=True))
    return 0


def _run_composition(args: argparse.Namespace) -> int:
    with LaikaProcessingEngine(
        composition_model_path=args.model,
        strategy=args.strategy,
        threads=args.threads,
        max_upload_bytes=args.max_upload_bytes,
        max_pixels=args.max_pixels,
    ) as engine:
        result = engine.suggest_composition(
            args.input, target_long_edge=args.target_long_edge
        )
    print(json.dumps(result.as_dict(), ensure_ascii=False, sort_keys=True))
    return 0


def _run_crop(args: argparse.Namespace) -> int:
    with LaikaProcessingEngine(
        max_upload_bytes=args.max_upload_bytes,
        max_pixels=args.max_pixels,
    ) as engine:
        result = engine.apply_crop_file(args.input, args.output, args.crop)
    print(json.dumps(result.as_dict(), ensure_ascii=False, sort_keys=True))
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "depth":
        try:
            return _run_depth(args)
        except (DepthInferenceError, ImageInputError, OSError, ValueError) as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
            return 2
    if args.command == "color":
        try:
            with M9ColorEngine(
                args.model,
                strategy=args.strategy,
                intra_op_num_threads=args.threads,
                limits=ImageLimits(
                    max_upload_bytes=args.max_upload_bytes,
                    max_pixels=args.max_pixels,
                ),
                row_chunk_height=args.row_chunk_height,
            ) as engine:
                result = engine.process_file(args.input, args.output)
        except (M9InferenceError, ImageInputError, OSError, ValueError) as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
            return 2
        print(json.dumps(result.as_dict(), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "bokeh":
        try:
            return _run_bokeh(args)
        except (
            BokehInferenceError,
            ImageInputError,
            OSError,
            ValueError,
        ) as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
            return 2
    if args.command == "pipeline":
        try:
            return _run_pipeline(args)
        except (
            BokehInferenceError,
            DepthInferenceError,
            ImageInputError,
            M9InferenceError,
            OSError,
            ProcessingError,
            ValueError,
        ) as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
            return 2
    if args.command == "composition":
        try:
            return _run_composition(args)
        except (
            CompositionInferenceError,
            ImageInputError,
            OSError,
            ProcessingError,
            ValueError,
        ) as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
            return 2
    if args.command == "crop":
        try:
            return _run_crop(args)
        except (ImageInputError, OSError, ProcessingError, ValueError) as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
            return 2
    try:
        result = process_image(args.input, args.output)
    except (ImageInputError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result.as_dict(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
