"""Command-line interface for the shared Laika image engine."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

from laika.depth import (
    DepthAnythingV2Small,
    DepthInferenceError,
    depth_input_geometry,
)
from laika.engine import ImageInputError, ImageLimitError, process_image

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
    return parser


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


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "depth":
        try:
            return _run_depth(args)
        except (DepthInferenceError, ImageInputError, OSError, ValueError) as exc:
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
