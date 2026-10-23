"""Compare one full-frame ARNet scale with lossless overlapping tiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from laika.bokeh import BokehMeARNet, full_frame_coc, normalize_depth_map
from laika.depth import DepthAnythingV2Small
from laika.highlights import extract_highlights


def _read_rgb(path: Path) -> np.ndarray[Any, np.dtype[np.uint8]]:
    with Image.open(path) as source:
        oriented = ImageOps.exif_transpose(source)
        return np.array(oriented.convert("RGB"), dtype=np.uint8, copy=True)


def _sha256(values: np.ndarray[Any, Any]) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(values))).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="original photo used for depth")
    parser.add_argument("m9_image", type=Path, help="M9 color-rendered RGB image")
    parser.add_argument("depth_model", type=Path)
    parser.add_argument("bokeh_model", type=Path)
    parser.add_argument("--scale", type=float, default=4.0)
    parser.add_argument("--focus", type=float, default=0.5)
    parser.add_argument("--aperture", type=float, default=2.8)
    parser.add_argument("--tile-edge", type=int, default=512)
    parser.add_argument("--full-threshold", type=int, default=4_000_000)
    parser.add_argument("--tiled-threshold", type=int, default=1_000_000)
    args = parser.parse_args()
    if args.scale < 1.0 or args.tile_edge < 64 or args.tile_edge % 2:
        parser.error("scale must be >= 1 and tile edge an even value >= 64")

    source = _read_rgb(args.source)
    m9_rgb = _read_rgb(args.m9_image)
    if source.shape != m9_rgb.shape:
        parser.error("source and M9 image must have matching EXIF-oriented sizes")
    height, width = source.shape[:2]

    depth_started = time.perf_counter()
    with DepthAnythingV2Small(args.depth_model, intra_op_num_threads=1) as depth_engine:
        raw_depth = depth_engine.infer_rgb(source).depth
        depth_model_sha256 = depth_engine.model_sha256
    depth_seconds = time.perf_counter() - depth_started
    normalized_depth = normalize_depth_map(raw_depth)
    del raw_depth

    split = extract_highlights(m9_rgb, row_chunk_height=256)
    image_float = split.base_rgb.astype(np.float32)
    np.divide(image_float, np.float32(255.0), out=image_float)
    coc = full_frame_coc(normalized_depth, args.focus, args.aperture)

    pass_width = max(2, math.ceil(width / args.scale))
    pass_height = max(2, math.ceil(height / args.scale))
    actual_scale = max(width / pass_width, height / pass_height)
    if (pass_width, pass_height) == (width, height):
        small_rgb = image_float
        small_coc = coc
    else:
        small_rgb = cv2.resize(
            image_float, (pass_width, pass_height), interpolation=cv2.INTER_AREA
        )
        small_coc = cv2.resize(
            coc, (pass_width, pass_height), interpolation=cv2.INTER_LINEAR
        )
    defocus = np.empty((pass_height, pass_width), dtype=np.float32)
    np.divide(small_coc, np.float32(actual_scale), out=defocus)
    np.divide(defocus, np.float32(10.0), out=defocus)
    np.clip(defocus, -1.0, 1.0, out=defocus)
    del small_coc, coc, normalized_depth, split, image_float

    engine = BokehMeARNet(
        args.bokeh_model,
        strategy="low-memory",
        intra_op_num_threads=1,
        inference_tile_edge=args.tile_edge,
        full_inference_pixel_limit=args.full_threshold,
    )
    try:
        full_started = time.perf_counter()
        full = engine.infer(small_rgb, defocus)
        full_seconds = time.perf_counter() - full_started

        engine.full_inference_pixel_limit = args.tiled_threshold
        tiled_started = time.perf_counter()
        tiled = engine.infer(small_rgb, defocus)
        tiled_seconds = time.perf_counter() - tiled_started

        if full.rgb_nchw is None or tiled.rgb_nchw is None:
            raise RuntimeError(
                "inference adapter did not return owned comparison tensors"
            )
        delta = np.abs(full.rgb_nchw - tiled.rgb_nchw)
        report = {
            "source": str(args.source),
            "m9_image": str(args.m9_image),
            "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
            "m9_image_sha256": hashlib.sha256(args.m9_image.read_bytes()).hexdigest(),
            "source_size": [width, height],
            "pass_size": [pass_width, pass_height],
            "actual_scale": actual_scale,
            "tile_edge": args.tile_edge,
            "full_threshold": args.full_threshold,
            "tiled_threshold": args.tiled_threshold,
            "full_tiled": full.tiled,
            "candidate_tiled": tiled.tiled,
            "candidate_tile_count": tiled.tile_count,
            "depth_seconds": depth_seconds,
            "full_inference_seconds": full_seconds,
            "tiled_inference_seconds": tiled_seconds,
            "output_sha256_full": _sha256(full.rgb_nchw),
            "output_sha256_tiled": _sha256(tiled.rgb_nchw),
            "bitwise_equal": bool(np.array_equal(full.rgb_nchw, tiled.rgb_nchw)),
            "max_abs_error": float(delta.max(initial=0.0)),
            "mean_abs_error": float(delta.mean()),
            "different_values": int(np.count_nonzero(delta)),
            "models": {
                "depth_sha256": depth_model_sha256,
                "bokeh_sha256": engine.model_sha256,
            },
        }
        print(json.dumps(report, sort_keys=True))
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
