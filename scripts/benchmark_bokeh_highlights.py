"""Benchmark the real-photo ARNet plus optical-highlight renderer."""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from laika.bokeh import (
    BokehMeARNet,
    focus_at_point,
    normalize_depth_map,
    render_bokeh,
    resize_rgb_cubic,
)
from laika.cli import _read_rgb, _save_rgb_png
from laika.depth import resize_bilinear_align_corners


class ProcessTreePeak:
    def __init__(self, interval_seconds: float = 0.01) -> None:
        self.process = psutil.Process()
        self.interval_seconds = interval_seconds
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.is_set():
            total = 0
            for process in [self.process, *self.process.children(recursive=True)]:
                try:
                    total += process.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            self.peak_bytes = max(self.peak_bytes, total)
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> ProcessTreePeak:
        self._thread.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self._stop.set()
        self._thread.join()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("depth", type=Path, help="raw float32 depth .npy")
    parser.add_argument("model", type=Path, help="hash-validated ARNet ONNX")
    parser.add_argument("output", type=Path)
    parser.add_argument("--long-edge", type=int, default=2048)
    parser.add_argument("--focus-x", type=float, default=0.5)
    parser.add_argument("--focus-y", type=float, default=0.5)
    parser.add_argument("--aperture", type=float, default=2.8)
    parser.add_argument("--row-chunk-height", type=int, default=64)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.long_edge < 2 or args.row_chunk_height < 1 or args.threads < 1:
        parser.error("long edge, row chunk height, and thread count must be positive")

    started = time.perf_counter()
    image = _read_rgb(args.image, max_bytes=512 * 1024 * 1024, max_pixels=80_000_000)
    raw_depth = np.load(args.depth, mmap_mode="r", allow_pickle=False)
    depth = normalize_depth_map(raw_depth)
    del raw_depth
    height, width = image.shape[:2]
    if depth.shape != (height, width):
        depth = resize_bilinear_align_corners(depth, height, width)
    scale = min(1.0, args.long_edge / max(width, height))
    output_width = max(2, int(round(width * scale)))
    output_height = max(2, int(round(height * scale)))
    if (output_width, output_height) != (width, height):
        image = resize_rgb_cubic(image, output_width, output_height)
        depth = resize_bilinear_align_corners(depth, output_height, output_width)
    focus = focus_at_point(depth, args.focus_x, args.focus_y)

    with ProcessTreePeak() as peak:
        load_started = time.perf_counter()
        model = BokehMeARNet(
            args.model,
            strategy="low-memory",
            intra_op_num_threads=args.threads,
        )
        model_load_seconds = time.perf_counter() - load_started
        result = render_bokeh(
            image,
            depth,
            focus,
            args.aperture,
            model,
            row_chunk_height=args.row_chunk_height,
        )
        output_bytes = _save_rgb_png(args.output, result.rgb)
        model_info = model.runtime_info.as_dict()
        model.close()
    report = {
        "image": str(args.image),
        "image_sha256": _sha256(args.image),
        "depth": str(args.depth),
        "model": model_info,
        "width": output_width,
        "height": output_height,
        "focus": [args.focus_x, args.focus_y, focus],
        "aperture": args.aperture,
        "row_chunk_height": args.row_chunk_height,
        "model_load_seconds": model_load_seconds,
        "stage_seconds": result.stage_seconds,
        "renderer_seconds": result.total_seconds,
        "task_wall_seconds": time.perf_counter() - started,
        "process_tree_peak_rss_bytes": peak.peak_bytes,
        "output": str(args.output),
        "output_bytes": output_bytes,
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
