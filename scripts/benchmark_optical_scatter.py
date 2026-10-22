"""Compare legacy and mirrored-kernel scatter on cached M9 and depth outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil

from laika import highlights
from laika.bokeh import (
    focus_at_point,
    full_frame_coc,
    normalize_depth_map,
    resize_rgb_cubic,
)
from laika.cgroup import memory_stats as _cgroup_memory
from laika.cli import _read_rgb
from laika.depth import resize_bilinear_align_corners
from laika.highlights import extract_highlights, scatter_highlights


class ProcessTreePeak:
    """Sample this process and descendants while the scatter stage runs."""

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
    parser.add_argument("m9_rgb", type=Path, help="M9-rendered RGB image")
    parser.add_argument("depth_npy", type=Path, help="raw Depth Anything float32 .npy")
    parser.add_argument("output", type=Path, help="output float32 scatter .npy")
    parser.add_argument("--long-edge", type=int, default=2048)
    parser.add_argument("--focus-x", type=float, default=0.5)
    parser.add_argument("--focus-y", type=float, default=0.5)
    parser.add_argument("--aperture", type=float, default=2.8)
    parser.add_argument("--row-chunk-height", type=int, default=128)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    if min(args.long_edge, args.row_chunk_height, args.repeat) < 1:
        parser.error("long edge, row chunk, and repeat must be positive")

    started = time.perf_counter()
    image = _read_rgb(args.m9_rgb, max_bytes=512 * 1024 * 1024, max_pixels=80_000_000)
    raw_depth = np.load(args.depth_npy, mmap_mode="r", allow_pickle=False)
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
    coc = full_frame_coc(depth, focus, args.aperture)
    split = extract_highlights(image)
    preprocess_seconds = time.perf_counter() - started

    legacy_seconds: list[float] = []
    legacy_peak_rss: list[int] = []
    legacy_hashes: list[str] = []
    reused_seconds: list[float] = []
    reused_peak_rss: list[int] = []
    reused_hashes: list[str] = []
    result: Any | None = None
    original_builder = highlights._field_pupil_kernels

    def run_scatter(*, reuse: bool) -> tuple[Any, float, int, str]:
        if not reuse:
            highlights._field_pupil_kernels = lambda shapes, radius, aperture: [
                highlights._optical_pupil_kernel(radius, aperture, *shape)
                for shape in shapes
            ]
        stage_started = time.perf_counter()
        try:
            with ProcessTreePeak() as peak:
                scattered = scatter_highlights(
                    split.highlight_linear,
                    coc,
                    args.aperture,
                    radius_scale=3.0,
                    field_optics=True,
                    row_chunk_height=args.row_chunk_height,
                )
        finally:
            highlights._field_pupil_kernels = original_builder
        elapsed = time.perf_counter() - stage_started
        digest = hashlib.sha256(scattered.tobytes()).hexdigest()
        return scattered, elapsed, peak.peak_bytes, digest

    for _ in range(args.repeat):
        legacy, elapsed, peak_bytes, digest = run_scatter(reuse=False)
        legacy_seconds.append(elapsed)
        legacy_peak_rss.append(peak_bytes)
        legacy_hashes.append(digest)
        del legacy
        result, elapsed, peak_bytes, digest = run_scatter(reuse=True)
        reused_seconds.append(elapsed)
        reused_peak_rss.append(peak_bytes)
        reused_hashes.append(digest)

    assert result is not None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, result, allow_pickle=False)
    report = {
        "m9_rgb": str(args.m9_rgb),
        "m9_rgb_file_sha256": _sha256(args.m9_rgb),
        "depth_npy": str(args.depth_npy),
        "depth_file_sha256": _sha256(args.depth_npy),
        "prepared_rgb_shape": list(image.shape),
        "prepared_highlight_nonzero": int(np.count_nonzero(split.highlight_linear)),
        "prepared_highlight_sum": float(split.highlight_linear.sum()),
        "focus": [args.focus_x, args.focus_y, focus],
        "aperture": args.aperture,
        "row_chunk_height": args.row_chunk_height,
        "opencv_threads_observed": cv2.getNumThreads(),
        "legacy_25_kernel_seconds": legacy_seconds,
        "legacy_peak_rss_bytes": legacy_peak_rss,
        "legacy_output_array_sha256": legacy_hashes,
        "reused_13_kernel_seconds": reused_seconds,
        "reused_peak_rss_bytes": reused_peak_rss,
        "reused_output_array_sha256": reused_hashes,
        "preprocess_seconds": preprocess_seconds,
        "cgroup_memory": _cgroup_memory(),
        "output_npy": str(args.output),
        "output_file_sha256": _sha256(args.output),
        "output_array_sha256": hashlib.sha256(result.tobytes()).hexdigest(),
        "output_sum": float(result.sum()),
        "output_nonzero": int(np.count_nonzero(result)),
        "total_wall_seconds": time.perf_counter() - started,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "logical_cpus": psutil.cpu_count(logical=True),
            "memory_total_bytes": psutil.virtual_memory().total,
        },
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
