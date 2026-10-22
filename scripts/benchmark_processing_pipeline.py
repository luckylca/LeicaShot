"""Benchmark one real-photo task through the public Python processing engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import threading
import time
from pathlib import Path
from typing import Any

import psutil

from laika.processing import LaikaProcessingEngine, ProcessingOptions


class ProcessTreePeak:
    """Sample this benchmark and all descendants while the task is running."""

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
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--m9-model", type=Path)
    parser.add_argument("--depth-model", type=Path)
    parser.add_argument("--bokeh-model", type=Path)
    parser.add_argument(
        "--mode", choices=("color", "bokeh", "color+bokeh"), default="color+bokeh"
    )
    parser.add_argument(
        "--strategy", choices=("low-memory", "balanced"), default="low-memory"
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--focus-x", type=float, default=0.5)
    parser.add_argument("--focus-y", type=float, default=0.5)
    parser.add_argument("--aperture", type=float, default=2.8)
    parser.add_argument("--bokeh-row-chunk-height", type=int, default=128)
    args = parser.parse_args()
    if (args.width is None) != (args.height is None):
        parser.error("--width and --height must be supplied together")
    if args.threads < 1 or args.bokeh_row_chunk_height < 1:
        parser.error("threads and row chunk height must be positive")

    started = time.perf_counter()
    with (
        LaikaProcessingEngine(
            m9_model_path=args.m9_model,
            depth_model_path=args.depth_model,
            bokeh_model_path=args.bokeh_model,
            strategy=args.strategy,
            threads=args.threads,
        ) as engine,
        ProcessTreePeak() as peak,
    ):
        result = engine.process_file(
            args.input,
            args.output,
            options=ProcessingOptions(
                mode=args.mode,
                strategy=args.strategy,
                focus_x=args.focus_x,
                focus_y=args.focus_y,
                aperture=args.aperture,
                output_size=(args.width, args.height)
                if args.width is not None
                else None,
                bokeh_row_chunk_height=args.bokeh_row_chunk_height,
            ),
        )
    report = {
        "input": str(args.input),
        "input_sha256": _sha256(args.input),
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
        "mode": args.mode,
        "strategy": args.strategy,
        "width": result.width,
        "height": result.height,
        "threads": args.threads,
        "bokeh_row_chunk_height": args.bokeh_row_chunk_height,
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "memory_total_bytes": psutil.virtual_memory().total,
        },
        "models": result.models,
        "stage_seconds": result.stage_seconds,
        "task_wall_seconds": time.perf_counter() - started,
        "process_tree_peak_rss_bytes": peak.peak_bytes,
        "output_bytes": result.output_bytes,
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
