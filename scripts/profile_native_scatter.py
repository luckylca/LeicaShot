"""Profile the full Python processing pipeline's optical highlight scatter."""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import platform
import pstats
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import psutil


class ScatterProfiler:
    """Profile one scatter call and time OpenCV filters by kernel geometry."""

    def __init__(self, highlights_module: Any, profile_path: Path) -> None:
        self.module = highlights_module
        self.profile_path = profile_path
        self.profile = cProfile.Profile()
        self.calls = 0
        self.filter_groups: dict[str, dict[str, float | int]] = defaultdict(
            lambda: {"calls": 0, "seconds": 0.0}
        )
        self._scatter = highlights_module.scatter_highlights
        self._filter2d = highlights_module.cv2.filter2D
        self._installed = False

    def install(self) -> None:
        if self._installed:
            raise RuntimeError("scatter profiler is already installed")
        self.module.scatter_highlights = self._profiled_scatter
        self._installed = True

    def restore(self) -> None:
        if self._installed:
            self.module.scatter_highlights = self._scatter
            self.module.cv2.filter2D = self._filter2d
            self._installed = False

    def _profiled_scatter(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1

        def timed_filter2d(*filter_args: Any, **filter_kwargs: Any) -> Any:
            source = filter_args[0]
            depth = filter_kwargs.get(
                "ddepth", filter_args[1] if len(filter_args) > 1 else None
            )
            kernel = filter_kwargs.get(
                "kernel", filter_args[2] if len(filter_args) > 2 else None
            )
            channels = source.shape[2] if source.ndim == 3 else 1
            side = int(kernel.shape[0])
            key = (
                f"kernel={side}x{int(kernel.shape[1])};channels={channels};"
                f"dtype={source.dtype};ddepth={depth}"
            )
            started = time.perf_counter()
            try:
                return self._filter2d(*filter_args, **filter_kwargs)
            finally:
                item = self.filter_groups[key]
                item["calls"] += 1
                item["seconds"] += time.perf_counter() - started

        self.module.cv2.filter2D = timed_filter2d
        try:
            return self.profile.runcall(self._scatter, *args, **kwargs)
        finally:
            self.module.cv2.filter2D = self._filter2d

    def write(self) -> dict[str, Any]:
        if self.calls == 0:
            return {
                "calls": 0,
                "profile_file": None,
                "top_functions": [],
                "filter2d_groups": [],
                "skipped": "renderer had no highlight radiance to scatter",
            }
        if self.calls != 1:
            raise RuntimeError(f"expected one scatter call, observed {self.calls}")
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        self.profile.dump_stats(str(self.profile_path))
        stats = pstats.Stats(self.profile)
        top_functions = []
        for (filename, line, function), values in sorted(
            stats.stats.items(), key=lambda item: item[1][3], reverse=True
        )[:30]:
            primitive_calls, total_calls, self_seconds, cumulative_seconds, _ = values
            top_functions.append(
                {
                    "file": Path(filename).name,
                    "line": line,
                    "function": function,
                    "primitive_calls": primitive_calls,
                    "calls": total_calls,
                    "self_seconds": self_seconds,
                    "cumulative_seconds": cumulative_seconds,
                }
            )
        filters = [
            {"group": key, **value}
            for key, value in sorted(
                self.filter_groups.items(),
                key=lambda item: float(item[1]["seconds"]),
                reverse=True,
            )
        ]
        return {
            "calls": self.calls,
            "profile_file": str(self.profile_path),
            "top_functions": top_functions,
            "filter2d_groups": filters,
        }


class ProcessTreePeak:
    def __init__(self, interval_seconds: float = 0.01) -> None:
        self.process = psutil.Process()
        self.interval_seconds = interval_seconds
        self.peak_rss_bytes = 0
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
            self.peak_rss_bytes = max(self.peak_rss_bytes, total)
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


def _point(value: str) -> tuple[float, float]:
    try:
        x_text, y_text = value.split(",", 1)
        point = float(x_text), float(y_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("point must be written as x,y") from exc
    if not all(0.0 <= coordinate < 1.0 for coordinate in point):
        raise argparse.ArgumentTypeError("point coordinates must be in [0, 1)")
    return point


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--m9-model", type=Path, required=True)
    parser.add_argument("--depth-model", type=Path, required=True)
    parser.add_argument("--bokeh-model", type=Path, required=True)
    parser.add_argument("--sam-model-dir", type=Path, required=True)
    parser.add_argument("--vitmatte-model", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--focus-x", type=float, default=0.5)
    parser.add_argument("--focus-y", type=float, default=0.5)
    parser.add_argument("--aperture", type=float, default=2.8)
    parser.add_argument("--matte-tile-size", type=int, default=512)
    parser.add_argument("--subject-point", type=_point, action="append")
    args = parser.parse_args()

    import cv2
    import numpy as np
    import onnxruntime as ort

    from laika import highlights
    from laika.benchmark import _cgroup_v2_snapshot
    from laika.processing import LaikaProcessingEngine, ProcessingOptions

    points = tuple(args.subject_point or [(0.72, 0.66)])
    options = ProcessingOptions(
        mode="color+bokeh",
        strategy="low-memory",
        focus_x=args.focus_x,
        focus_y=args.focus_y,
        aperture=args.aperture,
        subject_points=points,
        matte_tile_size=args.matte_tile_size,
    )
    profiler = ScatterProfiler(highlights, args.profile)
    profiler.install()
    started = time.perf_counter()
    cpu_before = psutil.Process().cpu_times()
    cgroup_before = _cgroup_v2_snapshot()
    try:
        with (
            ProcessTreePeak() as peak,
            LaikaProcessingEngine(
                m9_model_path=args.m9_model,
                depth_model_path=args.depth_model,
                bokeh_model_path=args.bokeh_model,
                sam_model_dir=args.sam_model_dir,
                vitmatte_model_path=args.vitmatte_model,
                strategy="low-memory",
                threads=1,
            ) as engine,
        ):
            result = engine.process_file(args.input, args.output, options=options)
    finally:
        profiler.restore()
    wall_seconds = time.perf_counter() - started
    cpu_after = psutil.Process().cpu_times()
    profiler_report = profiler.write()
    from PIL import Image

    with Image.open(args.output) as image:
        output_size = image.size
        output_format = image.format
    report = {
        "input": str(args.input),
        "input_sha256": _sha256(args.input),
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
        "output_bytes": args.output.stat().st_size,
        "output_size": list(output_size),
        "output_format": output_format,
        "options": {
            "mode": options.mode,
            "strategy": options.strategy,
            "focus": [options.focus_x, options.focus_y],
            "aperture": options.aperture,
            "subject_points": [list(point) for point in points],
            "matte_tile_size": options.matte_tile_size,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": psutil.cpu_count(logical=True),
            "memory_total_bytes": psutil.virtual_memory().total,
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "onnxruntime": ort.__version__,
            "onnxruntime_providers": ort.get_available_providers(),
        },
        "task_wall_seconds": wall_seconds,
        "process_cpu_seconds": (cpu_after.user + cpu_after.system)
        - (cpu_before.user + cpu_before.system),
        "process_tree_peak_rss_bytes": peak.peak_rss_bytes,
        "cgroup_v2_before": cgroup_before,
        "cgroup_v2_after": _cgroup_v2_snapshot(),
        "engine_result": result.as_dict(),
        "scatter_profile": profiler_report,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
