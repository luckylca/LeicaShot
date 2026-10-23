"""Benchmark one real-photo task through the public Python processing engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

# Benchmarking from a source mount must measure that checkout, not a potentially
# stale package baked into the benchmark container image.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
_PACKAGE_ROOT = _SOURCE_ROOT / "laika"
if not (_PACKAGE_ROOT / "__init__.py").is_file():
    raise RuntimeError(f"Laika source package is missing from {_PACKAGE_ROOT}")
sys.path.insert(0, str(_SOURCE_ROOT))

import psutil  # noqa: E402

import laika  # noqa: E402
from laika.cgroup import memory_stats as _cgroup_memory  # noqa: E402
from laika.composition import CompositionCrop  # noqa: E402
from laika.processing import LaikaProcessingEngine, ProcessingOptions  # noqa: E402


def _source_tree_sha256() -> str:
    digest = hashlib.sha256()
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        digest.update(path.relative_to(_SOURCE_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _source_provenance() -> dict[str, Any]:
    package_file = Path(laika.__file__).resolve()
    if not package_file.is_relative_to(_PACKAGE_ROOT.resolve()):
        raise RuntimeError(
            f"Benchmark imported Laika from {package_file}, outside {_PACKAGE_ROOT}"
        )
    try:
        revision = subprocess.run(
            ["git", "-C", str(_REPOSITORY_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(_REPOSITORY_ROOT), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        revision = None
        dirty = None
    return {
        "repository_root": str(_REPOSITORY_ROOT),
        "package_file": str(package_file),
        "source_tree_sha256": _source_tree_sha256(),
        "git_revision": revision,
        "git_worktree_dirty": dirty,
    }


class ProcessTreePeak:
    """Sample this benchmark and all descendants while the task is running."""

    def __init__(self, interval_seconds: float = 0.01) -> None:
        self.process = psutil.Process()
        self.interval_seconds = interval_seconds
        self.peak_bytes = 0
        self._stage = "startup"
        self._stage_peaks: dict[str, int] = {}
        self._stage_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def set_stage(self, stage: str) -> None:
        with self._stage_lock:
            self._stage = stage

    def stage_peaks(self) -> dict[str, int]:
        with self._stage_lock:
            return dict(self._stage_peaks)

    def current_bytes(self) -> int:
        total = 0
        for process in [self.process, *self.process.children(recursive=True)]:
            try:
                total += process.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return total

    def _sample(self) -> None:
        while not self._stop.is_set():
            total = self.current_bytes()
            self.peak_bytes = max(self.peak_bytes, total)
            with self._stage_lock:
                self._stage_peaks[self._stage] = max(
                    self._stage_peaks.get(self._stage, 0), total
                )
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


def _parse_crop(value: str) -> CompositionCrop:
    try:
        coordinates = tuple(float(item) for item in value.split(","))
        if len(coordinates) != 4:
            raise ValueError
        return CompositionCrop(*coordinates)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "crop must be LEFT,TOP,RIGHT,BOTTOM in normalized coordinates"
        ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--m9-model", type=Path)
    parser.add_argument("--depth-model", type=Path)
    parser.add_argument("--bokeh-model", type=Path)
    parser.add_argument("--sam-model-dir", type=Path)
    parser.add_argument("--vitmatte-model", type=Path)
    parser.add_argument(
        "--subject-point",
        action="append",
        default=[],
        metavar="X,Y",
        help="normalized foreground point; repeat for multiple subjects",
    )
    parser.add_argument(
        "--matte-tile-size", type=int, choices=(512, 768, 1024), default=1024
    )
    parser.add_argument(
        "--mode", choices=("color", "bokeh", "color+bokeh"), default="color+bokeh"
    )
    parser.add_argument(
        "--strategy", choices=("low-memory", "balanced"), default="low-memory"
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument(
        "--crop",
        type=_parse_crop,
        metavar="LEFT,TOP,RIGHT,BOTTOM",
        help="apply an explicitly selected normalized crop after AI processing",
    )
    parser.add_argument("--focus-x", type=float, default=0.5)
    parser.add_argument("--focus-y", type=float, default=0.5)
    parser.add_argument("--aperture", type=float, default=2.8)
    parser.add_argument("--bokeh-row-chunk-height", type=int, default=256)
    parser.add_argument(
        "--progress-log",
        type=Path,
        help="append progress and sampled RSS records, including before a cgroup OOM",
    )
    args = parser.parse_args()
    if (args.width is None) != (args.height is None):
        parser.error("--width and --height must be supplied together")
    if args.threads < 1 or args.bokeh_row_chunk_height < 1:
        parser.error("threads and row chunk height must be positive")
    subject_points = []
    for item in args.subject_point:
        try:
            x_text, y_text = item.split(",", maxsplit=1)
            subject_points.append((float(x_text), float(y_text)))
        except (ValueError, AttributeError):
            parser.error("--subject-point must use normalized X,Y coordinates")

    source_provenance = _source_provenance()
    started = time.perf_counter()
    progress_context = (
        args.progress_log.open("w", encoding="utf-8")
        if args.progress_log is not None
        else nullcontext(None)
    )
    with (
        progress_context as progress_stream,
        LaikaProcessingEngine(
            m9_model_path=args.m9_model,
            depth_model_path=args.depth_model,
            bokeh_model_path=args.bokeh_model,
            sam_model_dir=args.sam_model_dir,
            vitmatte_model_path=args.vitmatte_model,
            strategy=args.strategy,
            threads=args.threads,
        ) as engine,
        ProcessTreePeak() as peak,
    ):

        def on_progress(event: Any) -> None:
            if event.stage == "bokeh":
                if (
                    getattr(event, "details", None) is not None
                    and event.details.get("component") == "arnet_tile"
                ):
                    stage = "bokeh.arnet_tile"
                elif event.fraction < 0.69:
                    stage = "bokeh.highlight_extraction"
                elif event.fraction < 0.82:
                    stage = "bokeh.arnet"
                elif event.fraction < 0.83:
                    stage = "bokeh.highlight_recovery"
                elif event.fraction < 0.91:
                    stage = "bokeh.highlight_scatter"
                elif event.fraction < 0.92:
                    stage = "bokeh.highlight_softening"
                elif event.fraction < 0.94:
                    stage = "bokeh.highlight_composite"
                else:
                    stage = "bokeh.finalize"
            else:
                stage = event.stage
            peak.set_stage(stage)
            if progress_stream is not None:
                record = {
                    "stage": stage,
                    "fraction": event.fraction,
                    "elapsed_seconds": event.elapsed_seconds,
                    "process_tree_rss_bytes": peak.current_bytes(),
                    "cgroup_memory": _cgroup_memory(),
                }
                if getattr(event, "details", None) is not None:
                    record["details"] = event.details
                progress_stream.write(json.dumps(record, sort_keys=True) + "\n")
                progress_stream.flush()
            print(
                f"progress stage={stage} fraction={event.fraction:.4f} "
                f"rss={peak.current_bytes()}",
                file=sys.stderr,
                flush=True,
            )

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
                subject_points=tuple(subject_points),
                matte_tile_size=args.matte_tile_size,
                crop=args.crop,
            ),
            progress_callback=on_progress,
        )
    report = {
        "source": source_provenance,
        "input": str(args.input),
        "input_sha256": _sha256(args.input),
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
        "mode": args.mode,
        "strategy": args.strategy,
        "width": result.width,
        "height": result.height,
        "crop": args.crop.as_tuple() if args.crop is not None else None,
        "processing_options": {
            "focus_x": args.focus_x,
            "focus_y": args.focus_y,
            "aperture": args.aperture,
            "subject_points": [list(point) for point in subject_points],
            "output_size": [args.width, args.height]
            if args.width is not None
            else None,
            "matte_tile_size": args.matte_tile_size,
        },
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
        "process_tree_stage_peak_rss_bytes": peak.stage_peaks(),
        "cgroup_memory": _cgroup_memory(),
        "output_bytes": result.output_bytes,
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
