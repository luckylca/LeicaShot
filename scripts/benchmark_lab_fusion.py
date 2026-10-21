#!/usr/bin/env python3
"""Measure full-array and row-streamed Lab fusion in fresh CPU workers."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _worker_main(args: argparse.Namespace) -> int:
    import cv2
    import numpy as np

    from laika.color import fuse_lab_detail

    cv2.setNumThreads(args.threads)
    started = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    original = rng.integers(0, 256, (args.height, args.width, 3), dtype=np.uint8)
    predicted = rng.integers(0, 256, (args.height, args.width, 3), dtype=np.uint8)
    prepared = time.perf_counter()
    output = fuse_lab_detail(
        original,
        predicted,
        row_chunk_height=args.row_chunk_height,
    )
    fused = time.perf_counter()
    np.save(args.output, output, allow_pickle=False)
    saved = time.perf_counter()
    print(
        json.dumps(
            {
                "width": args.width,
                "height": args.height,
                "seed": args.seed,
                "row_chunk_height": args.row_chunk_height,
                "prepare_seconds": prepared - started,
                "fusion_seconds": fused - prepared,
                "save_seconds": saved - fused,
                "worker_seconds": saved - started,
                "numpy": np.__version__,
                "opencv": cv2.__version__,
                "opencv_threads": cv2.getNumThreads(),
            }
        )
    )
    return 0


def _process_tree_sample(pid: int) -> tuple[int, int | None, float, int]:
    import psutil

    try:
        root = psutil.Process(pid)
        processes = [root, *root.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0, None, 0.0, 0
    rss = 0
    uss = 0
    cpu = 0.0
    observed = 0
    uss_available = True
    for process in processes:
        try:
            rss += process.memory_info().rss
            observed += 1
            try:
                times = process.cpu_times()
                cpu += times.user + times.system
            except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
                pass
            try:
                uss += process.memory_full_info().uss
            except (
                AttributeError,
                psutil.AccessDenied,
                psutil.NoSuchProcess,
                OSError,
                RuntimeError,
            ):
                uss_available = False
        except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
            continue
    return rss, uss if uss_available and observed else None, cpu, observed


def _run_worker(
    *,
    width: int,
    height: int,
    row_chunk_height: int,
    threads: int,
    seed: int,
    output: Path,
    sample_interval_ms: int,
) -> dict[str, Any]:
    import psutil

    from laika.benchmark import _cgroup_v2_snapshot

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--width",
        str(width),
        "--height",
        str(height),
        "--row-chunk-height",
        str(row_chunk_height),
        "--threads",
        str(threads),
        "--seed",
        str(seed),
        "--output",
        str(output),
    ]
    cgroup_before = _cgroup_v2_snapshot()
    host_memory_before = psutil.virtual_memory()
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    started = time.perf_counter()
    peak_rss = peak_uss = peak_cpu = process_count = samples = 0
    uss_available = False
    interval = sample_interval_ms / 1000
    while process.poll() is None:
        rss, uss, cpu, count = _process_tree_sample(process.pid)
        peak_rss = max(peak_rss, rss)
        peak_cpu = max(peak_cpu, cpu)
        process_count = max(process_count, count)
        samples += 1
        if uss is not None:
            uss_available = True
            peak_uss = max(peak_uss, uss)
        time.sleep(interval)
    rss, uss, cpu, count = _process_tree_sample(process.pid)
    peak_rss = max(peak_rss, rss)
    peak_cpu = max(peak_cpu, cpu)
    process_count = max(process_count, count)
    if uss is not None:
        uss_available = True
        peak_uss = max(peak_uss, uss)
    stdout, stderr = process.communicate()
    elapsed = time.perf_counter() - started
    if process.returncode != 0:
        raise RuntimeError(stderr.strip() or f"worker exited with {process.returncode}")
    try:
        worker = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("worker returned invalid JSON") from exc
    return {
        "worker": worker,
        "process_tree_peak_rss_bytes": peak_rss,
        "process_tree_peak_uss_bytes": peak_uss if uss_available else None,
        "process_tree_cpu_seconds": peak_cpu,
        "peak_process_count": process_count,
        "sample_interval_ms": sample_interval_ms,
        "samples": samples,
        "parent_wall_seconds": elapsed,
        "host_available_memory_before_bytes": host_memory_before.available,
        "cgroup_v2_before": cgroup_before,
        "cgroup_v2_after": _cgroup_v2_snapshot(),
    }


def _error_metrics(actual: Any, reference: Any) -> dict[str, float | int]:
    import numpy as np

    delta = np.abs(actual.astype(np.int16) - reference.astype(np.int16))
    return {
        "max_absolute_pixel_error": int(delta.max(initial=0)),
        "mean_absolute_pixel_error": float(delta.mean()),
        "p99_absolute_pixel_error": float(np.percentile(delta, 99)),
        "pixels_differing": int(np.count_nonzero(delta)),
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    import cv2
    import numpy as np
    import psutil

    from laika.benchmark import _cgroup_v2_snapshot

    trials = []
    requested_chunks = sorted(set(args.row_chunk_heights))
    for trial in range(args.trials):
        seed = 20261021 + trial
        with tempfile.TemporaryDirectory(prefix="laika-lab-fusion-") as temp_dir:
            directory = Path(temp_dir)
            profiles = [args.height, *requested_chunks]
            profile_rows = []
            baseline: Any | None = None
            for row_chunk_height in dict.fromkeys(profiles):
                output_path = directory / f"fusion-{row_chunk_height}.npy"
                run = _run_worker(
                    width=args.width,
                    height=args.height,
                    row_chunk_height=row_chunk_height,
                    threads=args.threads,
                    seed=seed,
                    output=output_path,
                    sample_interval_ms=args.sample_interval_ms,
                )
                result = np.load(output_path, allow_pickle=False)
                if baseline is None:
                    baseline = result
                    comparison = {
                        "max_absolute_pixel_error": 0,
                        "mean_absolute_pixel_error": 0.0,
                        "p99_absolute_pixel_error": 0.0,
                        "pixels_differing": 0,
                    }
                else:
                    comparison = _error_metrics(result, baseline)
                profile_rows.append({**run, "comparison_to_full_frame": comparison})
            trials.append({"trial": trial + 1, "profiles": profile_rows})

    summaries = []
    for requested in dict.fromkeys([args.height, *requested_chunks]):
        measured = [
            profile
            for trial in trials
            for profile in trial["profiles"]
            if profile["worker"]["row_chunk_height"] == requested
        ]
        summaries.append(
            {
                "row_chunk_height": requested,
                "median_fusion_seconds": statistics.median(
                    row["worker"]["fusion_seconds"] for row in measured
                ),
                "median_worker_process_tree_peak_rss_bytes": statistics.median(
                    row["process_tree_peak_rss_bytes"] for row in measured
                ),
                "median_worker_process_tree_peak_uss_bytes": (
                    statistics.median(
                        row["process_tree_peak_uss_bytes"]
                        for row in measured
                        if row["process_tree_peak_uss_bytes"] is not None
                    )
                    if any(
                        row["process_tree_peak_uss_bytes"] is not None
                        for row in measured
                    )
                    else None
                ),
                "max_absolute_pixel_error_vs_full_frame": max(
                    row["comparison_to_full_frame"]["max_absolute_pixel_error"]
                    for row in measured
                ),
            }
        )

    return {
        "benchmark": "synthetic RGB Lab fusion, fresh worker per trial/profile",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "opencv_threads": args.threads,
            "available_memory_bytes": psutil.virtual_memory().available,
            "cgroup_v2": _cgroup_v2_snapshot(),
        },
        "input": {
            "width": args.width,
            "height": args.height,
            "format": "synthetic HWC RGB uint8 arrays; no photos or model weights",
            "trials": args.trials,
        },
        "sampling": {
            "interval_ms": args.sample_interval_ms,
            "rss": "sampled sum of worker process-tree RSS; short peaks can be missed",
            "cgroup": (
                "values describe the enclosing cgroup and may include other processes"
            ),
        },
        "summaries": summaries,
        "trials": trials,
        "interpretation": (
            "The full-height profile materializes full-frame Lab and float scratch "
            "arrays. Smaller row chunks recompute halo rows but should match the "
            "same full-frame Gaussian result. This is color postprocessing only, "
            "not AI color inference or a complete image pipeline."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=1536)
    parser.add_argument(
        "--row-chunk-heights", type=int, nargs="+", default=[128, 256, 512]
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--sample-interval-ms", type=int, default=5)
    parser.add_argument("--output", type=Path)
    return parser


def _build_worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--row-chunk-height", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    if sys.argv[1:2] == ["--worker"]:
        return _worker_main(_build_worker_parser().parse_args())
    args = build_parser().parse_args()
    if (
        args.width < 1
        or args.height < 1
        or args.trials < 1
        or args.threads < 1
        or args.sample_interval_ms < 1
        or any(size < 1 for size in args.row_chunk_heights)
    ):
        print(
            "width, height, chunks, trials, threads and sample interval "
            "must be positive",
            file=sys.stderr,
        )
        return 2
    try:
        report = run_benchmark(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
