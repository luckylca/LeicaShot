#!/usr/bin/env python3
"""Measure one real-photo GAIC request including its worker process tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil

from laika.benchmark import _cgroup_v2_snapshot, _process_tree_metrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _measure_gaic_trial(
    input_path: Path,
    model_path: Path,
    *,
    target_long_edge: int | None = None,
    sample_interval_ms: int = 5,
) -> dict[str, Any]:
    if sample_interval_ms < 1:
        raise ValueError("sample interval must be positive")
    command = [
        sys.executable,
        "-m",
        "laika.cli",
        "composition",
        str(input_path),
        "--model",
        str(model_path),
        "--strategy",
        "low-memory",
        "--threads",
        "1",
    ]
    if target_long_edge is not None:
        command.extend(("--target-long-edge", str(target_long_edge)))
    cgroup_before = _cgroup_v2_snapshot()
    memory_before = psutil.virtual_memory()
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    started = time.perf_counter()
    peak_rss = peak_uss = peak_cpu = samples = peak_processes = 0
    uss_available = False
    while process.poll() is None:
        rss, uss, cpu, process_count = _process_tree_metrics(process.pid)
        peak_rss = max(peak_rss, rss)
        peak_cpu = max(peak_cpu, cpu)
        peak_processes = max(peak_processes, process_count)
        if uss is not None:
            uss_available = True
            peak_uss = max(peak_uss, uss)
        samples += 1
        time.sleep(sample_interval_ms / 1000)
    rss, uss, cpu, process_count = _process_tree_metrics(process.pid)
    peak_rss = max(peak_rss, rss)
    peak_cpu = max(peak_cpu, cpu)
    peak_processes = max(peak_processes, process_count)
    if uss is not None:
        uss_available = True
        peak_uss = max(peak_uss, uss)
    stdout, stderr = process.communicate()
    wall_seconds = time.perf_counter() - started
    if process.returncode != 0:
        raise RuntimeError(stderr.strip() or f"GAIC worker exited {process.returncode}")
    result = json.loads(stdout)
    memory_after = psutil.virtual_memory()
    return {
        "measurement": "sampled subprocess process-tree RSS/USS",
        "sample_interval_ms": sample_interval_ms,
        "samples": samples,
        "process_count_peak": peak_processes,
        "process_tree_peak_rss_bytes": peak_rss,
        "process_tree_peak_uss_bytes": peak_uss if uss_available else None,
        "process_tree_cpu_seconds_peak": peak_cpu,
        "wall_seconds": wall_seconds,
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "python": platform.python_version(),
            "onnxruntime": __import__("onnxruntime").__version__,
            "available_memory_before_bytes": memory_before.available,
            "available_memory_after_bytes": memory_after.available,
        },
        "cgroup_v2_before": cgroup_before,
        "cgroup_v2_after": _cgroup_v2_snapshot(),
        "input": {
            "path": str(input_path),
            "sha256": _sha256(input_path),
            "bytes": input_path.stat().st_size,
        },
        "model": {
            "path": str(model_path),
            "sha256": _sha256(model_path),
            "bytes": model_path.stat().st_size,
        },
        "result": result,
        "interpretation": (
            "RSS is sampled across the worker process tree; shared pages can be "
            "counted more than once and brief peaks can fall between samples."
        ),
    }


def measure_gaic(
    input_path: Path,
    model_path: Path,
    *,
    target_long_edge: int | None = None,
    sample_interval_ms: int = 5,
    repeat: int = 1,
) -> dict[str, Any]:
    if repeat < 1:
        raise ValueError("repeat count must be positive")
    trials = [
        _measure_gaic_trial(
            input_path,
            model_path,
            target_long_edge=target_long_edge,
            sample_interval_ms=sample_interval_ms,
        )
        for _ in range(repeat)
    ]
    if repeat == 1:
        return trials[0]
    return {
        "measurement": "repeated fresh Low Memory GAIC CLI workers",
        "repeat": repeat,
        "summary": {
            field: {
                "minimum": min(trial[field] for trial in trials),
                "median": statistics.median(trial[field] for trial in trials),
                "maximum": max(trial[field] for trial in trials),
            }
            for field in ("process_tree_peak_rss_bytes", "wall_seconds")
        },
        "stage_medians": {
            stage: statistics.median(
                trial["result"]["stage_seconds"][stage] for trial in trials
            )
            for stage in trials[0]["result"]["stage_seconds"]
        },
        "candidate_count": trials[0]["result"]["candidate_count"],
        "host": trials[0]["host"],
        "input": trials[0]["input"],
        "model": trials[0]["model"],
        "trials": trials,
        "interpretation": trials[0]["interpretation"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--target-long-edge", type=int)
    parser.add_argument("--sample-interval-ms", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = measure_gaic(
        args.input,
        args.model,
        target_long_edge=args.target_long_edge,
        sample_interval_ms=args.sample_interval_ms,
        repeat=args.repeat,
    )
    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
