#!/usr/bin/env python3
"""Run one depth CLI benchmark and capture its enclosing cgroup outcome."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from laika.benchmark import (  # noqa: E402
    _cgroup_v2_snapshot,
    _classify_cgroup_worker_outcome,
)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    command = [
        sys.executable,
        str(ROOT / "scripts/benchmark_depth_inference.py"),
        "--model",
        str(args.model.resolve()),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--threads",
        str(args.threads),
        "--sample-interval-ms",
        str(args.sample_interval_ms),
    ]
    if args.input_size is not None:
        command.extend(("--input-size", str(args.input_size)))
    else:
        command.extend(("--analysis-max-edge", str(args.analysis_max_edge)))

    cgroup_before = _cgroup_v2_snapshot()
    started = time.perf_counter()
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - started
    cgroup_after = _cgroup_v2_snapshot()

    benchmark_report = None
    if process.returncode == 0:
        try:
            benchmark_report = json.loads(process.stdout)
        except json.JSONDecodeError:
            pass
    outcome = _classify_cgroup_worker_outcome(
        process.returncode, cgroup_before, cgroup_after
    )
    if outcome == "success" and benchmark_report is None:
        outcome = "error"

    return {
        "outcome": outcome,
        "worker_exit_code": process.returncode,
        "benchmark_process_wall_seconds": elapsed,
        "cgroup_v2_before": cgroup_before,
        "cgroup_v2_after": cgroup_after,
        "benchmark": benchmark_report,
        "worker_stderr": process.stderr.strip()[-4000:] or None,
        "worker_stdout": (
            process.stdout.strip()[-1000:]
            if process.returncode == 0 and benchmark_report is None
            else None
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    size_group = parser.add_mutually_exclusive_group()
    size_group.add_argument("--analysis-max-edge", type=int, default=1220)
    size_group.add_argument("--input-size", type=int)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--sample-interval-ms", type=int, default=5)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if min(args.width, args.height, args.threads, args.sample_interval_ms) < 1:
        print(json.dumps({"error": "dimensions and sampling values must be positive"}))
        return 2
    try:
        report = _run(args)
    except OSError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
