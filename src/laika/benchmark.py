"""Measure end-to-end image processing with process-tree memory sampling."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil
from PIL import __version__ as pillow_version


def _parse_macos_peak_rss(stderr: str) -> int | None:
    """Parse the byte-valued maximum RSS reported by macOS ``time -l``."""
    match = re.search(r"(?m)^\s*(\d+)\s+maximum resident set size\s*$", stderr)
    return int(match.group(1)) if match else None


def _cgroup_v2_snapshot() -> dict[str, Any] | None:
    """Read cgroup v2 counters when the current process is inside a visible cgroup."""
    cgroup_file = Path("/proc/self/cgroup")
    try:
        entries = cgroup_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    relative_path = next(
        (line.split("::", maxsplit=1)[1] for line in entries if "::" in line),
        None,
    )
    if relative_path is None:
        return None
    root = Path("/sys/fs/cgroup").resolve()
    cgroup = (root / relative_path.lstrip("/")).resolve()
    if cgroup != root and root not in cgroup.parents:
        return None
    result: dict[str, Any] = {}
    for filename, key in (
        ("memory.current", "current_bytes"),
        ("memory.max", "limit_bytes"),
        ("memory.peak", "peak_bytes"),
    ):
        try:
            raw = (cgroup / filename).read_text(encoding="ascii").strip()
        except OSError:
            continue
        if raw == "max":
            result[key] = raw
        else:
            try:
                result[key] = int(raw)
            except ValueError:
                continue
    try:
        event_text = (cgroup / "memory.events").read_text(encoding="ascii")
    except OSError:
        event_text = ""
    events = _parse_cgroup_v2_memory_events(event_text)
    if events:
        result["memory_events"] = events
    return result or None


def _parse_cgroup_v2_memory_events(content: str) -> dict[str, int]:
    """Parse Linux cgroup v2 memory event counters, ignoring malformed rows."""
    events: dict[str, int] = {}
    for line in content.splitlines():
        fields = line.split()
        if len(fields) != 2:
            continue
        try:
            value = int(fields[1])
        except ValueError:
            continue
        if value >= 0:
            events[fields[0]] = value
    return events


def _classify_cgroup_worker_outcome(
    return_code: int, before: dict[str, Any] | None, after: dict[str, Any] | None
) -> str:
    """Distinguish a successful worker, an observed OOM kill, and other errors."""
    if return_code == 0:
        return "success"
    before_events = (before or {}).get("memory_events", {})
    after_events = (after or {}).get("memory_events", {})
    if after_events.get("oom_kill", 0) > before_events.get("oom_kill", 0):
        return "oom"
    return "error"


def _process_tree_metrics(root_pid: int) -> tuple[int, int | None, float, int]:
    try:
        root = psutil.Process(root_pid)
        processes = [root, *root.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0, None, 0.0, 0

    rss_total = 0
    uss_total = 0
    cpu_seconds = 0.0
    uss_available = True
    observed = 0
    for process in processes:
        try:
            rss_total += process.memory_info().rss
            observed += 1
            try:
                cpu = process.cpu_times()
                cpu_seconds += cpu.user + cpu.system
            except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
                pass
            try:
                uss_total += process.memory_full_info().uss
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
    return (
        rss_total,
        uss_total if uss_available and observed else None,
        cpu_seconds,
        observed,
    )


def measure_image(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    sample_interval_ms: int = 5,
) -> dict[str, Any]:
    """Run the CLI as a child and sample the full worker process tree."""
    if sample_interval_ms <= 0:
        raise ValueError("sample interval must be positive")
    command = [
        sys.executable,
        "-m",
        "laika.cli",
        "process",
        os.fspath(input_path),
        os.fspath(output_path),
    ]
    cgroup_before = _cgroup_v2_snapshot()
    host_memory_before = psutil.virtual_memory()
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    started = time.perf_counter()
    peak_rss = 0
    peak_uss = 0
    peak_cpu_seconds = 0.0
    uss_observed = False
    peak_process_count = 0
    samples = 0
    interval = sample_interval_ms / 1000
    while True:
        rss, uss, cpu_seconds, count = _process_tree_metrics(process.pid)
        peak_rss = max(peak_rss, rss)
        peak_cpu_seconds = max(peak_cpu_seconds, cpu_seconds)
        if uss is not None:
            uss_observed = True
            peak_uss = max(peak_uss, uss)
        peak_process_count = max(peak_process_count, count)
        samples += 1
        if process.poll() is not None:
            break
        time.sleep(interval)
    stdout, stderr = process.communicate()
    elapsed = time.perf_counter() - started
    if process.returncode != 0:
        raise RuntimeError(
            stderr.strip() or f"Laika worker exited with {process.returncode}"
        )
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Laika worker returned invalid JSON") from exc
    host_memory_after = psutil.virtual_memory()
    cgroup_after = _cgroup_v2_snapshot()

    return {
        "measurement": "subprocess process-tree sampling",
        "sample_interval_ms": sample_interval_ms,
        "samples": samples,
        "peak_process_count": peak_process_count,
        "process_tree_peak_rss_bytes": peak_rss,
        "process_tree_peak_uss_bytes": peak_uss if uss_observed else None,
        "process_tree_cpu_seconds_peak": peak_cpu_seconds,
        "wall_seconds": elapsed,
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "python": platform.python_version(),
            "pillow": pillow_version,
            "psutil": psutil.__version__,
            "available_memory_before_bytes": host_memory_before.available,
            "available_memory_after_bytes": host_memory_after.available,
        },
        "cgroup_v2_before": cgroup_before,
        "cgroup_v2_after": cgroup_after,
        "image": result,
        "interpretation": {
            "rss": (
                "Summed RSS of sampled worker processes; shared pages may be counted "
                "more than once."
            ),
            "uss": "Summed unique-set-size samples when the platform exposes them.",
            "cgroup": (
                "Cgroup values describe the containing cgroup and may include other "
                "processes."
            ),
            "sampling": (
                "Peak can miss allocations shorter than the configured sampling "
                "interval."
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="laika-bench")
    parser.add_argument("input", help="input JPEG, PNG, or WebP")
    parser.add_argument("output", help="output .png path")
    parser.add_argument("--sample-interval-ms", type=int, default=5)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = measure_image(
            args.input,
            args.output,
            sample_interval_ms=args.sample_interval_ms,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
