"""Compare the API benchmark's process sampling overhead on Linux or macOS."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import threading
import time
from typing import Any

import psutil
from benchmark_api_task import ServerProcessPeak


def _cpu_seconds(process: psutil.Process) -> float:
    times = process.cpu_times()
    return times.user + times.system


def _run_legacy_sampler(
    target_pid: int, duration_seconds: float, interval_seconds: float
) -> dict[str, Any]:
    stop = threading.Event()
    counts = {"rss": 0, "uss": 0}

    def sample() -> None:
        while not stop.is_set():
            try:
                root = psutil.Process(target_pid)
                processes = [root, *root.children(recursive=True)]
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            for process in processes:
                try:
                    process.memory_info()
                    process.memory_full_info()
                    root.cpu_times()
                    counts["rss"] += 1
                    counts["uss"] += 1
                except (
                    psutil.NoSuchProcess,
                    psutil.AccessDenied,
                    psutil.ZombieProcess,
                    OSError,
                ):
                    continue
            stop.wait(interval_seconds)

    parent = psutil.Process()
    thread = threading.Thread(target=sample, daemon=True)
    cpu_before = _cpu_seconds(parent)
    wall_started = time.monotonic()
    thread.start()
    time.sleep(duration_seconds)
    stop.set()
    thread.join()
    return {
        "wall_seconds": time.monotonic() - wall_started,
        "observer_process_cpu_seconds": _cpu_seconds(parent) - cpu_before,
        "rss_samples": counts["rss"],
        "uss_scans": counts["uss"],
        "rss_interval_seconds": interval_seconds,
        "uss_interval_seconds": interval_seconds,
    }


def _run_current_sampler(
    target_pid: int,
    duration_seconds: float,
    rss_interval_seconds: float,
    uss_interval_seconds: float,
) -> dict[str, Any]:
    parent = psutil.Process()
    sampler = ServerProcessPeak(
        target_pid,
        rss_interval_seconds=rss_interval_seconds,
        uss_interval_seconds=uss_interval_seconds,
    )
    cpu_before = _cpu_seconds(parent)
    wall_started = time.monotonic()
    with sampler:
        time.sleep(duration_seconds)
    return {
        "wall_seconds": time.monotonic() - wall_started,
        "observer_process_cpu_seconds": _cpu_seconds(parent) - cpu_before,
        "rss_samples": sampler.rss_sample_count,
        "uss_scans": sampler.uss_sample_count,
        "rss_interval_seconds": rss_interval_seconds,
        "uss_interval_seconds": uss_interval_seconds,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    target_code = (
        "import time\n"
        f"memory = bytearray({args.target_memory_mib} * 1024 * 1024)\n"
        "for offset in range(0, len(memory), 4096): memory[offset] = 1\n"
        "value = 1\n"
        "while True: value = (value * 1664525 + 1013904223) & 0xffffffff\n"
    )
    target = subprocess.Popen([sys.executable, "-c", target_code])
    try:
        target_process = psutil.Process(target.pid)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                target_rss_bytes = target_process.memory_info().rss
                if target_rss_bytes >= args.target_memory_mib * 1024 * 1024:
                    break
            except psutil.NoSuchProcess as error:
                raise RuntimeError("sampling target exited during setup") from error
            time.sleep(0.05)
        else:
            raise TimeoutError("sampling target did not allocate the requested memory")

        return {
            "platform": platform.platform(),
            "python": sys.version,
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "target_rss_bytes": target_rss_bytes,
            "target_memory_mib": args.target_memory_mib,
            "duration_seconds_per_sampler": args.duration_seconds,
            "legacy_10ms_rss_uss": _run_legacy_sampler(
                target.pid, args.duration_seconds, args.legacy_interval_seconds
            ),
            "current_sampler": _run_current_sampler(
                target.pid,
                args.duration_seconds,
                args.rss_interval_seconds,
                args.uss_interval_seconds,
            ),
        }
    finally:
        target.terminate()
        target.wait(timeout=10)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-seconds", type=float, default=20.0)
    parser.add_argument("--target-memory-mib", type=int, default=256)
    parser.add_argument("--legacy-interval-seconds", type=float, default=0.01)
    parser.add_argument("--rss-interval-seconds", type=float, default=0.1)
    parser.add_argument("--uss-interval-seconds", type=float, default=1.0)
    args = parser.parse_args()
    if (
        args.duration_seconds <= 0
        or args.target_memory_mib <= 0
        or args.legacy_interval_seconds <= 0
        or args.rss_interval_seconds <= 0
        or args.uss_interval_seconds <= 0
    ):
        parser.error("duration, memory size, and intervals must be positive")
    print(json.dumps(run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
