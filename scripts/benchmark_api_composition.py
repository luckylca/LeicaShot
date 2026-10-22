#!/usr/bin/env python3
"""Benchmark GAIC crop suggestions through a local FastAPI worker."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run(args: argparse.Namespace) -> dict[str, Any]:
    import httpx
    import psutil

    from laika.benchmark import _cgroup_v2_snapshot
    from scripts.benchmark_api_task import ServerProcessPeak, _reserve_port, _sha256

    source = Path(args.input)
    host = "127.0.0.1"
    port = _reserve_port()
    base_url = f"http://{host}:{port}"
    child_env = dict(os.environ)
    child_env.update(
        {
            "LAIKA_API_HOST": host,
            "LAIKA_API_PORT": str(port),
        }
    )
    temporary_storage: Path | None = None
    if not child_env.get("LAIKA_API_STORAGE"):
        temporary_storage = Path(tempfile.mkdtemp(prefix="leicashot-api-gaic-"))
        child_env["LAIKA_API_STORAGE"] = str(temporary_storage)

    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "laika.api:create_app",
        "--factory",
        "--host",
        host,
        "--port",
        str(port),
        "--workers",
        "1",
        "--log-level",
        "warning",
    ]
    server = subprocess.Popen(
        command,
        cwd=ROOT,
        env=child_env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    peak = ServerProcessPeak(server.pid)
    available_before = psutil.virtual_memory().available
    cgroup_before = _cgroup_v2_snapshot()
    report: dict[str, Any] | None = None
    try:
        with peak, httpx.Client(timeout=120.0) as client:
            for _ in range(200):
                if server.poll() is not None:
                    break
                try:
                    if client.get(f"{base_url}/api/v1/health").status_code == 200:
                        break
                except httpx.HTTPError:
                    time.sleep(0.1)
            else:
                raise TimeoutError("local Laika API did not become healthy")
            if server.poll() is not None:
                logs = server.communicate(timeout=1)[0]
                raise RuntimeError(f"Laika API exited during startup:\n{logs}")

            media_type = (
                mimetypes.guess_type(source.name)[0] or "application/octet-stream"
            )
            data = {"strategy": args.strategy}
            if args.target_long_edge is not None:
                data["target_long_edge"] = str(args.target_long_edge)
            started = time.perf_counter()
            with source.open("rb") as image_file:
                response = client.post(
                    f"{base_url}/api/v1/compositions",
                    data=data,
                    files={"file": (source.name, image_file, media_type)},
                )
            task_wall_seconds = time.perf_counter() - started
            if response.status_code != 200:
                raise RuntimeError(
                    f"API composition returned {response.status_code}: {response.text}"
                )
            result = response.json()
            report = {
                "operation": "gaic-composition-suggestion",
                "input": str(source),
                "input_sha256": _sha256(source),
                "input_bytes": source.stat().st_size,
                "strategy": args.strategy,
                "target_long_edge": args.target_long_edge,
                "host": {
                    "platform": platform.platform(),
                    "processor": platform.processor(),
                    "cpu_count_logical": psutil.cpu_count(logical=True),
                    "memory_total_bytes": psutil.virtual_memory().total,
                    "memory_available_before_bytes": available_before,
                },
                "python": sys.version,
                "api_versioned_path": "/api/v1/compositions",
                "task_wall_seconds": task_wall_seconds,
                "server_process_tree_peak_rss_bytes": peak.peak_rss_bytes,
                "server_process_tree_peak_uss_bytes": peak.peak_uss_bytes,
                "server_cpu_seconds": max(
                    0.0, peak.cpu_seconds_after - peak.cpu_seconds_before
                ),
                "cgroup_v2_before": cgroup_before,
                "cgroup_v2_after": _cgroup_v2_snapshot(),
                "result": result,
            }
    finally:
        if server.poll() is None:
            server.terminate()
        try:
            logs = server.communicate(timeout=30)[0]
        except subprocess.TimeoutExpired:
            server.kill()
            logs = server.communicate()[0]
        if server.returncode not in (0, -15, 143) and report is None:
            raise RuntimeError(f"Laika API process failed:\n{logs}")
        if temporary_storage is not None:
            shutil.rmtree(temporary_storage, ignore_errors=True)

    if report is None:
        raise RuntimeError("GAIC API benchmark ended without a report")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--strategy", choices=("low-memory", "balanced"), default="low-memory"
    )
    parser.add_argument("--target-long-edge", type=int, default=1200)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.target_long_edge is not None and not 1 <= args.target_long_edge <= 8192:
        parser.error("target-long-edge must be in [1, 8192]")
    report = run(args)
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
