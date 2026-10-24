#!/usr/bin/env python3
"""Benchmark real tasks through one local Uvicorn API worker pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import platform
import socket
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class ServerProcessPeak:
    """Sample API process-tree RSS and periodic USS scans."""

    def __init__(
        self,
        pid: int,
        rss_interval_seconds: float = 0.1,
        uss_interval_seconds: float = 1.0,
    ) -> None:
        import psutil

        if rss_interval_seconds <= 0 or uss_interval_seconds <= 0:
            raise ValueError("process sampling intervals must be positive")
        self.pid = pid
        self.rss_interval_seconds = rss_interval_seconds
        self.uss_interval_seconds = uss_interval_seconds
        self.peak_rss_bytes = 0
        self.peak_uss_bytes: int | None = None
        self.rss_sample_count = 0
        self.uss_sample_count = 0
        self._psutil = psutil
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        next_uss_sample = 0.0
        while not self._stop.is_set():
            try:
                root = self._psutil.Process(self.pid)
                processes = [root, *root.children(recursive=True)]
            except (self._psutil.NoSuchProcess, self._psutil.AccessDenied):
                break
            now = time.monotonic()
            sample_uss = now >= next_uss_sample
            rss = 0
            uss = 0
            uss_available = True
            for process in processes:
                try:
                    rss += process.memory_info().rss
                    if sample_uss:
                        uss += process.memory_full_info().uss
                except (
                    self._psutil.NoSuchProcess,
                    self._psutil.AccessDenied,
                    OSError,
                    RuntimeError,
                ):
                    uss_available = False
            self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
            self.rss_sample_count += 1
            if sample_uss:
                self.uss_sample_count += 1
                next_uss_sample = now + self.uss_interval_seconds
                if uss_available:
                    self.peak_uss_bytes = max(self.peak_uss_bytes or 0, uss)
            self._stop.wait(self.rss_interval_seconds)

    def __enter__(self) -> ServerProcessPeak:
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


def _reserve_port() -> int:
    with socket.socket() as server_socket:
        server_socket.bind(("127.0.0.1", 0))
        return int(server_socket.getsockname()[1])


def _cgroup_v2_cpu_snapshot() -> dict[str, int] | None:
    """Read cgroup CPU counters when this benchmark runs in a v2 cgroup."""
    stats_path = Path("/sys/fs/cgroup/cpu.stat")
    if not stats_path.is_file():
        return None
    snapshot: dict[str, int] = {}
    try:
        for line in stats_path.read_text(encoding="ascii").splitlines():
            key, value = line.split(maxsplit=1)
            snapshot[key] = int(value)
    except (OSError, ValueError):
        return None
    return snapshot or None


def _cpu_seconds(cpu_times: Any) -> float:
    return float(cpu_times.user) + float(cpu_times.system)


def _process_tree_cpu_seconds(pid: int) -> float | None:
    import psutil

    processes = _server_process_tree(pid)
    if not processes:
        return None
    total = 0.0
    for process in processes:
        try:
            total += _cpu_seconds(process.cpu_times())
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue
    return total


def _server_process_tree(pid: int) -> list[Any]:
    import psutil

    try:
        root = psutil.Process(pid)
        return [root, *root.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return []


def _validate_task_count(task_count: int, max_pending_tasks: int) -> None:
    if task_count < 1:
        raise ValueError("task count must be positive")
    if task_count > max_pending_tasks:
        raise ValueError(
            f"task count {task_count} exceeds API admission capacity "
            f"{max_pending_tasks}"
        )


def _stream_artifact(response: Any, destination: Path | None) -> tuple[int, str]:
    """Hash a streamed PNG response and optionally retain it on disk."""
    media_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
    if media_type != "image/png":
        raise RuntimeError(f"API image endpoint returned unexpected type: {media_type}")
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    byte_count = 0
    stream = destination.open("wb") if destination is not None else None
    try:
        for block in response.iter_bytes(1024 * 1024):
            digest.update(block)
            byte_count += len(block)
            if stream is not None:
                stream.write(block)
    finally:
        if stream is not None:
            stream.close()
    if byte_count == 0:
        raise RuntimeError("API image endpoint returned an empty PNG")
    return byte_count, digest.hexdigest()


def _get_task_status(
    client: Any,
    url: str,
    headers: dict[str, str],
    *,
    deadline: float,
    server_process: subprocess.Popen[str] | None = None,
    max_consecutive_errors: int = 12,
) -> tuple[Any, int]:
    """Retry transient transport errors while polling the idempotent status route."""
    import httpx

    retries = 0
    consecutive_errors = 0
    while time.monotonic() < deadline:
        try:
            response = client.get(url, headers=headers)
        except httpx.TransportError as exc:
            retries += 1
            consecutive_errors += 1
            if server_process is not None and server_process.poll() is not None:
                raise RuntimeError(
                    "Laika API process exited while polling task status "
                    f"(exit code {server_process.returncode})"
                ) from exc
            if consecutive_errors >= max_consecutive_errors:
                raise RuntimeError(
                    "Laika API status route failed with "
                    f"{consecutive_errors} consecutive transport errors"
                ) from exc
            time.sleep(min(0.25 * (2 ** min(consecutive_errors - 1, 3)), 2.0))
            continue
        response.raise_for_status()
        return response, retries
    raise TimeoutError("API task status polling deadline expired")


def _indexed_path(path: Path, task_count: int, index: int) -> Path:
    if task_count == 1:
        return path
    return path.with_name(f"{path.stem}-{index + 1}{path.suffix}")


def _run_task_request(
    *,
    index: int,
    task_count: int,
    source: Path,
    body: str,
    base_url: str,
    timeout_seconds: float,
    poll_interval: float,
    save_output: Path | None,
    save_preview: Path | None,
    started_at: float,
    server_process: subprocess.Popen[str] | None = None,
) -> dict[str, Any]:
    import httpx

    started = time.perf_counter()
    with httpx.Client(timeout=30.0) as client:
        media_type = mimetypes.guess_type(source.name)[0] or "application/octet-stream"
        with source.open("rb") as image_file:
            created = client.post(
                f"{base_url}/api/v1/tasks",
                data={"options": body},
                files={"file": (source.name, image_file, media_type)},
            )
        if created.status_code != 202:
            raise RuntimeError(
                f"API task {index + 1} creation returned {created.status_code}: "
                f"{created.text}"
            )
        task = created.json()
        headers = {"X-Laika-Task-Token": task["access_token"]}
        deadline = time.monotonic() + timeout_seconds
        status_poll_retries = 0
        while time.monotonic() < deadline:
            status_response, retries = _get_task_status(
                client,
                f"{base_url}{task['status_url']}",
                headers,
                deadline=deadline,
                server_process=server_process,
            )
            status_poll_retries += retries
            status = status_response.json()
            if status["status"] in ("completed", "failed", "cancelled"):
                break
            time.sleep(poll_interval)
        else:
            client.post(f"{base_url}{task['cancel_url']}", headers=headers)
            raise TimeoutError(
                f"API task {index + 1} exceeded {timeout_seconds} seconds"
            )
        task_wall_seconds = time.perf_counter() - started
        if status["status"] != "completed":
            raise RuntimeError(
                f"API task {index + 1} {status['status']}: {status['error']}"
            )
        task_completed_elapsed_seconds = time.perf_counter() - started

        destination = (
            _indexed_path(save_output, task_count, index)
            if save_output is not None
            else None
        )
        with client.stream(
            "GET", f"{base_url}{status['download_url']}", headers=headers
        ) as response:
            response.raise_for_status()
            output_bytes, output_sha256 = _stream_artifact(response, destination)

        preview_destination = (
            _indexed_path(save_preview, task_count, index)
            if save_preview is not None
            else None
        )
        preview_bytes = 0
        preview_sha256 = None
        if save_preview is not None:
            preview_url = status.get("preview_url")
            if not preview_url:
                raise RuntimeError("completed API task has no preview URL")
            with client.stream(
                "GET", f"{base_url}{preview_url}", headers=headers
            ) as response:
                response.raise_for_status()
                preview_bytes, preview_sha256 = _stream_artifact(
                    response, preview_destination
                )

    return {
        "task_id": task["id"],
        "status_poll_retries": status_poll_retries,
        "task_wall_seconds": task_wall_seconds,
        "task_completed_elapsed_seconds": task_completed_elapsed_seconds,
        "task_finished_elapsed_seconds": time.perf_counter() - started_at,
        "model_results": status["result"],
        "stage_seconds": status["result"].get("stage_seconds", {}),
        "downloaded_output_bytes": output_bytes,
        "output_sha256": output_sha256,
        "saved_output": str(destination) if destination else None,
        "downloaded_preview_bytes": preview_bytes,
        "preview_sha256": preview_sha256,
        "saved_preview": str(preview_destination) if preview_destination else None,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import httpx
    import psutil

    from laika.api import ApiSettings
    from laika.benchmark import _cgroup_v2_snapshot

    source = Path(args.input)
    options = json.loads(args.options)
    if not isinstance(options, dict):
        raise ValueError("--options must be a JSON object")
    settings = ApiSettings.from_environment()
    _validate_task_count(args.task_count, settings.max_pending_tasks)
    if "strategy" not in options:
        options["strategy"] = settings.default_strategy
    if "mode" not in options:
        options["mode"] = "color+bokeh"
    body = json.dumps(options, separators=(",", ":"))
    host = "127.0.0.1"
    port = _reserve_port()
    base_url = f"http://{host}:{port}"
    child_env = dict(os.environ)
    child_env["LAIKA_API_HOST"] = host
    child_env["LAIKA_API_PORT"] = str(port)
    temporary_storage: Path | None = None
    if not child_env.get("LAIKA_API_STORAGE"):
        temporary_storage = Path(tempfile.mkdtemp(prefix="laikashot-api-bench-"))
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
    peak = ServerProcessPeak(
        server.pid,
        rss_interval_seconds=args.rss_sample_interval_seconds,
        uss_interval_seconds=args.uss_sample_interval_seconds,
    )
    available_before = psutil.virtual_memory().available
    cgroup_before = _cgroup_v2_snapshot()
    started = time.perf_counter()
    report: dict[str, Any] | None = None
    try:
        with peak, httpx.Client(timeout=30.0) as client:
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

            benchmark_process = psutil.Process()
            benchmark_cpu_before = benchmark_process.cpu_times()
            server_cpu_before = _process_tree_cpu_seconds(server.pid)
            cgroup_cpu_before = _cgroup_v2_cpu_snapshot()
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=args.task_count) as executor:
                task_runs = list(
                    executor.map(
                        lambda index: _run_task_request(
                            index=index,
                            task_count=args.task_count,
                            source=source,
                            body=body,
                            base_url=base_url,
                            timeout_seconds=args.timeout_seconds,
                            poll_interval=args.poll_interval,
                            save_output=Path(args.save_output)
                            if args.save_output
                            else None,
                            save_preview=Path(args.save_preview)
                            if args.save_preview
                            else None,
                            started_at=started,
                            server_process=server,
                        ),
                        range(args.task_count),
                    )
                )
            task_wall_seconds = time.perf_counter() - started
            benchmark_cpu_after = benchmark_process.cpu_times()
            server_cpu_after = _process_tree_cpu_seconds(server.pid)
            cgroup_cpu_after = _cgroup_v2_cpu_snapshot()

            host_info = {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "cpu_count_logical": psutil.cpu_count(logical=True),
                "memory_total_bytes": psutil.virtual_memory().total,
                "memory_available_before_bytes": available_before,
            }
            report = {
                "input": str(source),
                "input_sha256": _sha256(source),
                "mode": options["mode"],
                "strategy": options["strategy"],
                "options": options,
                "host": host_info,
                "api_versioned_path": "/api/v1/tasks",
                "runtime": {
                    "python": sys.version,
                    "platform": platform.platform(),
                },
                "task_wall_seconds": task_wall_seconds,
                "task_count": args.task_count,
                "scheduler": {
                    "max_workers": settings.max_workers,
                    "queue_capacity": settings.queue_capacity,
                    "max_pending_tasks": settings.max_pending_tasks,
                },
                "tasks": task_runs,
                "server_process_tree_peak_rss_bytes": peak.peak_rss_bytes,
                "server_process_tree_peak_uss_bytes": peak.peak_uss_bytes,
                "server_cpu_seconds": (
                    max(0.0, server_cpu_after - server_cpu_before)
                    if server_cpu_before is not None and server_cpu_after is not None
                    else None
                ),
                "benchmark_client_process_cpu_seconds": max(
                    0.0,
                    _cpu_seconds(benchmark_cpu_after)
                    - _cpu_seconds(benchmark_cpu_before),
                ),
                "server_process_tree_sampling": {
                    "rss_interval_seconds": peak.rss_interval_seconds,
                    "uss_interval_seconds": peak.uss_interval_seconds,
                    "rss_sample_count": peak.rss_sample_count,
                    "uss_sample_count": peak.uss_sample_count,
                },
                "server_process_count": len(_server_process_tree(server.pid)),
                "cgroup_v2_before": cgroup_before,
                "cgroup_v2_after": _cgroup_v2_snapshot(),
                "cgroup_cpu_before": cgroup_cpu_before,
                "cgroup_cpu_after": cgroup_cpu_after,
                "cgroup_cpu_seconds": (
                    max(
                        0,
                        cgroup_cpu_after.get("usage_usec", 0)
                        - cgroup_cpu_before.get("usage_usec", 0),
                    )
                    / 1_000_000
                    if cgroup_cpu_before is not None and cgroup_cpu_after is not None
                    else None
                ),
            }
            if args.task_count == 1:
                report.update(
                    {
                        "model_results": task_runs[0]["model_results"],
                        "stage_seconds": task_runs[0]["stage_seconds"],
                        "downloaded_output_bytes": task_runs[0][
                            "downloaded_output_bytes"
                        ],
                        "output_sha256": task_runs[0]["output_sha256"],
                        "saved_output": task_runs[0]["saved_output"],
                        "downloaded_preview_bytes": task_runs[0][
                            "downloaded_preview_bytes"
                        ],
                        "preview_sha256": task_runs[0]["preview_sha256"],
                        "saved_preview": task_runs[0]["saved_preview"],
                    }
                )
    finally:
        if server.poll() is None:
            server.terminate()
        try:
            server_logs = server.communicate(timeout=30)[0]
        except subprocess.TimeoutExpired:
            server.kill()
            server_logs = server.communicate()[0]
        if server.returncode not in (0, -15, 143) and report is None:
            raise RuntimeError(f"Laika API process failed:\n{server_logs}")
        if report is None and server_logs:
            print(f"Laika API server logs:\n{server_logs}", file=sys.stderr)
        if temporary_storage is not None:
            temporary_storage.rmdir()

    if report is None:
        raise RuntimeError("API benchmark ended without a completed task report")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--options",
        default='{"mode":"color+bokeh","strategy":"low-memory"}',
        help="JSON ProcessingOptions object, including an optional selected crop",
    )
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--poll-interval", type=float, default=0.2)
    parser.add_argument("--rss-sample-interval-seconds", type=float, default=0.1)
    parser.add_argument("--uss-sample-interval-seconds", type=float, default=1.0)
    parser.add_argument(
        "--task-count",
        type=int,
        default=1,
        help="submit this many tasks to one API process; configure workers to match",
    )
    parser.add_argument("--save-output", type=Path)
    parser.add_argument(
        "--save-preview",
        type=Path,
        help="download and save the API preview PNG as well as the full result",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if (
        args.timeout_seconds <= 0
        or args.poll_interval <= 0
        or args.task_count < 1
        or args.rss_sample_interval_seconds <= 0
        or args.uss_sample_interval_seconds <= 0
    ):
        parser.error("timeouts, sample intervals, and task count must be positive")
    report = run(args)
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
