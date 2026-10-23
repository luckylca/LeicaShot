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
    """Sample RSS and CPU for the API server and all child processes."""

    def __init__(self, pid: int, interval_seconds: float = 0.01) -> None:
        import psutil

        self.pid = pid
        self.interval_seconds = interval_seconds
        self.peak_rss_bytes = 0
        self.peak_uss_bytes: int | None = None
        self.cpu_seconds_before = 0.0
        self.cpu_seconds_after = 0.0
        self._psutil = psutil
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                root = self._psutil.Process(self.pid)
                processes = [root, *root.children(recursive=True)]
            except (self._psutil.NoSuchProcess, self._psutil.AccessDenied):
                break
            rss = 0
            uss = 0
            uss_available = True
            for process in processes:
                try:
                    rss += process.memory_info().rss
                    uss += process.memory_full_info().uss
                except (
                    self._psutil.NoSuchProcess,
                    self._psutil.AccessDenied,
                    OSError,
                    RuntimeError,
                ):
                    uss_available = False
            self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
            if uss_available:
                self.peak_uss_bytes = max(self.peak_uss_bytes or 0, uss)
            try:
                times = root.cpu_times()
                self.cpu_seconds_after = times.user + times.system
            except (self._psutil.NoSuchProcess, self._psutil.AccessDenied):
                pass
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> ServerProcessPeak:
        self._thread.start()
        time.sleep(0.02)
        try:
            process = self._psutil.Process(self.pid)
            times = process.cpu_times()
            self.cpu_seconds_before = times.user + times.system
        except (self._psutil.NoSuchProcess, self._psutil.AccessDenied):
            pass
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
        while time.monotonic() < deadline:
            status_response = client.get(
                f"{base_url}{task['status_url']}", headers=headers
            )
            status_response.raise_for_status()
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
    peak = ServerProcessPeak(server.pid)
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
                        ),
                        range(args.task_count),
                    )
                )
            task_wall_seconds = time.perf_counter() - started

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
                "server_cpu_seconds": max(
                    0.0, peak.cpu_seconds_after - peak.cpu_seconds_before
                ),
                "server_process_count": len(_server_process_tree(server.pid)),
                "cgroup_v2_before": cgroup_before,
                "cgroup_v2_after": _cgroup_v2_snapshot(),
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
    if args.timeout_seconds <= 0 or args.poll_interval <= 0 or args.task_count < 1:
        parser.error("timeout, poll interval, and task count must be positive")
    report = run(args)
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
