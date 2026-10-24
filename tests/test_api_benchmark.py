from __future__ import annotations

import hashlib
import runpy
from pathlib import Path

import pytest

_benchmark_api = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "benchmark_api_task.py")
)
_validate_task_count = _benchmark_api["_validate_task_count"]
_stream_artifact = _benchmark_api["_stream_artifact"]
_get_task_status = _benchmark_api["_get_task_status"]
_ServerProcessPeak = _benchmark_api["ServerProcessPeak"]
_process_tree_cpu_seconds = _benchmark_api["_process_tree_cpu_seconds"]


class _PNGResponse:
    headers = {"content-type": "image/png"}

    def __init__(self, blocks: tuple[bytes, ...]) -> None:
        self.blocks = blocks

    def iter_bytes(self, _chunk_size: int):
        return iter(self.blocks)


class _StatusResponse:
    def raise_for_status(self) -> None:
        return None


class _RetryingStatusClient:
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    def get(self, _url: str, *, headers: dict[str, str]):
        import httpx

        assert headers == {"X-Laika-Task-Token": "token"}
        self.calls += 1
        if self.calls <= self.failures:
            request = httpx.Request("GET", "http://localhost/status")
            raise httpx.ReadError("connection reset", request=request)
        return _StatusResponse()


def test_api_benchmark_streams_and_hashes_png(tmp_path: Path) -> None:
    blocks = (b"png-header", b"pixel-data")
    destination = tmp_path / "preview.png"

    byte_count, sha256 = _stream_artifact(_PNGResponse(blocks), destination)

    payload = b"".join(blocks)
    assert destination.read_bytes() == payload
    assert byte_count == len(payload)
    assert sha256 == hashlib.sha256(payload).hexdigest()


def test_api_benchmark_rejects_non_png_image_response() -> None:
    response = _PNGResponse((b"image",))
    response.headers = {"content-type": "image/jpeg"}

    with pytest.raises(RuntimeError, match="unexpected type: image/jpeg"):
        _stream_artifact(response, None)


def test_api_benchmark_rejects_empty_png_response() -> None:
    with pytest.raises(RuntimeError, match="empty PNG"):
        _stream_artifact(_PNGResponse(()), None)


def test_api_benchmark_task_count_must_be_positive() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        _validate_task_count(0, 2)


def test_api_benchmark_task_count_cannot_exceed_admission_capacity() -> None:
    with pytest.raises(ValueError, match="exceeds API admission capacity"):
        _validate_task_count(3, 2)


def test_api_benchmark_accepts_tasks_within_admission_capacity() -> None:
    _validate_task_count(2, 2)


def test_api_status_poll_retries_transient_connection_reset(monkeypatch) -> None:
    monkeypatch.setattr(_benchmark_api["time"], "sleep", lambda _seconds: None)
    client = _RetryingStatusClient(failures=1)

    response, retries = _get_task_status(
        client,
        "http://localhost/status",
        {"X-Laika-Task-Token": "token"},
        deadline=_benchmark_api["time"].monotonic() + 5,
    )

    assert isinstance(response, _StatusResponse)
    assert retries == 1
    assert client.calls == 2


def test_api_status_poll_bounds_persistent_transport_errors(monkeypatch) -> None:
    monkeypatch.setattr(_benchmark_api["time"], "sleep", lambda _seconds: None)
    client = _RetryingStatusClient(failures=10)

    with pytest.raises(RuntimeError, match="2 consecutive transport errors"):
        _get_task_status(
            client,
            "http://localhost/status",
            {"X-Laika-Task-Token": "token"},
            deadline=_benchmark_api["time"].monotonic() + 5,
            max_consecutive_errors=2,
        )


def test_process_peak_scans_uss_less_often_than_rss(monkeypatch) -> None:
    from types import SimpleNamespace

    class _StopAfterSamples:
        def __init__(self) -> None:
            self.checks = 0

        def is_set(self) -> bool:
            self.checks += 1
            return self.checks > 4

        def wait(self, _seconds: float) -> None:
            return None

    class _Process:
        def __init__(self) -> None:
            self.rss_calls = 0
            self.uss_calls = 0

        def children(self, *, recursive: bool) -> list[object]:
            assert recursive
            return []

        def memory_info(self):
            self.rss_calls += 1
            return SimpleNamespace(rss=100)

        def memory_full_info(self):
            self.uss_calls += 1
            return SimpleNamespace(uss=80)

        def cpu_times(self):
            return SimpleNamespace(user=1.0, system=2.0)

    class _FakePsutil:
        class NoSuchProcess(Exception):
            pass

        class AccessDenied(Exception):
            pass

        def __init__(self, process: _Process) -> None:
            self.process = process

        def Process(self, _pid: int) -> _Process:
            return self.process

    process = _Process()
    peak = _ServerProcessPeak(1, rss_interval_seconds=0.1, uss_interval_seconds=1.0)
    peak._psutil = _FakePsutil(process)
    peak._stop = _StopAfterSamples()
    ticks = iter((0.0, 0.25, 1.1, 2.1))
    monkeypatch.setattr(_benchmark_api["time"], "monotonic", lambda: next(ticks))

    peak._sample()

    assert process.rss_calls == 4
    assert process.uss_calls == 3
    assert peak.rss_sample_count == 4
    assert peak.uss_sample_count == 3
    assert peak.peak_rss_bytes == 100
    assert peak.peak_uss_bytes == 80


def test_process_tree_cpu_sums_root_and_child_processes(monkeypatch) -> None:
    from types import SimpleNamespace

    class _Process:
        def __init__(self, user: float, system: float) -> None:
            self.user = user
            self.system = system

        def cpu_times(self):
            return SimpleNamespace(user=self.user, system=self.system)

    root = _Process(1.25, 0.25)
    child = _Process(2.0, 0.5)
    monkeypatch.setitem(
        _process_tree_cpu_seconds.__globals__,
        "_server_process_tree",
        lambda pid: [root, child],
    )

    assert _process_tree_cpu_seconds(123) == 4.0
