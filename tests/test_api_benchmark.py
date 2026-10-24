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
