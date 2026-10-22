from __future__ import annotations

import io
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient
from PIL import Image

from laika.api import (
    ApiSettings,
    QueueFullError,
    StorageLimitError,
    TaskManager,
    TaskOptions,
    create_app,
)
from laika.composition import apply_crop_rgb
from laika.processing import ProcessingCancelled


def _png_bytes(size: tuple[int, int] = (32, 20)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (40, 90, 130)).save(buffer, format="PNG")
    return buffer.getvalue()


class FakeEngine:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.closed = False
        self.options: Any | None = None
        self.suggest_target_long_edge: int | None = None

    def process_file(
        self,
        source: Path,
        destination: Path,
        *,
        options: Any,
        progress_callback: Any,
        cancel_event: threading.Event,
    ) -> Any:
        self.options = options
        if cancel_event.is_set():
            raise ProcessingCancelled("processing was cancelled")
        progress_callback(SimpleNamespace(stage="render", fraction=0.5))
        with Image.open(source) as image:
            rgb = image.convert("RGB")
            try:
                rendered = (
                    apply_crop_rgb(rgb, options.crop)
                    if options.crop is not None
                    else rgb
                )
                Image.fromarray(rendered).save(destination, format="PNG")
            finally:
                rgb.close()
        with Image.open(destination) as image:
            width, height = image.size
        progress_callback(SimpleNamespace(stage="complete", fraction=1.0))
        return SimpleNamespace(
            as_dict=lambda: {
                "mode": options.mode,
                "output_path": str(destination),
                "width": width,
                "height": height,
                "output_bytes": destination.stat().st_size,
                "models": {},
            }
        )

    def suggest_composition(
        self,
        source: Path,
        *,
        target_long_edge: int | None,
    ) -> Any:
        assert source.is_file()
        self.suggest_target_long_edge = target_long_edge
        return SimpleNamespace(
            as_dict=lambda: {
                "width": 32,
                "height": 20,
                "analysis_width": 32,
                "analysis_height": 20,
                "candidate_count": 1,
                "suggestions": [
                    {
                        "crop": {
                            "left": 0.1,
                            "top": 0.1,
                            "right": 0.9,
                            "bottom": 0.9,
                        },
                        "score": 0.8,
                        "improvement_over_original": 0.1,
                    }
                ],
            }
        )

    def close(self) -> None:
        self.closed = True


class BlockingEngine(FakeEngine):
    def __init__(self, started: threading.Event, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.started = started

    def process_file(
        self,
        _source: Path,
        _destination: Path,
        *,
        options: Any,
        progress_callback: Any,
        cancel_event: threading.Event,
    ) -> Any:
        self.started.set()
        progress_callback(SimpleNamespace(stage="inference", fraction=0.2))
        while not cancel_event.wait(0.01):
            pass
        raise ProcessingCancelled("processing was cancelled")


def _settings(storage: Path, **kwargs: Any) -> ApiSettings:
    values: dict[str, Any] = {
        "storage_dir": storage,
        "max_upload_bytes": 1024 * 1024,
        "max_pixels": 100_000,
        "max_storage_bytes": 64 * 1024 * 1024,
        "max_workers": 1,
        "queue_capacity": 1,
        "result_ttl_seconds": 3600,
    }
    values.update(kwargs)
    return ApiSettings(**values)


def _submit(client: TestClient, image: bytes, options: dict[str, Any] | None = None):
    return client.post(
        "/api/v1/tasks",
        files={"file": ("../../untrusted-name.png", image, "application/octet-stream")},
        data={"options": json.dumps(options or {"mode": "color"})},
    )


def _wait_for_status(
    client: TestClient, task_id: str, token: str, expected: str
) -> dict[str, Any]:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get(
            f"/api/v1/tasks/{task_id}", headers={"X-Laika-Task-Token": token}
        )
        assert response.status_code == 200
        data = response.json()
        if data["status"] == expected:
            return data
        time.sleep(0.01)
    raise AssertionError(f"task {task_id} did not reach {expected}")


def test_api_upload_process_progress_preview_download_and_task_token(
    tmp_path: Path,
) -> None:
    image_bytes = _png_bytes()
    app = create_app(_settings(tmp_path / "tasks"), engine_factory=FakeEngine)

    with TestClient(app) as client:
        assert client.get("/api/v1/health").json()["status"] == "ok"
        created = _submit(
            client,
            image_bytes,
            {
                "mode": "color",
                "focus_x": 0.3,
                "focus_y": 0.7,
                "aperture": 4.0,
                "crop": [0.1, 0.1, 0.9, 0.9],
            },
        )
        assert created.status_code == 202, created.text
        task = created.json()
        task_id, token = task["id"], task["access_token"]
        assert task["status_url"] == f"/api/v1/tasks/{task_id}"

        completed = _wait_for_status(client, task_id, token, "completed")
        assert completed["progress"] == 1.0
        assert completed["result"]["width"] == 26
        assert completed["result"]["height"] == 16
        assert "output_path" not in completed["result"]

        denied = client.get(f"/api/v1/tasks/{task_id}")
        assert denied.status_code == 422
        wrong_token = client.get(
            f"/api/v1/tasks/{task_id}",
            headers={"X-Laika-Task-Token": "wrong-token"},
        )
        assert wrong_token.status_code == 404

        source = client.get(
            f"/api/v1/tasks/{task_id}/source",
            headers={"X-Laika-Task-Token": token},
        )
        preview = client.get(
            f"/api/v1/tasks/{task_id}/preview",
            headers={"X-Laika-Task-Token": token},
        )
        download = client.get(
            f"/api/v1/tasks/{task_id}/download",
            headers={"X-Laika-Task-Token": token},
        )
        assert source.content == image_bytes
        assert preview.status_code == 200
        assert download.status_code == 200
        assert download.headers["content-disposition"].endswith(
            'filename="leicashot-result.png"'
        )
        with Image.open(io.BytesIO(preview.content)) as image:
            assert image.size == (26, 16)
        with Image.open(io.BytesIO(download.content)) as image:
            assert image.size == (26, 16)

    assert list((tmp_path / "tasks").iterdir()) == []


def test_api_rejects_invalid_options_bytes_format_and_pixel_count(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path / "tasks", max_upload_bytes=4096, max_pixels=10_000)
    app = create_app(settings, engine_factory=FakeEngine)

    with TestClient(app) as client:
        invalid_options = _submit(client, _png_bytes(), {"mode": "invalid"})
        assert invalid_options.status_code == 422

        too_large = _submit(client, b"x" * 5000)
        assert too_large.status_code == 413

        malformed = _submit(client, b"not an image")
        assert malformed.status_code == 422

        oversized_pixels = _submit(client, _png_bytes((101, 101)))
        assert oversized_pixels.status_code == 413

    assert list((tmp_path / "tasks").iterdir()) == []


def test_api_reserves_a_bounded_disk_budget_before_accepting_a_task(
    tmp_path: Path,
) -> None:
    app = create_app(
        _settings(
            tmp_path / "tasks",
            max_upload_bytes=512,
            max_pixels=10,
            max_storage_bytes=1024,
        ),
        engine_factory=FakeEngine,
    )

    with TestClient(app) as client:
        response = _submit(client, _png_bytes())
        assert response.status_code == 507
        assert response.json()["detail"] == "task storage is full; retry after cleanup"

    assert list((tmp_path / "tasks").iterdir()) == []


def test_api_disk_budget_accounts_for_unfilled_active_task_reservations(
    tmp_path: Path,
) -> None:
    reservation = 1024 + 100 * 4 + 1600 * 1600 * 4
    settings = _settings(
        tmp_path / "tasks",
        max_upload_bytes=1024,
        max_pixels=100,
        max_storage_bytes=2 * reservation - 1,
    )
    manager = TaskManager(settings, engine_factory=FakeEngine)
    options = TaskOptions(mode="color").to_processing_options(settings)

    try:
        first, _token = manager.reserve(options)
        assert first.status == "uploading"
        try:
            manager.reserve(options)
        except StorageLimitError:
            pass
        else:
            raise AssertionError("active task storage reservations were overcommitted")
    finally:
        manager.close()


def test_api_composition_upload_returns_ranked_crop_and_cleans_temp_data(
    tmp_path: Path,
) -> None:
    storage = tmp_path / "tasks"
    engines: list[FakeEngine] = []

    def make_engine(**kwargs: Any) -> FakeEngine:
        engine = FakeEngine(**kwargs)
        engines.append(engine)
        return engine

    app = create_app(_settings(storage), engine_factory=make_engine)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/compositions",
            files={"file": ("portrait.png", _png_bytes(), "image/png")},
            data={"strategy": "low-memory", "target_long_edge": "1200"},
        )
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["candidate_count"] == 1
        assert result["suggestions"][0]["crop"] == {
            "left": 0.1,
            "top": 0.1,
            "right": 0.9,
            "bottom": 0.9,
        }
        assert engines[0].suggest_target_long_edge == 1200
        assert engines[0].closed
        assert client.get("/api/v1/health").json()["active_or_queued_tasks"] == 0

    assert list(storage.iterdir()) == []


def test_api_composition_shares_the_bounded_processing_queue(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path / "tasks",
        max_workers=1,
        queue_capacity=0,
    )
    manager = TaskManager(settings, engine_factory=FakeEngine)
    reservation_id, _directory = manager.reserve_composition()
    options = TaskOptions(mode="color").to_processing_options(settings)

    try:
        try:
            manager.reserve(options)
        except QueueFullError:
            pass
        else:
            raise AssertionError("composition and render exceeded shared queue limit")
    finally:
        manager.release_composition(reservation_id)
        manager.close()


def test_api_applies_total_body_limit_to_composition_uploads(tmp_path: Path) -> None:
    app = create_app(
        _settings(tmp_path / "tasks", max_upload_bytes=4096),
        engine_factory=FakeEngine,
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/compositions",
            content=b"x" * 200_000,
            headers={"content-type": "application/octet-stream"},
        )

    assert response.status_code == 413


def test_api_configures_exact_cross_origin_web_clients(tmp_path: Path) -> None:
    app = create_app(
        _settings(
            tmp_path / "tasks",
            cors_origins=("https://editor.example",),
        ),
        engine_factory=FakeEngine,
    )

    with TestClient(app) as client:
        preflight = client.options(
            "/api/v1/tasks/task-1/cancel",
            headers={
                "Origin": "https://editor.example",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "x-laika-task-token",
            },
        )
        assert preflight.status_code == 200
        assert (
            preflight.headers["access-control-allow-origin"] == "https://editor.example"
        )
        assert (
            "x-laika-task-token"
            in preflight.headers["access-control-allow-headers"].lower()
        )

        health = client.get(
            "/api/v1/health",
            headers={"Origin": "https://editor.example"},
        )
        assert health.status_code == 200
        assert health.headers["access-control-allow-origin"] == "https://editor.example"

        denied = client.get(
            "/api/v1/health",
            headers={"Origin": "https://other.example"},
        )
        assert "access-control-allow-origin" not in denied.headers


def test_api_rejects_wildcard_cors_origin(tmp_path: Path) -> None:
    try:
        _settings(tmp_path / "tasks", cors_origins=("*",))
    except ValueError as exc:
        assert "exact HTTP(S) origins" in str(exc)
    else:
        raise AssertionError("wildcard CORS origins must be rejected")


def test_api_rejects_non_origin_cors_urls(tmp_path: Path) -> None:
    for invalid in ("https://editor.example/", "https://editor.example/path"):
        try:
            _settings(tmp_path / "tasks", cors_origins=(invalid,))
        except ValueError as exc:
            assert "exact HTTP(S) origins" in str(exc)
        else:
            raise AssertionError(f"CORS URL with a path must be rejected: {invalid}")


def test_api_bounds_active_tasks_and_cancels_running_work(tmp_path: Path) -> None:
    started = threading.Event()
    engines: list[BlockingEngine] = []

    def factory(**kwargs: Any) -> BlockingEngine:
        engine = BlockingEngine(started, **kwargs)
        engines.append(engine)
        return engine

    app = create_app(
        _settings(tmp_path / "tasks", queue_capacity=0), engine_factory=factory
    )

    with TestClient(app) as client:
        first = _submit(client, _png_bytes())
        assert first.status_code == 202
        task = first.json()
        assert started.wait(timeout=2)

        second = _submit(client, _png_bytes())
        assert second.status_code == 429

        cancelling = client.post(
            f"/api/v1/tasks/{task['id']}/cancel",
            headers={"X-Laika-Task-Token": task["access_token"]},
        )
        assert cancelling.status_code == 200
        completed = _wait_for_status(
            client, task["id"], task["access_token"], "cancelled"
        )
        assert completed["error"] is None

    assert engines and engines[0].closed
