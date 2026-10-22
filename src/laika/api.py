"""Versioned FastAPI task service over the shared Python processing engine."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import secrets
import shutil
import tempfile
import threading
import time
import uuid
import warnings
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from laika.bokeh import APERTURES
from laika.composition import CompositionCrop
from laika.engine import ImageInputError, ImageLimitError
from laika.processing import (
    LaikaProcessingEngine,
    ProcessingCancelled,
    ProcessingError,
    ProcessingOptions,
)

logger = logging.getLogger(__name__)
SUPPORTED_UPLOAD_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})
UPLOAD_CHUNK_BYTES = 1024 * 1024
MAX_OPTIONS_JSON_BYTES = 16 * 1024
MULTIPART_OVERHEAD_BYTES = 128 * 1024
TASK_MARKER = ".laika-owned-task"
TaskState = Literal[
    "uploading", "queued", "running", "cancelling", "completed", "cancelled", "failed"
]
TERMINAL_STATES = frozenset({"completed", "cancelled", "failed"})


class QueueFullError(RuntimeError):
    """The bounded service queue has reached its configured capacity."""


class StorageLimitError(RuntimeError):
    """The managed task workspace cannot reserve another worst-case result."""


class TaskNotFoundError(LookupError):
    """A task does not exist or its per-task access token is invalid."""


class ApiUploadError(ValueError):
    """A streamed upload is malformed, unsupported, or over its resource limit."""


class _RequestTooLarge(Exception):
    pass


class UploadBodyLimitMiddleware:
    """Bound multipart parsing, including requests without Content-Length."""

    def __init__(self, app: Any, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or not (
            scope.get("method") == "POST" and scope.get("path") == "/api/v1/tasks"
        ):
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", ()))
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                if int(content_length) > self.max_bytes:
                    response = JSONResponse(
                        {"detail": "request exceeds the configured upload limit"},
                        status_code=413,
                    )
                    response.headers["connection"] = "close"
                    await response(scope, receive, send)
                    return
            except ValueError:
                pass

        received = 0

        async def limited_receive() -> Any:
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _RequestTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _RequestTooLarge:
            response = JSONResponse(
                {"detail": "request exceeds the configured upload limit"},
                status_code=413,
            )
            response.headers["connection"] = "close"
            await response(scope, receive, send)


@dataclass(frozen=True)
class ApiSettings:
    """Service resource limits and local model paths."""

    storage_dir: Path = field(
        default_factory=lambda: Path(tempfile.gettempdir()) / "leicashot-api"
    )
    max_upload_bytes: int = 50 * 1024 * 1024
    max_storage_bytes: int = 10 * 1024 * 1024 * 1024
    max_pixels: int = 40_000_000
    max_network_pixels: int = 4_000_000
    max_workers: int = 1
    queue_capacity: int = 1
    result_ttl_seconds: int = 6 * 60 * 60
    threads: int | None = 1
    default_strategy: Literal["low-memory", "balanced"] = "low-memory"
    m9_model_path: Path | None = None
    depth_model_path: Path | None = None
    bokeh_model_path: Path | None = None
    sam_model_dir: Path | None = None
    vitmatte_model_path: Path | None = None
    composition_model_path: Path | None = None

    def __post_init__(self) -> None:
        if (
            min(
                self.max_upload_bytes,
                self.max_storage_bytes,
                self.max_pixels,
                self.max_network_pixels,
                self.max_workers,
                self.result_ttl_seconds,
            )
            < 1
        ):
            raise ValueError("API limits, workers, and task TTL must be positive")
        if self.queue_capacity < 0:
            raise ValueError("queue_capacity cannot be negative")
        if self.threads is not None and self.threads < 1:
            raise ValueError("threads must be positive when configured")
        if self.default_strategy not in ("low-memory", "balanced"):
            raise ValueError("default_strategy must be low-memory or balanced")

    @property
    def max_pending_tasks(self) -> int:
        return self.max_workers + self.queue_capacity

    @classmethod
    def from_environment(cls) -> ApiSettings:
        def integer(name: str, default: int) -> int:
            raw = os.getenv(name)
            if raw is None:
                return default
            try:
                return int(raw)
            except ValueError as exc:
                raise ValueError(f"{name} must be an integer") from exc

        def path(name: str) -> Path | None:
            raw = os.getenv(name)
            return Path(raw).expanduser() if raw else None

        strategy = os.getenv("LAIKA_API_STRATEGY", "low-memory")
        if strategy not in ("low-memory", "balanced"):
            raise ValueError("LAIKA_API_STRATEGY must be low-memory or balanced")
        thread_count = integer("LAIKA_API_THREADS", 1)
        return cls(
            storage_dir=Path(
                os.getenv(
                    "LAIKA_API_STORAGE",
                    str(Path(tempfile.gettempdir()) / "leicashot-api"),
                )
            ).expanduser(),
            max_upload_bytes=integer("LAIKA_API_MAX_UPLOAD_BYTES", 50 * 1024 * 1024),
            max_storage_bytes=integer(
                "LAIKA_API_MAX_STORAGE_BYTES", 10 * 1024 * 1024 * 1024
            ),
            max_pixels=integer("LAIKA_API_MAX_PIXELS", 40_000_000),
            max_network_pixels=integer("LAIKA_API_MAX_NETWORK_PIXELS", 4_000_000),
            max_workers=integer("LAIKA_API_MAX_WORKERS", 1),
            queue_capacity=integer("LAIKA_API_QUEUE_CAPACITY", 1),
            result_ttl_seconds=integer("LAIKA_API_RESULT_TTL_SECONDS", 6 * 60 * 60),
            threads=thread_count,
            default_strategy=strategy,  # type: ignore[arg-type]
            m9_model_path=path("LAIKA_M9_MODEL"),
            depth_model_path=path("LAIKA_DEPTH_MODEL"),
            bokeh_model_path=path("LAIKA_BOKEH_MODEL"),
            sam_model_dir=path("LAIKA_SAM_MODEL_DIR"),
            vitmatte_model_path=path("LAIKA_VITMATTE_MODEL"),
            composition_model_path=path("LAIKA_GAIC_MODEL"),
        )

    def engine_kwargs(self, strategy: str, temp_root: Path) -> dict[str, Any]:
        return {
            "m9_model_path": self.m9_model_path,
            "depth_model_path": self.depth_model_path,
            "bokeh_model_path": self.bokeh_model_path,
            "sam_model_dir": self.sam_model_dir,
            "vitmatte_model_path": self.vitmatte_model_path,
            "composition_model_path": self.composition_model_path,
            "strategy": strategy,
            "threads": self.threads,
            "max_upload_bytes": self.max_upload_bytes,
            "max_pixels": self.max_pixels,
            "max_network_pixels": self.max_network_pixels,
            "temp_root": temp_root,
        }


class TaskOptions(BaseModel):
    """Editable AI parameters accepted with each uploaded photo."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["color", "bokeh", "color+bokeh"] = "color+bokeh"
    strategy: Literal["low-memory", "balanced"] = "low-memory"
    focus_x: float = Field(default=0.5, ge=0.0, le=1.0)
    focus_y: float = Field(default=0.5, ge=0.0, le=1.0)
    aperture: float = 2.8
    output_size: tuple[int, int] | None = None
    crop: tuple[float, float, float, float] | None = None
    subject_points: list[tuple[float, float]] = Field(default_factory=list)
    matte_tile_size: Literal[512, 768, 1024] = 1024
    m9_row_chunk_height: int = Field(default=512, ge=1)
    bokeh_row_chunk_height: int = Field(default=128, ge=1)

    @field_validator("aperture")
    @classmethod
    def validate_aperture(cls, value: float) -> float:
        if value not in APERTURES:
            raise ValueError(f"aperture must be one of {APERTURES}")
        return value

    def to_processing_options(self, settings: ApiSettings) -> ProcessingOptions:
        if self.output_size is not None and min(self.output_size) < 2:
            raise ValueError("output_size dimensions must be at least 2 pixels")
        crop = CompositionCrop(*self.crop) if self.crop is not None else None
        return ProcessingOptions(
            mode=self.mode,
            strategy=self.strategy,
            focus_x=self.focus_x,
            focus_y=self.focus_y,
            aperture=self.aperture,
            output_size=self.output_size,
            max_upload_bytes=settings.max_upload_bytes,
            max_pixels=settings.max_pixels,
            max_network_pixels=settings.max_network_pixels,
            m9_row_chunk_height=self.m9_row_chunk_height,
            bokeh_row_chunk_height=self.bokeh_row_chunk_height,
            subject_points=tuple(self.subject_points),
            matte_tile_size=self.matte_tile_size,
            crop=crop,
        )


@dataclass
class TaskRecord:
    task_id: str
    token_digest: str
    directory: Path
    input_path: Path
    output_path: Path
    preview_path: Path
    options: ProcessingOptions
    created_at: float
    status: TaskState = "uploading"
    stage: str = "uploading"
    progress: float = 0.0
    error: str | None = None
    input_format: str | None = None
    result: dict[str, Any] | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    future: Future[Any] | None = None


class TaskManager:
    """Bounded in-process scheduler with isolated, token-protected task storage."""

    def __init__(
        self,
        settings: ApiSettings,
        *,
        engine_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.settings = settings
        self.settings.storage_dir.mkdir(parents=True, exist_ok=True)
        self._engine_factory = engine_factory or LaikaProcessingEngine
        self._executor = ThreadPoolExecutor(
            max_workers=settings.max_workers,
            thread_name_prefix="laika-api-worker",
        )
        self._lock = threading.RLock()
        self._tasks: dict[str, TaskRecord] = {}
        self._closed = False
        self._thread_local = threading.local()
        self._resident_engines: list[Any] = []
        self._cleanup_stop = threading.Event()
        self._purge_expired_orphans()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            name="laika-api-cleanup",
            daemon=True,
        )
        self._cleanup_thread.start()

    def reserve(self, options: ProcessingOptions) -> tuple[TaskRecord, str]:
        with self._lock:
            self._expire_old_tasks()
            pending = sum(
                task.status not in TERMINAL_STATES for task in self._tasks.values()
            )
            if self._closed:
                raise RuntimeError("task service is shutting down")
            if pending >= self.settings.max_pending_tasks:
                raise QueueFullError("processing queue is full")
            reservation_bytes = (
                self.settings.max_upload_bytes
                + self.settings.max_pixels * 4
                + 1600 * 1600 * 4
            )
            active_reservations = 0
            for active_task in self._tasks.values():
                if active_task.status in TERMINAL_STATES:
                    continue
                active_bytes = self._directory_size_bytes(active_task.directory)
                active_reservations += max(0, reservation_bytes - active_bytes)
            if (
                self._storage_size_bytes() + active_reservations + reservation_bytes
                > self.settings.max_storage_bytes
            ):
                raise StorageLimitError("task storage is full; retry after cleanup")
            task_id = str(uuid.uuid4())
            token = secrets.token_urlsafe(32)
            directory = self.settings.storage_dir / task_id
            directory.mkdir(mode=0o700)
            (directory / TASK_MARKER).write_text("laikashot temporary task data\n")
            record = TaskRecord(
                task_id=task_id,
                token_digest=self._digest_token(token),
                directory=directory,
                input_path=directory / "source.upload",
                output_path=directory / "result.png",
                preview_path=directory / "preview.png",
                options=options,
                created_at=time.time(),
            )
            self._tasks[task_id] = record
            return record, token

    def abort_upload(self, task_id: str) -> None:
        with self._lock:
            record = self._tasks.pop(task_id, None)
            if record is not None:
                shutil.rmtree(record.directory, ignore_errors=True)

    def enqueue(self, task_id: str, input_format: str) -> TaskRecord:
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None or record.status != "uploading":
                raise RuntimeError("task upload reservation is no longer active")
            record.input_format = input_format
            record.status = "queued"
            record.stage = "queued"
            try:
                record.future = self._executor.submit(self._run_task, task_id)
            except RuntimeError:
                self._tasks.pop(task_id, None)
                shutil.rmtree(record.directory, ignore_errors=True)
                raise
            return record

    def cancel(self, task_id: str, token: str) -> TaskRecord:
        with self._lock:
            record = self._authorized_task(task_id, token)
            if record.status in TERMINAL_STATES:
                return record
            record.cancel_event.set()
            if record.status == "queued" and record.future is not None:
                if record.future.cancel():
                    record.status = "cancelled"
                    record.stage = "cancelled"
                    shutil.rmtree(record.directory, ignore_errors=True)
                    return record
            record.status = "cancelling"
            record.stage = "cancelling"
            return record

    def get(self, task_id: str, token: str) -> TaskRecord:
        with self._lock:
            self._expire_old_tasks()
            return self._authorized_task(task_id, token)

    def public_info(self, task_id: str, token: str) -> dict[str, Any]:
        with self._lock:
            record = self.get(task_id, token)
            data: dict[str, Any] = {
                "id": record.task_id,
                "status": record.status,
                "stage": record.stage,
                "progress": record.progress,
                "error": record.error,
                "created_at": datetime.fromtimestamp(
                    record.created_at, tz=UTC
                ).isoformat(),
                "cancellable": record.status in ("queued", "running", "cancelling"),
            }
            if record.status == "completed":
                data["result"] = record.result
                data["source_url"] = f"/api/v1/tasks/{task_id}/source"
                data["preview_url"] = f"/api/v1/tasks/{task_id}/preview"
                data["download_url"] = f"/api/v1/tasks/{task_id}/download"
            return data

    def file_for(self, task_id: str, token: str, kind: str) -> tuple[Path, str]:
        with self._lock:
            record = self.get(task_id, token)
            if kind == "source":
                if not record.input_path.is_file() or record.input_format is None:
                    raise HTTPException(
                        status_code=404, detail="source image unavailable"
                    )
                media_type = {
                    "JPEG": "image/jpeg",
                    "PNG": "image/png",
                    "WEBP": "image/webp",
                }[record.input_format]
                return record.input_path, media_type
            if record.status != "completed":
                raise HTTPException(
                    status_code=409, detail="task has no completed result"
                )
            if kind == "preview":
                return record.preview_path, "image/png"
            return record.output_path, "image/png"

    def _authorized_task(self, task_id: str, token: str) -> TaskRecord:
        try:
            canonical_id = str(uuid.UUID(task_id))
        except (ValueError, AttributeError):
            raise TaskNotFoundError(task_id) from None
        record = self._tasks.get(canonical_id)
        if record is None or not hmac.compare_digest(
            record.token_digest, self._digest_token(token)
        ):
            raise TaskNotFoundError(task_id)
        return record

    @staticmethod
    def _digest_token(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def _run_task(self, task_id: str) -> None:
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                return
            if record.cancel_event.is_set():
                record.status = "cancelled"
                record.stage = "cancelled"
                shutil.rmtree(record.directory, ignore_errors=True)
                return
            record.status = "running"
            record.stage = "starting"

        engine: Any | None = None
        owned = False
        try:
            engine, owned = self._acquire_engine(record)

            def report_progress(event: Any) -> None:
                with self._lock:
                    current = self._tasks.get(task_id)
                    if current is None or current.status in TERMINAL_STATES:
                        return
                    current.stage = event.stage
                    current.progress = min(1.0, max(0.0, float(event.fraction)))

            result = engine.process_file(
                record.input_path,
                record.output_path,
                options=record.options,
                progress_callback=report_progress,
                cancel_event=record.cancel_event,
            )
            if record.cancel_event.is_set():
                raise ProcessingCancelled("processing was cancelled")
            self._make_preview(record.output_path, record.preview_path)
            with self._lock:
                current = self._tasks.get(task_id)
                if current is None:
                    return
                result_data = result.as_dict()
                result_data.pop("output_path", None)
                current.result = result_data
                current.status = "completed"
                current.stage = "complete"
                current.progress = 1.0
        except ProcessingCancelled:
            with self._lock:
                current = self._tasks.get(task_id)
                if current is not None:
                    current.status = "cancelled"
                    current.stage = "cancelled"
                    current.error = None
            record.output_path.unlink(missing_ok=True)
            record.preview_path.unlink(missing_ok=True)
            shutil.rmtree(record.directory, ignore_errors=True)
        except (ImageInputError, ImageLimitError, ProcessingError, ValueError) as exc:
            with self._lock:
                current = self._tasks.get(task_id)
                if current is not None:
                    current.status = "failed"
                    current.stage = "failed"
                    current.error = str(exc)
            record.output_path.unlink(missing_ok=True)
            record.preview_path.unlink(missing_ok=True)
            shutil.rmtree(record.directory, ignore_errors=True)
        except Exception:
            logger.exception("Laika task %s failed", task_id)
            with self._lock:
                current = self._tasks.get(task_id)
                if current is not None:
                    current.status = "failed"
                    current.stage = "failed"
                    current.error = "processing failed; check server logs"
            record.output_path.unlink(missing_ok=True)
            record.preview_path.unlink(missing_ok=True)
            shutil.rmtree(record.directory, ignore_errors=True)
        finally:
            if owned and engine is not None:
                try:
                    engine.close()
                except Exception:
                    logger.exception(
                        "Could not close Laika engine for task %s", task_id
                    )

    def _acquire_engine(self, record: TaskRecord) -> tuple[Any, bool]:
        strategy = record.options.strategy
        if strategy == "low-memory":
            return self._engine_factory(
                **self.settings.engine_kwargs(strategy, record.directory)
            ), True
        engines = getattr(self._thread_local, "engines", None)
        if engines is None:
            engines = {}
            self._thread_local.engines = engines
        engine = engines.get(strategy)
        if engine is None:
            engine = self._engine_factory(
                **self.settings.engine_kwargs(strategy, self.settings.storage_dir)
            )
            engines[strategy] = engine
            with self._lock:
                self._resident_engines.append(engine)
        return engine, False

    @staticmethod
    def _make_preview(source: Path, destination: Path) -> None:
        with Image.open(source) as opened:
            opened.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
            options = (
                {"icc_profile": opened.info["icc_profile"]}
                if opened.info.get("icc_profile")
                else {}
            )
            temporary = destination.with_name(f".{destination.name}.tmp")
            if opened.mode == "RGB":
                opened.save(temporary, format="PNG", **options)
            else:
                preview = opened.convert("RGB")
                try:
                    preview.save(temporary, format="PNG", **options)
                finally:
                    preview.close()
            os.replace(temporary, destination)

    def _storage_size_bytes(self) -> int:
        total = 0
        for directory, _subdirectories, filenames in os.walk(
            self.settings.storage_dir, followlinks=False
        ):
            for filename in filenames:
                try:
                    path = Path(directory) / filename
                    if not path.is_symlink():
                        total += path.stat().st_size
                except OSError:
                    continue
        return total

    @staticmethod
    def _directory_size_bytes(directory: Path) -> int:
        total = 0
        for root, _subdirectories, filenames in os.walk(directory, followlinks=False):
            for filename in filenames:
                try:
                    path = Path(root) / filename
                    if not path.is_symlink():
                        total += path.stat().st_size
                except OSError:
                    continue
        return total

    def _cleanup_loop(self) -> None:
        interval = min(60.0, max(1.0, self.settings.result_ttl_seconds / 4))
        while not self._cleanup_stop.wait(interval):
            with self._lock:
                self._expire_old_tasks()

    def _cleanup(self) -> None:
        self._cleanup_stop.set()
        self._cleanup_thread.join()

    def _expire_old_tasks(self) -> None:
        now = time.time()
        expired = [
            task_id
            for task_id, task in self._tasks.items()
            if task.status in TERMINAL_STATES
            and now - task.created_at >= self.settings.result_ttl_seconds
        ]
        for task_id in expired:
            task = self._tasks.pop(task_id)
            shutil.rmtree(task.directory, ignore_errors=True)

    def _purge_expired_orphans(self) -> None:
        now = time.time()
        for directory in self.settings.storage_dir.iterdir():
            marker = directory / TASK_MARKER
            if directory.is_symlink() or not directory.is_dir() or not marker.is_file():
                continue
            try:
                age = now - directory.stat().st_mtime
            except OSError:
                continue
            if age >= self.settings.result_ttl_seconds:
                shutil.rmtree(directory, ignore_errors=True)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for task in self._tasks.values():
                if task.status not in TERMINAL_STATES:
                    task.cancel_event.set()
        self._cleanup()
        self._executor.shutdown(wait=True, cancel_futures=False)
        for engine in self._resident_engines:
            try:
                engine.close()
            except Exception:
                logger.exception("Could not close a resident Laika engine")
        for task in tuple(self._tasks.values()):
            shutil.rmtree(task.directory, ignore_errors=True)


def _validate_uploaded_image(path: Path, settings: ApiSettings) -> str:
    if path.stat().st_size > settings.max_upload_bytes:
        raise ImageLimitError("input file exceeds the configured byte limit")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                image_format = image.format or ""
                if image_format not in SUPPORTED_UPLOAD_FORMATS:
                    raise ApiUploadError(
                        "only JPEG, PNG, and WebP images are supported"
                    )
                width, height = image.size
                if width < 1 or height < 1:
                    raise ApiUploadError("image dimensions must be positive")
                if width * height > settings.max_pixels:
                    raise ImageLimitError("image exceeds the configured pixel limit")
                if getattr(image, "is_animated", False):
                    raise ApiUploadError("animated images are not supported")
                image.verify()
    except ImageLimitError:
        raise
    except ApiUploadError:
        raise
    except (
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ApiUploadError("uploaded data is not a valid supported image") from exc
    return image_format


def _parse_options(raw: str, settings: ApiSettings) -> ProcessingOptions:
    if len(raw.encode("utf-8")) > MAX_OPTIONS_JSON_BYTES:
        raise ValueError("processing options are too large")
    try:
        values = json.loads(raw)
        if not isinstance(values, dict):
            raise ValueError("options must be a JSON object")
        values.setdefault("strategy", settings.default_strategy)
        request = TaskOptions.model_validate(values)
        return request.to_processing_options(settings)
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise ValueError("invalid processing options") from exc


def create_app(
    settings: ApiSettings | None = None,
    *,
    engine_factory: Callable[..., Any] | None = None,
) -> FastAPI:
    """Create the API application; FastAPI remains outside the image engine."""
    selected_settings = settings or ApiSettings.from_environment()
    manager = TaskManager(selected_settings, engine_factory=engine_factory)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            await asyncio.to_thread(manager.close)

    app = FastAPI(
        title="LeicaShot Image Processing API",
        version="1.0.0",
        description=(
            "Asynchronous CPU-first image processing through the Laika Python engine."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        UploadBodyLimitMiddleware,
        max_bytes=selected_settings.max_upload_bytes + MULTIPART_OVERHEAD_BYTES,
    )
    app.state.task_manager = manager

    @app.get("/api/v1/health")
    def health() -> dict[str, Any]:
        with manager._lock:
            pending = sum(
                task.status not in TERMINAL_STATES for task in manager._tasks.values()
            )
        return {
            "status": "ok",
            "active_or_queued_tasks": pending,
            "max_pending_tasks": selected_settings.max_pending_tasks,
        }

    @app.post("/api/v1/tasks", status_code=202)
    async def create_task(
        file: Annotated[UploadFile, File()],
        options: Annotated[str, Form()] = "{}",
    ) -> dict[str, Any]:
        try:
            try:
                processing_options = _parse_options(options, selected_settings)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            try:
                record, token = manager.reserve(processing_options)
            except QueueFullError as exc:
                raise HTTPException(status_code=429, detail=str(exc)) from exc
            except StorageLimitError as exc:
                raise HTTPException(status_code=507, detail=str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(
                    status_code=503, detail="task service is unavailable"
                ) from exc

            try:
                total = 0
                with record.input_path.open("xb") as destination:
                    while chunk := await file.read(UPLOAD_CHUNK_BYTES):
                        total += len(chunk)
                        if total > selected_settings.max_upload_bytes:
                            raise ImageLimitError(
                                "input file exceeds the configured byte limit"
                            )
                        destination.write(chunk)
                if total == 0:
                    raise ApiUploadError("uploaded image is empty")
                image_format = _validate_uploaded_image(
                    record.input_path, selected_settings
                )
                enqueued = manager.enqueue(record.task_id, image_format)
            except ImageLimitError as exc:
                manager.abort_upload(record.task_id)
                raise HTTPException(status_code=413, detail=str(exc)) from exc
            except ApiUploadError as exc:
                manager.abort_upload(record.task_id)
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except Exception:
                manager.abort_upload(record.task_id)
                raise

            return {
                "id": enqueued.task_id,
                "access_token": token,
                "status": enqueued.status,
                "status_url": f"/api/v1/tasks/{enqueued.task_id}",
                "cancel_url": f"/api/v1/tasks/{enqueued.task_id}/cancel",
            }
        finally:
            await file.close()

    @app.get("/api/v1/tasks/{task_id}")
    def get_task(
        task_id: str,
        x_laika_task_token: str = Header(..., alias="X-Laika-Task-Token"),
    ) -> dict[str, Any]:
        try:
            return manager.public_info(task_id, x_laika_task_token)
        except TaskNotFoundError as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc

    @app.post("/api/v1/tasks/{task_id}/cancel")
    def cancel_task(
        task_id: str,
        x_laika_task_token: str = Header(..., alias="X-Laika-Task-Token"),
    ) -> dict[str, Any]:
        try:
            manager.cancel(task_id, x_laika_task_token)
            return manager.public_info(task_id, x_laika_task_token)
        except TaskNotFoundError as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc

    def task_file_response(task_id: str, token: str, kind: str) -> FileResponse:
        try:
            path, media_type = manager.file_for(task_id, token, kind)
        except TaskNotFoundError as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc
        return FileResponse(
            path,
            media_type=media_type,
            filename="leicashot-result.png" if kind == "download" else None,
        )

    @app.get("/api/v1/tasks/{task_id}/source")
    def get_source(
        task_id: str,
        x_laika_task_token: str = Header(..., alias="X-Laika-Task-Token"),
    ) -> FileResponse:
        return task_file_response(task_id, x_laika_task_token, "source")

    @app.get("/api/v1/tasks/{task_id}/preview")
    def get_preview(
        task_id: str,
        x_laika_task_token: str = Header(..., alias="X-Laika-Task-Token"),
    ) -> FileResponse:
        return task_file_response(task_id, x_laika_task_token, "preview")

    @app.get("/api/v1/tasks/{task_id}/download")
    def download_result(
        task_id: str,
        x_laika_task_token: str = Header(..., alias="X-Laika-Task-Token"),
    ) -> FileResponse:
        return task_file_response(task_id, x_laika_task_token, "download")

    @app.get("/api/v1/openapi.json", include_in_schema=False)
    def versioned_openapi() -> dict[str, Any]:
        return app.openapi()

    return app


def run() -> None:
    """Start a single-process Uvicorn service using environment configuration."""
    import uvicorn

    try:
        port = int(os.getenv("LAIKA_API_PORT", "8000"))
    except ValueError as exc:
        raise ValueError("LAIKA_API_PORT must be an integer") from exc
    uvicorn.run(
        "laika.api:create_app",
        factory=True,
        host=os.getenv("LAIKA_API_HOST", "127.0.0.1"),
        port=port,
        workers=1,
    )


__all__ = [
    "ApiSettings",
    "QueueFullError",
    "StorageLimitError",
    "TaskOptions",
    "TaskManager",
    "create_app",
    "run",
]
