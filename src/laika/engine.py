"""Small, model-free image I/O baseline used by the SDK, CLI, and benchmarks."""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

SUPPORTED_INPUT_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})
PNG_MODES = frozenset({"1", "L", "LA", "P", "RGB", "RGBA"})


class ImageInputError(ValueError):
    """The input is not a supported, valid still image."""


class ImageLimitError(ImageInputError):
    """The input exceeds a configured byte or decoded-pixel limit."""


@dataclass(frozen=True)
class ImageLimits:
    max_upload_bytes: int = 50 * 1024 * 1024
    max_pixels: int = 40_000_000

    def __post_init__(self) -> None:
        if self.max_upload_bytes <= 0 or self.max_pixels <= 0:
            raise ValueError("image limits must be positive")


@dataclass(frozen=True)
class ProcessResult:
    input_format: str
    input_bytes: int
    input_width: int
    input_height: int
    output_width: int
    output_height: int
    output_mode: str
    output_format: str
    output_bytes: int
    stage_seconds: dict[str, float]
    total_seconds: float
    operation: str = "decode-orient-lossless-png"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def process_image(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    limits: ImageLimits | None = None,
) -> ProcessResult:
    """Decode a still image, apply EXIF orientation, and write a lossless PNG.

    This is an I/O and memory-measurement baseline, not an AI enhancement. It
    does not resize the raster. It accepts JPEG, PNG, and WebP by decoded format,
    rejects multi-frame files, and retains ICC color-profile data when present.
    """
    selected_limits = limits or ImageLimits()
    source = Path(input_path)
    destination = Path(output_path)
    if not source.is_file():
        raise ImageInputError("input path is not a regular file")
    if source.resolve() == destination.resolve():
        raise ImageInputError("input and output paths must be different")
    if destination.suffix.lower() != ".png":
        raise ImageInputError("baseline output must use a .png filename")

    started = time.perf_counter()
    stage_seconds: dict[str, float] = {}
    stage_started = time.perf_counter()
    try:
        source_file = source.open("rb")
    except OSError as exc:
        raise ImageInputError("cannot open input file") from exc

    with source_file:
        file_stat = os.fstat(source_file.fileno())
        input_bytes = file_stat.st_size
        if input_bytes > selected_limits.max_upload_bytes:
            raise ImageLimitError("input file exceeds the configured byte limit")
        original_identity = (
            file_stat.st_dev,
            file_stat.st_ino,
            file_stat.st_size,
            file_stat.st_mtime_ns,
        )
        try:
            opened = Image.open(source_file)
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
            raise ImageInputError("input is not a valid supported image") from exc

        with opened:
            input_format = opened.format or ""
            if input_format not in SUPPORTED_INPUT_FORMATS:
                raise ImageInputError(
                    f"unsupported decoded image format: {input_format or 'unknown'}"
                )
            input_width, input_height = opened.size
            if input_width <= 0 or input_height <= 0:
                raise ImageInputError("image dimensions must be positive")
            if input_width * input_height > selected_limits.max_pixels:
                raise ImageLimitError(
                    "image exceeds the configured decoded-pixel limit"
                )
            if getattr(opened, "is_animated", False):
                raise ImageInputError(
                    "animated images are not supported by this baseline"
                )
            stage_seconds["validate"] = time.perf_counter() - stage_started

            stage_started = time.perf_counter()
            opened.load()
            current_stat = os.fstat(source_file.fileno())
            current_identity = (
                current_stat.st_dev,
                current_stat.st_ino,
                current_stat.st_size,
                current_stat.st_mtime_ns,
            )
            if current_identity != original_identity:
                raise ImageInputError("input changed while it was being processed")
            stage_seconds["decode"] = time.perf_counter() - stage_started

            stage_started = time.perf_counter()
            orientation = opened.getexif().get(274)
            image = (
                ImageOps.exif_transpose(opened)
                if orientation not in (None, 1)
                else opened
            )
            if image.mode not in PNG_MODES:
                converted = image.convert("RGB")
                if image is not opened:
                    image.close()
                image = converted
            output_width, output_height = image.size
            output_mode = image.mode
            icc_profile = image.info.get("icc_profile")
            stage_seconds["orient_normalize"] = time.perf_counter() - stage_started

            destination.parent.mkdir(parents=True, exist_ok=True)
            file_descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.stem}.",
                suffix=".tmp",
                dir=destination.parent,
            )
            os.close(file_descriptor)
            temporary_path = Path(temporary_name)
            stage_started = time.perf_counter()
            try:
                image.save(temporary_path, format="PNG", icc_profile=icc_profile)
                os.replace(temporary_path, destination)
            except Exception:
                temporary_path.unlink(missing_ok=True)
                raise
            finally:
                if image is not opened:
                    image.close()
            stage_seconds["encode"] = time.perf_counter() - stage_started

    total_seconds = time.perf_counter() - started
    return ProcessResult(
        input_format=input_format,
        input_bytes=input_bytes,
        input_width=input_width,
        input_height=input_height,
        output_width=output_width,
        output_height=output_height,
        output_mode=output_mode,
        output_format="PNG",
        output_bytes=destination.stat().st_size,
        stage_seconds=stage_seconds,
        total_seconds=total_seconds,
    )
