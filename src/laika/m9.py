"""M9 ONNX color prediction and reference-style Lab detail fusion."""

from __future__ import annotations

import gc
import hashlib
import os
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageOps, UnidentifiedImageError

from laika.color import fuse_lab_detail, fuse_lab_detail_to_sink
from laika.engine import ImageInputError, ImageLimitError, ImageLimits
from laika.png import PngRGBStreamWriter

M9_MODEL_FILENAME = "leica-m9-v3-epoch11.fp16.onnx"
M9_MODEL_SHA256 = "8936b8c86656265ab07c30b9b4cbbd7fa195d6e44cb3be1d4fd059bb36781e33"
M9_MODEL_BYTES = 10_053_669
M9_INPUT_NAME = "input"
M9_OUTPUT_NAME = "output"
M9_LONG_EDGE = 1024
M9_SIGMA_L = 25.0
M9_SIGMA_AB = 20.0
M9_STRATEGIES = ("low-memory", "balanced")
SUPPORTED_IMAGE_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})


class M9InferenceError(RuntimeError):
    """The local M9 model or one of its tensor contracts is invalid."""


@dataclass(frozen=True)
class M9Prediction:
    """8-bit RGB model output and dimensions for one M9 inference."""

    rgb: Any
    width: int
    height: int
    preprocess_seconds: float
    inference_seconds: float


@dataclass(frozen=True)
class M9ColorResult:
    """Saved M9+Lab result with stage timings and the preserved output size."""

    input_format: str
    output_format: str
    width: int
    height: int
    model_width: int
    model_height: int
    strategy: str
    model_sha256: str
    runtime_provider: str
    runtime_version: str
    intra_op_num_threads: int
    session_reused: bool
    stage_seconds: dict[str, float]
    total_seconds: float
    output_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _ImageMetadata:
    image_format: str
    width: int
    height: int
    identity: tuple[int, int, int, int]
    icc_profile: bytes | None


def m9_model_size(width: int, height: int) -> tuple[int, int]:
    """Fit the source to a 1024-pixel long edge using integer half-up rounding."""
    if width < 1 or height < 1:
        raise ValueError("image dimensions must be positive")
    longest = max(width, height)
    return (
        max(1, (width * M9_LONG_EDGE + longest // 2) // longest),
        max(1, (height * M9_LONG_EDGE + longest // 2) // longest),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_identity(source: Any) -> tuple[int, int, int, int]:
    stat = os.fstat(source.fileno())
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _open_oriented_rgb(
    path: Path,
    limits: ImageLimits,
    *,
    expected_identity: tuple[int, int, int, int] | None = None,
) -> tuple[Any, _ImageMetadata]:
    """Decode a still image, apply EXIF orientation, and return RGB PIL data."""
    if not path.is_file():
        raise ImageInputError("input path is not a regular file")
    try:
        source = path.open("rb")
    except OSError as exc:
        raise ImageInputError("cannot open input file") from exc

    with source:
        identity = _image_identity(source)
        if identity[2] > limits.max_upload_bytes:
            raise ImageLimitError("input file exceeds the configured byte limit")
        if expected_identity is not None and identity != expected_identity:
            raise ImageInputError("input changed between Low Memory processing stages")
        try:
            opened = Image.open(source)
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
            raise ImageInputError("input is not a valid supported image") from exc

        with opened:
            image_format = opened.format or ""
            if image_format not in SUPPORTED_IMAGE_FORMATS:
                raise ImageInputError(
                    f"unsupported decoded image format: {image_format or 'unknown'}"
                )
            width, height = opened.size
            if width < 1 or height < 1:
                raise ImageInputError("image dimensions must be positive")
            if width * height > limits.max_pixels:
                raise ImageLimitError(
                    "image exceeds the configured decoded-pixel limit"
                )
            if getattr(opened, "is_animated", False):
                raise ImageInputError("animated images are not supported")
            try:
                opened.load()
            except (OSError, Image.DecompressionBombError) as exc:
                raise ImageInputError("image pixels could not be decoded") from exc
            if _image_identity(source) != identity:
                raise ImageInputError("input changed while it was being decoded")

            oriented = ImageOps.exif_transpose(opened)
            try:
                rgb = oriented.convert("RGB")
            finally:
                if oriented is not opened:
                    oriented.close()

            metadata = _ImageMetadata(
                image_format=image_format,
                width=rgb.width,
                height=rgb.height,
                identity=identity,
                icc_profile=opened.info.get("icc_profile"),
            )
            # The converted RGB image owns its pixel storage after the file closes.
            rgb.load()
            return rgb, metadata


def _prepare_tensor(image: Any) -> tuple[Any, int, int, float]:
    """Resize RGB with Pillow bilinear and normalize directly into one NCHW buffer."""
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional runtime
        raise M9InferenceError(
            "M9 inference requires NumPy; install the inference extra"
        ) from exc

    width, height = m9_model_size(image.width, image.height)
    started = time.perf_counter()
    if (width, height) == image.size:
        resized = image
    else:
        resized = image.resize((width, height), resample=Image.Resampling.BILINEAR)
    try:
        rgb = np.asarray(resized, dtype=np.uint8)
        if rgb.shape != (height, width, 3) or not rgb.flags.c_contiguous:
            rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        tensor = np.empty((1, 3, height, width), dtype=np.float32, order="C")
        for channel in range(3):
            plane = tensor[0, channel]
            plane[...] = rgb[:, :, channel]
            np.divide(plane, np.float32(255.0), out=plane)
            plane -= np.float32(0.5)
            np.divide(plane, np.float32(0.5), out=plane)
        del rgb
    finally:
        if resized is not image:
            resized.close()
    return tensor, width, height, time.perf_counter() - started


class M9Onnx:
    """Hash-verified, CPU-only M9 ONNX session that callers may reuse."""

    def __init__(
        self,
        model_path: str | os.PathLike[str],
        *,
        strategy: Literal["low-memory", "balanced"] = "low-memory",
        intra_op_num_threads: int | None = None,
        expected_sha256: str = M9_MODEL_SHA256,
        session_factory: Any | None = None,
    ) -> None:
        if strategy not in M9_STRATEGIES:
            raise ValueError(f"strategy must be one of {', '.join(M9_STRATEGIES)}")
        if intra_op_num_threads is None:
            intra_op_num_threads = (
                1 if strategy == "low-memory" else min(4, os.cpu_count() or 1)
            )
        if intra_op_num_threads < 1:
            raise ValueError("intra_op_num_threads must be positive")
        self.model_path = Path(model_path)
        self.strategy = strategy
        self.intra_op_num_threads = intra_op_num_threads
        self._session: Any | None = None
        if not self.model_path.is_file():
            raise M9InferenceError(f"M9 model is unavailable: {self.model_path}")
        actual_sha256 = _sha256(self.model_path)
        if actual_sha256 != expected_sha256:
            raise M9InferenceError(
                "M9 model SHA-256 mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - optional runtime
            raise M9InferenceError(
                "M9 inference requires ONNX Runtime; install the inference extra"
            ) from exc
        self.model_sha256 = actual_sha256
        self.runtime_version = ort.__version__
        self.runtime_provider = "CPUExecutionProvider"

        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_op_num_threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.enable_cpu_mem_arena = strategy == "balanced"
        options.enable_mem_pattern = strategy == "balanced"
        options.enable_mem_reuse = True
        factory = session_factory or ort.InferenceSession
        try:
            self._session = factory(
                str(self.model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise M9InferenceError(f"cannot load M9 ONNX model: {exc}") from exc
        self._validate_contract()

    def _validate_contract(self) -> None:
        assert self._session is not None
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != M9_INPUT_NAME:
            raise M9InferenceError(
                "M9 model must expose the single input named 'input'"
            )
        if len(outputs) != 1 or outputs[0].name != M9_OUTPUT_NAME:
            raise M9InferenceError(
                "M9 model must expose the single output named 'output'"
            )
        for node in (*inputs, *outputs):
            if node.type != "tensor(float)" or len(node.shape) != 4:
                raise M9InferenceError(
                    "M9 external tensors must be rank-4 float32 NCHW"
                )
            if node.shape[0] != 1 or node.shape[1] != 3:
                raise M9InferenceError("M9 external tensors must have shape [1,3,H,W]")

    def infer_tensor(self, tensor: Any, width: int, height: int) -> M9Prediction:
        """Run one normalized FP32 NCHW tensor and quantize output like ToPILImage."""
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - optional runtime
            raise M9InferenceError("M9 inference requires NumPy") from exc
        if self._session is None:
            raise M9InferenceError("M9 ONNX session is closed")
        values = np.asarray(tensor)
        if (
            values.dtype != np.float32
            or values.shape != (1, 3, height, width)
            or not values.flags.c_contiguous
        ):
            raise ValueError("M9 input must be C-contiguous float32 NCHW [1,3,H,W]")
        if max(width, height) != M9_LONG_EDGE:
            raise ValueError("M9 model input long edge must be exactly 1024")

        started = time.perf_counter()
        try:
            outputs = self._session.run([M9_OUTPUT_NAME], {M9_INPUT_NAME: values})
        except Exception as exc:
            raise M9InferenceError(f"M9 CPU inference failed: {exc}") from exc
        inference_seconds = time.perf_counter() - started
        if len(outputs) != 1:
            raise M9InferenceError("M9 model returned an unexpected output count")
        output = np.asarray(outputs[0])
        if output.shape != (1, 3, height, width) or output.dtype != np.float32:
            raise M9InferenceError(
                f"M9 output must be float32 [1,3,{height},{width}], got "
                f"{output.dtype} {output.shape}"
            )
        if not np.isfinite(output).all():
            raise M9InferenceError("M9 model returned non-finite output values")
        if not output.flags.writeable:
            output = np.array(output, dtype=np.float32, order="C", copy=True)

        # Mutate the returned buffer in place to avoid a second full float32 output.
        for channel in range(3):
            plane = output[0, channel]
            np.multiply(plane, np.float32(0.5), out=plane)
            np.add(plane, np.float32(0.5), out=plane)
            np.clip(plane, np.float32(0), np.float32(1), out=plane)
            np.multiply(plane, np.float32(255), out=plane)
        rgb = np.empty((height, width, 3), dtype=np.uint8)
        for channel in range(3):
            np.copyto(rgb[:, :, channel], output[0, channel], casting="unsafe")
        del output, outputs
        return M9Prediction(
            rgb=rgb,
            width=width,
            height=height,
            preprocess_seconds=0.0,
            inference_seconds=inference_seconds,
        )

    def predict_pil(self, image: Any) -> M9Prediction:
        tensor, width, height, preprocess_seconds = _prepare_tensor(image)
        prediction = self.infer_tensor(tensor, width, height)
        del tensor
        return M9Prediction(
            rgb=prediction.rgb,
            width=width,
            height=height,
            preprocess_seconds=preprocess_seconds,
            inference_seconds=prediction.inference_seconds,
        )

    def close(self) -> None:
        self._session = None
        gc.collect()

    def __enter__(self) -> M9Onnx:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _resize_prediction(rgb: Any, width: int, height: int) -> Any:
    import numpy as np

    if rgb.shape == (height, width, 3):
        return np.ascontiguousarray(rgb)
    image = Image.fromarray(rgb, mode="RGB")
    try:
        resized = image.resize((width, height), resample=Image.Resampling.BICUBIC)
        try:
            return np.array(resized, dtype=np.uint8, order="C", copy=True)
        finally:
            resized.close()
    finally:
        image.close()


class M9ColorEngine:
    """Run M9 plus Lab fusion with serial task ownership and two memory profiles."""

    def __init__(
        self,
        model_path: str | os.PathLike[str],
        *,
        strategy: Literal["low-memory", "balanced"] = "low-memory",
        intra_op_num_threads: int | None = None,
        limits: ImageLimits | None = None,
        row_chunk_height: int = 512,
    ) -> None:
        if strategy not in M9_STRATEGIES:
            raise ValueError(f"strategy must be one of {', '.join(M9_STRATEGIES)}")
        if row_chunk_height < 1:
            raise ValueError("row_chunk_height must be positive")
        if intra_op_num_threads is not None and intra_op_num_threads < 1:
            raise ValueError("intra_op_num_threads must be positive")
        self.model_path = Path(model_path)
        self.strategy = strategy
        self.intra_op_num_threads = intra_op_num_threads or (
            1 if strategy == "low-memory" else min(4, os.cpu_count() or 1)
        )
        self.limits = limits or ImageLimits()
        self.row_chunk_height = row_chunk_height
        self._predictor: M9Onnx | None = None
        self._lock = threading.Lock()

    def _balanced_predictor(self) -> M9Onnx:
        if self._predictor is None:
            self._predictor = M9Onnx(
                self.model_path,
                strategy="balanced",
                intra_op_num_threads=self.intra_op_num_threads,
            )
        return self._predictor

    def process_file(
        self,
        input_path: str | os.PathLike[str],
        output_path: str | os.PathLike[str],
    ) -> M9ColorResult:
        """Process JPEG/PNG/WebP input and write full-oriented-size PNG or JPEG."""
        source_path = Path(input_path)
        destination = Path(output_path)
        if source_path.resolve() == destination.resolve():
            raise ImageInputError("input and output paths must be different")
        suffix = destination.suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg"}:
            raise ImageInputError("M9 output path must end in .png, .jpg, or .jpeg")
        output_format = "PNG" if suffix == ".png" else "JPEG"

        with self._lock:
            return self._process_locked(source_path, destination, output_format)

    def _process_locked(
        self, source_path: Path, destination: Path, output_format: str
    ) -> M9ColorResult:
        import numpy as np

        started = time.perf_counter()
        stages: dict[str, float] = {}
        model_width = model_height = 0
        session_reused = False
        if self.strategy == "low-memory":
            stage = time.perf_counter()
            model_image, metadata = _open_oriented_rgb(source_path, self.limits)
            try:
                tensor, model_width, model_height, preprocess_seconds = _prepare_tensor(
                    model_image
                )
            finally:
                model_image.close()
            stages["decode_model_input"] = time.perf_counter() - stage
            stages["preprocess"] = preprocess_seconds
            stage = time.perf_counter()
            predictor = M9Onnx(
                self.model_path,
                strategy="low-memory",
                intra_op_num_threads=self.intra_op_num_threads,
            )
            stages["model_load"] = time.perf_counter() - stage
            try:
                prediction = predictor.infer_tensor(tensor, model_width, model_height)
            finally:
                predictor.close()
            stages["inference"] = prediction.inference_seconds
            model_sha256 = predictor.model_sha256
            runtime_version = predictor.runtime_version
            runtime_provider = predictor.runtime_provider
            del predictor, tensor
            gc.collect()

            stage = time.perf_counter()
            original_image, confirmed = _open_oriented_rgb(
                source_path, self.limits, expected_identity=metadata.identity
            )
            try:
                original_rgb = np.array(
                    original_image, dtype=np.uint8, order="C", copy=True
                )
                icc_profile = confirmed.icc_profile
            finally:
                original_image.close()
            stages["decode_source"] = time.perf_counter() - stage
        else:
            stage = time.perf_counter()
            original_image, metadata = _open_oriented_rgb(source_path, self.limits)
            try:
                tensor, model_width, model_height, preprocess_seconds = _prepare_tensor(
                    original_image
                )
                original_rgb = np.array(
                    original_image, dtype=np.uint8, order="C", copy=True
                )
                icc_profile = metadata.icc_profile
            finally:
                original_image.close()
            stages["decode_source"] = time.perf_counter() - stage
            stages["preprocess"] = preprocess_seconds
            session_reused = self._predictor is not None
            stage = time.perf_counter()
            predictor = self._balanced_predictor()
            stages["model_load"] = (
                0.0 if session_reused else time.perf_counter() - stage
            )
            prediction = predictor.infer_tensor(tensor, model_width, model_height)
            stages["inference"] = prediction.inference_seconds
            model_sha256 = predictor.model_sha256
            runtime_version = predictor.runtime_version
            runtime_provider = predictor.runtime_provider
            del tensor

        output_height, output_width = original_rgb.shape[:2]
        stage = time.perf_counter()
        predicted_full = _resize_prediction(prediction.rgb, output_width, output_height)
        stages["prediction_resize"] = time.perf_counter() - stage
        del prediction
        try:
            import cv2

            cv2.setNumThreads(self.intra_op_num_threads)
        except ImportError as exc:  # pragma: no cover - optional runtime
            raise M9InferenceError(
                "M9 Lab fusion requires OpenCV; install the inference extra"
            ) from exc
        fused = None
        if output_format == "JPEG":
            stage = time.perf_counter()
            fused = fuse_lab_detail(
                original_rgb,
                predicted_full,
                sigma_l=M9_SIGMA_L,
                sigma_ab=M9_SIGMA_AB,
                row_chunk_height=self.row_chunk_height,
            )
            stages["lab_fusion"] = time.perf_counter() - stage
            del original_rgb, predicted_full

        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.stem}.", suffix=".tmp", dir=destination.parent
        )
        temporary_path = Path(temporary_name)
        stage = time.perf_counter()
        try:
            stream = os.fdopen(descriptor, "w+b")
            descriptor = -1
            with stream:
                if output_format == "PNG":
                    writer = PngRGBStreamWriter(
                        stream,
                        output_width,
                        output_height,
                        icc_profile=icc_profile,
                        compression_level=3,
                    )
                    fusion_seconds = fuse_lab_detail_to_sink(
                        original_rgb,
                        predicted_full,
                        lambda _row_start, rows: writer.write_rows(rows),
                        sigma_l=M9_SIGMA_L,
                        sigma_ab=M9_SIGMA_AB,
                        row_chunk_height=self.row_chunk_height,
                    )
                    writer.finish()
                    del original_rgb, predicted_full
                    stages["lab_fusion"] = fusion_seconds
                else:
                    image = Image.fromarray(fused, mode="RGB")
                    try:
                        save_options: dict[str, Any] = {}
                        if icc_profile:
                            save_options["icc_profile"] = icc_profile
                        save_options.update(quality=95, subsampling=0)
                        image.save(stream, format=output_format, **save_options)
                    finally:
                        image.close()
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, destination)
        except Exception:
            # If wrapping the descriptor itself failed, ensure it is not leaked.
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            temporary_path.unlink(missing_ok=True)
            raise
        stages["encode"] = time.perf_counter() - stage
        if output_format == "PNG":
            stages["encode"] = max(
                0.0, stages["encode"] - stages.get("lab_fusion", 0.0)
            )
        return M9ColorResult(
            input_format=metadata.image_format,
            output_format=output_format,
            width=output_width,
            height=output_height,
            model_width=model_width,
            model_height=model_height,
            strategy=self.strategy,
            model_sha256=model_sha256,
            runtime_provider=runtime_provider,
            runtime_version=runtime_version,
            intra_op_num_threads=self.intra_op_num_threads,
            session_reused=session_reused,
            stage_seconds=stages,
            total_seconds=time.perf_counter() - started,
            output_bytes=destination.stat().st_size,
        )

    def close(self) -> None:
        predictor = self._predictor
        self._predictor = None
        if predictor is not None:
            predictor.close()

    def __enter__(self) -> M9ColorEngine:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = [
    "M9ColorEngine",
    "M9ColorResult",
    "M9InferenceError",
    "M9Onnx",
    "M9Prediction",
    "M9_MODEL_BYTES",
    "M9_MODEL_FILENAME",
    "M9_MODEL_SHA256",
    "m9_model_size",
]
