from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from laika.m9 import (
    M9ColorEngine,
    M9InferenceError,
    M9Onnx,
    M9Prediction,
    _prepare_tensor,
    m9_model_size,
)


class FakeSession:
    def __init__(
        self, *, input_name: str = "input", output_name: str = "output"
    ) -> None:
        self.input = SimpleNamespace(
            name=input_name, type="tensor(float)", shape=[1, 3, "height", "width"]
        )
        self.output = SimpleNamespace(
            name=output_name, type="tensor(float)", shape=[1, 3, "height", "width"]
        )
        self.calls: list[tuple[list[str], dict[str, np.ndarray]]] = []

    def get_inputs(self) -> list[Any]:
        return [self.input]

    def get_outputs(self) -> list[Any]:
        return [self.output]

    def run(self, names: list[str], values: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.calls.append((names, values))
        return [values["input"].copy()]


def test_model_size_uses_reference_half_up_fit() -> None:
    assert m9_model_size(1920, 1080) == (1024, 576)
    assert m9_model_size(732, 475) == (1024, 664)
    assert m9_model_size(1, 2048) == (1, 1024)
    with pytest.raises(ValueError, match="positive"):
        m9_model_size(0, 10)


def test_preprocess_is_rgb_nchw_float32_and_reference_normalized() -> None:
    image = Image.new("RGB", (20, 10), (0, 128, 255))
    tensor, width, height, _ = _prepare_tensor(image)

    assert (width, height) == (1024, 512)
    assert tensor.shape == (1, 3, 512, 1024)
    assert tensor.dtype == np.float32
    assert tensor.flags.c_contiguous
    np.testing.assert_allclose(
        tensor[0, :, 100, 100],
        np.asarray([-1.0, (128 / 255 - 0.5) / 0.5, 1.0], dtype=np.float32),
        atol=1e-7,
    )
    image.close()


def test_m9_verifies_hash_and_enforces_io_contract(tmp_path: Path) -> None:
    model = tmp_path / "local.onnx"
    model.write_bytes(b"fixture model")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    session = FakeSession()
    options: dict[str, Any] = {}

    def factory(path: str, **kwargs: Any) -> FakeSession:
        options.update(kwargs)
        assert path == str(model)
        return session

    with M9Onnx(
        model,
        strategy="low-memory",
        intra_op_num_threads=1,
        expected_sha256=digest,
        session_factory=factory,
    ) as engine:
        tensor = np.zeros((1, 3, 1024, 1), dtype=np.float32)
        prediction = engine.infer_tensor(tensor, 1, 1024)

    assert prediction.rgb.shape == (1024, 1, 3)
    assert prediction.rgb.dtype == np.uint8
    assert len(session.calls) == 1
    assert session.calls[0][0] == ["output"]
    assert options["providers"] == ["CPUExecutionProvider"]

    with pytest.raises(M9InferenceError, match="SHA-256 mismatch"):
        M9Onnx(model, expected_sha256="0" * 64, session_factory=factory)


def test_m9_rejects_wrong_input_output_contract(tmp_path: Path) -> None:
    model = tmp_path / "local.onnx"
    model.write_bytes(b"fixture model")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()

    with pytest.raises(M9InferenceError, match="input named"):
        M9Onnx(
            model,
            expected_sha256=digest,
            session_factory=lambda *_args, **_kwargs: FakeSession(input_name="bad"),
        )


class FakePredictor:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.model_sha256 = "0" * 64
        self.runtime_version = "test"
        self.runtime_provider = "CPUExecutionProvider"

    def infer_tensor(self, tensor: np.ndarray, width: int, height: int) -> M9Prediction:
        rgb = np.empty((height, width, 3), dtype=np.uint8)
        rgb[:, :, 0] = 150
        rgb[:, :, 1] = 90
        rgb[:, :, 2] = 45
        return M9Prediction(rgb, width, height, 0.0, 0.01)

    def close(self) -> None:
        pass


@pytest.mark.parametrize(("suffix", "format_name"), [(".png", "PNG"), (".jpg", "JPEG")])
def test_low_memory_file_pipeline_preserves_exif_oriented_size_and_exports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
    format_name: str,
) -> None:
    import laika.m9 as m9

    monkeypatch.setattr(m9, "M9Onnx", FakePredictor)
    source = tmp_path / "source.jpg"
    original = Image.new("RGB", (19, 11), (70, 110, 160))
    exif = original.getexif()
    exif[274] = 6
    original.save(source, quality=100, subsampling=0, exif=exif)
    original.close()
    destination = tmp_path / f"result{suffix}"

    with M9ColorEngine("unused", strategy="low-memory", row_chunk_height=5) as engine:
        result = engine.process_file(source, destination)

    with Image.open(destination) as output:
        assert output.format == format_name
        assert output.size == (11, 19)
    assert (result.width, result.height) == (11, 19)
    assert result.strategy == "low-memory"
    assert "decode_model_input" in result.stage_seconds
    assert "lab_fusion" in result.stage_seconds


def test_balanced_engine_reuses_one_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import laika.m9 as m9

    constructed = 0

    class CountedPredictor(FakePredictor):
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            nonlocal constructed
            constructed += 1
            super().__init__(*_args, **_kwargs)

    monkeypatch.setattr(m9, "M9Onnx", CountedPredictor)
    source = tmp_path / "source.png"
    Image.new("RGB", (9, 5), (30, 60, 90)).save(source)
    with M9ColorEngine("unused", strategy="balanced") as engine:
        engine.process_file(source, tmp_path / "one.png")
        engine.process_file(source, tmp_path / "two.png")
    assert constructed == 1


def test_m9_file_pipeline_enforces_pixel_limit(tmp_path: Path) -> None:
    from laika.engine import ImageLimitError, ImageLimits

    source = tmp_path / "source.png"
    Image.new("RGB", (12, 10), (20, 40, 60)).save(source)
    with M9ColorEngine("unused", limits=ImageLimits(max_pixels=100)) as engine:
        with pytest.raises(ImageLimitError, match="pixel limit"):
            engine.process_file(source, tmp_path / "result.png")
