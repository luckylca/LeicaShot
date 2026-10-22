from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from laika.composition import CompositionCrop
from laika.processing import (
    LaikaProcessingEngine,
    ProcessingCancelled,
    ProcessingError,
    ProcessingOptions,
)


class FakeM9Engine:
    instances: list[FakeM9Engine] = []

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.closed = False
        self.instances.append(self)

    def process_file(
        self, source: Path, destination: Path, *, cancel_check=None
    ) -> Any:
        if cancel_check is not None:
            cancel_check()
        with Image.open(source) as opened:
            image = opened.convert("RGB")
            values = np.asarray(image, dtype=np.uint8).copy()
            image.close()
        values[:, :, 0] = np.minimum(values[:, :, 0].astype(np.uint16) + 9, 255)
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(values, mode="RGB").save(destination)
        return SimpleNamespace(
            stage_seconds={"model_load": 0.01, "lab_fusion": 0.02},
            model_sha256="m9-test-hash",
            runtime_version="ort-test",
            runtime_provider="CPUExecutionProvider",
            width=values.shape[1],
            height=values.shape[0],
        )

    def close(self) -> None:
        self.closed = True


class FakeDepthEngine:
    instances: list[FakeDepthEngine] = []
    observed_rgb: np.ndarray | None = None
    model_sha256 = "depth-test-hash"
    runtime_version = "ort-test"
    provider = "CPUExecutionProvider"

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.closed = False
        self.instances.append(self)

    def infer_rgb(self, rgb: np.ndarray, *, cancel_check=None) -> Any:
        if cancel_check is not None:
            cancel_check()
        self.observed_rgb = rgb.copy()
        height, width = rgb.shape[:2]
        rows = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        depth = np.broadcast_to(rows, (height, width)).copy()
        return SimpleNamespace(
            depth=depth,
            stage_seconds={"preprocess": 0.01, "inference": 0.02},
        )

    def close(self) -> None:
        self.closed = True


class FakeBokehEngine:
    instances: list[FakeBokehEngine] = []

    @staticmethod
    def _runtime_info() -> dict[str, str]:
        return {
            "model_sha256": "arnet-test-hash",
            "provider": "CPUExecutionProvider",
        }

    runtime_info = SimpleNamespace(as_dict=_runtime_info.__func__)

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.closed = False
        self.instances.append(self)

    def infer(self, rgb: np.ndarray, _defocus: np.ndarray) -> Any:
        nchw = np.transpose(rgb, (2, 0, 1))[None].copy()
        return SimpleNamespace(rgb_nchw=nchw)

    def close(self) -> None:
        self.closed = True


class FakeSamEngine:
    instances: list[FakeSamEngine] = []

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.closed = False
        self.predict_calls = 0
        self.cache = SimpleNamespace(hits=0)
        self.instances.append(self)

    def predict(self, _image: Any, _coords: Any, _labels: Any, *, cancel_check=None):
        from laika.subject import SubjectPrediction

        if cancel_check is not None:
            cancel_check()
        self.predict_calls += 1
        logits = np.full((1, 3, 256, 256), -1.0, dtype=np.float32)
        logits[0, 0, 32:224, 32:224] = 1.0
        scores = np.asarray([[0.9, 0.5, 0.1]], dtype=np.float32)
        return SubjectPrediction(
            mask_logits=logits,
            scores=scores,
            encoder_seconds=0.01 if self.predict_calls == 1 else 0.0,
            decoder_seconds=0.02,
            cache_hit=self.predict_calls > 1,
        )

    def close(self) -> None:
        self.closed = True


class FakeMatteEngine:
    instances: list[FakeMatteEngine] = []

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.closed = False
        self.model_sha256 = "vitmatte-test-hash"
        self.inference_calls = 0
        self.instances.append(self)

    def infer(self, pixel_values: np.ndarray) -> np.ndarray:
        self.inference_calls += 1
        return pixel_values[:, 3:4].copy()

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def clear_fake_instances() -> None:
    FakeM9Engine.instances.clear()
    FakeDepthEngine.instances.clear()
    FakeBokehEngine.instances.clear()
    FakeSamEngine.instances.clear()
    FakeMatteEngine.instances.clear()


def _source(path: Path, *, width: int = 80, height: int = 64) -> None:
    yy, xx = np.mgrid[:height, :width]
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[:, :, 0] = xx * 3
    rgb[:, :, 1] = yy * 3
    rgb[:, :, 2] = 90
    Image.fromarray(rgb, mode="RGB").save(path)


def _engine(
    *,
    mode: str,
    temp_root: Path | None = None,
    strategy: str = "low-memory",
) -> LaikaProcessingEngine:
    return LaikaProcessingEngine(
        m9_model_path="local-m9.onnx",
        depth_model_path="local-depth.onnx",
        bokeh_model_path="local-arnet.onnx",
        strategy=strategy,
        temp_root=temp_root,
        m9_factory=FakeM9Engine,
        depth_factory=FakeDepthEngine,
        bokeh_factory=FakeBokehEngine,
        sam_factory=FakeSamEngine,
        matte_factory=FakeMatteEngine,
    )


def test_unified_color_depth_bokeh_api_runs_in_order_and_preserves_size(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jpg"
    output = tmp_path / "result.png"
    _source(source)
    progress = []

    with _engine(mode="color+bokeh") as engine:
        result = engine.process_file(
            source,
            output,
            options=ProcessingOptions(mode="color+bokeh", aperture=2.8),
            progress_callback=progress.append,
        )

    assert output.is_file()
    with Image.open(output) as image:
        assert image.size == (80, 64)
        assert image.format == "PNG"
        output_rgb = np.asarray(image.convert("RGB"))
    with Image.open(source) as image:
        source_red = np.asarray(image.convert("RGB"))[:, :, 0]
    assert np.all(output_rgb[:, :, 0] >= source_red)
    assert result.mode == "color+bokeh"
    assert (result.width, result.height) == (80, 64)
    assert result.models["m9"]["sha256"] == "m9-test-hash"
    assert result.models["depth"]["sha256"] == "depth-test-hash"
    assert result.models["bokeh"]["model_sha256"] == "arnet-test-hash"
    assert FakeDepthEngine.instances[0].observed_rgb is not None
    with Image.open(source) as image:
        original_red = np.asarray(image.convert("RGB"))[0, 0, 0]
    assert FakeDepthEngine.instances[0].observed_rgb[0, 0, 0] == original_red
    assert FakeDepthEngine.instances[0].observed_rgb[0, 0, 0] != output_rgb[0, 0, 0]
    assert {event.stage for event in progress} == {
        "color",
        "depth",
        "bokeh",
        "export",
        "complete",
    }
    assert sum(event.stage == "bokeh" for event in progress) >= 2
    assert [event.fraction for event in progress if event.stage == "bokeh"] == sorted(
        event.fraction for event in progress if event.stage == "bokeh"
    )
    assert [event.fraction for event in progress][-1] == 1.0
    assert all(event.elapsed_seconds >= 0 for event in progress)
    assert [event.stage for event in progress[:2]] == ["depth", "depth"]
    assert {"depth.inference", "depth.model_load", "bokeh.arnet_pass_1"} <= set(
        result.stage_seconds
    )
    assert FakeM9Engine.instances[0].closed
    assert FakeDepthEngine.instances[0].closed
    assert FakeBokehEngine.instances[0].closed


@pytest.mark.parametrize("mode", ["color", "bokeh"])
def test_unified_engine_runs_each_single_feature_without_loading_unused_models(
    tmp_path: Path, mode: str
) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "result.png"
    _source(source)
    with _engine(mode=mode) as engine:
        result = engine.process_file(
            source,
            output,
            options=ProcessingOptions(mode=mode),  # type: ignore[arg-type]
        )

    if mode == "color":
        assert set(result.models) == {"m9"}
        assert not FakeDepthEngine.instances
        assert not FakeBokehEngine.instances
    else:
        assert set(result.models) == {"depth", "bokeh"}
        assert not FakeM9Engine.instances
    assert output.is_file()


@pytest.mark.parametrize("mode", ["color", "bokeh", "color+bokeh"])
def test_unified_engine_applies_selected_crop_after_requested_ai_modes(
    tmp_path: Path, mode: str
) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "cropped.png"
    _source(source)
    crop = CompositionCrop(0.125, 0.125, 0.875, 0.875)

    with _engine(mode=mode) as engine:
        result = engine.process_file(
            source,
            output,
            options=ProcessingOptions(mode=mode, crop=crop),  # type: ignore[arg-type]
        )

    with Image.open(output) as image:
        assert image.size == (60, 48)
        assert image.format == "PNG"
    assert (result.width, result.height) == (60, 48)
    assert result.stage_seconds["composition.crop"] >= 0
    if mode == "color":
        assert set(result.models) == {"m9"}
        assert not FakeDepthEngine.instances
        assert not FakeBokehEngine.instances
    elif mode == "bokeh":
        assert set(result.models) == {"depth", "bokeh"}
        assert not FakeM9Engine.instances
    else:
        assert set(result.models) == {"m9", "depth", "bokeh"}


def test_unified_engine_applies_explicit_output_size_and_reuses_balanced_sessions(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    output_a = tmp_path / "first.png"
    output_b = tmp_path / "second.png"
    _source(source, width=120, height=80)
    engine = _engine(mode="color+bokeh", strategy="balanced")
    options = ProcessingOptions(
        mode="color+bokeh", strategy="balanced", output_size=(60, 40)
    )
    try:
        first = engine.process_file(source, output_a, options=options)
        second = engine.process_file(source, output_b, options=options)
        assert first.width == second.width == 60
        assert first.height == second.height == 40
        assert len(FakeM9Engine.instances) == 1
        assert len(FakeDepthEngine.instances) == 1
        assert len(FakeBokehEngine.instances) == 1
        assert not FakeM9Engine.instances[0].closed
        assert not FakeDepthEngine.instances[0].closed
        assert not FakeBokehEngine.instances[0].closed
    finally:
        engine.close()
    assert FakeM9Engine.instances[0].closed
    assert FakeDepthEngine.instances[0].closed
    assert FakeBokehEngine.instances[0].closed
    with Image.open(output_b) as image:
        assert image.size == (60, 40)


def test_cancellation_removes_temporary_files_and_does_not_publish_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "cancelled.png"
    temp_root = tmp_path / "temporary"
    temp_root.mkdir()
    _source(source)
    cancellation = threading.Event()

    def cancel_on_depth(event: Any) -> None:
        if event.stage == "depth":
            cancellation.set()

    with _engine(mode="bokeh", temp_root=temp_root) as engine:
        with pytest.raises(ProcessingCancelled):
            engine.process_file(
                source,
                output,
                options=ProcessingOptions(mode="bokeh"),
                progress_callback=cancel_on_depth,
                cancel_event=cancellation,
            )

    assert not output.exists()
    assert list(temp_root.iterdir()) == []


def test_unified_engine_runs_sam_vitmatte_and_protected_bokeh_in_one_request(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "protected.png"
    _source(source)
    engine = _engine(mode="bokeh")
    engine.sam_model_dir = tmp_path / "sam-onnx"
    engine.vitmatte_model_path = tmp_path / "vitmatte.onnx"
    options = ProcessingOptions(
        mode="bokeh",
        subject_points=((0.5, 0.5), (0.6, 0.55)),
        matte_tile_size=512,
    )

    with engine:
        result = engine.process_file(source, output, options=options)

    with Image.open(output) as image:
        assert image.size == (80, 64)
        output_rgb = np.asarray(image.convert("RGB"))
    assert set(result.models) == {"depth", "bokeh", "sam", "vitmatte"}
    assert result.models["sam"]["embedding_cache_hits"] == 1
    assert result.models["vitmatte"]["onnx_sha256"] == "vitmatte-test-hash"
    assert result.stage_seconds["bokeh.subject_background_plate"] >= 0
    assert result.stage_seconds["bokeh.subject_composite"] >= 0
    assert FakeSamEngine.instances[0].predict_calls == 2
    assert FakeSamEngine.instances[0].closed
    assert FakeMatteEngine.instances[0].inference_calls >= 1
    assert FakeMatteEngine.instances[0].closed
    assert np.any(output_rgb != np.asarray(Image.open(source).convert("RGB")))


def test_subject_points_require_bokeh_and_both_local_models(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "result.png"
    _source(source)
    engine = _engine(mode="color")
    with pytest.raises(ProcessingError, match="requires a bokeh mode"):
        engine.process_file(
            source,
            output,
            options=ProcessingOptions(mode="color", subject_points=((0.5, 0.5),)),
        )
    with pytest.raises(ProcessingError, match="SAM and ViTMatte"):
        _engine(mode="bokeh").process_file(
            source,
            output,
            options=ProcessingOptions(mode="bokeh", subject_points=((0.5, 0.5),)),
        )
    assert not FakeDepthEngine.instances


def test_engine_requires_only_the_models_used_by_selected_mode(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _source(source)
    engine = LaikaProcessingEngine(strategy="low-memory")
    with pytest.raises(ProcessingError, match="M9 model"):
        engine.process_file(
            source, tmp_path / "color.png", options=ProcessingOptions(mode="color")
        )
    with pytest.raises(ProcessingError, match="Depth Anything and ARNet"):
        engine.process_file(
            source, tmp_path / "bokeh.png", options=ProcessingOptions(mode="bokeh")
        )


def test_processing_options_reject_implicit_upscale_and_invalid_controls() -> None:
    with pytest.raises(ValueError, match="aperture"):
        ProcessingOptions(aperture=3.2)
    with pytest.raises(ValueError, match="focus"):
        ProcessingOptions(focus_x=1.1)
    with pytest.raises(ValueError, match="row chunk"):
        ProcessingOptions(bokeh_row_chunk_height=0)
