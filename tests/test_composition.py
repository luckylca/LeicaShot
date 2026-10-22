from __future__ import annotations

import builtins
import hashlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from laika.composition import (
    GAIC_MODEL_ID,
    GAIC_SOURCE_CHECKPOINT_SHA256,
    GAIC_SOURCE_CODE_REVISION,
    CompositionCrop,
    CompositionInferenceError,
    GaicCompositionOnnx,
    analyze_composition_rgb,
    apply_crop_rgb,
    crop_pixel_box,
    enumerate_candidate_crops,
    prepare_gaic_analysis_rgb,
    prepare_gaic_tensors,
    select_composition_suggestions,
    snap_crop,
)
from laika.engine import ImageInputError
from laika.processing import (
    LaikaProcessingEngine,
    ProcessingCancelled,
    ProcessingError,
)


class FakeSession:
    def __init__(self) -> None:
        self.last_feed: dict[str, np.ndarray] | None = None
        self.closed = False

    def get_inputs(self) -> list[Any]:
        return [
            SimpleNamespace(name="image", type="tensor(float)", shape=[1, 3, "h", "w"]),
            SimpleNamespace(name="boxes", type="tensor(float)", shape=["n", 4]),
        ]

    def get_outputs(self) -> list[Any]:
        return [SimpleNamespace(name="scores", type="tensor(float)", shape=["n"])]

    def get_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]

    def run(
        self,
        _outputs: list[str],
        feed: dict[str, np.ndarray],
        *,
        run_options: Any,
    ) -> list[np.ndarray]:
        self.last_feed = feed
        self.run_options = run_options
        count = feed["boxes"].shape[0]
        scores = np.linspace(0.0, 1.0, count, dtype=np.float32)
        scores[0] = 0.0
        return [scores]


def _model_file(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "gaic.onnx"
    path.write_bytes(b"test graph bytes")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_model(tmp_path: Path) -> tuple[GaicCompositionOnnx, FakeSession]:
    path, digest = _model_file(tmp_path)
    session = FakeSession()
    model = GaicCompositionOnnx(
        path,
        model_sha256=digest,
        session_factory=lambda *_args, **_kwargs: session,
    )
    return model, session


def test_injected_session_does_not_require_optional_onnxruntime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_import = builtins.__import__

    def import_without_onnxruntime(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "onnxruntime":
            raise ModuleNotFoundError("onnxruntime is intentionally unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_onnxruntime)
    model, _session = _fake_model(tmp_path)

    assert model.runtime_version == "injected"


def test_crop_coordinates_snap_to_reference_pixel_rounding() -> None:
    crop = CompositionCrop(0.12, 0.2, 0.81, 0.91)

    assert crop_pixel_box(crop, 10, 10) == (1, 2, 8, 9)
    assert snap_crop(crop, 10, 10) == CompositionCrop(0.1, 0.2, 0.8, 0.9)
    assert crop_pixel_box(CompositionCrop(0.0, 0.0, 1.0, 1.0), 10, 7) == (
        0,
        0,
        10,
        7,
    )


def test_analysis_resize_uses_deterministic_bilinear_reference_kernel() -> None:
    import cv2

    y, x = np.mgrid[:1500, :800]
    rgb = np.stack((x % 256, y % 256, (x + y) % 256), axis=2).astype(np.uint8)
    actual = prepare_gaic_analysis_rgb(rgb)
    expected = cv2.resize(
        rgb,
        (651, 1220),
        interpolation=cv2.INTER_LINEAR_EXACT,
    )

    assert actual.shape == (1220, 651, 3)
    np.testing.assert_array_equal(actual, expected)


def test_candidate_grid_is_deterministic_and_respects_target_edge() -> None:
    first = enumerate_candidate_crops(1220, 813)
    second = enumerate_candidate_crops(1220, 813)
    constrained = enumerate_candidate_crops(1220, 813, target_long_edge=900)

    assert first == second
    assert first[0] == CompositionCrop(0.0, 0.0, 1.0, 1.0)
    assert len(first) > 400
    assert len(constrained) < len(first)
    assert all(
        max(
            crop_pixel_box(crop, 1220, 813)[2] - crop_pixel_box(crop, 1220, 813)[0],
            crop_pixel_box(crop, 1220, 813)[3] - crop_pixel_box(crop, 1220, 813)[1],
        )
        >= 900
        for crop in constrained
    )


def test_tensor_pack_uses_rgb_imagenet_normalization_and_replicates_padding() -> None:
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[0, 0] = (255, 128, 0)
    full = CompositionCrop(0.0, 0.0, 1.0, 1.0)

    tensor, boxes = prepare_gaic_tensors(image, (full,))

    assert tensor.shape == (1, 3, 32, 32)
    assert tensor.dtype == np.float32
    np.testing.assert_allclose(
        tensor[0, :, 0, 0],
        (
            (1.0 - 0.485) / 0.229,
            (128 / 255 - 0.456) / 0.224,
            (0 - 0.406) / 0.225,
        ),
        atol=1e-6,
    )
    np.testing.assert_array_equal(tensor[0, :, 1, 3], tensor[0, :, 1, 2])
    np.testing.assert_array_equal(boxes, np.array([[0, 0, 3, 2]], dtype=np.float32))


def test_suggestion_selection_keeps_full_frame_and_filters_duplicates() -> None:
    full = CompositionCrop(0, 0, 1, 1)
    crops = (
        full,
        CompositionCrop(0, 0, 0.9, 0.9),
        CompositionCrop(0.01, 0.01, 0.91, 0.91),
        CompositionCrop(0.4, 0.4, 0.8, 0.8),
        CompositionCrop(0.1, 0.1, 0.8, 0.8),
    )

    suggestions = select_composition_suggestions(crops, [1, 3, 2.9, 2, 1])

    assert suggestions[0].crop == crops[1]
    assert suggestions[-1].crop == full
    assert all(item.improvement_over_original >= 0 for item in suggestions)
    with pytest.raises(CompositionInferenceError, match="non-finite"):
        select_composition_suggestions(crops[:2], [1, np.nan])


def test_onnx_adapter_checks_hash_graph_and_returns_candidate_scores(
    tmp_path: Path,
) -> None:
    model, session = _fake_model(tmp_path)
    rgb = np.full((64, 96, 3), 127, dtype=np.uint8)
    crops = enumerate_candidate_crops(96, 64)[:7]

    scores = model.score(rgb, crops)

    assert scores.shape == (7,)
    assert session.last_feed is not None
    assert session.last_feed["image"].shape == (1, 3, 64, 96)
    assert session.last_feed["boxes"].shape == (7, 4)
    model.close()
    with pytest.raises(CompositionInferenceError, match="closed"):
        model.score(rgb, crops)

    with pytest.raises(CompositionInferenceError, match="SHA-256"):
        GaicCompositionOnnx(
            model.model_path,
            model_sha256="0" * 64,
            session_factory=lambda *_args, **_kwargs: FakeSession(),
        )


def test_onnx_adapter_terminates_active_inference_on_cancellation(
    tmp_path: Path,
) -> None:
    path, digest = _model_file(tmp_path)
    entered = threading.Event()

    class BlockingSession(FakeSession):
        def run(
            self,
            _outputs: list[str],
            feed: dict[str, np.ndarray],
            *,
            run_options: Any,
        ) -> list[np.ndarray]:
            self.last_feed = feed
            entered.set()
            deadline = threading.Event()
            for _ in range(250):
                if run_options.terminate:
                    raise RuntimeError("inference terminated")
                deadline.wait(0.01)
            raise AssertionError("cancellation watcher did not terminate inference")

    model = GaicCompositionOnnx(
        path,
        model_sha256=digest,
        session_factory=lambda *_args, **_kwargs: BlockingSession(),
    )
    rgb = np.full((64, 96, 3), 127, dtype=np.uint8)
    crops = enumerate_candidate_crops(96, 64)[:2]
    cancel_event = threading.Event()
    errors: list[BaseException] = []

    def check_cancelled() -> None:
        if cancel_event.is_set():
            raise RuntimeError("request cancelled")

    def infer() -> None:
        try:
            model.score(
                rgb,
                crops,
                cancel_check=check_cancelled,
                cancel_event=cancel_event,
            )
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=infer)
    worker.start()
    assert entered.wait(timeout=2)
    cancel_event.set()
    worker.join(timeout=3)

    assert not worker.is_alive()
    assert errors
    assert isinstance(errors[0], RuntimeError)
    assert str(errors[0]) == "request cancelled"
    model.close()


def test_validation_report_requires_pinned_source_and_passed_export(
    tmp_path: Path,
) -> None:
    model_path, digest = _model_file(tmp_path)
    report = {
        "model_id": GAIC_MODEL_ID,
        "source_checkpoint_sha256": GAIC_SOURCE_CHECKPOINT_SHA256,
        "source_code_revision": GAIC_SOURCE_CODE_REVISION,
        "passed": True,
        "dynamic_shapes": True,
        "input_names": ["image", "boxes"],
        "output_name": "scores",
        "onnx_sha256": digest,
    }
    Path(f"{model_path}.validation.json").write_text(json.dumps(report))
    session = FakeSession()
    model = GaicCompositionOnnx.from_validation_report(
        model_path, session_factory=lambda *_args, **_kwargs: session
    )
    model.close()

    report["source_checkpoint_sha256"] = "0" * 64
    Path(f"{model_path}.validation.json").write_text(json.dumps(report))
    with pytest.raises(CompositionInferenceError, match="does not match"):
        GaicCompositionOnnx.from_validation_report(model_path)

    with pytest.raises(CompositionInferenceError, match="SHA-256"):
        GaicCompositionOnnx(
            model_path,
            model_sha256="0" * 64,
            session_factory=lambda *_args, **_kwargs: FakeSession(),
        )


def test_analyze_composition_resizes_scores_and_returns_full_frame(
    tmp_path: Path,
) -> None:
    model, _session = _fake_model(tmp_path)
    rgb = np.full((1400, 900, 3), 100, dtype=np.uint8)
    events: list[str] = []

    result = analyze_composition_rgb(
        rgb, model, cancel_check=lambda: events.append("checked")
    )

    assert (result.width, result.height) == (900, 1400)
    assert max(result.analysis_width, result.analysis_height) == 1220
    assert result.candidate_count > 400
    assert result.suggestions[-1].crop == CompositionCrop(0, 0, 1, 1)
    assert "model_inference" in result.stage_seconds
    assert result.total_seconds >= sum(result.stage_seconds.values())
    assert events


def test_analysis_boxes_use_original_source_pixel_snapping(tmp_path: Path) -> None:
    model, session = _fake_model(tmp_path)
    rgb = np.full((1620, 1080, 3), 100, dtype=np.uint8)
    analysis = prepare_gaic_analysis_rgb(rgb)

    result = analyze_composition_rgb(rgb, model)

    source_crops = enumerate_candidate_crops(1080, 1620)
    _image_tensor, expected_boxes = prepare_gaic_tensors(analysis, source_crops)
    assert result.candidate_count == len(source_crops) == 556
    assert session.last_feed is not None
    np.testing.assert_array_equal(session.last_feed["boxes"], expected_boxes)


def test_crop_apply_preserves_pixels_and_returns_compact_copy() -> None:
    rgb = np.arange(8 * 6 * 3, dtype=np.uint8).reshape(6, 8, 3)
    crop = CompositionCrop(0.25, 1 / 6, 0.75, 5 / 6)

    cropped = apply_crop_rgb(rgb, crop)

    np.testing.assert_array_equal(cropped, rgb[1:5, 2:6])
    assert cropped.shape == (4, 4, 3)
    assert cropped.flags.c_contiguous
    assert not np.shares_memory(cropped, rgb)


def test_processing_engine_composition_cache_cancel_and_crop_only(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    cropped_path = tmp_path / "crop.png"
    Image.new("RGB", (24, 16), (120, 40, 10)).save(source)
    made: list[Any] = []

    def factory(*_args: Any, **_kwargs: Any) -> Any:
        class Model:
            model_sha256 = "test-hash"
            runtime_version = "test-runtime"
            provider = "CPUExecutionProvider"

            def score(
                self,
                rgb: np.ndarray,
                crops: Any,
                *,
                cancel_check=None,
                cancel_event=None,
            ) -> Any:
                if cancel_check is not None:
                    cancel_check()
                if cancel_event is not None and cancel_event.is_set():
                    raise ProcessingCancelled("cancelled")
                return np.linspace(0, 1, len(crops), dtype=np.float32)

            def close(self) -> None:
                self.closed = True

        model = Model()
        made.append(model)
        return model

    engine = LaikaProcessingEngine(
        composition_model_path="local-gaic.onnx",
        strategy="balanced",
        composition_factory=factory,
    )
    first = engine.suggest_composition(source)
    second = engine.suggest_composition(source)
    assert first.candidate_count == second.candidate_count
    assert len(made) == 1

    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(ProcessingCancelled):
        engine.suggest_composition(source, cancel_event=cancelled)

    before = len(made)
    crop_result = engine.apply_crop_file(
        source,
        cropped_path,
        CompositionCrop(0.25, 0.25, 0.75, 0.75),
    )
    assert len(made) == before
    assert (crop_result.width, crop_result.height) == (12, 8)
    with Image.open(cropped_path) as image:
        assert image.size == (12, 8)
        assert image.getpixel((0, 0)) == (120, 40, 10)
    engine.close()
    assert made[0].closed


def test_composition_requires_model_and_crop_rejects_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (8, 6)).save(source)
    engine = LaikaProcessingEngine()
    with pytest.raises(ProcessingError, match="GAIC model"):
        engine.suggest_composition(source)
    with pytest.raises(ImageInputError, match="different"):
        engine.apply_crop_file(source, source, CompositionCrop(0, 0, 1, 1))
