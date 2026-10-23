from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from laika import highlights
from laika.bokeh import (
    BOKEH_SOURCE_CHECKPOINT_SHA256,
    BokehInferenceError,
    BokehMeARNet,
    BokehTileProgress,
    _blend_resized_nchw_inplace,
    _mask_coc_inplace,
    _resize_nchw_to_rgb,
    _smooth,
    focus_at_point,
    full_frame_coc,
    normalize_depth_map,
    render_bokeh,
    resize_rgb_cubic,
)


class FakeSession:
    def __init__(self, *, input_names: tuple[str, str] = ("image", "defocus")) -> None:
        self.inputs = [
            SimpleNamespace(
                name=input_names[0],
                type="tensor(float)",
                shape=[1, 3, "height", "width"],
            ),
            SimpleNamespace(
                name=input_names[1],
                type="tensor(float)",
                shape=[1, 1, "height", "width"],
            ),
        ]
        self.outputs = [
            SimpleNamespace(
                name="output",
                type="tensor(float)",
                shape=["batch", "channels", "height", "width"],
            )
        ]
        self.calls: list[tuple[list[str], dict[str, np.ndarray]]] = []

    def get_inputs(self) -> list[Any]:
        return self.inputs

    def get_outputs(self) -> list[Any]:
        return self.outputs

    def get_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]

    def run(self, names: list[str], values: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.calls.append((names, values))
        return [values["image"].copy() + np.float32(0.1)]


def _install_fake_onnxruntime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            __version__="test",
            SessionOptions=SimpleNamespace,
            ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
            GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
        ),
    )


def _model_file(tmp_path: Path) -> tuple[Path, str]:
    model = tmp_path / "arnet.onnx"
    model.write_bytes(b"model fixture")
    return model, hashlib.sha256(model.read_bytes()).hexdigest()


def test_bokeh_adapter_preserves_rgb_and_defocus_tensor_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_onnxruntime(monkeypatch)
    model, digest = _model_file(tmp_path)
    session = FakeSession()
    created: dict[str, Any] = {}

    def factory(path: str, **kwargs: Any) -> FakeSession:
        created.update(kwargs)
        assert path == str(model)
        return session

    rgb = np.zeros((3, 5, 3), dtype=np.uint8)
    rgb[:, :, 0] = 255
    rgb[:, :, 1] = 128
    defocus = np.linspace(-1, 1, 15, dtype=np.float32).reshape(3, 5)
    with BokehMeARNet(
        model,
        expected_sha256=digest,
        intra_op_num_threads=1,
        session_factory=factory,
    ) as engine:
        prediction = engine.infer(rgb, defocus)

    assert prediction.rgb_nchw.shape == (1, 3, 3, 5)
    assert prediction.rgb_nchw.dtype == np.float32
    assert prediction.rgb_nchw.flags.c_contiguous
    assert session.calls[0][0] == ["output"]
    values = session.calls[0][1]
    np.testing.assert_allclose(
        values["image"][0, :, 0, 0],
        np.asarray([1.0, 128 / 255, 0.0], dtype=np.float32),
        rtol=0,
        atol=1e-7,
    )
    assert values["defocus"].shape == (1, 1, 3, 5)
    np.testing.assert_array_equal(values["defocus"].reshape(3, 5), defocus)
    np.testing.assert_allclose(prediction.rgb_nchw[:, 0], 1.0)
    assert created["providers"] == ["CPUExecutionProvider"]
    assert engine.runtime_info.provider == "CPUExecutionProvider"


def test_bokeh_adapter_accepts_float_rgb_without_changing_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_onnxruntime(monkeypatch)
    model, digest = _model_file(tmp_path)
    session = FakeSession()
    rgb = np.zeros((2, 3, 3), dtype=np.float32)
    rgb[:, :, 2] = 0.75
    with BokehMeARNet(
        model,
        expected_sha256=digest,
        session_factory=lambda *_args, **_kwargs: session,
    ) as engine:
        prediction = engine.infer(rgb, np.zeros((2, 3), dtype=np.float32))

    np.testing.assert_allclose(session.calls[0][1]["image"][0, 2], 0.75)
    np.testing.assert_allclose(prediction.rgb_nchw[0, 2], 0.85, atol=1e-7)


def test_low_memory_arnet_tiles_even_images_and_matches_full_fake_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_onnxruntime(monkeypatch)
    model, digest = _model_file(tmp_path)
    rng = np.random.default_rng(42)
    rgb = rng.random((130, 150, 3), dtype=np.float32) * np.float32(0.7)
    defocus = rng.uniform(-1.0, 1.0, size=(130, 150)).astype(np.float32)
    full_engine = BokehMeARNet(
        model,
        strategy="balanced",
        expected_sha256=digest,
        session_factory=lambda *_args, **_kwargs: FakeSession(),
    )
    tiled_session = FakeSession()
    tiled_engine = BokehMeARNet(
        model,
        strategy="low-memory",
        inference_tile_edge=64,
        full_inference_pixel_limit=100,
        expected_sha256=digest,
        session_factory=lambda *_args, **_kwargs: tiled_session,
    )

    full = full_engine.infer(rgb, defocus)
    tiled = tiled_engine.infer(rgb, defocus)

    np.testing.assert_array_equal(tiled.rgb_nchw, full.rgb_nchw)
    assert not full.tiled
    assert tiled.tiled
    assert tiled.tile_count == 9
    assert len(tiled_session.calls) == tiled.tile_count
    assert tiled_engine.runtime_info.inference_tile_edge == 64

    direct_engine = BokehMeARNet(
        model,
        strategy="low-memory",
        inference_tile_edge=64,
        full_inference_pixel_limit=100,
        expected_sha256=digest,
        session_factory=lambda *_args, **_kwargs: FakeSession(),
    )
    direct_rgb = np.empty_like(rgb)
    tile_events: list[BokehTileProgress] = []
    direct = direct_engine.infer_into_rgb(
        rgb, defocus, direct_rgb, tile_callback=tile_events.append
    )

    np.testing.assert_array_equal(direct_rgb, full.rgb_nchw[0].transpose(1, 2, 0))
    assert direct.rgb_nchw is None
    assert direct.tiled
    assert direct.tile_count == tiled.tile_count
    assert len(tile_events) == direct.tile_count * 2
    assert [event.phase for event in tile_events] == [
        phase for _ in range(direct.tile_count) for phase in ("started", "complete")
    ]
    assert [event.index for event in tile_events[::2]] == list(
        range(1, direct.tile_count + 1)
    )
    assert all(event.count == direct.tile_count for event in tile_events)
    assert all(
        (event.frame_width, event.frame_height) == (150, 130) for event in tile_events
    )


def test_low_memory_arnet_default_tiles_frames_above_one_megapixel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_onnxruntime(monkeypatch)
    model, digest = _model_file(tmp_path)
    engine = BokehMeARNet(
        model,
        strategy="low-memory",
        expected_sha256=digest,
        session_factory=lambda *_args, **_kwargs: FakeSession(),
    )

    assert engine.runtime_info.inference_tile_edge == 256
    assert engine.runtime_info.full_inference_pixel_limit == 1_000_000
    assert not engine.will_tile(1000, 1000)
    assert engine.will_tile(1024, 1024)


def test_balanced_arnet_keeps_512_pixel_tiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_onnxruntime(monkeypatch)
    model, digest = _model_file(tmp_path)
    engine = BokehMeARNet(
        model,
        strategy="balanced",
        expected_sha256=digest,
        session_factory=lambda *_args, **_kwargs: FakeSession(),
    )

    assert engine.runtime_info.inference_tile_edge == 512


def test_low_memory_arnet_keeps_odd_dimensions_on_validated_full_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_onnxruntime(monkeypatch)
    model, digest = _model_file(tmp_path)
    session = FakeSession()
    engine = BokehMeARNet(
        model,
        strategy="low-memory",
        inference_tile_edge=64,
        full_inference_pixel_limit=1,
        expected_sha256=digest,
        session_factory=lambda *_args, **_kwargs: session,
    )
    rgb = np.zeros((130, 151, 3), dtype=np.float32)
    defocus = np.zeros((130, 151), dtype=np.float32)

    prediction = engine.infer(rgb, defocus)

    assert not prediction.tiled
    assert prediction.tile_count == 1
    assert len(session.calls) == 1


def test_depth_normalization_and_point_focus_match_reference_percentiles() -> None:
    rows = np.linspace(0.0, 99.0, 100, dtype=np.float32)[:, None]
    raw = np.broadcast_to(rows, (100, 100)).copy()
    normalized = normalize_depth_map(raw)
    assert normalized.dtype == np.float32
    assert normalized.flags.c_contiguous
    assert normalized.min() == 0.0
    assert normalized.max() == 1.0
    assert focus_at_point(normalized, 0.5, 0.5) == pytest.approx(
        float(normalized[50, 50]), abs=1e-6
    )

    coc = full_frame_coc(normalized, 0.5, 2.8)
    assert coc.shape == raw.shape
    assert coc.dtype == np.float32
    assert coc[0, 0] < 0 < coc[-1, -1]


def test_subject_coc_row_chunks_match_full_expression_without_mutating_alpha() -> None:
    rng = np.random.default_rng(14)
    original_coc = rng.uniform(-5.0, 5.0, size=(23, 29)).astype(np.float32)
    alpha = rng.random((23, 29), dtype=np.float32)
    expected = original_coc * (np.float32(1.0) - alpha)
    actual = original_coc.copy()
    alpha_before = alpha.copy()

    _mask_coc_inplace(actual, alpha, row_chunk_height=7)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(alpha, alpha_before)


def test_channelwise_multiscale_blend_is_pixel_identical_to_hwc_baseline() -> None:
    import cv2

    rng = np.random.default_rng(105)
    height, width = 50, 70
    nchw = rng.random((1, 3, 25, 35), dtype=np.float32)
    plate_coc = rng.uniform(-16.0, 16.0, (height, width)).astype(np.float32)
    baseline = rng.random((height, width, 3), dtype=np.float32)
    actual = baseline.copy()
    previous_scale = 1.0
    row_chunk_height = 13

    enlarged = _resize_nchw_to_rgb(nchw, width, height, cv2)
    for row_start in range(0, height, row_chunk_height):
        row_stop = min(height, row_start + row_chunk_height)
        weight = _smooth(
            (np.abs(plate_coc[row_start:row_stop]) - np.float32(5.0 * previous_scale))
            / np.float32(3.0 * previous_scale)
        )[:, :, None]
        current = baseline[row_start:row_stop]
        current *= 1.0 - weight
        current += enlarged[row_start:row_stop] * weight

    channel_buffer = np.empty((height, width), dtype=np.float32)
    _blend_resized_nchw_inplace(
        actual,
        nchw,
        plate_coc,
        previous_scale,
        row_chunk_height,
        cv2,
        channel_buffer,
    )

    np.testing.assert_array_equal(actual, baseline)


def test_bokeh_multiscale_renderer_uses_model_and_preserves_full_dimensions() -> None:
    class DarkenModel:
        strategy = "low-memory"

        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []
            self.direct_calls: list[tuple[int, int]] = []

        def infer(self, rgb: np.ndarray, defocus: np.ndarray) -> Any:
            height, width = rgb.shape[:2]
            self.calls.append((width, height))
            assert defocus.shape == (height, width)
            return SimpleNamespace(
                rgb_nchw=np.zeros((1, 3, height, width), dtype=np.float32)
            )

        def infer_into_rgb(
            self, rgb: np.ndarray, defocus: np.ndarray, output: np.ndarray
        ) -> Any:
            height, width = rgb.shape[:2]
            self.direct_calls.append((width, height))
            assert defocus.shape == (height, width)
            output.fill(0.0)
            return SimpleNamespace(width=width, height=height)

        def will_tile(self, width: int, height: int) -> bool:
            return width == 700 and height == 500

    rgb = np.full((500, 700, 3), 180, dtype=np.uint8)
    y = np.linspace(0.0, 1.0, 500, dtype=np.float32)[:, None]
    depth = np.broadcast_to(y, (500, 700)).copy()
    model = DarkenModel()
    completed_passes: list[tuple[int, int, float]] = []
    result = render_bokeh(
        rgb,
        depth,
        0.5,
        0.95,
        model,
        row_chunk_height=13,
        pass_callback=lambda index, total, scale: completed_passes.append(
            (index, total, scale)
        ),
    )

    assert result.rgb.shape == rgb.shape
    assert result.rgb.dtype == np.uint8
    assert result.scales[0] == 1.0
    assert len(result.scales) == len(model.calls) + len(model.direct_calls) >= 2
    assert [item[0] for item in completed_passes] == list(
        range(1, len(result.scales) + 1)
    )
    assert all(item[1] == len(result.scales) for item in completed_passes)
    assert model.direct_calls == [(700, 500)]
    assert np.all(result.rgb <= rgb)
    assert np.any(result.rgb < rgb)
    assert result.total_seconds >= 0
    assert "optics" in result.stage_seconds
    assert "composite" in result.stage_seconds
    assert "highlight_extraction" in result.stage_seconds
    assert "highlight_radiance" in result.stage_seconds
    assert "highlight_scatter" in result.stage_seconds


def test_bokeh_renderer_integrates_highlight_scatter_after_arnet() -> None:
    class IdentityModel:
        def infer(self, rgb: np.ndarray, defocus: np.ndarray) -> Any:
            height, width = rgb.shape[:2]
            return SimpleNamespace(
                rgb_nchw=rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
            )

    rgb = np.zeros((192, 192, 3), dtype=np.uint8)
    rgb[94:98, 94:98] = (255, 220, 160)
    depth = np.full((192, 192), 0.8, dtype=np.float32)
    stages: list[tuple[str, float]] = []

    result = render_bokeh(
        rgb,
        depth,
        0.1,
        2.8,
        IdentityModel(),
        row_chunk_height=17,
        stage_callback=lambda name, fraction: stages.append((name, fraction)),
    )

    assert result.rgb.shape == rgb.shape
    assert result.stage_seconds["highlight_extraction"] >= 0.0
    assert result.stage_seconds["highlight_radiance"] >= 0.0
    assert result.stage_seconds["highlight_scatter"] > 0.0
    assert "highlight-scatter" in {name for name, _ in stages}
    assert result.rgb[94:98, 94:98].max() > 0
    outside = result.rgb[:, :, 0].copy()
    outside[94:98, 94:98] = 0
    assert np.count_nonzero(outside) > 0


def test_balanced_bokeh_keeps_eager_radiance_and_matches_low_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deferred_calls = 0
    eager_builder = highlights.extract_highlight_radiance

    def count_deferred(*args: Any, **kwargs: Any) -> Any:
        nonlocal deferred_calls
        deferred_calls += 1
        return eager_builder(*args, **kwargs)

    monkeypatch.setattr(highlights, "extract_highlight_radiance", count_deferred)

    class IdentityModel:
        def __init__(self, strategy: str) -> None:
            self.strategy = strategy

        def infer(self, rgb: np.ndarray, defocus: np.ndarray) -> Any:
            height, width = rgb.shape[:2]
            return SimpleNamespace(
                rgb_nchw=rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
            )

    rgb = np.zeros((192, 192, 3), dtype=np.uint8)
    rgb[94:98, 94:98] = (255, 220, 160)
    depth = np.full((192, 192), 0.8, dtype=np.float32)

    low_memory = render_bokeh(
        rgb,
        depth,
        0.1,
        2.8,
        IdentityModel("low-memory"),
        row_chunk_height=17,
    )
    balanced = render_bokeh(
        rgb,
        depth,
        0.1,
        2.8,
        IdentityModel("balanced"),
        row_chunk_height=17,
    )

    assert deferred_calls == 1
    np.testing.assert_array_equal(low_memory.rgb, balanced.rgb)
    assert low_memory.stage_seconds["highlight_radiance"] > 0.0
    assert balanced.stage_seconds["highlight_radiance"] == 0.0


def test_explicit_rgb_resize_uses_opencv_cubic_and_keeps_rgb_order() -> None:
    import cv2

    rgb = np.zeros((7, 11, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.arange(11, dtype=np.uint8)[None, :] * 17
    rgb[:, :, 1] = np.arange(7, dtype=np.uint8)[:, None] * 23
    rgb[:, :, 2] = 91
    actual = resize_rgb_cubic(rgb, 5, 3)
    expected = cv2.resize(rgb, (5, 3), interpolation=cv2.INTER_CUBIC)
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (3, 5, 3)
    assert actual.dtype == np.uint8
    assert resize_rgb_cubic(rgb, 11, 7) is rgb


def test_bokeh_adapter_rejects_wrong_hash_and_tensor_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_onnxruntime(monkeypatch)
    model, digest = _model_file(tmp_path)

    def factory(*_args: Any, **_kwargs: Any) -> FakeSession:
        return FakeSession()

    with pytest.raises(BokehInferenceError, match="SHA-256 mismatch"):
        BokehMeARNet(model, expected_sha256="0" * 64, session_factory=factory)
    with pytest.raises(BokehInferenceError, match="inputs named"):
        BokehMeARNet(
            model,
            expected_sha256=digest,
            session_factory=lambda *_args, **_kwargs: FakeSession(
                input_names=("bad", "defocus")
            ),
        )


def test_bokeh_adapter_binds_onnx_hash_to_export_validation_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_onnxruntime(monkeypatch)
    model, digest = _model_file(tmp_path)
    Path(f"{model}.validation.json").write_text(
        json.dumps(
            {
                "model_id": "bokehme-arnet-fp16",
                "checkpoint_sha256": BOKEH_SOURCE_CHECKPOINT_SHA256,
                "onnx_sha256": digest,
                "dynamic_shapes": True,
                "passed": True,
            }
        ),
        encoding="utf-8",
    )
    with BokehMeARNet(
        model,
        session_factory=lambda *_args, **_kwargs: FakeSession(),
    ) as engine:
        assert engine.model_sha256 == digest

    model.write_bytes(b"modified model")
    with pytest.raises(BokehInferenceError, match="SHA-256 mismatch"):
        BokehMeARNet(
            model,
            session_factory=lambda *_args, **_kwargs: FakeSession(),
        )


@pytest.mark.parametrize(
    ("image", "defocus", "message"),
    [
        (np.zeros((3, 5), dtype=np.uint8), np.zeros((3, 5), dtype=np.float32), "HWC"),
        (
            np.zeros((3, 5, 3), dtype=np.float32),
            np.zeros((5, 3), dtype=np.float32),
            "matching rgb",
        ),
        (
            np.zeros((3, 5, 3), dtype=np.float32),
            np.full((3, 5), 1.1, dtype=np.float32),
            r"\[-1.0, 1.0\]",
        ),
        (
            np.full((3, 5, 3), np.nan, dtype=np.float32),
            np.zeros((3, 5), dtype=np.float32),
            "finite",
        ),
    ],
)
def test_bokeh_adapter_rejects_invalid_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    image: np.ndarray,
    defocus: np.ndarray,
    message: str,
) -> None:
    _install_fake_onnxruntime(monkeypatch)
    model, digest = _model_file(tmp_path)
    with BokehMeARNet(
        model,
        expected_sha256=digest,
        session_factory=lambda *_args, **_kwargs: FakeSession(),
    ) as engine:
        with pytest.raises(ValueError, match=message):
            engine.infer(image, defocus)
