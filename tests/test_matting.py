from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from laika.matting import (
    MatteInferenceError,
    MatteTileConfig,
    ViTMatteOnnx,
    plan_matte_tiles,
    prepare_vitmatte_input,
    refine_matte_tiled,
)


def make_input(width: int, height: int, *, unknown_band: bool = True) -> np.ndarray:
    values = np.empty((1, 4, height, width), dtype=np.float32)
    y, x = np.mgrid[:height, :width]
    values[0, 0] = (x % 251) / 125.0 - 1.0
    values[0, 1] = (y % 239) / 119.0 - 1.0
    values[0, 2] = ((x + y) % 233) / 116.0 - 1.0
    trimap = np.zeros((height, width), dtype=np.float32)
    trimap[:, width // 3 : 2 * width // 3] = 1.0
    if unknown_band:
        trimap[:, width // 3 - 16 : width // 3 + 16] = np.float32(128 / 255)
        trimap[:, 2 * width // 3 - 16 : 2 * width // 3 + 16] = np.float32(128 / 255)
        trimap[height // 3 - 16 : height // 3 + 16, :] = np.float32(128 / 255)
    values[0, 3] = trimap
    return np.ascontiguousarray(values)


def pointwise_matte(values: np.ndarray) -> np.ndarray:
    alpha = (
        0.5
        + values[:, 0:1] * np.float32(0.20)
        + values[:, 1:2] * np.float32(0.17)
        + values[:, 2:3] * np.float32(0.13)
    )
    return np.clip(alpha, 0, 1).astype(np.float32)


@pytest.mark.parametrize("tile_size", [512, 768, 1024])
def test_tile_plan_covers_unknown_pixels_with_aligned_overlap(tile_size: int) -> None:
    values = make_input(1536, 2048)
    config = MatteTileConfig(tile_size=tile_size)
    tiles = plan_matte_tiles(values, config)

    assert tiles
    max_width = max(tile.width for tile in tiles)
    max_height = max(tile.height for tile in tiles)
    assert max(max_width, max_height) <= tile_size
    for tile in tiles:
        assert tile.x % 32 == tile.y % 32 == 0
        assert tile.width % 32 == tile.height % 32 == 0
        assert tile.x + tile.width <= values.shape[3]
        assert tile.y + tile.height <= values.shape[2]

    trimap = values[0, 3]
    covered_unknown = np.zeros(trimap.shape, dtype=bool)
    for tile in tiles:
        covered_unknown[tile.y : tile.y + tile.height, tile.x : tile.x + tile.width] = (
            True
        )
    assert np.all(covered_unknown[(trimap > 0) & (trimap < 1)])


@pytest.mark.parametrize("tile_size", [512, 768, 1024])
def test_tiled_pointwise_alpha_matches_full_frame_and_preserves_trimaps(
    tile_size: int,
) -> None:
    values = make_input(1536, 2048)
    calls: list[tuple[int, int]] = []

    def infer(tile: np.ndarray) -> np.ndarray:
        calls.append(tile.shape[2:])
        return pointwise_matte(tile)

    actual = refine_matte_tiled(
        values, infer, config=MatteTileConfig(tile_size=tile_size)
    )
    expected = pointwise_matte(values)[0, 0]
    trimap = values[0, 3]
    expected[trimap == 0] = 0
    expected[trimap == 1] = 1

    assert len(calls) > 1
    assert max(height * width for height, width in calls) <= tile_size**2
    np.testing.assert_allclose(actual[0, 0], expected, rtol=0, atol=3e-7)
    np.testing.assert_array_equal(actual[0, 0][trimap == 0], 0)
    np.testing.assert_array_equal(actual[0, 0][trimap == 1], 1)


def test_known_only_trimap_skips_model_and_returns_exact_endpoints() -> None:
    values = make_input(640, 512, unknown_band=False)

    def fail_if_called(_tile: np.ndarray) -> np.ndarray:
        pytest.fail("known-only regions must not run the matte model")

    result = refine_matte_tiled(values, fail_if_called)

    np.testing.assert_array_equal(result[0, 0], values[0, 3])


def test_reference_area_threshold_and_tile_only_override() -> None:
    values = make_input(1536, 736)
    reference = plan_matte_tiles(values, MatteTileConfig(tile_size=512))
    forced_tiles = plan_matte_tiles(
        values,
        MatteTileConfig(tile_size=512, single_pass_max_pixels=0),
    )

    assert len(reference) == 1
    assert (reference[0].x, reference[0].y) == (0, 0)
    assert (reference[0].width, reference[0].height) == (1536, 736)
    assert len(forced_tiles) > 1
    assert all(tile.width <= 512 and tile.height <= 512 for tile in forced_tiles)


def test_frame_inside_tile_budget_uses_one_full_model_call() -> None:
    values = make_input(768, 640)
    calls = 0

    def infer(tile: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        assert tile is values
        return pointwise_matte(tile)

    result = refine_matte_tiled(values, infer)
    expected = pointwise_matte(values).copy()
    expected[0, 0][values[0, 3] == 0] = 0
    expected[0, 0][values[0, 3] == 1] = 1

    assert calls == 1
    np.testing.assert_allclose(result, expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_prepare_vitmatte_input_normalizes_rgb_and_pads_bottom_right(dtype) -> None:
    raw = np.full((35, 65, 3), 128 if dtype is np.uint8 else 128 / 255, dtype=dtype)
    trimap = np.full((35, 65), np.float32(128 / 255), dtype=np.float32)

    tensor = prepare_vitmatte_input(raw, trimap)

    assert tensor.shape == (1, 4, 64, 96)
    assert tensor.dtype == np.float32
    assert tensor[0, 0, 0, 0] == pytest.approx(2 * (128 / 255) - 1, abs=1e-7)
    np.testing.assert_array_equal(tensor[0, :3, 35:, :], 0)
    np.testing.assert_array_equal(tensor[0, :3, :, 65:], 0)
    np.testing.assert_array_equal(tensor[0, 3, 35:, :], 0)
    np.testing.assert_array_equal(tensor[0, 3, :, 65:], 0)


def test_prepare_vitmatte_input_rejects_oversized_analysis_instead_of_resizing() -> (
    None
):
    rgb = np.zeros((64, 96, 3), dtype=np.uint8)
    trimap = np.zeros((64, 96), dtype=np.float32)
    with pytest.raises(ValueError, match="analysis long edge"):
        prepare_vitmatte_input(rgb, trimap, max_analysis_long_edge=64)


@pytest.mark.parametrize(
    "values",
    [
        np.zeros((1, 3, 32, 32), dtype=np.float32),
        np.zeros((1, 4, 33, 32), dtype=np.float32),
        np.zeros((1, 4, 32, 32), dtype=np.float64),
        np.zeros((1, 4, 32, 32), dtype=np.float32)[:, :, :, ::2],
    ],
)
def test_tile_planning_rejects_invalid_input(values: np.ndarray) -> None:
    with pytest.raises(ValueError, match="pixel_values|matte dimensions"):
        plan_matte_tiles(values)


def test_tile_config_rejects_overlap_smaller_than_feather_budget() -> None:
    with pytest.raises(ValueError, match="overlap"):
        MatteTileConfig(tile_size=512, feather=80, max_step=384)


class FakeSession:
    def __init__(
        self,
        input_names: tuple[str, ...],
        output_names: tuple[str, ...],
        input_shape: tuple[object, ...] | None = None,
    ):
        self.inputs = [
            SimpleNamespace(name=name, shape=input_shape) for name in input_names
        ]
        self.outputs = [SimpleNamespace(name=name) for name in output_names]
        self.calls: list[dict[str, np.ndarray]] = []

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    def run(self, names: list[str], inputs: dict[str, np.ndarray]):
        assert names == ["alpha"]
        self.calls.append(inputs)
        return [pointwise_matte(inputs["pixel_values"])]


def test_onnx_adapter_uses_hash_checked_dynamic_cpu_graph(tmp_path) -> None:
    model = tmp_path / "local.onnx"
    model.write_bytes(b"synthetic onnx placeholder")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    session = FakeSession(("pixel_values",), ("alpha",), (1, 4, "height", "width"))
    engine = ViTMatteOnnx(
        model,
        model_sha256=digest,
        session_factory=lambda _path, _threads: session,
    )
    tile = make_input(64, 96)

    actual = engine.infer(tile)

    np.testing.assert_array_equal(actual, pointwise_matte(tile))
    assert session.calls[0]["pixel_values"] is tile
    engine.close()
    assert engine._session is None
    with pytest.raises(MatteInferenceError, match="closed"):
        engine.infer(tile)


def test_onnx_adapter_rejects_wrong_graph_names(tmp_path) -> None:
    session = FakeSession(("image",), ("output",))
    model = tmp_path / "local.onnx"
    model.write_bytes(b"synthetic onnx placeholder")
    with pytest.raises(MatteInferenceError, match="pixel_values"):
        ViTMatteOnnx(
            model,
            model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
            session_factory=lambda _path, _threads: session,
        )


def test_onnx_adapter_rejects_hash_mismatch_before_session_creation(tmp_path) -> None:
    model = tmp_path / "local.onnx"
    model.write_bytes(b"synthetic onnx placeholder")
    created = False

    def factory(_path, _threads):
        nonlocal created
        created = True
        return FakeSession(("pixel_values",), ("alpha",))

    with pytest.raises(MatteInferenceError, match="SHA-256"):
        ViTMatteOnnx(model, model_sha256="0" * 64, session_factory=factory)
    assert not created


def test_onnx_adapter_rejects_static_spatial_graph(tmp_path) -> None:
    model = tmp_path / "local.onnx"
    model.write_bytes(b"synthetic onnx placeholder")
    session = FakeSession(("pixel_values",), ("alpha",), (1, 4, 1024, 1024))

    with pytest.raises(MatteInferenceError, match="dynamic height and width"):
        ViTMatteOnnx(
            model,
            model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
            session_factory=lambda _path, _threads: session,
        )


@pytest.mark.parametrize(
    "bad_output",
    [
        np.zeros((1, 1, 32, 64), dtype=np.float32),
        np.zeros((1, 1, 64, 64), dtype=np.float64),
        np.full((1, 1, 64, 32), np.nan, dtype=np.float32),
    ],
)
def test_tile_inference_rejects_wrong_output_contract(bad_output: np.ndarray) -> None:
    values = make_input(64, 64)
    with pytest.raises(MatteInferenceError):
        refine_matte_tiled(values, lambda _tile: bad_output)
