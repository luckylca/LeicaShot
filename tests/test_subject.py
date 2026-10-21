from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from laika.subject import (
    SAM_DECODER_INPUTS,
    SAM_FEATURE_OUTPUTS,
    SAM_FEATURE_SHAPES,
    ImageEmbeddingCache,
    Sam2TinyOnnx,
    _prepare_sam_rgb,
    _resize_mask_bilinear,
    normalized_points_to_sam,
    prepare_sam_input,
    select_subject_mask,
)


class FakeSession:
    def __init__(self, graph: str) -> None:
        self.graph = graph
        self.calls: list[dict[str, np.ndarray]] = []
        if graph == "encoder":
            self.inputs = [SimpleNamespace(name="image")]
            self.outputs = [SimpleNamespace(name=name) for name in SAM_FEATURE_OUTPUTS]
        else:
            self.inputs = [SimpleNamespace(name=name) for name in SAM_DECODER_INPUTS]
            self.outputs = [
                SimpleNamespace(name="masks"),
                SimpleNamespace(name="iou_predictions"),
                SimpleNamespace(name="low_res_masks"),
            ]

    def get_inputs(self) -> list[SimpleNamespace]:
        return self.inputs

    def get_outputs(self) -> list[SimpleNamespace]:
        return self.outputs

    def run(
        self, output_names: list[str], inputs: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        self.calls.append(inputs)
        if self.graph == "encoder":
            features = [
                np.zeros(shape, dtype=np.float32) for shape in SAM_FEATURE_SHAPES
            ]
            return [features[SAM_FEATURE_OUTPUTS.index(name)] for name in output_names]
        outputs = {
            "low_res_masks": np.zeros((1, 3, 256, 256), dtype=np.float32),
            "iou_predictions": np.asarray([[0.7, 0.8, 0.9]], dtype=np.float32),
        }
        return [outputs[name] for name in output_names]


def make_engine(cache_max_bytes: int = 32 * 1024 * 1024):
    sessions: dict[str, FakeSession] = {}

    def factory(path: str, _threads: int) -> FakeSession:
        graph = "encoder" if "encoder" in path else "decoder"
        sessions[graph] = FakeSession(graph)
        return sessions[graph]

    engine = Sam2TinyOnnx(
        "image_encoder.onnx",
        "image_decoder.onnx",
        cache_max_bytes=cache_max_bytes,
        session_factory=factory,
    )
    return engine, sessions


def image_tensor(value: float = 0.0) -> np.ndarray:
    return np.full((1, 3, 1024, 1024), value, dtype=np.float32)


def scalar_reference_axis(image: np.ndarray, output_size: int, axis: int) -> np.ndarray:
    input_size = image.shape[axis]
    ratio = input_size / output_size
    support = max(1.0, ratio)
    output_shape = list(image.shape)
    output_shape[axis] = output_size
    output = np.empty(output_shape, dtype=np.float32)
    for target in range(output_size):
        center = (target + 0.5) * ratio - 0.5
        low = max(0, int(np.ceil(center - support)))
        high = min(input_size - 1, int(np.floor(center + support)))
        taps = np.arange(low, high + 1)
        weights = np.maximum(0.0, 1.0 - np.abs(taps - center) / support)
        weights /= np.sum(weights)
        selected = np.take(image, taps, axis=axis).astype(np.float64)
        weight_shape = [1] * selected.ndim
        weight_shape[axis] = len(weights)
        values = np.sum(
            selected * weights.reshape(weight_shape),
            axis=axis,
        )
        index = [slice(None)] * image.ndim
        index[axis] = target
        output[tuple(index)] = values
    return output


def test_sam_preprocessing_matches_reference_separable_antialias() -> None:
    source = np.arange(3 * 5 * 3, dtype=np.float32).reshape(3, 5, 3) / 50
    expected = scalar_reference_axis(source, 7, axis=1)
    expected = scalar_reference_axis(expected, 7, axis=0)
    expected = expected.transpose(2, 0, 1)[None, ...].copy()
    for channel, (mean, std) in enumerate(
        zip((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), strict=True)
    ):
        expected[0, channel] = (expected[0, channel] - mean) / std

    actual = _prepare_sam_rgb(source, 7)

    assert actual.shape == (1, 3, 7, 7)
    assert actual.dtype == np.float32
    assert actual.flags.c_contiguous
    np.testing.assert_allclose(actual, expected, rtol=0, atol=4e-7)


def test_sam_preprocessing_accepts_rgb_bytes_and_normalizes() -> None:
    image = np.full((2, 5, 3), 128, dtype=np.uint8)

    tensor = prepare_sam_input(image)

    assert tensor.shape == (1, 3, 1024, 1024)
    assert tensor.dtype == np.float32
    assert tensor.flags.c_contiguous
    for channel, (mean, std) in enumerate(
        zip((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), strict=True)
    ):
        assert tensor[0, channel, 500, 500] == pytest.approx(
            (128 / 255 - mean) / std, abs=2e-6
        )


def test_sam_preprocessing_rejects_full_rasters_above_analysis_budget() -> None:
    image = np.zeros((1, 2049, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="long edge at most 2048"):
        prepare_sam_input(image)


def test_normalized_prompt_coordinates_map_to_model_frame() -> None:
    mapped = normalized_points_to_sam([[0.5, 0.25], [0.999, 0.0]])

    assert mapped.shape == (1, 2, 2)
    assert mapped.dtype == np.float32
    np.testing.assert_allclose(mapped[0], [[512, 256], [1022.976, 0]], atol=1e-5)


@pytest.mark.parametrize(
    "coords", [[[1.0, 0.0]], [[-0.01, 0.0]], [[float("nan"), 0.2]]]
)
def test_normalized_prompt_coordinates_reject_out_of_range_values(coords) -> None:
    with pytest.raises(ValueError, match="normalized prompt coordinates"):
        normalized_points_to_sam(coords)


def test_mask_selection_uses_highest_scoring_candidate_covering_prompt() -> None:
    logits = np.full((1, 3, 256, 256), -1.0, dtype=np.float32)
    logits[0, 1] = 1.0
    logits[0, 2] = 1.0
    scores = np.asarray([[0.99, 0.8, 0.9]], dtype=np.float32)

    selected = select_subject_mask(logits, scores, [0.5, 0.5], 64, 48)

    assert selected.candidate_index == 2
    assert selected.score == pytest.approx(0.9)
    assert selected.mask.shape == (48, 64)
    assert selected.mask.dtype == np.float32
    assert set(np.unique(selected.mask)) == {1.0}
    assert logits[0, 2, 0, 0] == 1.0


def test_mask_resize_matches_reference_half_pixel_bilinear() -> None:
    low_res = np.random.default_rng(13).normal(size=(256, 256)).astype(np.float32)
    expected = np.empty((11, 17), dtype=np.float32)
    for y in range(expected.shape[0]):
        source_y = np.float32(y + 0.5) * np.float32(256) / np.float32(11) - np.float32(
            0.5
        )
        iy = int(np.floor(source_y))
        fy = source_y - iy
        for x in range(expected.shape[1]):
            source_x = np.float32(x + 0.5) * np.float32(256) / np.float32(
                17
            ) - np.float32(0.5)
            ix = int(np.floor(source_x))
            fx = source_x - ix
            value = np.float32(0.0)
            for dy in range(2):
                py = min(max(iy + dy, 0), 255)
                wy = np.float32(1.0) - fy if dy == 0 else fy
                for dx in range(2):
                    px = min(max(ix + dx, 0), 255)
                    wx = np.float32(1.0) - fx if dx == 0 else fx
                    value += low_res[py, px] * wy * wx
            expected[y, x] = value

    actual = _resize_mask_bilinear(low_res, 17, 11)

    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-7)


def test_mask_selection_fails_when_no_candidate_covers_prompt() -> None:
    logits = np.full((1, 3, 256, 256), -1.0, dtype=np.float32)

    with pytest.raises(ValueError, match="no SAM candidate"):
        select_subject_mask(logits, np.zeros(3, dtype=np.float32), [0.5, 0.5], 64, 48)


@pytest.mark.parametrize(
    ("logits", "scores", "point", "width", "height", "message"),
    [
        (
            np.zeros((3, 256, 256), dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            [0.5, 0.5],
            64,
            48,
            "mask_logits",
        ),
        (
            np.zeros((1, 3, 256, 256), dtype=np.float32),
            np.zeros(3, dtype=np.float64),
            [0.5, 0.5],
            64,
            48,
            "float32",
        ),
        (
            np.zeros((1, 3, 256, 256), dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            [1.0, 0.5],
            64,
            48,
            "normalized range",
        ),
        (
            np.zeros((1, 3, 256, 256), dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            [0.5, 0.5],
            2049,
            48,
            "2048-pixel budget",
        ),
    ],
)
def test_mask_selection_rejects_invalid_inputs(
    logits, scores, point, width, height, message
) -> None:
    with pytest.raises(ValueError, match=message):
        select_subject_mask(logits, scores, point, width, height)


def test_reuses_image_features_for_repeated_prompts() -> None:
    engine, sessions = make_engine()
    image = image_tensor()

    first = engine.predict(image, [[100, 200]], [1])
    second = engine.predict(image, [[100, 200], [300, 400]], [1, 0])

    assert first.mask_logits.shape == (1, 3, 256, 256)
    assert first.scores.shape == (1, 3)
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.encoder_seconds >= 0
    assert second.encoder_seconds == 0
    assert len(sessions["encoder"].calls) == 1
    assert len(sessions["decoder"].calls) == 2
    assert engine.cache.hits == 1
    assert engine.cache.entry_count == 1
    assert engine.cache.used_bytes == sum(
        int(np.prod(shape)) * np.dtype(np.float32).itemsize
        for shape in SAM_FEATURE_SHAPES
    )


def test_concurrent_prompts_share_one_encoder_pass() -> None:
    engine, sessions = make_engine()
    image = image_tensor()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(engine.predict, image, [[index + 10, 20]], [1])
            for index in range(2)
        ]
    predictions = [future.result() for future in futures]

    assert sorted(prediction.cache_hit for prediction in predictions) == [False, True]
    assert len(sessions["encoder"].calls) == 1
    assert len(sessions["decoder"].calls) == 2


def test_cache_budget_evicts_oldest_image_embedding() -> None:
    one_embedding_bytes = sum(
        int(np.prod(shape)) * np.dtype(np.float32).itemsize
        for shape in SAM_FEATURE_SHAPES
    )
    engine, sessions = make_engine(cache_max_bytes=one_embedding_bytes)

    engine.predict(image_tensor(0.0), [[10, 20]], [1])
    engine.predict(image_tensor(1.0), [[10, 20]], [1])

    assert len(sessions["encoder"].calls) == 2
    assert engine.cache.entry_count == 1
    assert engine.cache.used_bytes == one_embedding_bytes
    assert engine.cache.evictions == 1


def test_invalidate_and_close_release_embedding_references() -> None:
    engine, _ = make_engine()
    image = image_tensor()
    engine.predict(image, [[10, 20]], [1])
    assert engine.cache.entry_count == 1

    engine.invalidate(image)
    assert engine.cache.entry_count == 0
    engine.predict(image, [[10, 20]], [1])
    engine.close()
    assert engine.cache.entry_count == 0
    assert engine.cache.used_bytes == 0


def test_cache_rejects_negative_capacity() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ImageEmbeddingCache(-1)


@pytest.mark.parametrize(
    ("coords", "labels", "message"),
    [
        ([[1024, 0]], [1], "1024×1024 frame"),
        ([[-1, 0]], [1], "1024×1024 frame"),
        ([[0, 0]], [2], "labels must be"),
        ([[0, 0], [1, 1]], [1], "match the point count"),
    ],
)
def test_invalid_prompts_are_rejected(coords, labels, message) -> None:
    engine, sessions = make_engine()
    with pytest.raises(ValueError, match=message):
        engine.predict(image_tensor(), coords, labels)
    assert sessions["encoder"].calls == []


def test_input_tensor_must_have_expected_shape_and_contiguity() -> None:
    engine, _ = make_engine()
    with pytest.raises(ValueError, match=r"shape \[1, 3, 1024, 1024\]"):
        engine.predict(np.zeros((1, 3, 512, 512), dtype=np.float32), [[1, 1]], [1])
    with pytest.raises(ValueError, match="C-contiguous"):
        non_contiguous = np.zeros((1, 3, 1024, 1025), dtype=np.float32)[:, :, :, :1024]
        engine.predict(non_contiguous, [[1, 1]], [1])
