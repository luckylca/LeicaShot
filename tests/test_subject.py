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
