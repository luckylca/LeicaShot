"""Fast model-free tests for the full-pipeline performance harness."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.benchmark_speed_matrix import (
    _decode_result,
    _image_difference,
    _launch,
    _parse_threads,
)


def test_speed_matrix_rejects_invalid_thread_counts() -> None:
    assert _parse_threads("1, 2,4") == [1, 2, 4]
    for invalid in ("", "0", "33", "1,1", "abc", "1,"):
        with pytest.raises(ValueError, match="threads"):
            _parse_threads(invalid)


def test_speed_matrix_decodes_only_successful_pipeline_json() -> None:
    payload = {"total_seconds": 1.5, "stage_seconds": {"depth": 0.75}}
    result = _decode_result(
        '{"progress":"start"}\n' + json.dumps(payload) + "\n"
    )
    assert result == payload
    with pytest.raises(RuntimeError, match="ProcessingResult"):
        _decode_result('{"error":"not a pipeline result"}')


def test_speed_matrix_uses_decoded_rgb_pixel_error(tmp_path: Path) -> None:
    a = np.zeros((3, 5, 3), dtype=np.uint8)
    b = a.copy()
    b[1, 2, 1] = 1
    before = tmp_path / "before.png"
    after = tmp_path / "after.png"
    Image.fromarray(a, mode="RGB").save(before)
    Image.fromarray(b, mode="RGB").save(after)
    measured = _image_difference(before, after)
    assert measured["max_8bit_channel_diff"] == 1
    assert measured["changed_pixel_fraction"] == pytest.approx(1 / 15)
    assert measured["mean_8bit_channel_diff"] == pytest.approx(1 / 45)


def test_speed_matrix_measures_subprocess_without_model_weights() -> None:
    code = (
        "import json;"
        "print(json.dumps({'total_seconds':0.1,"
        "'stage_seconds':{'mock_inference':0.01}}))"
    )
    elapsed, rss, result = _launch(
        [sys.executable, "-c", code],
        {},
        timeout=20,
        interval=0.01,
    )
    assert elapsed >= 0
    assert rss >= 0
    assert result["stage_seconds"]["mock_inference"] == 0.01
