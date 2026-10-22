from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from laika.cli import main


def test_pipeline_cli_routes_options_through_shared_engine(
    monkeypatch: Any, capsys: Any
) -> None:
    calls: dict[str, Any] = {}

    class FakeEngine:
        def __init__(self, **kwargs: Any) -> None:
            calls["engine"] = kwargs

        def __enter__(self) -> FakeEngine:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def process_file(self, input_path: str, output_path: str, **kwargs: Any) -> Any:
            calls["request"] = (input_path, output_path, kwargs)
            return SimpleNamespace(
                as_dict=lambda: {"mode": "color+bokeh", "width": 120}
            )

    monkeypatch.setattr("laika.cli.LaikaProcessingEngine", FakeEngine)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "laika",
            "pipeline",
            "input.jpg",
            "output.png",
            "--mode",
            "color+bokeh",
            "--m9-model",
            "m9.onnx",
            "--depth-model",
            "depth.onnx",
            "--bokeh-model",
            "arnet.onnx",
            "--strategy",
            "balanced",
            "--threads",
            "3",
            "--width",
            "120",
            "--height",
            "80",
            "--focus-x",
            "0.3",
            "--focus-y",
            "0.7",
            "--aperture",
            "4.0",
            "--crop",
            "0.1,0.2,0.9,0.8",
        ],
    )

    assert main() == 0

    assert calls["engine"] == {
        "m9_model_path": "m9.onnx",
        "depth_model_path": "depth.onnx",
        "bokeh_model_path": "arnet.onnx",
        "sam_model_dir": None,
        "vitmatte_model_path": None,
        "strategy": "balanced",
        "threads": 3,
        "max_upload_bytes": 50 * 1024 * 1024,
        "max_pixels": 40_000_000,
        "max_network_pixels": 4_000_000,
    }
    input_path, output_path, request = calls["request"]
    assert (input_path, output_path) == ("input.jpg", "output.png")
    assert request["options"].mode == "color+bokeh"
    assert request["options"].strategy == "balanced"
    assert request["options"].output_size == (120, 80)
    assert (request["options"].focus_x, request["options"].focus_y) == (0.3, 0.7)
    assert request["options"].aperture == 4.0
    assert request["options"].crop.as_tuple() == (0.1, 0.2, 0.9, 0.8)
    assert request["progress_callback"] is None
    assert json.loads(capsys.readouterr().out) == {"mode": "color+bokeh", "width": 120}


def test_pipeline_cli_requires_both_output_dimensions(
    monkeypatch: Any, capsys: Any
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["laika", "pipeline", "input.jpg", "output.png", "--width", "120"],
    )

    assert main() == 2

    assert json.loads(capsys.readouterr().err) == {
        "error": "--width and --height must be supplied together"
    }


@pytest.mark.parametrize(
    ("argv", "method", "expected"),
    [
        (
            [
                "laika",
                "composition",
                "input.jpg",
                "--model",
                "gaic.onnx",
                "--target-long-edge",
                "1200",
            ],
            "suggest_composition",
            {"candidate_count": 7},
        ),
        (
            [
                "laika",
                "crop",
                "input.jpg",
                "output.png",
                "--crop",
                "0.1,0.2,0.9,0.8",
            ],
            "apply_crop_file",
            {"width": 80},
        ),
    ],
)
def test_composition_cli_routes_through_shared_engine(
    argv: list[str],
    method: str,
    expected: dict[str, Any],
    monkeypatch: Any,
    capsys: Any,
) -> None:
    calls: dict[str, Any] = {}

    class FakeEngine:
        def __init__(self, **kwargs: Any) -> None:
            calls["engine"] = kwargs

        def __enter__(self) -> FakeEngine:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def suggest_composition(self, *args: Any, **kwargs: Any) -> Any:
            calls[method] = (args, kwargs)
            return SimpleNamespace(as_dict=lambda: expected)

        def apply_crop_file(self, *args: Any, **kwargs: Any) -> Any:
            calls[method] = (args, kwargs)
            return SimpleNamespace(as_dict=lambda: expected)

    monkeypatch.setattr("laika.cli.LaikaProcessingEngine", FakeEngine)
    monkeypatch.setattr(sys, "argv", argv)

    assert main() == 0

    output = json.loads(capsys.readouterr().out)
    assert output == expected
    assert method in calls
    args, kwargs = calls[method]
    if method == "suggest_composition":
        assert args == ("input.jpg",)
        assert kwargs == {"target_long_edge": 1200}
        assert calls["engine"]["composition_model_path"] == "gaic.onnx"
    else:
        assert args[:2] == ("input.jpg", "output.png")
        assert args[2].as_tuple() == (0.1, 0.2, 0.9, 0.8)
        assert not calls["engine"].get("composition_model_path")
