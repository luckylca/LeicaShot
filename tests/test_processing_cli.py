from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from typing import Any

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
