from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from laika.benchmark import measure_image


def test_cli_calls_shared_engine_and_returns_json(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    destination = tmp_path / "result.png"
    Image.new("RGB", (11, 7), (40, 80, 120)).save(source)

    completed = subprocess.run(
        [sys.executable, "-m", "laika.cli", "process", str(source), str(destination)],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(completed.stdout)
    assert report["operation"] == "decode-orient-lossless-png"
    assert report["output_width"] == 11
    assert report["output_height"] == 7
    assert destination.is_file()


def test_benchmark_samples_worker_memory_and_cpu(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    destination = tmp_path / "result.png"
    Image.new("RGB", (8, 6), (40, 80, 120)).save(source)

    report = measure_image(source, destination, sample_interval_ms=2)

    assert report["measurement"] == "subprocess process-tree sampling"
    assert report["samples"] > 0
    assert report["process_tree_peak_rss_bytes"] > 0
    assert report["process_tree_cpu_seconds_peak"] >= 0
    assert report["image"]["output_width"] == 8
    assert report["image"]["output_height"] == 6
