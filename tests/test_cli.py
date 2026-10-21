from __future__ import annotations

import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import psutil
import pytest
from PIL import Image

from laika.benchmark import measure_image
from laika.cli import _read_rgb, _run_depth, _save_depth
from laika.engine import ImageInputError, ImageLimitError


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


def test_benchmark_samples_worker_memory_and_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.png"
    destination = tmp_path / "result.png"
    Image.new("RGB", (8, 6), (40, 80, 120)).save(source)

    def uss_unavailable(_process: psutil.Process) -> psutil._psplatform.pmem:
        raise RuntimeError("platform does not expose USS for this process")

    monkeypatch.setattr(psutil.Process, "memory_full_info", uss_unavailable)

    report = measure_image(source, destination, sample_interval_ms=2)

    assert report["measurement"] == "subprocess process-tree sampling"
    assert report["samples"] > 0
    assert report["process_tree_peak_rss_bytes"] > 0
    assert report["process_tree_cpu_seconds_peak"] >= 0
    assert report["process_tree_peak_uss_bytes"] is None
    assert report["image"]["output_width"] == 8
    assert report["image"]["output_height"] == 6


def test_depth_cli_decoder_applies_orientation_and_checks_limits(
    tmp_path: Path,
) -> None:
    source = tmp_path / "oriented.jpg"
    image = Image.new("RGB", (17, 9), (23, 94, 175))
    exif = image.getexif()
    exif[274] = 6
    image.save(source, quality=100, exif=exif)

    rgb = _read_rgb(source, max_bytes=1024 * 1024, max_pixels=1_000)

    assert rgb.shape == (17, 9, 3)
    assert rgb.dtype.name == "uint8"
    with pytest.raises(ImageLimitError, match="byte limit"):
        _read_rgb(source, max_bytes=1, max_pixels=1_000)
    with pytest.raises(ImageLimitError, match="pixel limit"):
        _read_rgb(source, max_bytes=1024 * 1024, max_pixels=100)


def test_depth_cli_decoder_uses_actual_image_format(tmp_path: Path) -> None:
    mislabeled = tmp_path / "photo.jpg"
    Image.new("RGB", (4, 3), (17, 22, 33)).save(mislabeled, format="BMP")

    with pytest.raises(ImageInputError, match="unsupported decoded image format"):
        _read_rgb(mislabeled, max_bytes=1024 * 1024, max_pixels=100)


def test_depth_cli_writes_numpy_output_without_pickle(tmp_path: Path) -> None:
    import numpy as np

    output = tmp_path / "nested" / "depth.npy"
    depth = np.arange(20, dtype=np.float32).reshape(4, 5)

    _save_depth(output, depth)

    np.testing.assert_array_equal(np.load(output, allow_pickle=False), depth)
    assert not list(output.parent.glob("*.tmp"))


def test_depth_cli_rejects_pathological_network_aspect_ratio_before_loading_model(
    tmp_path: Path,
) -> None:
    source = tmp_path / "extreme.png"
    Image.new("RGB", (8_000, 1), (10, 20, 30)).save(source)
    args = Namespace(
        input=str(source),
        output=str(tmp_path / "depth.npy"),
        model=str(tmp_path / "missing.onnx"),
        input_size=518,
        analysis_max_edge=None,
        threads=1,
        max_upload_bytes=1024 * 1024,
        max_pixels=40_000_000,
        max_network_pixels=1_000_000,
    )

    with pytest.raises(ImageLimitError, match="network input exceeds"):
        _run_depth(args)


def test_reference_cli_caps_analysis_and_network_pixel_budgets_before_model_load(
    tmp_path: Path,
) -> None:
    source = tmp_path / "square.png"
    Image.new("RGB", (4000, 4000), (10, 20, 30)).save(source)
    args = Namespace(
        input=str(source),
        output=str(tmp_path / "depth.npy"),
        model=str(tmp_path / "missing.onnx"),
        input_size=None,
        analysis_max_edge=None,
        threads=1,
        max_upload_bytes=1024 * 1024,
        max_pixels=40_000_000,
        max_network_pixels=1_000_000,
    )

    with pytest.raises(ImageLimitError, match="network input exceeds"):
        _run_depth(args)
