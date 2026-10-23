from __future__ import annotations

import importlib.util
import io
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from laika import highlights


def _load_profiler_module():
    path = Path(__file__).parents[1] / "scripts" / "profile_native_scatter.py"
    spec = importlib.util.spec_from_file_location("profile_native_scatter", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scatter_profiler_preserves_result_and_restores_filter(tmp_path: Path) -> None:
    module = _load_profiler_module()
    source = np.arange(24, dtype=np.float32).reshape(3, 4, 2)
    kernel = np.ones((3, 3), dtype=np.float32)

    def filter2d(src, ddepth, kernel, **kwargs):
        return src.copy()

    cv_module = SimpleNamespace(filter2D=filter2d)

    def scatter(value):
        return cv_module.filter2D(value, ddepth=-1, kernel=kernel, borderType=0)

    highlights_module = SimpleNamespace(
        scatter_highlights=scatter,
        cv2=cv_module,
    )
    profiler = module.ScatterProfiler(highlights_module, tmp_path / "scatter.prof")
    profiler.install()
    try:
        actual = highlights_module.scatter_highlights(source)
    finally:
        profiler.restore()

    np.testing.assert_array_equal(actual, source)
    assert highlights_module.scatter_highlights is scatter
    assert cv_module.filter2D is filter2d
    report = profiler.write()
    assert Path(report["profile_file"]).is_file()
    assert profiler.calls == 1
    assert len(report["filter2d_groups"]) == 1
    group = report["filter2d_groups"][0]
    assert group["group"] == "kernel=3x3;channels=2;dtype=float32;ddepth=-1"
    assert group["calls"] == 1
    assert group["seconds"] >= 0.0


def test_scatter_profiler_is_exact_on_real_scatter_function(tmp_path: Path) -> None:
    module = _load_profiler_module()
    random = np.random.default_rng(20261023)
    high = random.random((19, 23, 3), dtype=np.float32)
    coc = random.uniform(-2.5, 2.5, size=(19, 23)).astype(np.float32)
    expected = highlights.scatter_highlights(
        high,
        coc,
        2.8,
        radius_scale=1.0,
        field_optics=True,
        row_chunk_height=5,
    )
    profiler = module.ScatterProfiler(highlights, tmp_path / "real-scatter.prof")
    profiler.install()
    try:
        actual = highlights.scatter_highlights(
            high,
            coc,
            2.8,
            radius_scale=1.0,
            field_optics=True,
            row_chunk_height=5,
        )
    finally:
        profiler.restore()

    np.testing.assert_array_equal(actual, expected)
    report = profiler.write()
    assert profiler.calls == 1
    assert report["filter2d_groups"]


def test_scatter_profiler_tracks_fft_mass_filter_calls(tmp_path: Path) -> None:
    module = _load_profiler_module()
    rng = np.random.default_rng(9123)
    high = rng.random((19, 23, 3), dtype=np.float32)
    coc = rng.uniform(-2.5, 2.5, size=(19, 23)).astype(np.float32)
    profiler = module.ScatterProfiler(
        highlights,
        tmp_path / "fft-mass.prof",
        fft32_outer_mass_filter=True,
    )
    profiler.install()
    try:
        highlights.scatter_highlights(
            high,
            coc,
            2.8,
            radius_scale=1.0,
            field_optics=True,
            row_chunk_height=5,
        )
    finally:
        profiler.restore()

    report = profiler.write()
    assert profiler.calls == 1
    assert report["fft_mass_filter_calls"] > 0
    assert report["fft_mass_filter_seconds"] > 0.0
    assert report["fft_kernel_cache_peak_bytes"] > 0


def test_cgroup_snapshot_includes_memory_categories_and_swap_limits(
    tmp_path: Path,
) -> None:
    module = _load_profiler_module()
    (tmp_path / "memory.stat").write_text("anon 4096\nfile 2048\nkernel 512\n")
    (tmp_path / "memory.swap.current").write_text("0\n")
    (tmp_path / "memory.swap.max").write_text("0\n")

    snapshot = module._augment_cgroup_snapshot(
        {"current_bytes": 6144, "peak_bytes": 8192}, tmp_path
    )

    assert snapshot == {
        "current_bytes": 6144,
        "peak_bytes": 8192,
        "memory_stat_bytes": {"anon": 4096, "file": 2048, "kernel": 512},
        "swap_current_bytes": 0,
        "swap_limit_bytes": 0,
    }


def test_process_tree_peak_flushes_periodic_stage_and_cgroup_telemetry(
    monkeypatch,
) -> None:
    module = _load_profiler_module()

    class FakeProcess:
        def memory_info(self):
            return SimpleNamespace(rss=4096)

        def children(self, recursive: bool):
            assert recursive
            return []

    monkeypatch.setattr(module.psutil, "Process", FakeProcess)

    class TelemetryStream(io.StringIO):
        def __init__(self) -> None:
            super().__init__()
            self.memory_sample_written = threading.Event()

        def write(self, value: str) -> int:
            result = super().write(value)
            if '"record_type": "memory_sample"' in value:
                self.memory_sample_written.set()
            return result

    stream = TelemetryStream()
    progress = SimpleNamespace(
        fraction=0.5,
        elapsed_seconds=2.0,
        details={"component": "arnet_tile"},
    )

    peak = module.ProcessTreePeak(
        interval_seconds=0.001,
        telemetry_stream=stream,
        cgroup_snapshot=lambda: {"current_bytes": 6144, "peak_bytes": 8192},
        telemetry_interval_seconds=0.003,
    )
    peak.set_stage("bokeh.arnet_tile")
    with peak:
        peak._progress_record(progress, "bokeh.arnet_tile")
        assert stream.memory_sample_written.wait(timeout=1.0)

    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert any(record["record_type"] == "progress" for record in records)
    samples = [record for record in records if record["record_type"] == "memory_sample"]
    assert samples
    assert all(record["stage"] == "bokeh.arnet_tile" for record in samples)
    assert all(record["process_tree_rss_bytes"] == 4096 for record in samples)
    assert all(record["cgroup_v2"]["peak_bytes"] == 8192 for record in samples)
    report = peak.report()
    assert report["process_tree_stage_peak_rss_bytes"]["bokeh.arnet_tile"] == 4096
    assert (
        report["cgroup_memory_high_water_observed_by_stage_bytes"]["bokeh.arnet_tile"]
        == 8192
    )
