from __future__ import annotations

import importlib.util
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
