from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_benchmark_module():
    path = Path(__file__).parents[1] / "scripts" / "benchmark_mass_convolution.py"
    spec = importlib.util.spec_from_file_location("benchmark_mass_convolution", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cached_fft_mass_convolution_matches_opencv_in_interior_crop() -> None:
    benchmark = _load_benchmark_module()
    rng = np.random.default_rng(7241)
    source = rng.random((83, 97), dtype=np.float64)
    margin = 7
    kernel = benchmark._optical_pupil_kernel(
        5, 2.8, 0.7071067811865476, 0.7071067811865476, 0.82
    )
    dft_shape = benchmark._dft_shape(source.shape, kernel.shape)
    spectrum = benchmark._kernel_spectrum(kernel, dft_shape)

    expected = benchmark._interior_filter2d(source, kernel, margin)
    actual = benchmark._fft_interior(
        source, kernel, margin, cached_kernel_spectrum=spectrum
    )
    kernel_spectrum32 = benchmark._kernel_spectrum(kernel, dft_shape, dtype=np.float32)
    actual32 = benchmark._fft_interior(
        source,
        kernel,
        margin,
        cached_kernel_spectrum=kernel_spectrum32,
        fft_dtype=np.float32,
    )

    assert actual.shape == expected.shape == (69, 83)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(actual32, expected, rtol=0.0, atol=1e-6)
