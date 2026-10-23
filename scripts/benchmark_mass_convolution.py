"""Compare OpenCV filtering with a reusable-kernel DFT convolution prototype."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil

from laika.highlights import _optical_pupil_kernel


def _interior_filter2d(source: Any, kernel: Any, margin: int) -> Any:
    filtered = cv2.filter2D(
        source,
        ddepth=cv2.CV_64F,
        kernel=kernel,
        borderType=cv2.BORDER_REFLECT_101,
    )
    height = source.shape[0] - 2 * margin
    width = source.shape[1] - 2 * margin
    return filtered[margin : margin + height, margin : margin + width]


def _dft_shape(
    source_shape: tuple[int, int], kernel_shape: tuple[int, int]
) -> tuple[int, int]:
    return (
        cv2.getOptimalDFTSize(source_shape[0] + kernel_shape[0] - 1),
        cv2.getOptimalDFTSize(source_shape[1] + kernel_shape[1] - 1),
    )


def _kernel_spectrum(
    kernel: Any,
    dft_shape: tuple[int, int],
    dtype: Any = np.float64,
) -> Any:
    kernel_pad = np.zeros(dft_shape, dtype=dtype)
    kernel_pad[: kernel.shape[0], : kernel.shape[1]] = np.asarray(
        kernel[::-1, ::-1], dtype=dtype
    )
    return cv2.dft(kernel_pad, flags=cv2.DFT_COMPLEX_OUTPUT)


def _fft_interior(
    source: Any,
    kernel: Any,
    margin: int,
    *,
    cached_kernel_spectrum: Any | None = None,
    fft_dtype: Any = np.float64,
) -> Any:
    dft_shape = _dft_shape(source.shape, kernel.shape)
    source_pad = np.zeros(dft_shape, dtype=fft_dtype)
    source_pad[: source.shape[0], : source.shape[1]] = np.asarray(
        source, dtype=fft_dtype
    )
    source_spectrum = cv2.dft(source_pad, flags=cv2.DFT_COMPLEX_OUTPUT)
    kernel_spectrum = (
        _kernel_spectrum(kernel, dft_shape, fft_dtype)
        if cached_kernel_spectrum is None
        else cached_kernel_spectrum
    )
    product = cv2.mulSpectrums(source_spectrum, kernel_spectrum, 0)
    del source_pad, source_spectrum, kernel_spectrum
    full = cv2.idft(product, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    del product
    height = source.shape[0] - 2 * margin
    width = source.shape[1] - 2 * margin
    start_y = kernel.shape[0] // 2 + margin
    start_x = kernel.shape[1] // 2 + margin
    return full[start_y : start_y + height, start_x : start_x + width]


def _median_seconds(function: Any, repeats: int) -> tuple[float, list[float], Any]:
    function()
    samples = []
    output = None
    for _ in range(repeats):
        started = time.perf_counter()
        output = function()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples), samples, output


def _memory_snapshot() -> dict[str, int]:
    process_info = psutil.Process().memory_info()
    snapshot: dict[str, int] = {}
    for attribute in ("peak_rss", "peak_wset"):
        value = getattr(process_info, attribute, None)
        if value is not None:
            snapshot["process_peak_rss_bytes"] = int(value)
            break
    for name in ("current", "peak", "max"):
        path = Path(f"/sys/fs/cgroup/memory.{name}")
        try:
            value = path.read_text().strip()
            if value.isdigit():
                snapshot[f"cgroup_memory_{name}_bytes"] = int(value)
        except OSError:
            continue
    return snapshot


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=498)
    parser.add_argument("--width", type=int, default=1682)
    parser.add_argument("--kernel-radius", type=int, default=120)
    parser.add_argument("--margin", type=int, default=121)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument(
        "--fft-dtype", choices=("float32", "float64"), default="float64"
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.height <= 2 * args.margin or args.width <= 2 * args.margin:
        raise SystemExit("source dimensions must exceed twice the interior margin")
    if args.kernel_radius < 1 or args.margin < args.kernel_radius:
        raise SystemExit("margin must be at least the positive kernel radius")
    if args.repeats < 1 or args.threads < 1:
        raise SystemExit("repeats and threads must be positive")

    cv2.setNumThreads(args.threads)
    rng = np.random.default_rng(args.seed)
    source = rng.random((args.height, args.width), dtype=np.float64)
    kernel = _optical_pupil_kernel(
        args.kernel_radius, 2.8, 0.7071067811865476, 0.7071067811865476, 0.82
    )
    if args.margin < kernel.shape[0] // 2:
        raise SystemExit("margin must cover the complete convolution halo")

    shape = _dft_shape(source.shape, kernel.shape)
    fft_dtype = np.dtype(args.fft_dtype)
    kernel_spectrum = _kernel_spectrum(kernel, shape, fft_dtype)
    baseline_seconds, baseline_samples, baseline = _median_seconds(
        lambda: _interior_filter2d(source, kernel, args.margin), args.repeats
    )
    uncached_seconds, uncached_samples, uncached = _median_seconds(
        lambda: _fft_interior(source, kernel, args.margin, fft_dtype=fft_dtype),
        args.repeats,
    )
    cached_seconds, cached_samples, cached = _median_seconds(
        lambda: _fft_interior(
            source,
            kernel,
            args.margin,
            cached_kernel_spectrum=kernel_spectrum,
            fft_dtype=fft_dtype,
        ),
        args.repeats,
    )
    delta = np.abs(cached - baseline)
    process = psutil.Process()
    report = {
        "configuration": {
            "source_size": [args.width, args.height],
            "kernel_size": list(kernel.shape),
            "interior_size": [
                args.width - 2 * args.margin,
                args.height - 2 * args.margin,
            ],
            "dft_size": [shape[1], shape[0]],
            "baseline_dtype": "float64",
            "fft_dtype": args.fft_dtype,
            "threads": args.threads,
            "repeats": args.repeats,
            "seed": args.seed,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
        "measurements": {
            "opencv_filter2d_median_seconds": baseline_seconds,
            "opencv_filter2d_samples_seconds": baseline_samples,
            "fft_uncached_kernel_median_seconds": uncached_seconds,
            "fft_uncached_kernel_samples_seconds": uncached_samples,
            "fft_cached_kernel_median_seconds": cached_seconds,
            "fft_cached_kernel_samples_seconds": cached_samples,
            "cached_kernel_spectrum_bytes": int(kernel_spectrum.nbytes),
            **_memory_snapshot(),
            "process_rss_after_benchmark_bytes": int(process.memory_info().rss),
        },
        "numerical_error_cached_vs_opencv": {
            "max_absolute": float(delta.max(initial=0.0)),
            "mean_absolute": float(delta.mean()),
            "p99_absolute": float(np.quantile(delta, 0.99)),
            "exact_array_equal": bool(np.array_equal(cached, baseline)),
        },
        "uncached_max_absolute_vs_opencv": float(
            np.abs(uncached - baseline).max(initial=0.0)
        ),
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
