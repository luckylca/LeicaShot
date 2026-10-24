# Windows exact-support optical mass convolution experiment

Date: 2026-10-24. Branch perf/win-sparse-optical-convolution.
Baseline develop f6fb217. This experiment optimizes the algorithm
rather than output dimensions, kernel, aperture, CoC layers or precision.

## Problem and algorithm

Optical scatter computes 5x5 spatial pupil field nodes, multiple CoC layers
and row chunks. Many layer-weight tiles contain broad zero margins yet
the old filter2D convolves the whole halo-extended region in float64.
The optimized path bounds the nonzero source support and dilates by the
same pupil-kernel half-width; it convolves the *unchanged* float64 kernel
only on this region. Explicit zero padding avoids false reflected source
contributions along the cropped edge. If compact padded area is >=55% of
the original area, it falls back to the original dense convolution. The
previous FP32 and FFT experiments remain separate and unchanged.
The algorithm retains node order and coverage/occlusion mathematics.
Use _sparse_mass_roi=False to disable this path for independent comparison.

## Measured Windows data

CPU only, Windows 10.0.22631, NumPy and OpenCV. Input is the
5472x3648 CC0 portrait downsampled for each scatter test, using a
real Depth Anything V2 depth map resampled to test dimensions.
Aperture 2.8, optical field on, scale 3, 128-row chunks.

| Width | Reference seconds | Sparse seconds | Speedup on means |
|---|---:|---:|---:|
| 512 | 0.871 / 0.763 | 0.692 / 0.687 | ~15% |
| 1024 | 4.518 | 3.932 | 13.0% |
| 2048 | 44.327 / 41.843 | 34.996 / 36.234 | 17.3% |

All these scatter outputs were bitwise-identical float32 arrays, with
identical final 8-bit compositing on the 512 and 1024 tests. At 1024
1,016 of 1,409 nonempty candidate nodes used sparse convolution;
at 2048 3,335 of 4,647 used sparse convolution. The control and
candidate runs at 2048 were conducted in both execution orders.

At 2048x1365 the reference fresh-process sampled RSS peaks were
311,201,792 and 311,177,216 bytes. The candidate peaks were
310,890,496 and 309,899,264 bytes. These are process RSS samples
not whole-pipeline or cgroup high-water measurements.

The full real-photo 1024x683 five-model CPU pipeline (M9, Depth,
ARNet, SAM and ViTMatte) produced byte-identical PNGs before and
after enabling the default sparse support algorithm. Both PNGs have
SHA256 3401f85308a2754771a4a41489facb3d8613f610ddd3b1ed4bfb35e62ffc03a8.
The optimized full-task wall time was 30.06s vs prior baseline 28.53s.
Unrelated neural model stages fluctuated, so the whole-task speed
cannot be claimed improved from one sample. Optical scatter stage
was ~4.61s optimized vs ~5.08s previously.

## Acceptance, limitations and reproduction

222 Python tests pass, Ruff lint/format pass and git diff --check passes.
Includes randomized depth masks, edge/border cases, dense-mask
fallback, direct compact convolution reference and quantized composites.

Matching hashes for these test images do not prove cross-backend
bitwise equality for all images. No new native 20MP full-pipeline
or strict 2GiB/4GiB Linux cgroup measurement was conducted on
Windows; those must be measured separately before production
memory tier claims. Measured isolated scatter speedups cannot be
projected directly onto the full application wall clock.
