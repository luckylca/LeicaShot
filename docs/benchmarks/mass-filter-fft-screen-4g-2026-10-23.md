# Cached FFT screen for field-optical mass convolution

Updated: 2026-10-23

## Question

Determine whether caching FP32 DFT pupil spectra for the narrow edge-node mass tiles reduces the cost of the 12,041 scalar highlight-mass filters in the full CPU pipeline.

## Representative tile screen

The benchmark used the production `_optical_pupil_kernel` geometry at radius 120 (243×243 kernel), a deterministic float64 source (`seed=20261023`), one OpenCV thread and a target row band of 256 pixels. The source tile includes the full kernel halo. Each case ran seven repeats; the table shows median time per filter call. It ran in a Linux ARM64 Docker container limited to 2 CPU and 4 GiB memory; all trials exited successfully without OOM.

| Tile class | Source → output size | OpenCV FP64 | Cached FFT FP64 | Cached FFT FP32 | FP32 max absolute error vs OpenCV | FP32 cgroup peak |
|---|---:|---:|---:|---:|---:|---:|
| Edge node | 1682×498 → 1440×256 | 32.383 ms | 30.307 ms | 21.206 ms | 2.59e−7 | 106,061,824 B |
| Inner node | 3121×498 → 2879×256 | 44.683 ms | 43.531 ms | 46.063 ms | 3.29e−7 | 161,992,704 B |

The FP32 FFT is 34.5% faster on the edge tile and 3.1% slower on the inner tile in these paired measurements. At 2 edge and 3 inner node widths, the weighted estimate is about 9% faster for the scalar filter stage. It is only a tile estimate. FP64 FFT was 6.4% slower on the edge tile and a 30.0% inner-tile loss. The cached FP32 kernel spectrum occupied 11,664,000 B for the edge tile and 20,250,000 B for the inner tile; the cgroup peak includes all benchmark working buffers and Python runtime. Max absolute convolution error was below 3.3e−7, but this alone does not establish a final-photo quality bound.

## Full five-model task

The candidate used `scripts/profile_native_scatter.py --fft32-outer-mass-filter`; it applied FFT only when the source tile width was at most 1700 pixels, and left other mass convolutions on the existing FP64 OpenCV path. All other settings matched the one-run current-source FP64 control from the [FP32 mass-filter report](nonzero-mass-fp32-native-4g-2026-10-23.md): same 5760×3840 CC0 mountains JPEG, M9, Depth Anything V2, SAM 2.1 Tiny, ViTMatte Small, BokehMe ARNet, focus, subject point, aperture, 256-row scatter chunks, one CPU thread and a 4 GiB cgroup-v2 limit. Both used Linux ARM64, Python 3.11.17, NumPy 2.4.6, OpenCV 5.0.0 and ONNX Runtime 1.31.0.

| Measurement | FP64 OpenCV control | FFT32 edge-tile candidate |
|---|---:|---:|
| Complete task wall time | 1,305.825 s | 1,385.620 s |
| Highlight scatter stage | 984.026 s | 1,023.005 s |
| Scalar mass filter calls | 12,041 OpenCV | 3,571 FFT + 8,470 OpenCV |
| Candidate FFT time | — | 51.557 s |
| Total measured scalar mass-filter time | 360.133 s | 360.590 s |
| Process-tree peak RSS | 2,546,049,024 B | 2,551,111,680 B |
| cgroup-v2 peak | 2,629,607,424 B | 2,622,283,776 B |
| FFT pupil-spectrum cache peak | — | 101,025,792 B |
| OOM / cgroup-limit events | 0 / 0 | 0 / 0 |

Both exported 5760×3840 PNG files are byte-identical (SHA-256 `0a6282df408319950f0bd4c8f56bcfce78ed9e7eba95ce9c50b2c70f5555ade2`), so decoded pixel error is zero for this image. Candidate and control ran on the same 4 GiB cgroup limit. The candidate ARNet first pass took 215.284 s versus 165.900 s in the control, so whole-task wall time is not a balanced timing comparison; the remaining model stages also varied. The scatter stage was 3.96% slower in the candidate, while its combined FFT plus OpenCV scalar-filter time was effectively unchanged (+0.13%).

## Decision

Do not enable the FFT backend by default. The representative tile screen suggested a gain, but it did not reduce measured full-task mass-filter time or scatter time. The cgroup peak was 7,323,648 B lower, while process-tree peak RSS was 5,062,656 B higher; these small differences do not show a useful memory improvement. Keep the explicit profiler switch and regression coverage only as an experimental path for future hardware/backends; production remains FP64 OpenCV.

The tile reports, full-task JSON, cProfile data and output image remain local-only under `/tmp/laika-mass-fft-screen/` on the Mac and the same path in the Lima guest. No models or photographs are committed.
