# CPU thread scaling on the native-resolution five-model task

Updated: 2026-10-23

## Question

Determine whether increasing the engine's shared ONNX Runtime/OpenCV thread setting from one to two materially accelerates the CPU-only Low Memory pipeline under a two-CPU quota, and measure the memory and pixel-quality tradeoff.

## Method

- Both native-resolution runs used the same 5760×3840 CC0 mountain JPEG (`4e6661606c195b2462c0f2321a7feefe2ae2fc081499f3462a30fbb108cd7f8e`), model assets, M9 + Depth + SAM + ViTMatte + ARNet/highlights flow, focus `(0.5, 0.5)`, aperture `2.8`, subject point `(0.72, 0.66)`, tile size 512, Low Memory strategy, and CPUExecutionProvider.
- The one-thread baseline is PR #44's native-resolution candidate run. The two-thread run used the same source and profiling harness with `--threads 2`; this value configures the ONNX Runtime intra-op sessions and OpenCV thread pool. Runs were sequential, in fresh containers, with no other inference workload.
- Each container had a 4 GiB cgroup, swap capped at the same limit, two CPUs, one task, and read-only source/model mounts. Runtime was Linux ARM64, Python 3.11.17, ONNX Runtime 1.31.0, NumPy 2.4.6, OpenCV 5.0.0.
- M9, Depth, SAM encoder/decoder, ViTMatte, and ARNet model/checkpoint hashes matched between runs.

## Native-resolution results

| Measurement | One thread | Two threads | Change |
|---|---:|---:|---:|
| Complete task wall time | 1,569.613 s | 1,529.419 s | −40.194 s (−2.56%) |
| Engine time | 1,569.601 s | 1,529.405 s | −2.56% |
| Highlight scatter stage | 1,272.015 s | 1,252.931 s | −19.084 s (−1.50%) |
| Process CPU time | 1,579.810 s | 1,697.810 s | +7.47% |
| Process-tree peak RSS | 2,551,914,496 B | 2,564,661,248 B | +12,746,752 B |
| cgroup peak memory | 2,844,807,168 B | 2,830,872,576 B | −13,934,592 B |
| OpenCV filter calls | 22,414 | 22,414 | unchanged |
| OOM / cgroup limit events | 0 / 0 | 0 / 0 | unchanged |

The one-thread output SHA-256 was `0a6282df408319950f0bd4c8f56bcfce78ed9e7eba95ce9c50b2c70f5555ade2`; the two-thread output SHA-256 was `077ec0eb2d2decf1ef54c134796794c0cf1d0f9a526331ddd9a9ab1187a8868d`. Comparing decoded 5760×3840 RGB pixels found one channel value different by one 8-bit level: MAE `1.507e-8`, p99 absolute error `0`, maximum error `1`, PSNR `126.35 dB`.

## Same-photo smoke and decision

A 1440×960 resize of the same source completed in 67.708 s at one thread and 43.581 s at two threads, with exact pixel equality. That small-input result overstated the native-size gain: the full task was only 2.56% faster and used 7.47% more process CPU time, while the output differed by one least-significant channel value.

Keep the Low Memory default at one thread. Existing SDK/CLI/API thread configuration remains available to callers who prefer this small, measured time reduction and accept the one-LSB output variation. This single paired run does not establish a stable cross-device or cross-scene scaling result.

Raw full-size reports, profiles, and images remain outside Git under `/tmp/laika-scatter-skip-empty-mass-4g-20261023/` on the Mac; the run used the `thread-2-native` directory and the PR #44 `candidate` directory.
