# Native-resolution optical highlight scatter profile

Date: 2026-10-23. This run profiles the real shared Python engine at the
source photo's full 5760×3840 resolution. Profiling wrapped one call to the
existing scatter function and measured OpenCV `filter2D` calls by kernel size
and channel count. No image-processing math, model or requested output size was
changed.

## Setup

- Source revision: `31f61c9` (`perf: validate native-resolution API task at 4 GiB`),
  mounted read-only into the container.
- Host: MacBook Air M1, 8 GiB RAM. Linux guest: Ubuntu 26.04, ARM64,
  4,085,092,352 B memory. The task container used a 4 GiB cgroup v2 memory and
  swap limit, two-CPU quota and 128 PID limit.
- Runtime: Python 3.11.17, NumPy 2.4.6, OpenCV 5.0.0, ONNX Runtime 1.31.0;
  Low Memory strategy, one ONNX intra-op thread. M9, Depth, SAM, ViTMatte and
  ARNet all reported `CPUExecutionProvider`.
- Input: CC0 mountains JPEG, 5760×3840, 3,878,306 bytes, SHA-256
  `4e6661606c195b2462c0f2321a7feefe2ae2fc081499f3462a30fbb108cd7f8e`.
- Options: `color+bokeh`, focus `(0.5, 0.5)`, aperture `2.8`, subject point
  `(0.72, 0.66)`, 512-pixel ViTMatte tiles, no output-size override.
- Model hashes: M9
  `8936b8c86656265ab07c30b9b4cbbd7fa195d6e44cb3be1d4fd059bb36781e33`;
  Depth Anything V2
  `a8ce30941c85b2c9af29fb26f36fbf7be3c1ba1dcf348187a51f6eaa856a238e`;
  ARNet
  `f0696c16a6eb02099292176f25b47e4dc53863e7e4f3f7748e66470d54cf0173`;
  SAM encoder/decoder
  `37de035410f11fb505c210ce1f8b10b450cb018fbfa7f226ec7e7ab542955c75` /
  `3a6a96c62fd39c1369641ea626e5b2e4aed10de877f85291e9993a044d5bf981`;
  ViTMatte ONNX
  `a5c0b49a664867021eaf2b9cadcdc236ea511028d5e709303b418431f20e223b`.

## End-to-end result

| Measurement | Result |
|---|---:|
| Task wall time, decode through PNG export | 3,164.246 s (52.74 min) |
| Engine-reported total | 3,164.231 s |
| Process CPU time | 3,155.810 s |
| Process-tree sampled peak RSS | 2,521,657,344 B (2.348 GiB) |
| cgroup v2 peak | 2,913,480,704 B (2.713 GiB; 67.83% of limit) |
| cgroup headroom at peak | 1,381,486,592 B (1.287 GiB) |
| cgroup OOM / limit events | 0 / 0 |
| Output PNG | 16,655,520 B; 5760×3840 |
| Output SHA-256 | `0a6282df408319950f0bd4c8f56bcfce78ed9e7eba95ce9c50b2c70f5555ade2` |

The output hash is byte-identical to the full-resolution API download and the
previous source-verified direct-engine result. The measured stages were:

| Stage | Seconds |
|---|---:|
| M9 inference / Lab fusion | 3.577 / 2.998 |
| Depth inference | 12.968 |
| SAM encoder / decoder | 4.500 / 0.083 |
| ViTMatte inference | 5.675 |
| ARNet passes 1–4 | 185.938 / 39.102 / 12.050 / 3.934 |
| Highlight extraction / recovery | 3.458 / 1.694 |
| Highlight scatter | 2,845.631 |
| Highlight softening / composite | 21.699 / 1.171 |
| PNG export | 6.934 |

Scatter accounted for 89.93% of engine time. Relative to the prior 4 GiB API
task, this direct SDK task was 398.544 s shorter overall and 274.364 s shorter
in scatter, while cgroup peak was 114,221,056 B lower. Both runs used the same
photo, model hashes, CPU quota, memory cap and processing parameters and
produced the exact same PNG. They were sequential single trials without host
CPU isolation, so the timing and memory differences are not attributed to the
API layer or any one runtime setting.

## Scatter profile

The profiler captured one call to `scatter_highlights`. It recorded 55,132
`cv2.filter2D` calls with 1,610.058 s accumulated timer time (56.58% of the
scatter stage). The cProfile record for the underlying OpenCV function was
1,608.964 s; the small difference is wrapper timing overhead.

| Filter input | Calls | Time | Share of scatter |
|---|---:|---:|---:|
| Single-channel float64 depth-layer mass | 53,940 | 1,497.169 s | 52.61% |
| Three-channel float64 highlight radiance | 1,192 | 112.888 s | 3.97% |

The largest individual grouped cost was the 285×285 single-channel kernel:
310 calls and 50.872 s. The next three single-channel groups were 281×281
(38.647 s), 283×283 (33.933 s), and 279×279 (30.964 s), each across 310
calls. The profile also attributed 40.476 s cumulative to NumPy `any` and
28.797 s to array `astype`; `_scatter_field_optical` itself had 1,129.582 s
self time outside its profiled child calls.

The measured next experiment is to count depth-layer mass inputs that are
exactly zero and test skipping their filter calls. The current profile grouped
calls by kernel geometry and channel count but did not count zero inputs, so
the opportunity and expected gain remain unverified. Any candidate must keep
full key coverage, compare scatter arrays and final PNG bytes, and be measured
at native size before merging.

## Reproduction and artifacts

Run in a Linux ARM64 container configured with `--memory=4g
--memory-swap=4g --cpus=2 --pids-limit=128`. Mount the repository and model
assets read-only; write output/profile/report to a separate volume:

```sh
python scripts/profile_native_scatter.py /data/mountains-5760x3840.jpg \
  /results/profiled.png \
  --m9-model /models/leica-m9-v3-epoch11.fp16.onnx \
  --depth-model /models/depth-v2-small-dynamic-q64.onnx \
  --bokeh-model /models/legendary-1.0-arnet.fp16.onnx \
  --sam-model-dir /sam \
  --vitmatte-model /models/vitmatte-onnx/vitmatte-small-composition-1k.fp32.onnx \
  --profile /results/scatter.prof \
  --report /results/report.json \
  --focus-x 0.5 --focus-y 0.5 --aperture 2.8 \
  --matte-tile-size 512 --subject-point 0.72,0.66
```

The retained `report.json`, `scatter.prof`, and native PNG are outside Git under
`/tmp/laika-native-scatter-profile-4g-20261023/` on the Mac. The downsampled
720×480 smoke image correctly had no highlight radiance and did not invoke
scatter; it is not used as a scatter measurement.
