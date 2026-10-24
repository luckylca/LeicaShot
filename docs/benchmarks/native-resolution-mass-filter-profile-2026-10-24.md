# Current-source native-resolution highlight mass-filter profile

Date: 2026-10-24. This run profiles the current Python `color+bokeh` engine on
the full 5760×3840 CC0 landscape photograph and measures mass and radiance
`filter2D` calls by kernel size, channel count and dtype. It uses the merged
float32 radiance workspace and exact-empty mass skip. No resolution, model,
filter kernel, or processing math was changed.

## Setup and provenance

- Source: current `develop` code after PR #66, mounted read-only. Python source
  tree SHA-256 `c03f5c618e9eac022eccd66272041f86b6cb6d4387aa0d81e65c15b6f35705c2`.
- Input: [Mountains Landscape on Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Mountains_Landscape.jpg),
  CC0 1.0, 5760×3840, SHA-256
  `4e6661606c195b2462c0f2321a7feefe2ae2fc081499f3462a30fbb108cd7f8e`.
- Request: full Python `color+bokeh` path with M9 + LabFusion, Depth Anything
  V2, SAM 2.1 Tiny, ViTMatte Small, BokehMe ARNet and optical highlights;
  Low Memory, CPUExecutionProvider, one inference thread, focus `(0.5, 0.5)`,
  aperture `2.8`, subject point `(0.72, 0.66)`, 512-pixel ViTMatte tiles,
  256-row highlight chunks and native-size output.
- Host: MacBook Air M1; Lima Linux ARM64 guest with 4,085,092,352 B memory.
  Task cgroup v2 limit 4,294,967,296 B, swap disabled, two-CPU quota, 128 PID
  limit. Runtime: Python 3.11.17, NumPy 2.4.6, OpenCV 5.0.0, ONNX Runtime
  1.31.0. All inference used CPUExecutionProvider.
- Model hashes: M9 `8936b8c86656265ab07c30b9b4cbbd7fa195d6e44cb3be1d4fd059bb36781e33`;
  Depth `a8ce30941c85b2c9af29fb26f36fbf7be3c1ba1dcf348187a51f6eaa856a238e`;
  ARNet `f0696c16a6eb02099292176f25b47e4dc53863e7e4f3f7748e66470d54cf0173`;
  SAM encoder/decoder
  `37de035410f11fb505c210ce1f8b10b450cb018fbfa7f226ec7e7ab542955c75` /
  `3a6a96c62fd39c1369641ea626e5b2e4aed10de877f85291e9993a044d5bf981`;
  ViTMatte `a5c0b49a664867021eaf2b9cadcdc236ea511028d5e709303b418431f20e223b`.

## Full-task result

| Measurement | Result |
|---|---:|
| Full task wall time | 3,087.131 s |
| Engine-reported time | 3,087.078 s |
| Process CPU time | 2,933.010 s |
| Highlight scatter stage | 2,320.979 s (75.18% of engine time) |
| Process-tree peak RSS | 1,770,176,512 B |
| Scatter-stage process-tree peak RSS | 1,770,176,512 B |
| Exact cgroup `memory.peak` | 2,115,444,736 B (49.25% of 4 GiB) |
| cgroup `max` / `oom` / `oom_kill` | 0 / 0 / 0 |
| Output | 5760×3840 RGB PNG, 16,655,520 B |
| Output SHA-256 | `0a6282df408319950f0bd4c8f56bcfce78ed9e7eba95ce9c50b2c70f5555ade2` |

The output is byte-identical to the previously recorded full-resolution
reference PNG. The 4 GiB run completed with 2,179,522,560 B cgroup headroom.
The lower peak and task time than the 2026-10-23 profile are recorded as current
measurements; these separate runs do not isolate the cause of their difference.

## Scatter profile

One call to `scatter_highlights` took 2,320.979 s. The profiler recorded
12,954 OpenCV `filter2D` calls taking 1,013.238 s in aggregate (43.66% of the
scatter stage):

| Filter input | Calls | Time | Share of scatter |
|---|---:|---:|---:|
| Single-channel float64 depth-layer mass | 12,041 | 826.621 s | 35.61% |
| Three-channel float32 highlight radiance | 913 | 186.617 s | 8.04% |

The exact-empty mass skip was enabled. None of the 12,041 executed mass-filter
inputs were zero, so another empty-input skip will not reduce calls for this
scene. The largest mass-filter groups included 197×197 (81 calls / 23.744 s),
241×241 (67 / 22.009 s), 239×239 (69 / 19.647 s), and 231×231 (77 / 16.611 s)
kernels. The filter calls spanned 273 kernel-size/channel/dtype groups.

The cProfile report attributed 62.770 s across 189,848 NumPy `astype` calls,
34.974 s to ufunc reductions, 28.889 s to `numpy.any`, and about 22.9 s each to
`_field_pupil_kernels` and `_optical_pupil_kernel`. A local OpenCV check showed
that passing float32 input with a requested float64 `filter2D` output is not a
supported source/destination combination. The production float64 input
conversion therefore cannot simply be removed by changing `ddepth`; any
replacement must retain the double-precision convolution reference or measure
and document its numerical error.

The current profile confirms mass convolution remains the largest discrete
CPU hotspot, while also confirming that the earlier exact-zero skip is already
effective. A source-verified 512-row candidate and a fresh 256-row repeat now
exist. Their decoded and encoded outputs have the same PNG SHA-256, but the
first 256-row profile is a clear timing outlier: the repeat reduced its total
time from 3,087 s to 1,469 s and scatter time from 2,321 s to 1,032 s without
changing source or request settings. Do not use that first run as a timing
baseline for selecting a default.

| Row chunk | Full engine (s) | Scatter (s) | `filter2D` calls / s | Mass calls / s | Process-tree RSS peak | cgroup peak | Output SHA-256 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 256 initial | 3,087.078 | 2,320.979 | 12,954 / 1,013.238 | 12,041 / 826.621 | 1,770,176,512 B | 2,115,444,736 B | `0a6282df408319950f0bd4c8f56bcfce78ed9e7eba95ce9c50b2c70f5555ade2` |
| 512 first candidate | 1,423.087 | 1,126.855 | 8,338 / 467.793 | 7,527 / 355.936 | 1,957,703,680 B | 2,109,562,880 B | same |
| 256 repeat | 1,468.959 | 1,032.326 | 12,954 / 461.580 | 12,041 / 378.515 | 1,756,979,200 B | 2,028,392,448 B | same |
| 512 repeat | 2,464.445 | 2,034.701 | 8,338 / 674.452 | 7,527 / 536.453 | 1,969,725,440 B | 2,067,030,016 B | same |

The fresh same-source 512-row repeat ran substantially slower than the 256-row
repeat in scatter (2,034.701 s versus 1,032.326 s), filter2D aggregate time
(674.452 s versus 461.580 s), and full task (2,464.445 s versus 1,468.959 s).
Its process-tree peak was 212,746,240 B higher. Its cgroup peak was 38,637,568 B
higher, though both runs were well below the 4 GiB limit. Model-stage times
outside scatter were similar in this pair; the difference was concentrated in
the scatter stage. The first 512 run had also been slower in scatter than the
fresh 256 repeat, despite a faster total time caused by faster unrelated model
stages. All four runs produced byte-identical PNGs. Retain the 256-row default;
the larger chunk reduced call count but did not demonstrate a speed or memory
benefit, and its repeat materially increased CPU time.

## Reproduction and retained artifacts

Runs used `scripts/profile_native_scatter.py` from the same mounted, read-only
source tree, with only `--bokeh-row-chunk-height` varied between 256 and 512.
Common settings were `--focus-x 0.5 --focus-y 0.5 --aperture 2.8
--matte-tile-size 512 --subject-point 0.72,0.66 --threads 1`, with the same
models and progress telemetry. Input, models, generated PNGs, JSON reports,
cProfile data and progress logs remain outside Git under
`/Users/lucky/Desktop/project/leicashot-photo-validation/` in the
`mass-filter-profile-native-current`, `mass-filter-candidate-512-20261024`,
`mass-filter-repeat-256-20261024`, and `mass-filter-repeat-512-20261024`
directories.
