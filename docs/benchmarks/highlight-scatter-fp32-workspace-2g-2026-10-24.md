# Reuse a float32 tile workspace in optical highlight scatter

Date: 2026-10-24. This experiment reuses the float32 advanced-index tile copy
as the weighted radiance workspace in `_scatter_field_optical`. It also filters
the per-node RGB radiance tile in float32 while retaining float64 accumulation
for the full-scene radiance total. The change targets transient scatter-stage
memory; it does not remove the renderer's full-resolution output, own, or
coverage arrays.

## Configuration and provenance

- Candidate: `perf/highlight-scatter-live-buffers`, code commit `ff0b199`.
- Comparison: source-verified merged PR #61 (`085a369`), with the radiance
  lifetime change already enabled.
- Input: CC0 mountains JPEG, 5760×3840, SHA-256
  `4e6661606c195b2462c0f2321a7feefe2ae2fc081499f3462a30fbb108cd7f8e`.
- Request: M9 + LabFusion, Depth Anything V2, SAM 2.1 Tiny, ViTMatte Small,
  BokehMe ARNet, optical highlights, and one protected subject point
  `(0.72, 0.66)`. Low Memory, focus `(0.5, 0.5)`, aperture `2.8`, 512-pixel
  matte tiles, 256-pixel ARNet tiles, 128-row highlight chunks, one ORT thread,
  native output dimensions.
- Host: MacBook Air M1; Linux 7.0.0-34 ARM64 Lima guest with 4,085,092,352 B
  RAM and 2 logical CPUs. The task ran in cgroup v2 with
  `memory.max=2,147,483,648` B, swap disabled, and a 2-CPU quota.
- Runtime: Python 3.11.17, NumPy 2.4.6, OpenCV 5.0.0, ONNX Runtime 1.31.0;
  all ONNX sessions used `CPUExecutionProvider`.
- Input source was loaded from `/opt/laika-src/src/laika/__init__.py`; mounted
  checkout source-tree SHA-256:
  `9e184b095881c33a499bdffbf6f85aa8478e600b1caa109ab211e243f769a819`.
- Model hashes match the prior report: M9
  `8936b8c86656265ab07c30b9b4cbbd7fa195d6e44cb3be1d4fd059bb36781e33`, Depth
  `a8ce30941c85b2c9af29fb26f36fbf7be3c1ba1dcf348187a51f6eaa856a238e`, ARNet
  `f0696c16a6eb02099292176f25b47e4dc53863e7e4f3f7748e66470d54cf0173`, SAM
  encoder/decoder
  `37de035410f11fb505c210ce1f8b10b450cb018fbfa7f226ec7e7ab542955c75` /
  `3a6a96c62fd39c1369641ea626e5b2e4aed10de877f85291e9993a044d5bf981`, and
  ViTMatte `a5c0b49a664867021eaf2b9cadcdc236ea511028d5e709303b418431f20e223b`.

## Full-resolution result

Both runs completed the same Python engine request and produced a 5760×3840
RGB PNG with SHA-256
`0a6282df408319950f0bd4c8f56bcfce78ed9e7eba95ce9c50b2c70f5555ade2`.
The candidate output was independently hashed inside the Linux guest. Every
cgroup `max`, `oom`, and `oom_kill` event counter remained zero.

| Measure | PR #61 | Float32 workspace | Change |
|---|---:|---:|---:|
| Full task wall time | 1,502.613 s | 1,501.569 s | −1.045 s; not a speed claim |
| Process-tree peak RSS | 1,767,378,944 B | 1,724,833,792 B | −42,545,152 B |
| Scatter-stage process-tree peak RSS | 1,767,378,944 B | 1,659,846,656 B | −107,532,288 B |
| Optical scatter time | 1,181.810 s | 1,178.114 s | −3.696 s; timing variation |
| Exact cgroup `memory.peak` | 2,033,582,080 B | 2,004,103,168 B | −29,478,912 B |
| Maximum 1-second cgroup `memory.current` | 2,020,020,224 B | 1,934,520,320 B | −85,499,904 B |
| cgroup `max` / `oom` / `oom_kill` | 0 / 0 / 0 | 0 / 0 / 0 | no events |

The independent candidate sampler captured 1,420 one-second samples. At the
largest current-memory sample, anonymous memory was 1,603,031,040 B and file
memory was 323,252,224 B. Exact cgroup high-water left 143,380,480 B (136.7 MiB)
below the 2 GiB limit, compared with 113,901,568 B (108.6 MiB) in PR #61. This
is a single scene and candidate run; 2 GiB is still an experimental challenge,
not a supported service tier.

## Numerical and test checks

- The real-photo full pipeline output is byte-identical to PR #61.
- In a deterministic optical-scatter A/B fixture, the float32 workspace
  differed from the float64 tile reference by maximum absolute error
  `1.1920928955078125e-7`, mean absolute error `6.27e-11`, and 99th percentile
  absolute error `2.3283064365386963e-10`. Input radiance was not modified, and
  the final uint8 composite matched exactly.
- The local Python suite passed: 214 tests. Ruff lint, Ruff formatting, and
  `git diff --check` passed.
- A tiny trial showed no material runtime change; the native-size sequential
  timings also do not support a speed claim. Scatter remains the dominant CPU
  stage at 1,178.114 s.

This measurement supports the lower transient scatter memory use with preserved
final pixels for this source and fixture. It does not establish float-level
equivalence for every image, multi-scene quality, stable timing, API concurrency,
or safe 2 GiB deployment headroom. Repeat on additional scenes before expanding
the memory claim.
