# Release highlight radiance after scatter: full-resolution CPU run

Date: 2026-10-24. The renderer now drops its full-resolution float32 source
radiance immediately after `scatter_highlights()` returns, before highlight
softening and final compositing. This run uses the merged PR #60 implementation
as the comparison point; its deferred radiance was still retained through those
later stages.

## Configuration and provenance

- Input: CC0 mountains JPEG, 5760×3840, SHA-256
  `4e6661606c195b2462c0f2321a7feefe2ae2fc081499f3462a30fbb108cd7f8e`.
- Request: M9 + LabFusion, Depth Anything V2, SAM 2.1 Tiny, ViTMatte Small,
  BokehMe ARNet, optical highlights, and one subject point `(0.72, 0.66)`.
  Low Memory, focus `(0.5, 0.5)`, aperture `2.8`, 512-pixel ViTMatte tiles,
  256-pixel ARNet tiles, 128-row highlight chunks, one ORT thread, and no output
  size override.
- Host: MacBook Air M1; Linux 7.0.0-34 ARM64 Lima guest, 4,085,092,352 B RAM,
  2 logical CPUs. The task ran in a cgroup v2 with `memory.max=2,147,483,648` B,
  `memory.swap.max=0`, and a 2-CPU quota.
- Runtime: Python 3.11.17, NumPy 2.4.6, OpenCV 5.0.0, ONNX Runtime 1.31.0;
  CPUExecutionProvider for every model.
- The checked-out source was loaded from `/opt/laika-src/src/laika/__init__.py`.
  Its Python source tree SHA-256 was
  `78828920fc6b3c285b957c6f05c42ded4fe8e5d20eb1d7dfe1ee93d6a4e1e9b6`.
- Model hashes: M9
  `8936b8c86656265ab07c30b9b4cbbd7fa195d6e44cb3be1d4fd059bb36781e33`, Depth
  `a8ce30941c85b2c9af29fb26f36fbf7be3c1ba1dcf348187a51f6eaa856a238e`, ARNet
  `f0696c16a6eb02099292176f25b47e4dc53863e7e4f3f7748e66470d54cf0173`, SAM
  encoder/decoder
  `37de035410f11fb505c210ce1f8b10b450cb018fbfa7f226ec7e7ab542955c75` /
  `3a6a96c62fd39c1369641ea626e5b2e4aed10de877f85291e9993a044d5bf981`, and
  ViTMatte `a5c0b49a664867021eaf2b9cadcdc236ea511028d5e709303b418431f20e223b`.

## Comparison

The comparison row is the source-verified 2 GiB Low Memory run recorded after
PR #60 (`60740ee`), with the same photo, models, settings, guest, and cgroup
limit. The new task is one sequential repeat, so timing differences are
observations and do not establish a speed change.

| Measure | PR #60: retained after scatter | Release after scatter | Change |
|---|---:|---:|---:|
| Complete task wall | 1,553.704 s | 1,502.613 s | −51.091 s; not a speed claim |
| Process-tree peak RSS | 1,855,737,856 B | 1,767,378,944 B | −88,358,912 B |
| `bokeh.highlight_softening` process-tree peak | 1,855,737,856 B | 1,592,594,432 B | −263,143,424 B |
| `bokeh.highlight_scatter` process-tree peak | 1,756,311,552 B | 1,767,378,944 B | +11,067,392 B |
| Exact cgroup `memory.peak` | 2,145,607,680 B | 2,033,582,080 B | −112,025,600 B |
| Largest 1-second cgroup `memory.current` | 2,041,085,952 B | 2,020,020,224 B | −21,065,728 B |
| cgroup `max` / `oom` / `oom_kill` | 0 / 0 / 0 | 0 / 0 / 0 | no events in either run |
| Highlight scatter | 1,212.944 s | 1,181.810 s | timing varies between runs |
| Highlight radiance construction | 1.703 s | 0.997 s | one observation per run |

The independent sampler recorded 1,419 one-second cgroup samples. At its
maximum sample, `memory.current` was 2,020,020,224 B, with 1,654,239,232 B
anonymous and 322,326,528 B file memory. The exact cgroup high-water left
113,901,568 B (108.6 MiB) below the 2 GiB limit. The process-tree high-water
moved to scatter; the softening-stage sampled peak fell by about 251 MiB.

## Output and validation

- The task completed on CPU with a full-size 5760×3840 RGB PNG, 16,655,520 B.
- The engine-reported and independently computed output SHA-256 are both
  `0a6282df408319950f0bd4c8f56bcfce78ed9e7eba95ce9c50b2c70f5555ade2`, exactly
  matching the prior full-resolution five-model result.
- All five model assets were hash-verified and ran through the public Python
  processing engine in one call. The report identifies the mounted checkout
  source and its tree hash; model assets and output stayed outside Git.
- The radiance lifetime regression test uses weak references to prove the
  source buffer is gone before softening in both Low Memory and Balanced modes.
  The complete local Python suite passed (213 tests), and Ruff lint, formatting,
  and `git diff --check` passed.

This is one full-resolution scene and one candidate run. It records improved
process RSS at the softening stage and a lower exact cgroup peak than PR #60,
but does not establish robust 2 GiB API headroom, concurrency capacity, or a
stable runtime improvement. Scatter is now the measured process-tree peak and
remains the dominant CPU stage; optimize it next while preserving the exact
output hash.
