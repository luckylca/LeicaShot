# Row-major optical scatter accumulators: rejected candidate

Date: 2026-10-24. This experiment moved the field-optical scatter loop from
CoC-level-major traversal to row-band-major traversal, keeping the full-frame
output but restricting `own` and coverage accumulators to the current row band.
All kernels for the used radii were cached for reuse. The code was reverted after
the full task showed higher memory and longer runtime than merged PR #62.

## Configuration and provenance

- Candidate code commit: `31906f0`; reverted on the experiment branch by
  `d6b9bbf`. No candidate code is proposed for merge.
- Reference: merged PR #62 (`08af060`) and its
  [float32 workspace report](highlight-scatter-fp32-workspace-2g-2026-10-24.md).
- Source tree SHA-256 used by the container:
  `5267caa33e9e2065fd1b237d0ce2dfc4a970c6f3f1532b9ee77ab52e4b963db6`; mounted
  source path `/opt/laika-src/src/laika/__init__.py`.
- Input: CC0 mountains JPEG, 5760×3840, SHA-256
  `4e6661606c195b2462c0f2321a7feefe2ae2fc081499f3462a30fbb108cd7f8e`.
- Request: M9 + LabFusion, Depth Anything V2, SAM 2.1 Tiny, ViTMatte Small,
  BokehMe ARNet and optical highlights, with subject point `(0.72, 0.66)`. Low
  Memory, CPUExecutionProvider, one ORT thread, no size override, 512-pixel
  matte tiles, 256-pixel ARNet tiles and 128-row highlight chunks.
- Host: MacBook Air M1; Linux 7.0.0-34 ARM64 Lima guest, 4,085,092,352 B RAM,
  two logical CPUs. The container had a 2,147,483,648 B cgroup v2 memory limit,
  swap disabled and two CPUs.
- Runtime: Python 3.11.17, NumPy 2.4.6, OpenCV 5.0.0 and ONNX Runtime 1.31.0.
- The pinned M9, Depth, ARNet, SAM encoder/decoder and ViTMatte assets were
  hash-verified and ran through one public Python processing-engine request.
  Model hashes are identical to the PR #62 benchmark report.

## Result against PR #62

The complete 5760×3840 RGB PNG was byte-identical to PR #62:
`0a6282df408319950f0bd4c8f56bcfce78ed9e7eba95ce9c50b2c70f5555ade2`.
The image was independently hashed in the Linux guest. The task exited 0 and
did not OOM-kill, but it exhausted the cgroup headroom and was slower.

| Measure | PR #62 | Row-band traversal | Change |
|---|---:|---:|---:|
| Complete task wall | 1,501.569 s | 1,763.941 s | +262.372 s (+17.47%) |
| Process-tree peak RSS | 1,724,833,792 B | 1,940,873,216 B | +216,039,424 B (+12.53%) |
| Scatter-stage process-tree peak RSS | 1,659,846,656 B | 1,940,873,216 B | +281,026,560 B (+16.93%) |
| Optical scatter time | 1,178.114 s | 1,435.059 s | +256.946 s (+21.81%) |
| Exact cgroup `memory.peak` | 2,004,103,168 B | 2,147,483,648 B | +143,380,480 B; reached limit |
| Maximum 1-second `memory.current` | 1,934,520,320 B | 2,147,479,552 B | +212,959,232 B |
| cgroup `max` / `oom` / `oom_kill` | 0 / 0 / 0 | 113 / 0 / 0 | limit reclaim events |

The candidate sampler captured 1,543 one-second samples. At its largest sample,
anonymous memory was 1,840,345,088 B and file memory was 298,176,512 B. The
exact cgroup peak had zero headroom. The process-tree peak moved to scatter and
was higher than the PR #62 scatter-stage peak by 268 MiB. There is no speed or
memory benefit to retain in this loop order.

## Validation and disposition

- Full Python suite passed on the candidate source: 214 tests. Ruff lint, Ruff
  formatting and `git diff --check` passed.
- The native full pipeline preserved the prior PNG exactly, confirming the
  per-pixel calculations, but its scatter and whole-task timings worsened and
  the cgroup hit its hard limit. No cgroup OOM event occurred; reaching the
  memory limit and worsening measured RSS are sufficient to reject it.
- The implementation was reverted before PR creation. Keep the merged PR #62
  level-major traversal in production. This report is a negative result to
  prevent reintroducing row-major accumulator traversal without a new design.
