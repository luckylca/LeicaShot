# Full-resolution mass-filter conversion experiments

Status: both candidate implementations were rejected. The persistent
dual-workspace variant failed the matched 2 GiB comparison; direct
FP32-input / FP64-output filtering preserved the image at 4 GiB but did not
reduce process RSS and was substantially slower in the full task. Production
code remains at the PR #67 baseline.

## Task and reference

- Input: CC0 `Mountains_Landscape.jpg`, 5760×3840, SHA-256
  `4e6661606c195b2462c0f2321a7feefe2ae2fc081499f3462a30fbb108cd7f8e`.
- Task: CPU-only `color+bokeh`, one thread, M9 + LabFusion + Depth Anything V2
  Small + SAM 2.1 Tiny + ViTMatte Small + BokehMe ARNet + optical highlights,
  subject point `(0.72, 0.66)`, 512-pixel matte tiles and 256-row highlight
  chunks. Both tasks kept native output resolution.
- Model digests are recorded in the external JSON reports; the source,
  model files, input and outputs remain outside Git.
- Linux environment: ARM64 Ubuntu, Python 3.11.17, NumPy 2.4.6, OpenCV 5.0.0,
  ONNX Runtime 1.31.0, 2 CPU quota, one inference thread, swap disabled.
- Baseline: merged PR #67 implementation, mounted from commit `19d8623`.
- Candidate: initially mounted from `perf/highlight-mass-filter-workspace`;
  the direct-depth variant now in the same branch is evaluated separately.

## Matched 2 GiB comparison: persistent dual buffers rejected

The first candidate converted each FP32 mass tile into a reusable FP64 source
buffer and reused a second FP64 destination buffer. Both full tasks completed
without cgroup `high`, `max`, or OOM events and produced byte-identical PNGs.

| Measure | PR #67 baseline | Dual-buffer candidate |
|---|---:|---:|
| Task wall / engine | 1,422.586 / 1,422.561 s | 1,618.954 / 1,618.934 s |
| Highlight scatter | 1,110.879 s | 1,267.897 s |
| FP64 mass filters | 12,041 / 401.966 s | 12,041 / 453.597 s |
| Process-tree peak RSS | 1,776,005,120 B | 1,769,455,616 B |
| cgroup v2 peak | 1,916,710,912 B | 1,963,257,856 B |
| cgroup max/OOM events | 0 / 0 | 0 / 0 |
| PNG SHA-256 | `0a6282df408319950f0bd4c8f56bcfce78ed9e7eba95ce9c50b2c70f5555ade2` | same |

The candidate raised cgroup peak by 46,546,944 B, increased scatter time by
14.1%, and increased total task time by 13.8%. It is rejected despite the
slightly lower sampled process-tree RSS. External artifacts are in
`mass-workspace-baseline-2g-20261024/` and
`mass-workspace-candidate-2g-20261024/` under
`/Users/lucky/Desktop/project/leicashot-photo-validation/`.

## Direct FP32 input to FP64 output

The current Linux candidate calls OpenCV `filter2D` with the original FP32
mass tile, `ddepth=CV_64F`, the original FP64 aperture kernel and
`BORDER_REFLECT_101`. On Linux/OpenCV 5.0.0, its output was bitwise identical
to first converting the same input tile to FP64 and then filtering, for both
small and native-shaped tiles. A seven-repeat native-shaped OpenCV screen
measured median 0.03662 s for direct input versus 0.03757 s for the explicit
cast path (one kernel/shape; a screening result, not a full-pipeline speed
claim).

The local macOS OpenCV build rejects this source/destination depth pair. A
temporary adapter probed support once at module import and used the original
cast path on unsupported builds; all 216 local tests passed on that fallback.
The full 4 GiB task produced the exact reference PNG, but its mass-filter and
scatter timings were markedly slower than the 256-row baseline. Process RSS
was slightly higher, so the lower cgroup high-water cannot be attributed to a
process-memory reduction and may reflect host page-cache variation.

| Measure | PR #67 4 GiB baseline | Direct-depth candidate |
|---|---:|---:|
| Task wall / engine | 1,468.974 / 1,468.959 s | 2,160.203 / 2,160.180 s |
| Highlight scatter | 1,032.326 s | 1,707.947 s |
| FP64 mass filters | 12,041 / 378.515 s | 12,041 / 560.675 s |
| Process-tree peak RSS | 1,756,979,200 B | 1,773,395,968 B |
| cgroup v2 peak | 2,028,392,448 B | 1,868,193,792 B |
| cgroup max/OOM events | 0 / 0 | 0 / 0 |
| PNG SHA-256 | `0a6282df408319950f0bd4c8f56bcfce78ed9e7eba95ce9c50b2c70f5555ade2` | same |

The candidate's mass-filter time rose 48.1%, scatter time 65.4% and total
engine time 47.1%; process-tree peak RSS rose 16,416,768 B. The 160,198,656 B
lower cgroup peak does not outweigh this measured CPU regression and did not
correspond to lower process RSS. Do not use direct-depth filtering in
production. No 2 GiB direct-depth run was started after the native-size 4 GiB
candidate failed the CPU and process-RSS acceptance gates.

## Reproduction artifacts

The baseline report, profile, progress trace and PNG are outside Git under
`/Users/lucky/Desktop/project/leicashot-photo-validation/mass-workspace-baseline-2g-20261024/`.
The rejected dual-buffer candidate and direct-depth run artifacts are stored
under their corresponding `mass-workspace-candidate-2g-20261024/` and
`mass-filter-cast-elision-4g-20261024/` directories. The script was
`scripts/profile_native_scatter.py` from each read-only mounted source tree.
