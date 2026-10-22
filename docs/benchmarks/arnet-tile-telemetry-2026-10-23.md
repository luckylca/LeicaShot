# Full-resolution ARNet tile telemetry in a 3 GiB cgroup

Date: 2026-10-23. Run from the mounted `perf/fullres-cgroup-tile-telemetry`
checkout with the Python CPU engine. This validates tile progress and records a
completed no-subject full-resolution pipeline; it does not measure SAM or
ViTMatte.

## Configuration

- Input: CC0 Mountains Landscape JPEG, 5760×3840, SHA-256
  `4e6661606c195b2462c0f2321a7feefe2ae2fc081499f3462a30fbb108cd7f8e`.
- Mode and strategy: `color+bokeh`, Low Memory, CPU-only, no output-size
  override. Output remained 5760×3840.
- Platform: Linux ARM64 Lima guest, 2 logical CPUs, 4 GiB guest RAM, 2 CPU
  quota, one ONNX Runtime intra-op thread. Task cgroup v2 limit was
  3,221,225,472 B (3 GiB). Python 3.11, NumPy 2.4.6, OpenCV 5.0.0.93, and
  ONNX Runtime 1.31.0.
- Models: M9 SHA-256
  `8936b8c86656265ab07c30b9b4cbbd7fa195d6e44cb3be1d4fd059bb36781e33`;
  Depth Anything V2 SHA-256
  `a8ce30941c85b2c9af29fb26f36fbf7be3c1ba1dcf348187a51f6eaa856a238e`;
  ARNet SHA-256
  `f0696c16a6eb02099292176f25b47e4dc53863e7e4f3f7748e66470d54cf0173`.
- Bokeh highlight row chunk: 128. The mounted source was selected explicitly
  with `PYTHONPATH=/opt/laika-src`; source, model, and input mounts were
  read-only. The final report and progress JSONL were retrieved from a named
  Docker volume before the Lima Docker socket became unavailable.

## Result

| Measure | Result |
|---|---:|
| Complete task wall time | 4,789.789 s (79.83 min) |
| Process-tree sampled peak RSS | 2,357,911,552 B (2.197 GiB) |
| Linux cgroup peak | 2,696,331,264 B (2.511 GiB) of 3 GiB |
| Cgroup OOM / limit events | 0 / 0 |
| M9 inference / LabFusion | 3.471 s / 3.013 s |
| Depth inference | 13.886 s |
| ARNet passes | 151.616 / 37.077 / 12.530 / 9.035 s |
| Highlight scatter | 4,473.569 s |
| Output | RGB PNG, 5760×3840, 16,621,497 B |
| Output SHA-256 | `62bf1f9b29a93e15435d5f38eb9975d5e910ca773fc5eb4e6de43b16a8ca1aef` |

All four ARNet passes and highlight processing completed. The per-tile log
contains 120 tile starts and 120 completions across the two tiled ARNet scales
(96 tiles at 5760×3840 and 24 at 2880×1920). Progress fractions were monotonic
from 0.05 through 1.0. Tile-level cgroup samples reached 2,208,690,176 B;
later highlight softening sampled 2,319,450,112 B process-tree RSS. The global
cgroup peak was higher at 2,696,331,264 B.

The output SHA-256 is identical to the optimized 512-row full-resolution
result in the [paired highlight cgroup report](full-resolution-highlight-cgroup-2026-10-22.md).
That establishes byte-identical exported pixels for this photo. It is not a
controlled 128-versus-512 performance comparison: the reference run used a
different source revision and did not include the new per-tile callbacks. Its
recorded task time was 2,112.493 s and scatter time was 1,739.518 s, while this
128-row run took 4,789.789 s and 4,473.569 s respectively. The gap makes a
current-source, same-session chunk-size sweep the next performance experiment;
these two observations alone do not attribute the slowdown to row chunk size.

No model weights, photographs, generated outputs, or progress logs are stored
in Git. The local output and raw task data are under
`/tmp/laika-pipeline-results/arnet-tile-telemetry/`.

## Reproduction

Use the same validated local model assets and source photograph, mount the
current repository at `/opt/laika-src`, and set `PYTHONPATH=/opt/laika-src` in
the container. Run the benchmark with `--mode color+bokeh --strategy low-memory
--threads 1 --bokeh-row-chunk-height 128`, no output-size override, and a 3 GiB
cgroup (`--memory=3g --memory-swap=3g --cpus=2`). Preserve the generated report
and JSONL progress log outside the repository.
