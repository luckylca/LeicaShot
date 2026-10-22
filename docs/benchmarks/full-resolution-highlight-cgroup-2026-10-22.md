# Native-resolution highlight pipeline in a 3 GiB Linux cgroup

Updated: 2026-10-22

This records the pre-optimization full-resolution baseline. The paired run with
PR #32's exact-zero radiance optimization is still pending.

## Configuration

- Input: the CC0 [Mountains Landscape](https://commons.wikimedia.org/wiki/File:Mountains_Landscape.jpg) JPEG, 5760×3840, 3,878,306 bytes, SHA-256 `4e6661606c195b2462c0f2321a7feefe2ae2fc081499f3462a30fbb108cd7f8e`.
- Engine snapshot: `develop` at `762ff93` (PR #31's tiled ARNet, before PR #32's exact-zero highlight optimization).
- Models: M9 `8936b8c86656265ab07c30b9b4cbbd7fa195d6e44cb3be1d4fd059bb36781e33`; Depth `a8ce30941c85b2c9af29fb26f36fbf7be3c1ba1dcf348187a51f6eaa856a238e`; ARNet `f0696c16a6eb02099292176f25b47e4dc53863e7e4f3f7748e66470d54cf0173`.
- Platform: Ubuntu ARM64 Linux guest, 2 logical CPUs and 4,085,092,352 B total memory. The container had a 3,221,225,472 B memory cgroup limit, 2 CPU quota, one ONNX intra-op thread, CPUExecutionProvider, Low Memory strategy, and 512-row highlight chunks.
- Operation: full `color+bokeh` with highlights, no output-size override. The requested and returned size remained 5760×3840.

## Baseline result

| Measure | Result |
|---|---:|
| Complete task wall time | 5,658.667 s (94.31 min) |
| Process-tree sampled peak RSS | 2,502,225,920 B |
| Linux cgroup peak | 2,676,203,520 B of 3,221,225,472 B |
| OOM / limit events | 0 / 0 |
| Highlight scatter stage | 5,323.159 s |
| M9 inference | 7.180 s |
| Depth inference | 15.353 s |
| ARNet passes | 193.806 / 56.912 / 12.182 / 4.916 s |
| Output | 5760×3840 RGB PNG, 16,621,497 bytes |
| Output SHA-256 | `62bf1f9b29a93e15435d5f38eb9975d5e910ca773fc5eb4e6de43b16a8ca1aef` |

The task completed successfully and Docker reported `OOMKilled=false`. The
largest measured stage was field-dependent highlight scatter, at about 94% of
total task time. The final PNG was copied from the completed container to
`/tmp/laika-pipeline-results/python-fullres-tiled-512.png`; no photo or model is
stored in Git.

## Comparison still required

The next run must use the same input, models, 5760×3840 output, CPU quota,
thread count, Low Memory strategy, and 512-row chunks with the merged PR #32
code. Record total time, stage times, process-tree RSS, cgroup peak/events,
output dimensions and hash. Compare decoded RGB against the baseline and report
max, mean, p95, p99, changed-pixel count, and whether the full output is
byte-identical. The baseline alone does not establish the optimized full-size
runtime or an 8/4/2 GiB acceptance result.
