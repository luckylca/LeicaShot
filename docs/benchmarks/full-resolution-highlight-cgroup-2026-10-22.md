# Native-resolution highlight pipeline in a 3 GiB Linux cgroup

Updated: 2026-10-22

This records paired full-resolution Linux runs before and after PR #32's
exact-zero radiance optimization. Both used the same source image, model assets,
output dimensions, ARM64 guest, and container resource limits.

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

## Optimized result

The optimized run used `develop` after PR #32 (`a503d0b`) and the same pinned
models, source, full-size output, ARM64 Linux guest, 3 GiB cgroup, 2 CPU quota,
one ONNX thread, Low Memory mode, and 512-row chunks. The task exited normally
with Docker `OOMKilled=false`; all cgroup OOM/limit event counters were zero.

| Measure | Before PR #32 | With PR #32 | Change |
|---|---:|---:|---:|
| Complete task wall | 5,658.667 s | 2,112.493 s | 62.7% shorter |
| Highlight scatter | 5,323.159 s | 1,739.518 s | 67.3% shorter |
| Process-tree sampled peak RSS | 2,502,225,920 B | 2,369,044,480 B | 5.3% lower |
| Linux cgroup peak | 2,676,203,520 B | 2,427,625,472 B | 9.3% lower |
| Cgroup limit | 3,221,225,472 B | 3,221,225,472 B | same |
| OOM / limit events | 0 / 0 | 0 / 0 | none |
| M9 inference | 7.180 s | 10.060 s | — |
| Depth inference | 15.353 s | 24.127 s | — |
| ARNet passes | 193.806 / 56.912 / 12.182 / 4.916 s | 212.942 / 49.564 / 22.195 / 4.333 s | — |
| Output | 5760×3840 RGB PNG, 16,621,497 B | 5760×3840 RGB PNG, 16,621,497 B | same |

The output SHA-256 is identical in both runs:
`62bf1f9b29a93e15435d5f38eb9975d5e910ca773fc5eb4e6de43b16a8ca1aef`.
Decoded RGB comparison also measured max absolute error 0, MAE 0, RMSE 0,
p95/p99 error 0, and zero changed pixels. This is a bit-for-bit image-quality
parity result at 5760×3840. Stage timings outside the optimized scatter path
varied between runs; these single paired observations do not establish a
throughput guarantee.

The optimized task's per-stage measurements were: depth inference 24.127 s,
M9 inference 10.060 s, LabFusion 3.310 s, ARNet passes 212.942 / 49.564 /
22.195 / 4.333 s, highlight scatter 1,739.518 s, and export 6.932 s. Its
process-tree sampled peak was 2,369,044,480 B (scatter stage); cgroup peak was
2,427,625,472 B under the 3,221,225,472 B cap. The task report and output were
retrieved from the exited container; the image is at
`/tmp/laika-pipeline-results/python-fullres-optimized-512.png`.

These measurements show that this full-resolution three-model color+bokeh task
fits the tested 3 GiB task cgroup and preserves every output pixel. They do not
establish full-resolution SAM/ViTMatte subject-protection memory, 2 GiB
acceptance, concurrent-service capacity, or performance on other CPUs/images.
