# Reuse depth weights across optical scatter field nodes

Date: 2026-10-23. Candidate implementation is commit `cd4e6f9` on
`perf/native-resolution-scatter-profile`; the parent is the PR #38 merge
`8360f91`. Scaled and full-resolution candidate tasks and a same-host parent
comparison are complete. The candidate preserved exact pixels but was slower
and had a higher total process-tree peak on the native-resolution paired run,
so the implementation is rejected for production and will be reverted.

## Change under test

`_scatter_field_optical()` evaluates a 5×5 field grid in bounded row chunks.
Each horizontal field node previously gathered the same depth pixels for a
given vertical node and recomputed the level's CoC layer weights. The candidate
computes the width-wide CoC and weight arrays once per active vertical node and
row chunk, then slices the required columns for each horizontal node. Kernel
generation, convolution, node accumulation order, occlusion math, output type,
aperture and image dimensions are unchanged.

## Fixed input and environment

- Source photo: CC0 Mountains Landscape JPEG, 5760×3840, SHA-256
  `4e6661606c195b2462c0f2321a7feefe2ae2fc081499f3462a30fbb108cd7f8e`.
- The cached M9 + LabFusion RGB image is SHA-256
  `e631ed1e7885e9ebdff8eefa2631ced973e3744472201fb0d01f3c2803f9edba`;
  the raw Depth Anything V2 map is
  `ed65f464cddb540d09838c80b5ebeef998900b79336e70a9c58a6deca43302cb`.
  Both were generated once from the source photo using the verified local
  models. The benchmark resized these same cached arrays for each paired run.
- Hardware: MacBook Air M1, 8 GiB RAM, CPU-only. Python 3.11.16 and OpenCV
  5.0.0. The scatter worker uses the same 128-row chunks and aperture 2.8 for
  both variants.
- Baseline source is `8360f91`; candidate source is `cd4e6f9`. Each paired run
  reads the same cached highlight-linear and CoC arrays. Both orders were run
  (`before, after, after, before`) to reduce ordering bias.

## Paired results

| Prepared size | Nonzero highlight pixels | Parent times | Candidate times | Mean change | Output array SHA-256 | Mean sampled process-tree peak RSS |
|---|---:|---:|---:|---:|---|---:|
| 1024×683 | 115 | 9.472 / 12.711 s | 8.407 / 8.551 s | 11.091 → 8.479 s (23.5% lower) | `f45f5f37d61a1ae03633a3c3f71eb552f654d989c23cb063020b33c6319c4f8f` for all runs | 163,790,848 → 167,329,792 B |
| 2048×1365 | 1,903 | 84.138 / 109.524 s | 81.581 / 80.077 s | 96.831 → 80.829 s (16.5% lower) | `f1d5001211f1a234209123fc365687908ba3915fd6463bd7ccd9b40c9be17e7d` for all runs | 323,198,976 → 325,632,000 B |

The 2048 output SHA proves the scatter float32 array is bitwise identical for
these runs. Mean sampled peak RSS rose by 2,433,024 B (0.75%) at 2048. The
timing comparison has two runs per variant and is a single-photo CPU result;
the parent times varied by about 26 s at 2048. Native-size same-host results
below reverse the apparent scaled-size timing benefit. These results do not
predict a service throughput guarantee.

The prepared 2048 arrays were held outside Git:

- Highlight-linear SHA-256:
  `2f93c3866f8316d7757fd52185c81ab72a79435fa4caf970969f8cdc6b4a798e`.
- CoC SHA-256:
  `a0bfb70065ac382ea232b5fd12ac683159fc4e18f90ef37d182a5574160533fa`.

## CPU profile and rejected probes

A 1024×683 `cProfile` run of the parent implementation took 19.995 s in
`scatter_highlights()`. It recorded 2,392 `cv2.filter2D` calls totaling
8.166 s, 0.954 s in optical pupil-kernel construction, and 19.935 s cumulative
in `_scatter_field_optical()`. This candidate reduces repeated per-node
depth/weight preparation; it does not reduce filter call count.

A prototype that packed mass and RGB radiance into one four-channel
`filter2D` call produced the same output hash but took 13.112 s at 1024 versus
11.202 s for the existing implementation, so it was discarded. OpenCV
5.0.0 also reports eight threads after `cv2.setNumThreads(1/2/4/8)`; those
thread-count probes did not establish a controllable thread setting and are
not used as performance evidence.

## Full-resolution candidate run

Both parent (`8360f91`) and candidate (`cd4e6f9`) processed the same 5760×3840
photo on a MacBook Air M1 with Low Memory, CPUExecutionProvider, one ONNX
thread, 512-pixel ViTMatte tiles, point `(0.72, 0.66)`, aperture 2.8, and
128-row highlight chunks. Each was one full task in sequential order; cgroup
measurement is unavailable on macOS.

| Metric | Parent | Candidate | Candidate change |
|---|---:|---:|---:|
| Whole-task wall | 3,169.499 s | 3,257.537 s | +88.038 s (+2.78%) |
| Optical scatter stage | 2,858.616 s | 2,936.921 s | +78.306 s (+2.74%) |
| Process-tree peak RSS | 1,362,968,576 B | 1,458,585,600 B | +95,617,024 B (+7.01%) |
| Scatter-stage sampled peak RSS | 1,093,779,456 B | 1,029,849,088 B | −63,930,368 B (−5.84%) |
| Output size / SHA-256 | 5760×3840 / `d97ab9d208c0d2956a3ea82369da398cf9d173253cba952c83218d4fcf949655` | same | identical |

Decoded RGB comparison found all 22,118,400 pixels and all channel values
identical. M9 inference took 5.112 s and Depth inference 23.870 s in the
candidate run. The candidate reduced the scatter-stage sampled RSS in this
single run but raised whole-task peak RSS, total wall time, and scatter wall
time; therefore it does not meet the project's full-pipeline optimization
acceptance and is rejected for production. A second identical native-size
trial would improve confidence in the timing difference, but is not needed to
justify declining a change that has no demonstrated end-to-end benefit.

The candidate Mac PNG compared to the available PR #38 Linux baseline had
MAE 0.2686, RMSE 0.5434, p95 absolute channel error 1, p99 2, maximum 43, and
98.82% exactly equal pixels. The same-Mac parent run now proves that the
candidate and parent outputs are exactly equal; the Linux/Mac discrepancy is
platform/backend numerical drift. The SAM point score differed slightly
(0.922584 on Linux vs 0.923018 on Mac).

Two candidate-source Linux attempts stopped in ARNet before this scatter
function ran. A 3 GiB task cgroup hit a memory-cgroup OOM at 3,137,860 KiB
anonymous RSS. A 3.25 GiB cgroup cap exceeded the 4 GiB guest's available
memory and produced a global OOM at 3,388,612 KiB anonymous RSS. These failures
do not measure the candidate scatter stage and must not be attributed to this
change. The successful PR #38 Linux baseline completed in the same nominal
4 GiB guest / 3 GiB cgroup, so candidate Linux low-memory reproducibility
remains open. These two failed attempts do not establish a regression because
they stopped before the changed function; they leave Linux candidate
full-resolution cgroup acceptance unverified. Since the native Mac paired run
shows no full-task benefit, the code will be reverted instead of retrying the
candidate on Linux.

## Reproduction

Create the M9 + LabFusion RGB and raw Depth Anything V2 map for the source
photo once, outside the repository. Run the existing paired harness twice,
using the same cached files and options, changing only the order:

```sh
PYTHONPATH=src .venv/bin/python scripts/benchmark_highlight_scatter.py \
  /tmp/laika-scatter-profile/mountains-m9.png \
  /tmp/laika-scatter-profile/mountains-depth.npy \
  /tmp/laika-scatter-profile/scatter-before-first.json \
  --baseline-revision 8360f91 --long-edge 2048 --row-chunk-height 128 \
  --order baseline-first

PYTHONPATH=src .venv/bin/python scripts/benchmark_highlight_scatter.py \
  /tmp/laika-scatter-profile/mountains-m9.png \
  /tmp/laika-scatter-profile/mountains-depth.npy \
  /tmp/laika-scatter-profile/scatter-after-first.json \
  --baseline-revision 8360f91 --long-edge 2048 --row-chunk-height 128 \
  --order optimized-first
```

The raw image, model assets, arrays, outputs and reports remain outside Git.
