# Verified ARNet split-path full-pipeline A/B — Windows CPU

Date: 2026-10-25. PR #75 merged source bf5dc98 plus the processing engine
wiring fix on branch fix/arnet-split-unified-engine-wiring.

## Motivation and critical benchmark correctness

PR #75 added the verified global-resize/tiled-core ONNX subgraphs and
standalone ARNet inference, but the unified LaikaProcessingEngine used by
the Python SDK, web API and photo pipeline was not passing the optional
LAIKA_ARNET_SPLIT_DIR setting to BokehMeARNet. The earlier experiment
with a 58.379-second five-model task and byte-identical PNG was in fact
using the original, UNSPLIT BokehMe ARNet. Those observations must not be
used as evidence of split-graph full-task throughput or pixel fidelity.

Fix: pass this environment opt-in to _acquire_bokeh low-memory only. The
existing balanced mode and unset variable behavior remain unchanged.
A regression test now inspects the factory kwargs for on/off/balanced
configurations. For this new measurement, the benchmark additionally
wraps BokehMeARNet._infer_split and **asserts it was entered**; traces
two calls at 2048x1365 and 1449x966 (multi-scale ARNet passes).

## Fixed workload

- Host: Windows 10.0.22631 on a laptop with 15.7 GiB memory, ORT
  CPUExecutionProvider 1.31.0, Python 3.11, model threads 1.
- Image: the existing CC0 portrait resized bicubic to 2048x1365.
- Complete processing: M9 color / Lab fusion, Depth Anything V2 Small,
  SAM 2.1 Tiny click, ViTMatte Small 512 tiles, two ARNet passes, optical
  highlight scatter with PR #74 support scheduling, output PNG.
- Two separate fresh worker processes per pair. One arm with
  LAIKA_ARNET_SPLIT_DIR unset, and the other with the SHA256-verified
  original-authorized ARNet graph prefix/core/suffix; all other model
  inputs and settings identical.
- Two execution orders: control/split then split/control. Both output
  2048x1365 PNGs. RSS sampled every 150ms within each Python process.
- The split mode performs its convolution on 256-feature-pixel blocks
  with 8 feature-pixel halo, retaining the original global alignment
  transformations before and after the convolutional network.

## Full five-model results

| Order | Control seconds | Split seconds | Control RSS B | Split RSS B |
|---|---:|---:|---:|---:|
| Control -> split | 59.0813 | 59.6444 | 1,295,273,984 | 983,580,672 |
| Split -> control | 59.0245 | 62.7642 | 1,296,072,704 | 1,013,878,784 |
| Mean | 59.0529 | 61.2043 | 1,295,673,344 | 998,729,728 |

The split path reduces sampled peak worker RSS by 296,943,616 B
(22.9%, 283.2 MiB on the two-pair means). Split mean whole-task time
is about 3.6% longer; CPU warm/cold effects remain and more photos
and runs are needed before estimating stable speed.

The ARNet stage is not exactly bitwise equivalent to original ONNX
because different CPU convolution tile sizes have slightly different
rounding. The decoded final PNG changed in 100,167 / 2,795,520 pixels
(3.583%), never by more than 1 RGB 8-bit channel value in this photo,
with mean channel absolute error 0.0122021. The changed pixel count,
max channel error and MAE are identical in both paired runs.

## Standalone ARNet validation

On the same machine with a 2048x1365 odd-height input:
full graph 12.409s and 1,217,134,592 B sampled peak RSS; optional
split path 14.413s and 305,594,368 B. Split ARNet 4.5% of pixels
had at most one channel unit difference. Odd width, odd height and
both odd at 512px also tested without crashes; max difference one
8-bit channel unit on those inputs.

These isolated model figures are NOT a substitute for the whole-engine
results above.

## Limits

- The old statement of approximately 58.379s exact-output identity for
  the split option was not a valid active split-path experiment.
- The 2 GiB Linux cgroup ~20MP CPU image task was not rerun. Sampled
  process RSS on a 2.8MP frame is not a Linux cgroup memory high-water
  and does not justify claiming 2 GiB / 4 GiB production readiness.
- Model subgraphs are not redistributed and must be locally exported
  from an authorized source ONNX and SHA256 verified.
- The opt-in should remain optional until more diverse real-photo
  fidelity/latency tests have been carried out.

## Local reproduction

The Windows local-data directory retains the authorized model weights,
test photo and scripts outside version control:
- E:\codex\LeicaShot-local-data\benchmark-full-support-2048.py
- E:\codex\LeicaShot-local-data\benchmark-real-split-2048.py
- E:\codex\LeicaShot-local-data\laika-models\arnet-split-verified

GitHub PRs must run cross-platform CI before merge.
