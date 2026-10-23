# Full-resolution five-model CPU task at a 2 GiB cgroup limit

Date: 2026-10-24

## Result

The current Python engine completed a single 5760×3840 task with M9, Depth
Anything V2, SAM 2.1 Tiny, ViTMatte Small, BokehMe ARNet, optical highlights,
subject protection, and PNG export. It ran in Low Memory mode using
CPUExecutionProvider and preserved the native output dimensions.

This is one completion at a 2 GiB cgroup limit. The cgroup `memory.peak`
reached the full limit, so it does not establish a safe service headroom or a
repeatable 2 GiB deployment target. It does establish that this exact task
completed without an OOM kill after lowering Low Memory ARNet's full-frame
threshold to 1 MP.

## Configuration

- Input: CC0 mountains JPEG, 5760×3840, SHA-256
  `4e6661606c195b2462c0f2321a7feefe2ae2fc081499f3462a30fbb108cd7f8e`.
- Request: `color+bokeh`, Low Memory, focus `(0.5, 0.5)`, aperture `2.8`,
  foreground subject point `(0.72, 0.66)`, 512-pixel ViTMatte tiles, 128-row
  highlight chunks, one ORT intra-op thread. No output-size override or crop.
- Source: `develop` after PR #56, commit `c4db3bccd714db586cd0f8c837128cb7bb85d81d`.
- Host: M1 Mac, Linux ARM64 Lima guest, 4,085,092,352 B guest memory, 2 vCPUs.
  Docker cgroup v2 limit `2,147,483,648` B, `memory.swap.max=0`, 2 CPU quota.
- Runtime: Python 3.11, NumPy 2.4.6, OpenCV 5.0.0, ONNX Runtime 1.31.0.
  All neural inference used `CPUExecutionProvider`.
- Models were mounted read-only from local authorized assets and their
  SHA-256 hashes were checked before execution:
  - M9: `8936b8c86656265ab07c30b9b4cbbd7fa195d6e44cb3be1d4fd059bb36781e33`.
  - Depth Anything V2 Small: `a8ce30941c85b2c9af29fb26f36fbf7be3c1ba1dcf348187a51f6eaa856a238e`.
  - BokehMe ARNet: `f0696c16a6eb02099292176f25b47e4dc53863e7e4f3f7748e66470d54cf0173`.
  - SAM encoder / decoder ONNX:
    `37de035410f11fb505c210ce1f8b10b450cb018fbfa7f226ec7e7ab542955c75` /
    `3a6a96c62fd39c1369641ea626e5b2e4aed10de877f85291e9993a044d5bf981`.
  - ViTMatte Small ONNX:
    `a5c0b49a664867021eaf2b9cadcdc236ea511028d5e709303b418431f20e223b`.

## Measurements

| Measurement | Result |
|---|---:|
| Complete task wall time | 1,475.646 s (24.59 min) |
| Process-tree peak RSS | 2,067,648,512 B |
| Process-tree peak stage | ARNet tiled inference |
| cgroup `memory.peak` | 2,147,483,648 B (limit reached) |
| Largest 1-second sampled `memory.current` | 2,138,918,912 B |
| Sampled anon / file at that sample | 1,984,520,192 / 148,275,200 B |
| cgroup `memory.events` max / oom / oom_kill | 285 / 0 / 0 |
| M9 inference / LabFusion | 3.816 / 2.950 s |
| Depth inference | 13.943 s |
| SAM encoder / decoder | 5.290 / 0.096 s |
| ViTMatte inference | 6.352 s |
| ARNet passes | 159.547 / 39.885 / 10.576 / 3.939 s |
| Optical highlight scatter | 1,180.937 s (80.0% of task wall time) |
| Output | RGB PNG, 5760×3840, 16,655,520 B |
| Output SHA-256 | `0a6282df408319950f0bd4c8f56bcfce78ed9e7eba95ce9c50b2c70f5555ade2` |

The engine report hash and an independent hash of the output file match. The
SHA-256 also matches the prior source-verified 3 GiB subject-protection output
for this photo and request. The SAM point score was `0.9225843`; one point was
used, so this run does not test multi-prompt embedding-cache reuse. GAIC was
not requested and was not loaded.

## Interpretation and next work

The Low Memory 1 MP ARNet threshold allowed the task to pass the 1440×960
scale as six 512×512 full-context tiles. During the run, cgroup `memory.peak`
reached `memory.max`, and the external one-second sampler observed only about
8.2 MiB below the limit at its highest sample. There were no cgroup OOM kills,
and the task wrote a complete native-resolution result. This is a measured
single-task completion, not adequate headroom for API concurrency or all
scenes.

Next, reduce the ARNet-stage anonymous-memory high-water without changing tile
coverage, output dimensions, or model precision. Repeat the same cgroup task
for any candidate and keep the exact output hash, OOM counters, and full
process-tree/cgroup measurements. The optical scatter stage remains the main
CPU-time cost. Input, model, output, progress, and cgroup sampling artifacts
are retained outside Git in Docker volume
`leicashot-subject-2g-20261024`.

## Reproduction

Run `scripts/benchmark_processing_pipeline.py` in a Linux ARM64 container with
the model, input, and output mounts used in the run, under
`--memory=2g --memory-swap=2g --cpus=2 --pids-limit=128`. Pass
`--mode color+bokeh --strategy low-memory --threads 1 --focus-x 0.5
--focus-y 0.5 --aperture 2.8 --matte-tile-size 512
--subject-point 0.72,0.66 --bokeh-row-chunk-height 128` and
`--progress-log /out/progress.jsonl`. Do not set an output size. The external
one-second cgroup sampler wrote `cgroup-memory.tsv` to the same results volume.
