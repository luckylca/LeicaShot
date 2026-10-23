# FP32 mass-filter experiment at native resolution

Updated: 2026-10-23

## Question

Test whether converting the single-channel highlight-mass convolutions from FP64 to FP32 improves the complete CPU task's speed or memory use without changing its encoded result.

## Method

- Each task processed the same 5760×3840 CC0 mountains JPEG (`4e6661606c195b2462c0f2321a7feefe2ae2fc081499f3462a30fbb108cd7f8e`) with M9, Depth Anything V2, SAM 2.1 Tiny, ViTMatte Small, BokehMe ARNet and highlight scatter. Mode was `color+bokeh`, Low Memory strategy, CPUExecutionProvider, one ONNX/OpenCV thread, focus `(0.5, 0.5)`, aperture `2.8`, subject point `(0.72, 0.66)`, 512-pixel ViTMatte tiles and 256-row scatter chunks.
- Runs used fresh Linux ARM64 containers with a 2-vCPU quota, 4 GiB cgroup-v2 memory limit, and memory+swap limited to 4 GiB. Runtime was Python 3.11.17, NumPy 2.4.6, OpenCV 5.0.0 and ONNX Runtime 1.31.0. All five inference models used CPU execution.
- The FP32 production-path implementation was measured twice. The FP64 comparison used the same source, profiler wrapper, inputs and options, with one full-resolution trial. The comparison is indicative rather than a statistically balanced A/B; CPU and model-stage timing varies between runs.
- The profiler-hook pilot produced two exact-output runs with 1,001.460 s mean scatter time. That apparent gain did not reproduce in the two source-loaded production-path trials below, so it is not treated as performance evidence for adoption.

## Results

| Measurement | FP64 control | FP32 production run 1 | FP32 production repeat | FP32 mean |
|---|---:|---:|---:|---:|
| Complete task wall time | 1,305.825 s | 1,425.094 s | 1,370.287 s | 1,397.691 s |
| Highlight scatter stage | 984.026 s | 1,123.879 s | 1,053.048 s | 1,088.464 s |
| Single-channel mass filter time | 360.133 s | 396.703 s | 374.070 s | 385.386 s |
| Process-tree peak RSS | 2,546,049,024 B | 2,589,167,616 B | 2,598,236,160 B | 2,593,701,888 B |
| cgroup-v2 peak memory | 2,629,607,424 B | 2,648,883,200 B | 2,617,556,992 B | 2,633,220,096 B |
| FP32 mass-filter calls | 0 | 12,041 | 12,041 | 12,041 |
| OOM / cgroup-limit events | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

All three output PNGs are 5760×3840, 16,655,520 bytes and have SHA-256 `0a6282df408319950f0bd4c8f56bcfce78ed9e7eba95ce9c50b2c70f5555ade2`. Their encoded bytes match exactly, so decoded pixel error is zero for these runs. All completed below the 4 GiB cgroup limit.

## Decision

The FP32 mean was 7.03% slower for the full task and 10.61% slower for the scatter stage than the single FP64 control. Measured mean process-tree RSS was 47,652,864 B higher; mean cgroup peak was 3,620,672 B higher. These differences are small relative to system variation, but they provide no demonstrated performance or memory gain. The pilot profiler-hook speedup also failed to reproduce in the source-loaded production runs.

Keep FP64 as the production default. Retain FP32 only as an explicit internal/profiler experiment so later backend or kernel changes can re-evaluate it. The exact PNG parity is useful evidence that FP32 can preserve quantized output for this photo and configuration, but does not justify changing the default. The experiment does not establish a general pixel-error bound across images.

The three JSON reports, cProfile files and output images remain local-only under `/tmp/laika-nonzero-mass-fp32-20261023/` on the Mac and matching Lima VM paths; no photo or model asset is committed.
