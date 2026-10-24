# Cross-scene verification of the float32 highlight workspace

Date: 2026-10-24. This experiment compares the merged PR #62 float32
per-node radiance workspace with the same Python engine's forced-FP64 scatter
reference on an independent full-resolution portrait. It also records the
float32 candidate's repeated 2 GiB cgroup failure. It does not establish a
supported 2 GiB service tier or a speed improvement.

## Input, source, and runtime

- Input: 5472×3648 CC0 portrait from
  [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Portrait_(CreativeCommons).jpg),
  SHA-256 `d237bbf13f886b69a376aad00836403dafdb8a6be0062dc98a50e1a6c7b32a48`.
  The JPEG has EXIF orientation 1. It remained outside Git.
- Both successful runs used the same mounted Python source tree, source-tree
  SHA-256 `c03f5c618e9eac022eccd66272041f86b6cb6d4387aa0d81e65c15b6f35705c2`.
  The float32 candidate used the normal engine path. The reference wrapper
  forced `scatter_highlights(..., _float32_radiance_workspace=False)`; all
  other source, model, and request settings were unchanged.
- Request: `bokeh`, Low Memory, CPUExecutionProvider, one inference thread,
  focus `(0.5, 0.5)`, aperture `2.8`, one subject point `(0.52, 0.56)`,
  512-pixel ViTMatte tiles, 128-row scatter chunks, and native 5472×3648 output.
  The pipeline ran Depth Anything V2, SAM 2.1 Tiny encoder/decoder, ViTMatte
  Small, BokehMe ARNet, highlight scatter, and subject compositing. **M9 was
  not loaded or run**; this was a bokeh-only request.
- Host: MacBook Air M1; Lima Linux 7.0.0-34 ARM64 guest with
  4,085,092,352 B total RAM and 2 logical CPUs. Runtime: Python 3.11.17,
  NumPy 2.4.6, OpenCV 5.0.0, ONNX Runtime 1.31.0. All inference used CPU.
- Model SHA-256: Depth Anything V2 q64
  `a8ce30941c85b2c9af29fb26f36fbf7be3c1ba1dcf348187a51f6eaa856a238e`; ARNet
  `f0696c16a6eb02099292176f25b47e4dc53863e7e4f3f7748e66470d54cf0173`; SAM
  checkpoint `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`,
  encoder `37de035410f11fb505c210ce1f8b10b450cb018fbfa7f226ec7e7ab542955c75`,
  decoder `3a6a96c62fd39c1369641ea626e5b2e4aed10de877f85291e9993a044d5bf981`;
  ViTMatte checkpoint `bda9289db1bb6762d978b42d1c62ae3f34daf7497171a347a1d09657efd788cb`,
  ONNX `a5c0b49a664867021eaf2b9cadcdc236ea511028d5e709303b418431f20e223b`.

## Completed full-pipeline runs

| Measurement | Forced FP64 reference | Float32 workspace candidate |
|---|---:|---:|
| cgroup memory / swap limit | 2,147,483,648 B / disabled | 2,684,354,560 B / disabled |
| Full task wall time | 1,675.976 s | 3,150.830 s |
| Highlight scatter time | 1,118.537 s | 2,603.757 s |
| Process-tree peak RSS | 2,191,032,320 B | 2,212,343,808 B |
| ARNet-stage process-tree peak | 2,191,032,320 B | 2,212,343,808 B |
| Scatter-stage process-tree peak | 1,543,364,608 B | 1,564,803,072 B |
| Exact cgroup `memory.peak` | 2,147,483,648 B | 2,264,207,360 B |
| cgroup `max` / `oom` / `oom_kill` | 952 / 0 / 0 | 0 / 0 / 0 |
| Output size | 11,155,476 B | 11,155,479 B |
| Output SHA-256 | `8a121c345a29fc2e54bdcef41df7583743df94c0168510219bf1a1ac93a0cd17` | `434472393856314836d8fe17f406c3e5299939c549d1f96f77116bdca2554446` |

Both completed outputs are RGB PNGs at the original 5472×3648 dimensions. The
forced-FP64 reference touched the 2 GiB cgroup limit and repeatedly triggered
`max` events despite completing without OOM. The float32 candidate used a
different 2.5 GiB limit and left 420,147,200 B below that limit. Because the
limits differed and runs were sequential on a shared host, these measurements
are not a controlled memory or speed A/B.

## Pixel comparison

The decoded output arrays have identical dimensions. Exactly one pixel differs:
at `(x=540, y=1617)`, the forced-FP64 RGB value is `[71, 66, 62]` and the float32
candidate is `[71, 66, 61]`. Channel mean absolute error is
`1.6698514072706132e-8`; maximum channel error is 1 on the 8-bit scale; PSNR is
125.904 dB. No output pixel differs by more than one channel LSB. This measured
difference is negligible in the rendered comparison, but is not byte identity.

The candidate scatter took 2,603.757 s versus 1,118.537 s in the reference.
The candidate completed under a larger cgroup cap, so this timing difference
cannot be attributed to workspace precision and does not demonstrate a speed
benefit. The maximum process-tree RSS in both successful runs occurred during
ARNet, not scatter; scatter's process-tree peak was about 1.54–1.56 GB.

## Repeated 2 GiB candidate failure

Two additional float32 full-pipeline runs used a no-swap 2,147,483,648 B cgroup
and the same image, model assets, source-tree digest, and processing arguments.
Both containers exited 137 and Docker reported `OOMKilled=true` during the
`bokeh.arnet` stage, before highlight scatter. The first sampler observed
2,147,409,920 B current memory and 981 cgroup `max` events; the wrapped repeat
observed up to 2,140,786,688 B current memory. The sampler's OOM counters were
zero in the snapshots captured before teardown. No PNG was produced. The
available evidence identifies the failure stage but does not establish a more
specific causal allocation.

Therefore the float32 candidate has not passed a 2 GiB end-to-end run on this
scene. The successful 2 GiB forced-FP64 run itself reached the cgroup cap and
is not evidence of safe 2 GiB service headroom either.

## Reproduction and artifacts

The runs used `scripts/benchmark_processing_pipeline.py` from the measured
source checkout, mounted read-only in the Linux guest. The common request
arguments were `--mode bokeh --strategy low-memory --threads 1 --focus-x 0.5
--focus-y 0.5 --aperture 2.8 --subject-point 0.52,0.56 --matte-tile-size 512
--bokeh-row-chunk-height 128`, with the pinned Depth, ARNet, SAM, and ViTMatte
model paths and `--progress-log` enabled. The candidate ran the default
float32 workspace; the reference used the external wrapper described above.
The input, model files, wrappers, logs, cgroup samples, and output PNGs are
stored outside Git under `/Users/lucky/Desktop/project/leicashot-photo-validation/`.

The full CPU test suite is run separately as part of this documentation PR.
No model weights, source photographs, or generated output images are committed.
