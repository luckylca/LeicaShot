# Subject-protection quality and full-resolution memory benchmark

Date: 2026-10-22. This work continues from PR #34 on
`perf/fullres-subject-protection`. All processing and measurements used the
Python engine and CPUExecutionProvider. The connected Android emulator and
phone were not used for image processing.

The aim was to test three things: one real five-model request on a larger
portrait, subject-edge quality on more than one photo, and the memory behavior
of a native 5760×3840 request with subject protection. Images and authorized
model assets remain outside Git.

## Larger five-model portrait request

The input is a 1080×2123 non-privacy `P3M_500_NP` validation photo from the
[P3M-10K dataset](https://huggingface.co/datasets/nobg/P3M-10K). The dataset card
describes the release as MIT licensed. Input SHA-256:
`f04e9b8efef5bf0cd2f2f8123c7691106ad521dfa8cedb3238a790151dce7b75`. Its JPEG
alpha reference SHA-256 is
`61b4124cbbc2847b43cd7f96397e9e2e15321ddf69b3d10f1e9314396855625f`.

M9 + LabFusion, Depth Anything V2, SAM 2.1 Tiny, ViTMatte Small, ARNet and
highlight processing ran in one `LaikaProcessingEngine.process_file()` request.
The three positive subject points were `(0.50, 0.23)`, `(0.52, 0.55)`, and
`(0.50, 0.85)`. ViTMatte used 512-pixel tiles. Output remained 1080×2123.

| Host and limit | Task wall | Process-tree peak RSS | cgroup peak | Output SHA-256 |
|---|---:|---:|---:|---|
| MacBook Air M1, 8 GiB, macOS 15.7.7 | 84.757 s | 905,445,376 B | unavailable | `736c9e253d58c6f864e1519341058ee0cc2f7f5dff511558579aca24c4c66750` |
| Ubuntu ARM64 Lima guest, 4 GiB, 3 GiB task cgroup, 2 vCPU | 103.264 s | 2,007,384,064 B | 2,322,145,280 B | `262be8d69b89bc99610a46a38bba128c90e416268c5d3d26c1cc1aecbd68cbd8` |

Both tasks used one ONNX Runtime intra-op thread. The Linux task completed with
zero cgroup OOM or limit events. Its principal stage times were Depth 9.103 s,
M9 inference 3.572 s, SAM encoder 5.424 s, ViTMatte inference 47.813 s, ARNet
passes 22.908 s and 5.232 s, and export 0.820 s. SAM encoded the image once and
reused its embedding for two further prompts. This image had no qualifying
highlight radiance, so the scatter stage took 0 s.

The decoded Mac/Linux result comparison had MAE 0.2621, RMSE 0.5417, p95 and
p99 channel errors 1, maximum error 48, and 74.74% exact channel values. These
are cross-platform results, not a same-platform bitwise regression. Both
outputs preserved the source dimensions.

A separate 1.5 GiB task-cgroup attempt on the same photo was OOM-killed during
the first ARNet pass after M9, Depth, SAM, and ViTMatte had run. The last saved
pre-kill sample showed 1,452,912,640 B cgroup peak and 1,213,317,120 B
process-tree RSS; those are not the final cgroup peak. The guest still had 4
GiB configured, so this was not a 2 GiB guest acceptance result.

## More hair and edge-quality cases

The following validation photos and reference mattes come from the
[AIM-500 dataset](https://huggingface.co/datasets/nobg/AIM-500), whose dataset
card describes it as MIT licensed. The images stay outside the repository.
Each experiment used the local M9/SAM/ViTMatte ONNX files, CPUExecutionProvider,
one thread, normalized positive points, and the dataset soft-alpha as ground
truth. The helper includes M9, SAM, and ViTMatte; it excludes Depth and ARNet.
Boundary precision and recall use a five-pixel tolerance around the binary
alpha edge.

### Long hair against a wheat field — 1080×1752

| ViTMatte tile | Helper wall | Peak process-tree RSS | ViTMatte time | IoU | Soft-alpha MAE | Boundary precision / recall |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 67.727 s | 906,625,024 B | 53.577 s | 0.989368 | 0.0030545 | 0.96484 / 0.93169 |
| 768 | 83.710 s | 1,083,654,144 B | 70.873 s | 0.989559 | 0.0029725 | 0.96443 / 0.93010 |
| 1024 | 143.249 s | 1,153,482,752 B | 129.957 s | 0.989276 | 0.0030222 | 0.96014 / 0.92707 |

The 768 tile's IoU and MAE changes are small; its helper took 15.98 s longer
and used 177 MB more sampled RSS than 512. The 1024 tile was much slower with
slightly worse IoU and edge recall. These measurements support retaining 512
for this image; they do not establish a universal tile-size optimum.

### Backlit wispy hair — 1714×1080

| ViTMatte tile | Helper wall | Peak process-tree RSS | ViTMatte time | IoU | Soft-alpha MAE | Boundary precision / recall |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 43.085 s | 905,674,752 B | 30.844 s | 0.990567 | 0.003184 | 0.98628 / 0.76981 |
| 768 | 78.843 s | 791,085,056 B | 63.691 s | 0.989853 | 0.003412 | 0.98250 / 0.75452 |

The foreground body and binary silhouette score well, while boundary recall is
lower on wispy backlit strands. The 768 tile did not improve the edge metrics
and took 35.76 s longer. The lower sampled RSS in that single 768 run is not a
repeatable memory comparison.

The earlier curly-hair P3M photo's tile sweep is recorded in the
[initial subject-protection report](subject-protection-integration-2026-10-22.md).
Across these three scenes, 512 gave the shortest measured ViTMatte time. Fine
hair recall and uncertain JPEG matte edges remain quality work; binary IoU
alone is not sufficient evidence of edge fidelity.

## Native-resolution 5760×3840 subject attempt

The existing 5760×3840 mountain photo
(`4e6661606c195b2462c0f2321a7feefe2ae2fc081499f3462a30fbb108cd7f8e`) ran
M9, Depth, SAM, ViTMatte, ARNet, subject background fill, and highlights at its
native dimensions. A positive point `(0.50, 0.45)` selected the mountain; the
focus point was `(0.50, 0.50)` with aperture 2.8. This is a memory stress case,
not a ground-truth subject-quality result.

Every attempt was OOM-killed in the first full-resolution ARNet pass before
producing a PNG. Docker reported exit 137 and `OOMKilled=true`.

| Low Memory ARNet setting | Task cgroup limit | Last saved cgroup current | Last saved cgroup peak | Last saved process-tree RSS |
|---|---:|---:|---:|---:|
| Existing NCHW assembly, 512 core | 3 GiB | 1,482,272,768 B | 1,947,783,168 B | 1,041,354,752 B |
| Direct HWC tile sink, 512 core | 3 GiB | 1,493,913,600 B | 2,002,157,568 B | 916,135,936 B |
| Direct HWC tile sink, 512 core | 3.125 GiB | 1,473,679,360 B | 1,986,117,632 B | 896,958,464 B |
| Direct HWC tile sink, 256 core | 3.125 GiB | 1,614,368,768 B | 2,079,621,120 B | 1,034,690,560 B |
| Direct HWC tile sink, 512 core, glibc trim prototype | 3.125 GiB | 1,530,454,016 B | 1,951,641,600 B | 951,885,824 B |

These are the last progress-log samples before termination, not final cgroup
peaks. The cgroup memory controller killed the task after the logged ARNet
stage began; the exact allocation that crossed the limit was not captured.
The 256-core and glibc-trim trials did not improve completion and were not
retained as defaults or code paths. The native-resolution subject task has not
passed a 3.125 GiB cgroup. The successful 1080×2123 result must not be
presented as 4 GiB native-resolution acceptance.

## Exact-preserving ARNet output-buffer experiment

Low Memory's tiled ARNet used to assemble a complete NCHW FP32 image and then
copy it into the HWC float32 image required by the renderer. The renderer now
lets tiled ARNet write each tile core directly into that HWC destination. The
normal full-frame route remains in place for non-tiled, odd-sized, and smaller
inputs. Subject CoC masking also uses row-bounded complement buffers instead
of materializing a full-frame `1 - alpha` array.

For an even 1024×682 real M9 photo input, a full-frame ARNet result and four
direct-output tiles were bitwise identical: zero changed values, maximum error
0, and MAE 0. A 512/256/128 core sweep on that same input also produced the
same tensor SHA-256
`4ee1679de673b1208afb36e68edaab0f1e66aacd5f5283264d53dbff4a76c0cf` for every
tile size. Single-trial times were 6.519 / 6.042 / 6.346 s respectively; this
does not establish a speed ranking. The tested 256 core increased last-sample
RSS in the 5760×3840 cgroup attempt, so the default remains 512.

At 5760×3840, the old intermediate full NCHW tensor contains
`1 × 3 × 3840 × 5760 × 4 = 265,420,800` bytes (253 MiB). Direct output removes
that full-frame allocation from the tiled path; this is an allocation-count
reduction, not a measured 253 MiB peak reduction. The full-resolution subject
task still OOMs, so further live-buffer and stage-lifecycle work is required.

A separate complete `render_bokeh` A/B on the 1080×2123 P3M image used the same
real M9 RGB and depth map, ground-truth alpha, and ARNet model. Both paths
exported RGB SHA-256
`3b9c87f6c337bbd29162eaf9abf76b42443214c9b28d8f86e6b04d53ae0dff79`; decoded
outputs had zero changed channels. The prototype took 22.464 s versus 20.823 s
for the full NCHW path. The renderer therefore selects direct output only when
the image already meets the existing native-size tiled-ARNet condition.

## Reproduction

Use authorized model files at the paths below and the downloaded dataset
fixtures under `/tmp`; never put model weights or photos into the Git tree.

```bash
PYTHONPATH=src .venv/bin/python scripts/benchmark_processing_pipeline.py \
  /tmp/laika-p3m-large/image.jpg /tmp/laika-p3m-large/result.png \
  --m9-model /tmp/laika-models/leica-m9-v3-epoch11.fp16.onnx \
  --depth-model /tmp/laika-models/depth-v2-small-dynamic-q64.onnx \
  --bokeh-model /tmp/laika-models/legendary-1.0-arnet.fp16.onnx \
  --sam-model-dir /tmp/laika-models/sam2-onnx \
  --vitmatte-model /tmp/laika-models/vitmatte-onnx/vitmatte-small-composition-1k.fp32.onnx \
  --subject-point 0.50,0.23 --subject-point 0.52,0.55 --subject-point 0.50,0.85 \
  --matte-tile-size 512 --mode color+bokeh --strategy low-memory --threads 1 \
  --focus-x 0.50 --focus-y 0.50 --aperture 2.8 --bokeh-row-chunk-height 512
```

For Linux acceptance, run the same command in a fresh ARM64 container with
`--memory=3g --memory-swap=3g --cpus=2`, mount `src`, `scripts`, the model
directory, and the input directory read-only, and use the cgroup benchmark
image built from `tools/cgroup/Dockerfile`. The benchmark records cgroup v2
memory and OOM counters alongside sampled process-tree RSS.

## Next work

- Reduce live high-resolution image, highlight, CoC, and subject buffers so
  native-size five-model processing can pass a Linux memory limit without
  changing output dimensions or model quality.
- Add per-tile memory progress to the cgroup harness so future OOM runs retain
  the last allocation stage and actual cgroup peak.
- Extend reference-matte testing to flyaway hair, thin objects, and background
  holes, and repeat the 512/768/1024 comparison on those scenes.
- Instrument cgroup memory sampling inside the ARNet tile loop; the existing
  task-level sampler missed the allocation that crossed the limit in these OOM
  runs.
