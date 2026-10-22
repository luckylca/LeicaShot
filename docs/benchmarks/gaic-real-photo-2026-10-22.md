# GAIC real-photo composition integration

Date: 2026-10-22

## Scope

This run validates GAIC proposal scoring, the shared Python engine entrypoints,
and applying a selected crop to an existing image. The crop command does not
load GAIC or any other model. It runs on the final raster supplied to it, so a
caller can crop an already processed image without repeating color or bokeh
inference.

The Python model is the authorized GAIC MobileNetV2 `reddim16` checkpoint,
converted to dynamic FP32 ONNX. The checkpoint is not in this repository. The
local checkpoint SHA-256 is
`eb1d5eed2710c79d3270add448111b9e6fa2475e246694c0ad995a6efb299e8c`; its ONNX
export SHA-256 is
`b9bd8f7b0af48f8628f6ace41eaaabe96fba7ab6f0b566a9cd30c417bcf81d7b`.
Export validation compared the pinned PyTorch implementation with CPU ONNX on
three dynamic shapes. Maximum absolute score error was between
`4.77e-7` and `7.15e-7`.

## Real photo and resource run

- Input: P3M-10K portrait, 1080×1620, oriented RGB; SHA-256
  `33bcc476421718673730a44b30d136b911dc97c8f086fc58fea55d63666d308b`.
- Analysis tensor: 813×1220 RGB, created with OpenCV `INTER_LINEAR_EXACT`, ImageNet
  normalization, and no upscaling.
- Candidates: 556, including the scored full frame. Crop geometry was snapped
  to the original 1080×1620 pixel grid, then mapped into the analysis tensor.
  No output-size filter was requested.
- Host: MacBook Air M1, 8 GiB, CPU-only, one ORT thread, Python 3.11.16,
  ONNX Runtime 1.31.0, macOS 15.7.7.
- Three fresh Low Memory workers sampled process-tree RSS every 5 ms. RSS
  median was **353,615,872 bytes**, with a **343,523,328–362,512,384-byte**
  observed range. Worker wall median was **1.016 s** (range **0.952–1.563 s**).
- Median stage times: model inference **0.773 s**, session load **0.054 s**,
  input decode **0.038 s**, analysis resize **0.050 s**. No visible cgroup was
  available; the sample range is reported because short RSS peaks can fall
  between samples.
- The `laika crop` operation on the selected result produced a **918×1224 PNG**
  from the 1080×1620 input in **0.176 s** total, with no model load and no
  rescaling. Output SHA-256:
  `682cd0351d6ac6fbdc5cc3f32f5003d00f775033d6bf9c535f919e09f6978193`.

The image was visually inspected. This single portrait shows a tighter face and
upper-body framing; it does not establish that every model suggestion is
preferred for every image. The original full-frame candidate remains available
when the model provides no better crop.

## Crop inside a real M9 processing request

After the shared processing options gained an explicit selected crop, a fresh
CPU-only request ran M9 and LabFusion and applied the first GAIC suggestion to
the result:

- Input: the same 1080×1620 P3M-10K portrait.
- Output: 918×1224 PNG; crop coordinates were
  `(0.15, 0.0611111111, 1.0, 0.8166666667)` in oriented source space.
- Host/runtime: MacBook Air M1, 8 GiB, macOS 15.7.7, Python 3.11, ONNX Runtime
  1.31.0, CPUExecutionProvider, one intra-op thread, Low Memory.
- M9 SHA-256: `8936b8c86656265ab07c30b9b4cbbd7fa195d6e44cb3be1d4fd059bb36781e33`.
- Total task wall time: **20.117 s**; sampled process-tree peak RSS:
  **356,876,288 bytes**. M9 inference took 14.365 s, model load 1.929 s,
  LabFusion 0.988 s, and crop application 0.009 s.
- The integrated output was compared against applying the same crop to the
  existing full-size M9+LabFusion PNG: all **1,123,632** output pixels matched
  exactly (maximum channel error 0). This confirms crop ordering introduces
  no image change relative to cropping the full-resolution M9 result.

This measures the M9-plus-crop task, not the full M9+depth+ARNet+subject
pipeline with a crop and not a 4/2 GiB constrained run. It preserves the model's
1024-pixel inference long edge and full-resolution Lab fusion before applying
the explicit final crop.

## Physical-phone Kotlin comparison

The same 813×1220 RGB tensor and ONNX file were run through the reference Kotlin
`GaicComposition` on physical ADB device `a4976c80` (Xiaomi product
`2405CPX3DC`, SoC `SM8650`, Android 16). The original installed application was
left untouched: the temporary reference build used a distinct
`xyz.shuncongci.leica.laikaref` application ID. No emulator was used.

The Kotlin and Python implementations returned the same four selected crop
boxes in the same order. Their reported score improvements matched exactly at
the serialized float precision (maximum difference **0.0**). Both scored 556
candidates with the full frame included. Android ORT logged 436 ms inference on
this tensor fixture.

A second phone run decoded and resized the actual JPEG through the reference
`CompositionFiles` path. Python's OpenCV `INTER_LINEAR_EXACT` analysis raster
had MAE **0.313**, RMSE **0.583**, and 55.04% exact RGB pixels versus the Android
bitmap raster; 95% of channel errors were at most one 8-bit value. Both paths
selected the same four source-coordinate crop boxes. Their model-score
improvements differed by at most **0.0131**, reflecting platform JPEG decode and
resize rounding. Android inference was 411 ms and Python's median was 773 ms;
these are cross-device timings, not a controlled speed comparison.

The same-tensor fixture validates model input layout/normalization, candidate
geometry, score execution, and selection order across CPU runtimes. The actual
JPEG run additionally validates both image-loading paths and confirms matching
crop choices despite small decode/resize differences. One portrait does not
establish broad composition quality or full edit-pipeline memory use.

## Commands

```bash
python scripts/export_gaic_mobilenetv2_onnx.py \
  --checkpoint /local/models/GAIC-mobilenetv2-reddim16.pth \
  --output-dir /local/models/gaic/export --threads 1

python scripts/benchmark_gaic.py input.jpg \
  /local/models/gaic/export/gaic-mobilenetv2-reddim16.fp32.onnx \
  --repeat 3 --report /tmp/gaic-report.json

laika composition input.jpg \
  --model /local/models/gaic/export/gaic-mobilenetv2-reddim16.fp32.onnx

laika crop processed.png crop.png --crop 0.15,0.0611111,1.0,0.8166667
laika pipeline input.jpg result.png --mode color --m9-model /local/models/m9.onnx \
  --crop 0.15,0.0611111,1.0,0.8166667 --strategy low-memory --threads 1
```

The validation JSON is required beside the ONNX graph. Model weights and user
photos stay outside Git.
