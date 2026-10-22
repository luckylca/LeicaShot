# Unified M9 + Depth + ARNet real-photo pipeline

Run date: 2026-10-22

## Scope

One `LaikaProcessingEngine.process_file()` call and one `laika pipeline` command coordinate the three model stages. The integration follows `LeicaEngine.kt`: estimate depth from the EXIF-oriented original, run M9 and Lab detail fusion, then render ARNet using the M9 image and the depth map from the original. The first implementation attempt estimated depth from the recolored image; that run was invalidated and its metrics are excluded here.

Modes exercised by automated orchestration tests are `color`, `bokeh`, and `color+bokeh`. The real-model end-to-end run used `color+bokeh`. No explicit output size was given for the full-size run, so output remained the source's oriented 5760×3840 pixels. The 2048 trial is a separately labeled preview-size experiment, not the full-resolution acceptance result.

## Environment and assets

- MacBook Air, Apple M1 / arm64, 8 logical CPUs, 8 GiB RAM, macOS 15.7.7.
- Python 3.11.16, NumPy 2.4.6, Pillow 13.0.0, ONNX Runtime 1.31.0.
- ONNX Runtime `CPUExecutionProvider` for all models, Low Memory strategy, one intra-op thread.
- The host had another active Python process using close to one logical CPU during the run. The recorded wall times are therefore measured under shared host load, not an isolated throughput benchmark.
- Input: real CC0 mountains photo, JPEG 5760×3840, 3,878,306 bytes. It is outside Git at `/tmp/laika-m9-results/photo/mountains-source.jpg`.
- M9: `leica-m9-v3-epoch11.fp16.onnx`, SHA-256 `8936b8c86656265ab07c30b9b4cbbd7fa195d6e44cb3be1d4fd059bb36781e33`.
- Depth Anything V2 Small: pinned checkpoint converted to dynamic q64 ONNX, SHA-256 `a8ce30941c85b2c9af29fb26f36fbf7be3c1ba1dcf348187a51f6eaa856a238e`.
- BokehMe ARNet: locally exported checkpoint-backed FP16 ONNX, SHA-256 `f0696c16a6eb02099292176f25b47e4dc53863e7e4f3f7748e66470d54cf0173`.
- Models are in `/tmp/laika-models`; none are committed.

## Results

| Run | Output | Engine wall | Process wall | Maximum RSS | Main stages |
|---|---:|---:|---:|---:|---|
| Explicit preview | 2048×1365 | 54.20 s | 54.38 s | 1,094,172,672 B (1.02 GiB) | Depth inference 13.63 s; M9 inference 3.77 s; Lab fusion 4.34 s; ARNet passes 20.34 / 4.48 / 3.10 s |
| Source resolution | 5760×3840 | 605.22 s | 605.43 s | 2,197,651,456 B (2.05 GiB) | Depth inference 13.26 s; M9 inference 3.80 s; Lab fusion 4.45 s; ARNet passes 514.45 / 48.25 / 10.43 / 3.87 s |

The full-size PNG is 14,275,242 bytes. Its dimensions and model metadata were returned by the engine. Maximum RSS is from macOS `/usr/bin/time -l`; this is one Python worker and the command had no additional processing child. No Linux cgroup was available, so this run does not prove operation under a 2 GiB or 4 GiB cgroup limit. The 2 GiB challenge remains unpassed: measured RSS is slightly above 2 GiB when expressed in binary units.

The first full-resolution ARNet pass accounts for about 85% of the 605 s task wall time. This is the first current full-pipeline optimization target; reducing the output dimensions is not proposed. The macOS command also printed a larger virtual/footprint figure, which is not process RSS and must not be compared directly with the RSS results above.

## Physical-device Kotlin comparison

The reference app was built from the external pinned source checkout and invoked with its test instrumentation on the USB-connected physical phone (serial `a4976c80`). The Android emulator was not used. For this local comparison only, the reference APK bundled the same local M9 model and the checkpoint-backed local Depth/ARNet exports. The external depth model's ONNX input is named `image` while the Kotlin source expects `input`; the temporary reference checkout bound that input name and its local model hashes, without changing the image algorithms. No Kotlin source or model assets were copied into Laika.

For M9-only, both engines processed the exact 5760×3840 JPEG and emitted 5760×3840 PNG. Compared pixel-by-pixel:

- Python M9/LabFusion: 10.53 s wall, 742,539,264 B maximum RSS.
- Kotlin reference M9/Lab: 71.393 s total on the phone; its reported Lab fusion stage was 62.4 s. This is a different device/runtime path, so time is recorded as context, not a CPU speed comparison.
- Image difference: MAE 0.215, RMSE 0.526, PSNR 53.71 dB, p95 absolute channel error 1, p99 2, maximum 15; 97.42% of channel values differ by at most 1.

The shared 2048×1365 `color+bokeh` photo was also compared. The phone completed the reference Kotlin render in 174.716 s, including 82.4 s reported in highlight synthesis. Python completed in 54.38 s and 1,094,172,672 B RSS. Image difference was MAE 1.278, RMSE 2.068, PSNR 41.82 dB, p95 4, p99 8, and maximum 69; 86.03% of channel values differ by at most 2. Both outputs look close on this diffuse mountain scene. This is not an exact equivalence test: the Kotlin path includes highlight scatter absent from Python, and its Lab fusion is done at the selected 2048 working size while this Python engine fuses at source resolution before the requested output resize. The comparison demonstrates real reference execution and identifies the next porting target; it does not certify highlight or overall quality parity.

Reference outputs and comparison input remain local under `/tmp/laika-pipeline-results/`. This Python implementation still omits highlight scattering, SAM/ViTMatte subject restoration, and GAIC composition; this report does not claim the complete reference finish or equivalent image quality.

## Reproduction

```bash
/usr/bin/time -l .venv/bin/laika pipeline \
  /tmp/laika-m9-results/photo/mountains-source.jpg \
  /tmp/laika-pipeline-results/mountains-reference-order-full.png \
  --mode color+bokeh \
  --m9-model /tmp/laika-models/leica-m9-v3-epoch11.fp16.onnx \
  --depth-model /tmp/laika-models/depth-v2-small-dynamic-q64.onnx \
  --bokeh-model /tmp/laika-models/legendary-1.0-arnet.fp16.onnx \
  --strategy low-memory --threads 1 \
  --focus-x 0.5 --focus-y 0.5 --aperture 2.8 --progress
```

For the preview comparison, append `--width 2048 --height 1365` and use a different output path. All images, model weights, and outputs stay outside the repository.

## Acceptance and follow-up

- [x] One Python SDK call coordinates model execution and image buffers for all three models.
- [x] CLI runs the same coordinator and preserves full-resolution output by default.
- [x] Actual M9, Depth Anything V2, and ARNet CPU sessions completed on one real photo; SHA-256 and runtime provider were emitted.
- [x] Corrected model order and original-size 5760×3840 output verified.
- [x] Progress, cancellation boundary, Low Memory release, Balanced session reuse, and modes have unit coverage.
- [x] Compare M9/Lab full-size output and 2048 bokeh output to the Kotlin reference on the physical device (`adb -s a4976c80`).
- [ ] Add high-light extraction, aperture pupils/scatter, and final radiance blend; the standalone ARNet plate is not the reference's complete bokeh finish.
- [ ] Measure full task in Linux cgroups at 8 / 4 / 2 GiB, including RSS/cgroup peak, no-OOM outcome, timing, and output quality.

Output artifact: `/tmp/laika-pipeline-results/mountains-reference-order-full.png`.
