# Unified subject-protection real-photo benchmark

Date: 2026-10-22. This report records the first single-request run of M9, Depth
Anything V2, BokehMe ARNet, SAM 2.1 Tiny, and ViTMatte on CPU, plus a numerical
comparison of the Python subject composite against the reference Kotlin
composite run on the physical Android phone. The Android runtime is a reference
measurement only; Laika's processing remains Python.

## Host and input

- MacBook Air M1, Apple arm64, macOS 15.7.7, 8 GiB RAM.
- Python 3.11.16; ONNX Runtime 1.31.0, NumPy 2.4.6, OpenCV 5.0.0, Pillow 13.0.0,
  psutil 7.2.2; CPUExecutionProvider, one intra-op thread, Low Memory strategy.
- Input: 1080×1620 non-privacy portrait from the validation portion of the
  [P3M-10K dataset](https://huggingface.co/datasets/nobg/P3M-10K). The dataset
  card describes its release as MIT. Input SHA-256:
  `33bcc476421718673730a44b30d136b911dc97c8f086fc58fea55d63666d308b`.
  Reference mask SHA-256:
  `2768edd89c41eecea3cd982cec09f0e3535b9cf241338ef1f72a2bdf08dd1aad`.
  The image and mask remain outside this repository.
- Model asset SHA-256 values:
  - M9 ONNX: `8936b8c86656265ab07c30b9b4cbbd7fa195d6e44cb3be1d4fd059bb36781e33`
  - Depth Anything V2 Small q64 ONNX:
    `a8ce30941c85b2c9af29fb26f36fbf7be3c1ba1dcf348187a51f6eaa856a238e`
  - BokehMe ARNet ONNX:
    `f0696c16a6eb02099292176f25b47e4dc53863e7e4f3f7748e66470d54cf0173`
  - SAM 2.1 Tiny encoder ONNX:
    `37de035410f11fb505c210ce1f8b10b450cb018fbfa7f226ec7e7ab542955c75`
  - SAM 2.1 Tiny decoder ONNX:
    `3a6a96c62fd39c1369641ea626e5b2e4aed10de877f85291e9993a044d5bf981`
  - ViTMatte Small ONNX:
    `a5c0b49a664867021eaf2b9cadcdc236ea511028d5e709303b418431f20e223b`
  - ViTMatte source safetensors checkpoint:
    `bda9289db1bb6762d978b42d1c62ae3f34daf7497171a347a1d09657efd788cb`

## Full request

One call to `LaikaProcessingEngine.process_file()` through the benchmark/CLI
wrapper ran all five model components on the same photo. The output remained at
the original 1080×1620 dimensions. No image-size reduction or model was skipped.

The normalized positive points were `(0.52, 0.30)` on the face, `(0.55, 0.80)`
on the lower body, and `(0.80, 0.65)` on the sleeve. This multi-point selection
is necessary for this pose: the single face point selected only the local head
and neck region, and two of the tested points did not select the full body.
SAM's encoder ran once; the next two decoder prompts reused its cached image
features.

| Measurement | Result |
|---|---:|
| Process-tree peak RSS | 788,070,400 B |
| Complete task wall time | 112.73 s |
| Output size | 1080×1620, PNG, 1,699,480 B |
| Output SHA-256 | `a50fd8d2dc8615392942c613dca99a7fecb027fe1d184f5d9ffa8064ebb85484` |
| Depth inference | 14.327 s |
| M9 inference | 3.808 s |
| SAM image encoder / prompts | 5.688 s / 0.347 s; 2 embedding-cache hits |
| ViTMatte inference / model setup | 60.174 s / 0.643 s |
| ARNet passes | 12.128 s and 9.928 s |
| Subject background fill / final composite | 0.294 s / 0.254 s |
| Export | 0.175 s |
| Highlight scatter | 0 s on this photo (no qualifying highlight regions) |

This is a sampled process-tree RSS measurement on macOS, not a Linux cgroup
measurement. The input is a small portrait, so it does not establish 4 GiB or
2 GiB acceptance for larger photos.

### Subject mask quality

The supplied JPEG mask is the reference; JPEG edges limit exact boundary
interpretation. Scores below use the full 1080×1620 raster.

| Selection | SAM mask IoU at 0.5 | Soft-alpha MAE | Boundary precision / recall within 5 px |
|---|---:|---:|---:|
| One face point | 0.2128 | — | — |
| Three points, 512-pixel ViTMatte tiles | 0.9879 | 0.00715 | 0.9059 / 0.8277 |

For the three-point alpha, RMSE was `0.05966`; p95 absolute alpha error was
`1.3232e-5`. The small p95 value reflects large regions that are confidently
foreground/background; remaining differences are concentrated around uncertain
boundaries and compression artifacts. These scores apply to this one image and
selection. They do not establish hair quality over a broad image set.

### Python and Kotlin final-composite comparison

To isolate final compositing, the M9 RGB image, no-subject ARNet background, and
three-point ViTMatte alpha were sent as the same byte fixture to
`SubjectProtectionMath.composite` in the reference instrumentation app. The
reference source revision was `f012dca8a845b8c37389345676e43a7e99a5ae93`.
Instrumentation ran on the physical `2405CPX3DC` phone through the explicitly
selected ADB serial `a4976c80`; the attached emulator was not used.

| Metric, compared with Kotlin output | Python result |
|---|---:|
| Maximum absolute channel error | 68 / 255 |
| Mean absolute channel error | 0.03678 / 255 |
| RMSE | 0.58272 / 255 |
| 95th-percentile channel error | 0 |
| Pixels with exact RGB match | 98.840% |
| Pixels with any channel error > 1 | 0.738% |

The maximum errors are confined to a small mixed-edge area, where Python uses a
chunked local background estimate and the reference additionally applies its
gradient and natural-edge refinements. This is a final-composite comparison;
the SAM and ViTMatte model inference comparison is separately covered by their
checkpoint-backed tensor and alpha metrics.

The generated artifacts were kept in `/tmp/laika-p3m-portrait/` and are not
committed: `result-protected-3point.png`, `python-composite.png`,
`android-reference-composite.png`, and `subject-comparison-contact.png`.

## Reproduction

With the model files already present locally, the same unified request can be
run with:

```bash
.venv/bin/python scripts/benchmark_processing_pipeline.py \
  /tmp/laika-p3m-portrait/image.jpg \
  /tmp/laika-p3m-portrait/reproduced.png \
  --mode color+bokeh --strategy low-memory --threads 1 \
  --m9-model /tmp/laika-models/leica-m9-v3-epoch11.fp16.onnx \
  --depth-model /tmp/laika-models/depth-v2-small-dynamic-q64.onnx \
  --bokeh-model /tmp/laika-models/legendary-1.0-arnet.fp16.onnx \
  --sam-model-dir /tmp/laika-models/sam2-onnx \
  --vitmatte-model /tmp/laika-models/vitmatte-onnx/vitmatte-small-composition-1k.fp32.onnx \
  --subject-point 0.52,0.30 \
  --subject-point 0.55,0.80 \
  --subject-point 0.80,0.65 \
  --matte-tile-size 512 \
  --focus-x 0.52 --focus-y 0.30 --aperture 2.8
```

`scripts/benchmark_subject_protection.py` runs M9, SAM, and ViTMatte with
optional reference-mask metrics and saves local alpha/mask artifacts. It does
not replace the unified full-pipeline benchmark above.

## Limits and next checks

- Only the 1080×1620 photo has a full five-model subject-protection run. Measure
  2048-pixel and original-resolution large photos, and test hair, thin objects,
  multiple subjects, and holes/background gaps.
- Compare 512/768/1024 tile sizes on the same photos; 512 was selected here for
  memory behavior, not declared a universal optimum.
- Measure the full highlighted pipeline with subject protection under Linux
  cgroup 8/4/2 GiB limits. The current result is neither a 4 GiB nor a 2 GiB
  pass.
- GAIC composition, FastAPI task service, and Web editor remain unimplemented.
