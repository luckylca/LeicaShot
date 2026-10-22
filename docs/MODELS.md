# Model artifact acquisition

Model files are not committed to Laika. `models/manifest.json` pins artifacts that have an explicit fetch workflow; `laika-models` downloads those into a user cache and makes a file available only after size and SHA-256 match. M9 has a separate local-only record because this repository does not assert public redistribution permission for its weights.

The manifest includes Depth Anything V2 Small, SAM 2.1 Hiera Tiny, and the BokehMe ARNet source checkpoint. The official SAM 2.1 model card declares Apache-2.0 and its pinned checkpoint page records SHA-256 `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69` and 156,008,466 bytes. The Meta implementation is separately pinned to `2b90b9f5ceec907a1c18123530e92e794ad901a4`. Laika's split encoder/decoder ONNX export passed checkpoint-backed tensor comparison with the upstream PyTorch model; a three-point real-photo request also confirmed encoder-feature reuse and strong mask agreement on one validation portrait. The 2 GiB single-stage synthetic CPU measurement is in the [SAM benchmark](benchmarks/m1-sam2-cgroup-2026-10-21.md); it is not a full-pipeline resource result.

## M9 series 3.0 epoch11

M9 is Laika's AI color predictor and connects directly to `laika.color.fuse_lab_detail`. The user confirmed on 2026-10-22 that this project has the required authorization to use M9 and the other planned model assets. The initial general license assessment has been superseded for project use; see [ADR 0002](adr/0002-m9-project-authorization.md). This confirmation does not establish public redistribution permission. Keep the ONNX file in a user-provided local path and outside Git.

The pinned local artifact is `leica-m9-v3-epoch11.fp16.onnx`, 10,053,669 bytes, SHA-256 `8936b8c86656265ab07c30b9b4cbbd7fa195d6e44cb3be1d4fd059bb36781e33`. The reference `models.json` lists source checkpoint SHA-256 `8456c139e98ce98ac8742a5d20bc1a65a0315f85f13ad37c02cb708af1c337c6`. The reference model record identifies internal FP16 and external FP32. Laika's adapter verifies the ONNX hash, requires the single float32 NCHW input `input` and output `output`, and selects only ONNX Runtime's CPU provider.

Processing applies EXIF orientation, resizes RGB to an aspect-preserving 1024-pixel long edge using Pillow bilinear, normalizes each channel with `((x / 255) - 0.5) / 0.5`, and runs M9 with float32 external tensors. Output values map through `clip(y * 0.5 + 0.5, 0, 1)` and truncating 8-bit conversion. The predicted RGB is resized to the full oriented source dimensions with Pillow bicubic and fused with source detail using Lab sigma L=25 and a/b=20. The final raster keeps the oriented source dimensions; the model working size never changes the export size.

Install the inference dependencies and run the shared engine from the CLI:

```bash
python -m pip install -e '.[inference]'
laika color input.jpg output.png \
  --model /path/to/leica-m9-v3-epoch11.fp16.onnx \
  --strategy low-memory --threads 1 --row-chunk-height 512
```

`low-memory` uses one ONNX intra-op thread by default, disables the CPU memory arena/pattern, releases the session after inference, and reopens the unchanged input for full-resolution fusion. `balanced` keeps an M9 session for reuse across calls on the same `M9ColorEngine` instance, enables those runtime allocators, and defaults to at most four intra-op threads. Both profiles run the same preprocessing, model and Lab algorithm. Explicit `--threads` applies to ONNX Runtime and OpenCV. Neither profile reduces the final dimensions. `M9Onnx` and `M9ColorEngine` are also available from the Python package.

The local `scripts/benchmark_m9.py` command runs fresh CLI workers and samples process-tree RSS. It can compare row chunk heights against full-height fusion without committing an image or model. On macOS it reports sampled RSS and cannot claim a cgroup memory limit; constrained 4/2 GiB service verification requires Linux cgroups.

## BokehMe ARNet

Laika uses the BokehMe ARNet checkpoint from upstream commit `8b3ed556dc146207c9bb39fccd078a113f8d1bd8`. The checkpoint is 11,101,736 bytes with SHA-256 `6c7dba6f32e3afc8e984eb5adfecfea5d13b6f4ebd132a5a890beba384be4752`. Its project use is authorized by the user's confirmation. Keep checkpoint and ONNX files local; this record does not assert public redistribution permission.

`tools/export_bokeh_arnet.py` checks the pinned upstream `neural_renderer.py` source hash and checkpoint hash, uses PyTorch's `weights_only=True` loader with a narrow NumPy metadata allowlist, and exports a dynamic opset-17 ONNX graph. It fixes gamma at 0.3, retains the even-size resize and pixel-unshuffle path, converts internal operators/weights to FP16, and keeps external tensors FP32. The output is stored beside a validation report; `BokehMeARNet` compares the actual model SHA-256 with that report before opening ONNX Runtime. The original reference ONNX hash is `a3aa5fa6a0818f36dbfa4be4dd5471d874cea5981357b7aeb28919954e70d05c` (1,867,929 bytes). The independently exported Laika graph in the recorded toolchain is `f0696c16a6eb02099292176f25b47e4dc53863e7e4f3f7748e66470d54cf0173` (1,868,251 bytes); it is not byte-identical to the reference artifact, so Laika validates tensor outputs numerically rather than claiming artifact identity.

The exporter passed three dynamic synthetic shapes and a real 5760 × 3840 CC0 photo resized explicitly to 1024 × 683 for comparison. For the photo, FP16 ONNX versus the pinned upstream PyTorch ARNet had maximum float error `0.000742674`, mean float error `0.0000915710`, 8-bit PSNR `63.63 dB`, maximum 8-bit channel error 1, and no channel errors greater than 2. The full multiscale renderer and CPU resource run are documented in the [BokehMe real-photo benchmark](benchmarks/bokeh-real-photo-2026-10-22.md). Reference optical highlight scatter is integrated; optional SAM and ViTMatte subject protection now runs through the unified engine as recorded in the [subject benchmark](benchmarks/subject-protection-integration-2026-10-22.md).

Fetch the source checkpoint, then export outside the repository using the pinned toolchain:

```bash
python -m pip install -e '.[inference]'
laika-models fetch bokehme-arnet-checkpoint
git clone --filter=blob:none --no-checkout \
  https://github.com/JuewenPeng/BokehMe.git /tmp/laika-bokehme
git -C /tmp/laika-bokehme checkout 8b3ed556dc146207c9bb39fccd078a113f8d1bd8
python3.11 -m venv /tmp/laika-bokeh-export-env
/tmp/laika-bokeh-export-env/bin/pip install -r tools/bokeh_export_requirements.txt
/tmp/laika-bokeh-export-env/bin/python -m pip install -e .
/tmp/laika-bokeh-export-env/bin/python tools/export_bokeh_arnet.py \
  --upstream-root /tmp/laika-bokehme \
  --checkpoint ~/.cache/laika/models/bokehme-arnet.pth \
  --output-dir /tmp/laika-bokeh-export \
  --validation-photo /path/to/photo.jpg \
  --validation-depth /path/to/depth.npy
```

The `laika bokeh` CLI takes the raw float32 map emitted by `laika depth`, normalizes it with the reference 2nd/98th percentiles, selects focus at normalized coordinates, and runs the reference thin-lens CoC, multiscale ARNet, and optical highlight schedule. It keeps the input's oriented dimensions unless an explicit `--width` and `--height` are supplied; a requested smaller size uses bicubic RGB and align-corners depth resizing. The unified `laika pipeline` also supports optional SAM/ViTMatte subject protection through repeated `--subject-point X,Y` flags and a validated local model directory/graph.

```bash
laika bokeh input.jpg depth.npy output.png \
  --model /tmp/laika-bokeh-export/legendary-1.0-arnet.fp16.onnx \
  --focus-x 0.5 --focus-y 0.5 --aperture 2.8 \
  --strategy low-memory --threads 1

# A smaller output is explicit and keeps its requested dimensions.
laika bokeh input.jpg depth.npy preview.png \
  --model /tmp/laika-bokeh-export/legendary-1.0-arnet.fp16.onnx \
  --width 1024 --height 683 --focus-x 0.5 --focus-y 0.5
```

`low-memory` defaults to one CPU thread and disables ONNX Runtime's CPU arena/pattern. `balanced` enables runtime allocation reuse and defaults to at most four threads. Both use the same image math and preserve the specified output dimensions. The `BokehMeARNet` and `render_bokeh` APIs are available to Python callers.

The Depth Anything V2 Small checkpoint's official model card identifies it as Apache-2.0; its file page records the checkpoint hash and byte size. The upstream model implementation is separately pinned to `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`. The checkpoint has been strictly loaded and run on CPU. Static and dynamic ONNX exports have passed seeded-tensor checks; the dynamic graph was tested at four multiple-of-14 shapes. The Python adapter also produced a finite full-resolution map from a real 5760 × 3840 photo on CPU; this run verifies execution and resource use, not depth accuracy. See the [real-photo baseline](benchmarks/m4-depth-real-photo-2026-10-22.md), [M1 benchmark](benchmarks/m1-depth-anything-2026-10-21.md), and [dynamic-shape experiment](benchmarks/m1-dynamic-depth-2026-10-21.md).

```bash
laika-models list
laika-models fetch depth-anything-v2-small
laika-models fetch sam2.1-hiera-tiny
laika-models fetch bokehme-arnet-checkpoint
laika-models verify depth-anything-v2-small \
  ~/.cache/laika/models/depth_anything_v2_vits.pth
laika-models verify sam2.1-hiera-tiny \
  ~/.cache/laika/models/sam2.1_hiera_tiny.pt
laika-models verify bokehme-arnet-checkpoint \
  ~/.cache/laika/models/bokehme-arnet.pth
```

Set `LAIKA_MODEL_CACHE` or pass `--cache-dir` to change the cache location. Downloads use a temporary file in the target directory and are atomically renamed only after validation. A mismatched existing file is reported and left untouched. Run `laika-models fetch` explicitly; installing Laika never downloads model weights.

## Reproduce static and dynamic ONNX exports

The converter uses the upstream model definition only from its manifest-pinned revision. It writes a graph, optional external tensor data, and a JSON validation report. It refuses to overwrite any of those files and checks framework and ONNX Runtime outputs on seeded synthetic tensors before publishing. By default the export is static; add `--dynamic` to use the dynamic-height/width path. The exporter validates its listed shapes; the adapter currently checks positive dimensions and caps the derived network raster before inference. The optional model smoke covers the 518-based upstream profile, a padded 225×231 source, and a 4K source under the reference sizing profile.

For the experimental DINO attention optimization, add `--query-chunk-size 128` together with `--dynamic`. Each query tile attends across all keys and values. The exporter checks equality against the unchunked PyTorch model and verifies that the ONNX graph retains its 12 loop nodes; it rejects chunked exports that lose this control flow. The corrected direct model benchmark and the full CLI image-to-depth comparison are recorded separately in the [query-chunking report](benchmarks/m1-depth-query-chunking-2026-10-21.md) and [M2 inference report](benchmarks/m2-depth-inference-2026-10-21.md). This remains experimental and is not yet part of the production edit pipeline.

```bash
git clone --filter=blob:none --no-checkout \
  https://github.com/DepthAnything/Depth-Anything-V2.git /tmp/laika-depth-anything-v2
git -C /tmp/laika-depth-anything-v2 checkout a561b849ebae10a6f5ef49e26c83cbbcd36c71bf

python3.11 -m venv /tmp/laika-depth-env
/tmp/laika-depth-env/bin/pip install \
  torch==2.14.1 torchvision==0.29.1 opencv-python-headless==5.0.0.93 \
  numpy==2.4.6 onnx==1.23.2 onnxruntime==1.31.0 onnxscript==0.7.2
/tmp/laika-depth-env/bin/python -m pip install -e .
/tmp/laika-depth-env/bin/laika-models fetch depth-anything-v2-small

/tmp/laika-depth-env/bin/python scripts/export_depth_anything_v2.py \
  --source-checkout /tmp/laika-depth-anything-v2 \
  --weights "$HOME/.cache/laika/models/depth_anything_v2_vits.pth" \
  --output /tmp/laika-depth-v2-small-518x686.onnx \
  --height 518 --width 686 --threads 1

# Dynamic image-shape graph; validates 518×686, 700×532, 1400×518, and 518×518.
/tmp/laika-depth-env/bin/python scripts/export_depth_anything_v2.py \
  --source-checkout /tmp/laika-depth-anything-v2 \
  --weights "$HOME/.cache/laika/models/depth_anything_v2_vits.pth" \
  --output /tmp/laika-depth-v2-small-dynamic.onnx \
  --height 518 --width 686 --threads 1 --dynamic

# Experimental full-global query chunking; use a separate output path.
/tmp/laika-depth-env/bin/python scripts/export_depth_anything_v2.py \
  --source-checkout /tmp/laika-depth-anything-v2 \
  --weights "$HOME/.cache/laika/models/depth_anything_v2_vits.pth" \
  --output /tmp/laika-depth-v2-small-dynamic-q128.onnx \
  --height 518 --width 686 --threads 1 --dynamic --query-chunk-size 128
```

Use a separate virtual environment for this large conversion toolchain. The exact dependency versions above reproduce the recorded macOS experiment; select and record compatible CPU wheels for other operating systems. Keep the `.onnx.data` file with the graph when the validation report lists nonzero external data bytes; the tested dynamic export embeds its weights in the graph. The JSON report records output hashes, versions, tested shapes, export/session/inference times, and maximum/mean/p99 absolute error. Dynamic output-shape support is validated for the four listed test sizes only.

GitHub Actions provides an opt-in `Optional model smoke` workflow. Add the `model-smoke` label to a pull request to run it; it downloads and verifies this checkpoint, exports dynamic baseline and query-chunked CPU ONNX models, checks four exporter shapes, compares the lower-bound preprocessing profile and both profile outputs against PyTorch, validates padded small and 4K sources, and prints process-tree benchmarks including full-resolution q16/q32/q64/q128 runs. It also runs the depth-only CLI for the full-attention and q16/q32/q64/q128 graphs in isolated Docker containers with enforced 4 GiB and 2 GiB cgroup v2 caps, records OOM events, and uploads one JSON report per trial. The [depth cgroup report](benchmarks/m3-depth-cgroup-memory-limits-2026-10-21.md) records the initial baseline/q128 results; the [chunk cgroup report](benchmarks/m3-depth-attention-cgroup-sweep-2026-10-21.md) compares all tested attention sizes. This measures only the depth stage and its benchmark harness; it does not establish that a complete Laika edit fits either limit. Manual dispatch is available once the workflow is present on the repository's default branch. Regular cross-platform CI remains model-free.

The SAM path accepts an oriented subject-analysis RGB raster whose long edge is at most 2048, resizes it to 1024×1024 with the reference's separable antialiasing, and applies RGB ImageNet normalization. Normalized `(x,y)` prompts map to the model frame. `Sam2TinyOnnx` caches its exact encoder outputs in a configurable byte-bounded LRU and reuses them for later prompt decoder calls; `invalidate()` and `close()` explicitly release cached features. `select_subject_mask()` picks the highest-scoring candidate covering the positive click, resizes its logits to the subject-analysis raster, and thresholds at zero. `build_subject_trimap()` marks the eroded foreground, eroded background, and unknown edge band using the reference's ellipse and scale-adjusted radius. The split FP32 ONNX graphs passed checkpoint-backed tensor comparison with the pinned upstream PyTorch model. A real-photo three-point SAM→ViTMatte path records two encoder-cache hits and a dataset-mask IoU of 0.9879; this photo evidence and complete task RSS are in the [subject benchmark](benchmarks/subject-protection-integration-2026-10-22.md). The earlier [SAM cgroup benchmark](benchmarks/m1-sam2-cgroup-2026-10-21.md) remains a single-stage synthetic harness measurement, not a full-pipeline resource result.

The model-gated `sam2-model-smoke` job checks out the pinned Meta source, downloads and hash-verifies the checkpoint, exports separate FP32 encoder/decoder graphs, runs ONNX checker, and compares the feature maps, mask logits, and scores against the same upstream PyTorch model on a deterministic synthetic tensor. A separate fresh worker samples process-tree RSS/USS, CPU time, cgroup values when visible, encoder/decoder time, and cached prompt latency. Reports include graph hashes and numerical errors. These synthetic measurements do not establish photo quality or complete subject extraction. To run it on a pull request, add the `sam2-model-smoke` label.

```bash
git clone --filter=blob:none --no-checkout \
  https://github.com/facebookresearch/sam2.git /tmp/laika-sam2
git -C /tmp/laika-sam2 \
  checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
python3.11 -m venv /tmp/laika-sam-env
/tmp/laika-sam-env/bin/pip install torch==2.14.1 torchvision==0.29.1 \
  numpy==2.4.6 onnx==1.23.2 onnxruntime==1.31.0
SAM2_BUILD_CUDA=0 /tmp/laika-sam-env/bin/pip install -e /tmp/laika-sam2
/tmp/laika-sam-env/bin/pip install -e .
/tmp/laika-sam-env/bin/laika-models fetch sam2.1-hiera-tiny \
  --cache-dir /tmp/laika-models
/tmp/laika-sam-env/bin/python scripts/export_sam2_tiny_onnx.py \
  --source-checkout /tmp/laika-sam2 \
  --weights /tmp/laika-models/sam2.1_hiera_tiny.pt \
  --output-dir /tmp/laika-sam-onnx --threads 1
```

The user confirmed that M9 and the other required model uses have project authorization. This supports development and use within Laika; it does not assert that any weights may be publicly redistributed. Keep model assets local by default. The model-card, dataset, and code-license notes below describe published terms and remain relevant to attribution and distribution decisions.

ViTMatte uses the pinned `hustvl/vitmatte-small-composition-1k` safetensors checkpoint at revision `6a58ad7646403c1df626fbd746900aec7361ea1d`, SHA-256 `bda9289db1bb6762d978b42d1c62ae3f34daf7497171a347a1d09657efd788cb`. `scripts/export_vitmatte_small_onnx.py` exports a dynamic-spatial CPU ONNX graph only from the matching local weights and refuses to overwrite existing outputs; `ViTMatteOnnx.from_validation_report()` verifies the checkpoint identity, validation status, graph shape contract and graph SHA before inference. PyTorch parity passed at 512×512, 640×768, and 1024×1024 with maximum absolute alpha error at or below `1.41e-5`. The integrated 512-tile real-photo result and resource measurements are in the [subject benchmark](benchmarks/subject-protection-integration-2026-10-22.md). Project use is authorized by the user. Keep the checkpoint and graph as local assets; this record does not assert public redistribution rights. See the [ViTMatte integration study](research/vitmatte-integration.md) for provenance and remaining validation limits.

## CPU inference adapter and CLI

### GAIC MobileNetV2 crop scoring

`laika.composition.GaicCompositionOnnx` loads the authorized GAIC MobileNetV2
`reddim16` export on CPU. The pinned local checkpoint is SHA-256
`eb1d5eed2710c79d3270add448111b9e6fa2475e246694c0ad995a6efb299e8c` from
`bo-zhang-cs/GAIC-Pytorch` commit `877d6a325943c0a098324dc6d4a975ff4097c3f6`.
The adapted implementation and its MIT notice are in
[`tools/gaic_mobile_port.py`](../tools/gaic_mobile_port.py) and
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md). The checkpoint is a
separate local model asset; project use is authorized by the user, and public
redistribution rights are not asserted. Its record is
[`models/gaic-local.json`](../models/gaic-local.json).

Export from the verified local checkpoint. The exporter refuses a hash mismatch
or overwrite, emits dynamic image and candidate axes, runs ONNX checker, and
compares PyTorch scores to CPU ONNX on three shapes. Keep the `.validation.json`
beside the graph because inference validates that report and the graph SHA.

```bash
python scripts/export_gaic_mobilenetv2_onnx.py \
  --checkpoint /local/models/GAIC-mobilenetv2-reddim16.pth \
  --output-dir /local/models/gaic/export --threads 1
```

The adapter accepts EXIF-oriented HWC `uint8` RGB, downsizes only the analysis
raster to a maximum long edge of 1220 with OpenCV `INTER_LINEAR_EXACT`, and leaves the
requested/source output dimensions intact. It uses ImageNet RGB normalization,
replicate-pads axes smaller than 32 without rescaling, packs dynamic `[N,4]`
crop boxes in analysis-image pixels, and scores all candidates in one CPU run.
The grid and selection match the reference, include the full frame in scoring,
and have no rule-score fallback. Cancellation uses ONNX Runtime `RunOptions` so
an active GAIC inference can be terminated.

```python
from laika import CompositionCrop, LaikaProcessingEngine

with LaikaProcessingEngine(
    composition_model_path="/local/models/gaic/export/gaic-mobilenetv2-reddim16.fp32.onnx",
    strategy="low-memory",
) as engine:
    proposals = engine.suggest_composition("input.jpg")
    best = proposals.suggestions[0].crop
    result = engine.apply_crop_file("processed.png", "crop.png", best)
```

`laika composition` prints proposals as JSON. `laika crop` applies a saved
normalized box to an existing image, performs no model load or inference, keeps
the image's original pixels and EXIF-oriented coordinate system, preserves ICC
data, and writes PNG. Across three fresh workers on the measured 1080×1620
P3M-10K portrait, 556 candidate scores took a median 1.016 s and process-tree
RSS median was 353,615,872 B (observed range 343,523,328–362,512,384 B) on a
MacBook Air M1 CPU. The applied 918×1224 crop took 0.176 s end-to-end. On a
physical Xiaomi `2405CPX3DC` Android reference phone, Python and Kotlin selected
the same top crop boxes from the actual photo; score deltas differed by at most
0.0131 due to platform decode/resize differences. Same-tensor score deltas
matched exactly. See the
[GAIC real-photo report](../docs/benchmarks/gaic-real-photo-2026-10-22.md).

The shared engine and CLI expose suggestion and crop operations;
`ProcessingOptions.crop` applies a selected crop during a render. FastAPI has a
bounded `/api/v1/compositions` suggestion route, and the Web editor can select
one of those crops before submitting a render. See the [API task benchmark](benchmarks/api-task-real-photo-2026-10-22.md)
for the first real M9 + Depth + ARNet upload-to-download run and
[`WEB.md`](WEB.md) for the browser smoke record.

Install the optional inference dependencies with `pip install -e '.[inference]'`. The `DepthAnythingV2Small` Python adapter accepts an HWC `uint8` RGB array and returns a float32 depth map at the same width and height. It applies EXIF orientation only when the caller decodes an image; `laika depth` handles orientation, format checks, upload-byte and pixel limits, and writes a NumPy `.npy` array atomically. Defaults cap uploads at 50 MiB, source images at 40 million pixels, and derived network rasters at 4 million pixels. Over-limit requests fail instead of silently changing analysis dimensions. The adapter accepts only dynamic exports with a matching `.onnx.validation.json` report and graph SHA-256. The session uses CPUExecutionProvider and releases its native buffers on `close()` or context exit.

```python
import numpy as np
from laika.depth import DepthAnythingV2Small

rgb: np.ndarray  # H×W×3 uint8, RGB order
with DepthAnythingV2Small("/path/to/depth.onnx") as depth_model:
    prediction = depth_model.infer_rgb(rgb)
    depth_map = prediction.depth  # float32, same H×W as rgb
```

```bash
laika depth input.jpg output-depth.npy \
  --model /path/to/depth.onnx --threads 1
```

By default, `laika depth` caps the analysis image's long edge at 1220, does not upscale smaller sources, uses Pillow bicubic resize, then replicate-pads the right and bottom edges to multiples of 14 before normalization and inference. Padding is cropped before the depth map is restored to the original image dimensions. This follows the reference sizing and resampling contract without importing its implementation. To test the upstream model wrapper's lower-bound resize policy, pass `--input-size 518`; this may upscale a small source and can produce a larger network raster. The full-size output is interpolated using the upstream `align_corners=True` coordinate rule. `scripts/validate_depth_inference.py` compares upstream preprocessing when applicable and compares output maps against the pinned PyTorch implementation on deterministic input. `scripts/benchmark_depth_inference.py` measures the complete CLI worker, including image decode, model load, inference, and full-size output. See the [reference-profile 4K measurements](benchmarks/m3-reference-depth-analysis-sizing-2026-10-21.md). Regular CI covers pure resize, input validation, CLI decoding/output, and package behavior without downloading weights.
