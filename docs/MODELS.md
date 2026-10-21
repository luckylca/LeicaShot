# Model artifact acquisition

Model files are not committed to Laika. `models/manifest.json` pins each supported artifact's upstream revision, filename, expected size, SHA-256, and license record. The `laika-models` command downloads into a user cache and publishes the file only after both size and SHA-256 match.

The manifest includes Depth Anything V2 Small and a pinned SAM 2.1 Hiera Tiny checkpoint. The official SAM 2.1 model card declares Apache-2.0 and its pinned checkpoint page records SHA-256 `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69` and 156,008,466 bytes. The Meta implementation is separately pinned to `2b90b9f5ceec907a1c18123530e92e794ad901a4`. Laika's split encoder/decoder ONNX export passed synthetic-tensor comparison with the upstream PyTorch model, and the bounded embedding cache is covered by mock-session tests. A 2 GiB single-stage CPU measurement is in the [SAM benchmark](benchmarks/m1-sam2-cgroup-2026-10-21.md); it is not a full-pipeline or real-photo quality result. The reference project's converted ONNX is not downloaded, redistributed, or treated as a validated Laika artifact.

The Depth Anything V2 Small checkpoint's official model card identifies it as Apache-2.0; its file page records the checkpoint hash and byte size. The upstream model implementation is separately pinned to `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`. The checkpoint has been strictly loaded and run on CPU. Static and dynamic ONNX exports have passed seeded-tensor checks; the dynamic graph was tested at four multiple-of-14 shapes. This does not validate other dimension constraints, photo quality, or the full Laika pipeline; see the [M1 benchmark](benchmarks/m1-depth-anything-2026-10-21.md) and [dynamic-shape experiment](benchmarks/m1-dynamic-depth-2026-10-21.md).

```bash
laika-models list
laika-models fetch depth-anything-v2-small
laika-models fetch sam2.1-hiera-tiny
laika-models verify depth-anything-v2-small \
  ~/.cache/laika/models/depth_anything_v2_vits.pth
laika-models verify sam2.1-hiera-tiny \
  ~/.cache/laika/models/sam2.1_hiera_tiny.pt
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

The SAM path accepts an oriented subject-analysis RGB raster whose long edge is at most 2048, resizes it to 1024×1024 with the reference's separable antialiasing, and applies RGB ImageNet normalization. Normalized `(x,y)` prompts map to the model frame. `Sam2TinyOnnx` caches its exact encoder outputs in a configurable byte-bounded LRU and reuses them for later prompt decoder calls; `invalidate()` and `close()` explicitly release cached features. `select_subject_mask()` picks the highest-scoring candidate covering the positive click, resizes its logits to the subject-analysis raster, and thresholds at zero. `build_subject_trimap()` marks the eroded foreground, eroded background, and unknown edge band using the reference's ellipse and scale-adjusted radius. The split FP32 ONNX graphs passed synthetic-tensor comparison with the pinned upstream PyTorch model. The model-gated benchmark is being extended to cover synthetic mask restoration and trimap generation; its masks do not establish photo quality. No ViTMatte refinement or real-photo mask-quality evaluation is implemented yet. The latest completed SAM memory measurements are in the [SAM benchmark](benchmarks/m1-sam2-cgroup-2026-10-21.md).

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

The M9 checkpoint remains excluded because its reference project license limits it to personal non-commercial use. ARNet and GAIC checkpoint permissions remain unresolved. The ViTMatte model card says Apache-2.0, while its Composition-1k training data requires separate license review before commercial use or redistribution. See the [reference study](research/reference-study.md).

## CPU inference adapter and CLI

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
