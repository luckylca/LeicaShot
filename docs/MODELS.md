# Model artifact acquisition

Model files are not committed to Laika. `models/manifest.json` pins each supported artifact's upstream revision, filename, expected size, SHA-256, and license record. The `laika-models` command downloads into a user cache and publishes the file only after both size and SHA-256 match.

The initial manifest entry is Depth Anything V2 Small. The official Hugging Face model card identifies this Small checkpoint as Apache-2.0; its file page records the checkpoint hash and byte size. The upstream model implementation is separately pinned to `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`. The checkpoint has been strictly loaded and run on CPU. Static and dynamic ONNX exports have passed seeded-tensor checks; the dynamic graph was tested at four multiple-of-14 shapes. This does not validate other dimension constraints, photo quality, or the full Laika pipeline; see the [M1 benchmark](benchmarks/m1-depth-anything-2026-10-21.md) and [dynamic-shape experiment](benchmarks/m1-dynamic-depth-2026-10-21.md).

```bash
laika-models list
laika-models fetch depth-anything-v2-small
laika-models verify depth-anything-v2-small \
  ~/.cache/laika/models/depth_anything_v2_vits.pth
```

Set `LAIKA_MODEL_CACHE` or pass `--cache-dir` to change the cache location. Downloads use a temporary file in the target directory and are atomically renamed only after validation. A mismatched existing file is reported and left untouched. Run `laika-models fetch` explicitly; installing Laika never downloads model weights.

## Reproduce static and dynamic ONNX exports

The converter uses the upstream model definition only from its manifest-pinned revision. It writes a graph, optional external tensor data, and a JSON validation report. It refuses to overwrite any of those files and checks framework and ONNX Runtime outputs on seeded synthetic tensors before publishing. By default the export is static; add `--dynamic` to use the validated dynamic-height/width path. That path requires each network dimension to be at least 518 and divisible by 14; the inference engine must enforce those conditions before invoking ONNX Runtime.

For the experimental DINO attention optimization, add `--query-chunk-size 128` together with `--dynamic`. Each query tile attends across all keys and values. The exporter checks equality against the unchunked PyTorch model and verifies that the ONNX graph retains its 12 loop nodes; it rejects chunked exports that lose this control flow. On the tested M1, chunk size 128 cut single-stage RSS by 61% with a 3.6% median latency increase at 1400×518. See the [query-chunking report](benchmarks/m1-depth-query-chunking-2026-10-21.md); this remains experimental and is not yet part of the production engine.

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

GitHub Actions provides an opt-in `Optional model smoke` workflow. Add the `model-smoke` label to a pull request to run it; it downloads and verifies this checkpoint, exports dynamic baseline and query-chunked CPU ONNX models, checks four input shapes, and prints PyTorch/ONNX process-tree benchmarks including baseline versus query-chunked 1400×518 runs. The Linux runner's cgroup has no configured memory limit, so its measurements do not represent a 4 GiB or 2 GiB test. Manual dispatch is available once the workflow is present on the repository's default branch. Regular cross-platform CI remains model-free.

The M9 checkpoint remains excluded because its reference project license limits it to personal non-commercial use. ARNet and GAIC checkpoint permissions remain unresolved. The ViTMatte model card says Apache-2.0, while its Composition-1k training data requires separate license review before commercial use or redistribution. See the [reference study](research/reference-study.md).
