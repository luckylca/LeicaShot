# Model artifact acquisition

Model files are not committed to Laika. `models/manifest.json` pins each supported artifact's upstream revision, filename, expected size, SHA-256, and license record. The `laika-models` command downloads into a user cache and publishes the file only after both size and SHA-256 match.

The initial manifest entry is Depth Anything V2 Small. The official Hugging Face model card identifies this Small checkpoint as Apache-2.0; its file page records the checkpoint hash and byte size. The upstream model implementation is separately pinned to `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`. The checkpoint has been strictly loaded and run on CPU, and a fixed-shape ONNX export has passed a numerical check. That does not yet validate arbitrary input sizes, photo quality, or the full Laika pipeline; see [the M1 benchmark](benchmarks/m1-depth-anything-2026-10-21.md).

```bash
laika-models list
laika-models fetch depth-anything-v2-small
laika-models verify depth-anything-v2-small \
  ~/.cache/laika/models/depth_anything_v2_vits.pth
```

Set `LAIKA_MODEL_CACHE` or pass `--cache-dir` to change the cache location. Downloads use a temporary file in the target directory and are atomically renamed only after validation. A mismatched existing file is reported and left untouched. Run `laika-models fetch` explicitly; installing Laika never downloads model weights.

## Reproduce the fixed-shape ONNX export

The converter uses the upstream model definition only from its manifest-pinned revision. It writes a graph, external tensor data, and a JSON validation report. It refuses to overwrite any of those files and checks the framework and ONNX Runtime outputs on a seeded synthetic tensor before publishing. The exported model has **static network input dimensions**; this first conversion experiment is not the final arbitrary-resolution inference engine.

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
```

Use a separate virtual environment for this large conversion toolchain. The exact dependency versions above reproduce the recorded macOS experiment; select and record compatible CPU wheels for other operating systems. Both `.onnx` and `.onnx.data` are required at runtime. The JSON report records output hashes, versions, fixed input shape, export/session/inference times, and maximum/mean/p99 absolute error. Do not infer dynamic-shape support from this fixed-shape export.

GitHub Actions provides an opt-in `Optional model smoke` workflow. Add the `model-smoke` label to a pull request to run it; it downloads and verifies this checkpoint, exports a 518×518 CPU ONNX model, checks numerical parity, and prints separate PyTorch/ONNX process-tree benchmarks. It also supports manual dispatch once the workflow is present on the repository's default branch. The regular cross-platform CI remains model-free.

The M9 checkpoint remains excluded because its reference project license limits it to personal non-commercial use. ARNet and GAIC checkpoint permissions remain unresolved. The ViTMatte model card says Apache-2.0, while its Composition-1k training data requires separate license review before commercial use or redistribution. See the [reference study](research/reference-study.md).
