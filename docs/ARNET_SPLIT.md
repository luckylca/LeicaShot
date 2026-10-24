# Optional ARNet global-resize / tiled-core inference

## Purpose

The original ARNet graph uses **global align_corners Resize** both before
and after eight 3x3 convolution layers. Local full-graph tiling is not
coordinate-equivalent for odd image dimensions and may change visible color
values. In the existing Low Memory path, large odd-sized inputs consequently
fell back to one full-frame CPU ONNX session, requiring >1GB RSS for a
2048x1365 ARNet pass on the tested Windows laptop.

An optional verified three-graph execution path:
1. Compute the original, **global** resize / channel permutation once;
2. Execute only the eight convolution layers on overlapping patches of
   the half-resolution feature tensor (8 feature pixels of halo);
3. Run the original, **global** output resize and crop.

This avoids the global feature maps of the convolution stack in memory.
No original weights are retrained or re-quantized, image output is not
resized, and the optics/depth/matting stages remain unchanged. Small
floating-point rounding differences can arise from tile-dependent ONNX
CPU kernels, so bitwise ARNet output identity is NOT claimed.

## Generate locally (once per source ONNX revision)

ONNX model weights and derived graph files are deliberately not in Git.
Source file must already be authorized and verified by LeicaShot.

  pip install onnx
  python scripts/export_arnet_split_graphs.py --model PATH_TO_VERIFIED_ARNET.onnx --output-dir PATH_TO_NEW_EMPTY_SPLIT_DIR

The exporter creates:
- arnet-prefix.onnx
- arnet-core.onnx
- arnet-suffix.onnx
- manifest.json with SHA256 of the source and each derived graph

The runtime refuses a split manifest with a different source model hash,
an absent model, or a modified subgraph. The runtime requires only the
normal NumPy + ONNX Runtime inference dependencies (not the onnx package).

To opt in on Windows PowerShell:

  $env:LAIKA_ARNET_SPLIT_DIR = "E:\codex\LeicaShot-local-data\laika-models\arnet-split-verified"

Then use the usual laika CLI or start the API from that same environment.
Unsetting the variable restores the previous behavior. The split execution
path is used only with strategy low-memory, inputs above the full-inference
pixel limit, and **odd width or height**. Even-sized frame inference,
small frames, and balanced strategy remain unchanged.
Half-resolution feature-tile edge defaults to 256; caller API also allows
an explicit split_graph_edge when constructing BokehMeARNet.

## Windows measured result, 2026-10-24

Host: Windows CPUExecutionProvider, one CPU inference thread. Input was
a 2048x1365 real-photo RGB frame and a deterministic float32 defocus map.
Native full-frame ARNet uses 12.409 s and 1,217,134,592 B sampled peak RSS;
the optional 256-feature-pixel tiled-core path uses 14.413 s and
305,594,368 B sampled peak RSS.

This is approximately **74.9% lower ARNet process RSS** for approximately
16.1% longer single-pass time in the adapter A/B. A lower-level graph
benchmark on the same 2048x1365 geometry reported 1,215,807,488 B
whole-frame vs 289,062,912 B split at tile edge 256, core runtime
12.295 vs 13.105 seconds (not counting graph prefix/suffix overhead).

The split and full versions differ by at most one 8-bit channel value
on this tested ARNet input; approximately 4.5% of output pixels have any
different channel value, with very low average RGB error. This is NOT
bitwise exact model equivalence. By contrast, naive whole-graph tiling
had up to 56 8-bit channel levels error and affected about 42% of pixels;
that method is rejected.

A single five-model 2048x1365 real-photo execution through CLI with the
new environment variable completed in 58.379 seconds, and its final PNG
was byte-identical to the prior whole-graph five-model test for that photo.
The whole process reached approximately 1.296 GB RSS regardless of ARNet
split usage, so the memory reduction did NOT yet materially lower the
whole service's peak. The 58.379s result was not a repeated or correctly
paired full-flow A/B for ARNet split performance, and must not be marketed
as a guaranteed overall speedup.

Isolated candidate half-resolution tile edges on 2048x1365:

| Feature tile edge | Sampled RSS | Core seconds |
|---|---:|---:|
| 128 | 228.2 MB | 15.0 |
| 256 | 289.1 MB | 13.1 |
| 384 | 414.4 MB | 12.6 |
| 512 | 597.0 MB | 12.5 |

256 is a speed-memory compromise. These are process memory estimates, not
Linux cgroup peak memory. The split core's output hash was the same for
all tile edges tested, indicating that 8 feature pixels' overlap covered
the eight 3x3 convolution receptive field on this runtime.

## Limitations and follow-up

- High-resolution (>20 MP) full-pipeline and strict 2/4GiB cgroup tests
  have not been repeated for this optional setting.
- Compare diverse images/defocus maps and API concurrency before offering
  a default change; do not assume one photo's final PNG will always match.
- Profile memory per full-pipeline stage to find the remaining ~1.3GB
  high-water mark that independent ARNet savings did not resolve.
- For image-quality-critical exact-reproduction use, leave the split graph
  option disabled. The existing full graph remains the reference.
