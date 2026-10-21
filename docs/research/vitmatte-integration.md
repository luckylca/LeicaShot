# ViTMatte edge-refinement study

**Status:** adapter and synthetic tiling regression are implemented on
`feat/vitmatte-matte-tiling`. No ViTMatte checkpoint or converted graph has been
downloaded, exported, or benchmarked by Laika.

## Reference contract

The inspected Android reference is pinned in
[`reference-study.md`](reference-study.md) to source revision
`f012dca8a845b8c37389345676e43a7e99a5ae93`. Its subject path passes
`pixel_values` as float32 NCHW with three RGB planes normalized to `[-1, 1]`
and a fourth trimap plane with values `0`, `128/255`, and `1`. It pads the
bottom and right to multiples of 32 and expects one alpha plane of the same
spatial size. The local ONNX exporter contract in that source names the output
`alpha`.

The reference runs one whole-frame pass when the padded area is at most
`1248 × 928` pixels. Larger frames use 1024-pixel square crops, at most 768
pixels between starts, and a 96-pixel linear feather at interior edges. It
skips tiles without unknown trimap pixels and forces known foreground and
background endpoints after matte inference. This is a bounded-memory
approximation; overlapping tiles do not guarantee numerical equality to a
whole-frame transformer pass.

Laika's `prepare_vitmatte_input`, `plan_matte_tiles`, and
`refine_matte_tiled` implement this tensor contract without copying the
reference source. The default preserves its whole-frame threshold at
`1248 × 928` pixels, then uses a configurable 512, 768, or 1024 pixel tile
budget. Set `single_pass_max_pixels=0` for a tile-only comparison sweep.
Input packing rejects analysis rasters whose long edge exceeds 2048 by default
and never resizes implicitly. Loading a local ONNX graph requires the caller's
expected SHA-256, rejects a mismatch, and rejects fixed spatial dimensions.
Starts remain aligned to 32 pixels, overlap is at least twice the feather
width, all tiles touching unknown pixels are included, and the accumulator
uses normalized float32 blend weights. Known trimap endpoints are restored
exactly. `ViTMatteOnnx` accepts a locally supplied CPU graph only; this feature
does not add a model download command or an entry to the fetchable artifact
manifest.

## License and provenance

- The [upstream ViTMatte source repository](https://github.com/hustvl/ViTMatte)
  contains an MIT license. Laika does not copy its source implementation.
- The [Hugging Face ViTMatte Small model card](https://huggingface.co/hustvl/vitmatte-small-composition-1k)
  labels that artifact Apache-2.0 and says the model was trained on
  Composition-1k. The card also says it was written by Hugging Face because the
  releasing team did not provide a model card.
- Adobe Research's [Deep Image Matting Dataset license](https://git.mitgai.net/adobe-research/Deep_Image_Matting_Dataset/blob/main/LICENSE.md)
  defines a model as configuration data resulting from machine learning using
  the dataset, restricts the dataset to non-commercial purposes, and expressly
  says models can only be used and distributed for non-commercial purposes and
  may not be sold. The model-card Apache label conflicts with that separate
  dataset restriction for this Composition-1k checkpoint. Laika therefore
  excludes this checkpoint and any conversion from commercial service,
  redistribution, and the fetchable model manifest absent a separate written
  grant. The adapter code remains usable for non-commercial research or a
  separately cleared compatible checkpoint.
- The reference `models.json` contains checkpoint and ONNX hashes, but Laika has
  not independently verified either artifact. Those values are not promoted to
  Laika's model manifest or treated as a verified baseline.

## Validation scope and next experiments

Current tests cover tensor normalization/padding, tile geometry, unknown-pixel
coverage, endpoint preservation, output contract checks, CPU-session wiring,
and a pointwise synthetic matte that is exactly independent of crop context.
These tests verify implementation invariants only. They cannot measure hair
matte quality, tile seams for the real ViTMatte model, ONNX parity, or memory
and speed.

Next, after checkpoint and dataset terms are reviewed, add a local-only
checkpoint hash gate and dynamic ONNX exporter, compare outputs against the
pinned PyTorch checkpoint on synthetic and licensed matte fixtures, and run
fresh-process tile sweeps at 512/768/1024. Record tile-level and full subject
stage RSS, cgroup peak, latency, max/mean/p99 alpha error, and seam-region
metrics. No complete 4 GiB or 2 GiB Laika pipeline claim follows from the
synthetic tests in this branch.
