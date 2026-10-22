# ViTMatte edge-refinement study

**Status:** the pinned local ViTMatte Small checkpoint has been hash-verified,
exported to dynamic-spatial CPU ONNX, and compared against PyTorch at three
resolutions. The tiled adapter is connected to the shared SAM/ViTMatte subject
path; a three-point 1080×1620 real-photo run and dataset-mask quality report are
recorded in
[`../benchmarks/subject-protection-integration-2026-10-22.md`](../benchmarks/subject-protection-integration-2026-10-22.md).

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
  dataset restriction for this Composition-1k checkpoint. The user confirmed
  project authorization for required models, including ViTMatte, so Laika
  continues with a user-provided local asset. That confirmation does not assert
  public redistribution rights; keep weights outside the fetchable manifest
  and Git history.
- The reference `models.json` contains checkpoint and ONNX hashes. Laika's
  exporter validates the local checkpoint independently and writes a distinct
  dynamic ONNX graph; both Laika artifact hashes are recorded in the integration
  report.

## Validation scope and next experiments

The checkpoint-backed exporter passed 512×512, 640×768, and 1024×1024
comparisons with maximum alpha errors of `1.41e-5`, `4.65e-6`, and `3.22e-6`.
On the P3M-10K non-privacy 1080×1620 portrait, three positive points refined
with 512-pixel tiles achieved binary IoU `0.98792`, soft alpha MAE `0.00715`,
and 5-pixel boundary precision/recall `0.9059` / `0.8277` against the supplied
JPEG mask. A single face point covered only a local head/neck region (IoU
`0.2128`); separated subject regions require additional positive points. This
is why the engine accepts repeated points and reuses one SAM image embedding.

The three-point M9 + Depth + ARNet + SAM + ViTMatte request peaked at
`788,070,400` bytes process-tree RSS and took `112.73 s` on the 8 GiB MacBook
Air M1. This is not a Linux cgroup run or a 4/2 GiB acceptance result. More
more photos, full-resolution tests and constrained Linux measurements remain
open. A same-photo 512/768/1024 tile sweep slightly favored 512 on runtime and
memory with essentially unchanged mask quality; see the integration report for
the exact local model hashes, timings and reproducible commands.
