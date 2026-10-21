# Reference implementation study (initial)

**Research snapshot:** 2026-10-21. Source reviewed at commit `f012dca8a845b8c37389345676e43a7e99a5ae93` (reference release documentation identifies version 2.1.22). The checkout is under `/tmp/laika-reference`, outside this repository. This document summarizes contracts and design choices; it does not copy source code.

## Pipeline and ownership observations

The reference engine sequences color, depth, bokeh, and optional subject protection in one engine, with a separate GAIC composition path. It normalizes photo orientation before mapping user points and caps analysis dimensions independently from export dimensions. A depth estimate is relative depth rather than calibrated metric distance. The color model predicts low-frequency color and the original detail is restored through Lab fusion. For bokeh, the pipeline derives a circle-of-confusion field from relative depth, focus, and aperture, then invokes ARNet at multiple blur scales and separately handles highlights and subject edges.

Subject protection uses one SAM image embedding for multiple prompt points, selects among mask candidates, creates a trimap, and runs ViTMatte on unknown-boundary areas. The reference computes per-point alpha results so incremental point edits can reuse completed results. It releases SAM before loading ViTMatte. Its large-image matting uses 1024-square tiles, up to 768-pixel steps, and feathered overlap; this is explicitly a memory/consistency tradeoff, not a guarantee of full-frame numerical equality. GAIC proposes/scored crops separately from final export and maintains original-image coordinate mapping.

The reference code includes separate edge and highlight modules (including `SoftBokehEdges`, `BokehRadiance`, `ClippedHighlights`, `HighlightExtraction`, `HighlightScatter`, `NaturalSubjectEdges`, `GradientSubjectEdges`, and `SharpSubjectEdges`). These are distinct from the learned ARNet pass and must be measured/tested as a group if implemented in Laika. Its regression suite includes synthetic math, image fixtures, edge, matte-tile, composition, cancellation, and workflow tests; a JVM test pass cannot validate ONNX model execution or service memory.

## Model mapping

| Role | Reference checkpoint / contract | Conversion and preprocessing notes | Permission status for Laika |
|---|---|---|---|
| AI color | M9 series 3.0 epoch11, bundled in reference; RGB NCHW, nominal `[-1, 1]`, long edge 1024; outputs low-frequency color for Lab fusion | Internal FP16, external FP32; reference fuses model color with original detail in Lab rather than treating output as the finished photo | **Excluded.** The reference license explicitly limits its own M9 and original material to personal non-commercial use. No commercial service permission is established. |
| Depth | Depth Anything V2 Small; reference ONNX input `[1,3,H,W]`, output `[1,H,W]`; RGB ImageNet normalization; relative depth | FP32/opset 17; input padded to a multiple of 14; exact global attention query chunks of 64 still attend to all keys | Upstream Small model card marks Apache-2.0; this conclusion does not cover Base/Large/Giant variants. Recheck exact checkpoint and notices before redistribution. |
| Neural bokeh | BokehMe ARNet; `image [1,3,H,W]`, signed normalized defocus `[1,1,H,W]`, RGB output `[1,3,H,W]` | Reference export takes renderer RGB, fixes gamma 0.3, preserves even-size resampling/pixel shuffle; internal FP16, external FP32 | Upstream code is Apache-2.0, but the reference explicitly says checkpoint/data rights still need confirmation. Keep unresolved until exact checkpoint terms are reviewed. |
| Subject prompts | SAM 2.1 Hiera Tiny; fixed `[1,3,1024,1024]` image and one positive `[1,1,2]` point; outputs three `[1,3,256,256]` mask logits plus scores | The reference exports image encoder and point/mask decoder separately enough to reuse image features; no video memory path | SAM 2 code/checkpoint are declared Apache-2.0 upstream. Validate the exact export and required notices before distributing a converted ONNX. |
| Edge matte | ViTMatte Small Composition-1k; `pixel_values [1,4,H,W]` (normalized RGB plus trimap), alpha `[1,1,H,W]`; dimensions padded to multiples of 32 | FP32, dynamic spatial sizes; reference performs full-frame inference only below a pixel budget and otherwise uses overlapping tiles | HF model card labels the model Apache-2.0, but the training dataset is Adobe Composition-1k and its separate terms need review. Treat commercial use and redistribution as **unresolved** until the model/dataset rights are reconciled. |
| Composition | GAIC MobileNetV2 `reddim16`; normalized RGB image + pixel-coordinate boxes, one score per candidate | FP32; reference preserves model-specific RoI/RoD 10×10 sampling and 9×9 aggregation, `RoI + 1`, edge extrapolation, and interior mask rather than substituting generic RoIAlign | Code repository is MIT, but its README points to pretrained weights hosted separately without an explicit checkpoint license. Weight redistribution/use rights remain unresolved. |

Model names alone are insufficient provenance. Before any model adapter is implemented, pin the source revision and checkpoint hash, record exact preprocessing and tensor names/shapes, reproduce conversion, compare framework and ONNX outputs, and preserve runtime hash checks.

## Initial low-memory hypotheses to measure

1. **Query-chunked exact global attention:** the reference export script chunks Q and attends each chunk to the complete K/V set. This bounds the materialized score matrix by chunk size while preserving global key coverage. Verify the exported graph does not re-materialize the full matrix and compare output error for chunk sizes 16/32/64/128.
2. **ViTMatte tiling:** reference tiles are 1024×1024, overlapping and feathered. Compare 512/768/1024 tiles only with a fixed full-frame or trusted reference output, including hair and boundary seam cases; preserve full analysis resolution.
3. **Sequential session lifetime:** the reference closes SAM before ViTMatte. Measure model load, inference, release, process RSS/USS, and cold/hot latency. Process exit may release native allocations more reliably, but can increase cold-start time.
4. **Feature reuse:** for multiple SAM clicks on the same image, cache image embeddings once and reuse them in the mask decoder. Bound cache size and invalidate on image identity/transform changes.
5. **Buffers and data movement:** quantify image decode, tensor conversion, copies, and postprocessing separately. Avoid keeping multiple full-resolution float RGB arrays alive; use byte planes, preallocation, row/tile transforms, or disk-backed intermediates only when measurements show a benefit.

These are experiments, not results. No 8GB, 4GB, or 2GB complete-pipeline target has been demonstrated.

## Source and license records

- [Reference source](https://github.com/shuncongci/homemade-legendary-moment/tree/f012dca8a845b8c37389345676e43a7e99a5ae93), especially `models.json`, `docs/MODELS.md`, `docs/ARCHITECTURE.md`, `LeicaEngine.kt`, `LegendaryMath.kt`, `LabFusion.kt`, `SubjectProtection.kt`, `SubjectMatteTiles.kt`, `GaicComposition.kt`, `BokehRadiance.kt`, `SoftBokehEdges.kt`, and `tools/model_export/chunk_attention_fp32.py`.
- The reference's custom [LICENSE](https://github.com/shuncongci/homemade-legendary-moment/blob/f012dca8a845b8c37389345676e43a7e99a5ae93/LICENSE) says it is not an OSI open-source license and limits covered original material and M9 to personal non-commercial use.
- [Depth Anything V2 Small model card](https://huggingface.co/depth-anything/Depth-Anything-V2-Small) identifies the Small checkpoint as Apache-2.0; other model sizes have separate terms.
- [BokehMe upstream](https://github.com/JuewenPeng/BokehMe) and the reference's model notes do not settle the exact `arnet.pth` checkpoint license.
- [GAIC-Pytorch](https://github.com/bo-zhang-cs/GAIC-Pytorch) identifies its code as MIT and links pretrained weights separately; the linked checkpoint has no explicit license in that README.
- [SAM 2 upstream license](https://github.com/facebookresearch/sam2/blob/main/LICENSE) declares Apache-2.0 for SAM 2.
- [ViTMatte model card](https://huggingface.co/hustvl/vitmatte-small-composition-1k) labels the artifact Apache-2.0 and states it was trained on Composition-1k. Separate dataset/checkpoint terms must still be checked before commercial use or redistribution.
