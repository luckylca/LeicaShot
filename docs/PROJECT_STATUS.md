# LeicaShot project status

Updated: 2026-10-22

## Current checkpoint

- Stable integration branch: `develop`, currently at `6c17b59` (PR #23).
- Active work: `feat/subject-protection-integration` — connects SAM 2.1, ViTMatte, and foreground compositing to the shared Python M9 + Depth + ARNet request.
- Most recent merged PR: [#23, reference optical highlight scattering](https://github.com/luckylca/LeicaShot/pull/23).
- Existing cross-platform CI passed on macOS, Ubuntu, and Windows with Python 3.11 and 3.13.
- Current subject-protection branch: 141 unit/integration tests pass locally; Ruff lint and formatting checks pass. Cross-platform PR CI remains the merge gate.
- Previously merged model work: M9 + Lab fusion, Depth Anything V2 Small CPU adapter, SAM 2.1 encoder/decoder and bounded feature cache, ViTMatte tiled adapter, and BokehMe ARNet multiscale renderer.

## Models and verified assets

| Model | Python implementation | Current verification |
|---|---|---|
| M9 series 3.0 epoch11 | `laika.m9.M9ColorEngine` | CPU inference, SHA-256 check, real-photo M9/Lab output and RSS report; local model at `/tmp/laika-models/leica-m9-v3-epoch11.fp16.onnx` |
| Depth Anything V2 Small | `laika.depth.DepthAnythingV2Small` | CPU adapter and full-size real-photo depth report; pinned checkpoint and validated dynamic q64 ONNX are in `/tmp/laika-models` |
| BokehMe ARNet + optical highlights | `laika.bokeh.BokehMeARNet` and `laika.highlights` | CPU multiscale real-photo run with extraction, clipped-light recovery, field-dependent scatter and composite; Python/Kotlin intermediate fixtures numerically compared; local ONNX at `/tmp/laika-models/legendary-1.0-arnet.fp16.onnx` |
| SAM 2.1 Hiera Tiny | `laika.subject.Sam2TinyOnnx` | Hash-verified split ONNX graphs, checkpoint-backed tensor parity, bounded encoder cache, multi-point real-photo pipeline validation |
| ViTMatte Small | `laika.matting.ViTMatteOnnx` | Pinned local checkpoint and dynamic ONNX export, three-shape PyTorch parity, tiled 512-pixel real-photo refinement and mask metrics |
| GAIC | Not integrated | Checkpoint conversion, scoring, and crop application remain outstanding |

Model weights and photographs stay outside Git. Project-use authorization is confirmed by the user; this does not assert public redistribution rights.

## Performance facts

- Host: MacBook Air M1, 8 GiB RAM, CPU-only ONNX Runtime.
- Standalone ARNet renderer: 1024×683 measured 658,980,864 B high-water RSS / 8.31 s; 2048×1365 measured 1,118,208,000 B / 27.74 s. Both reused precomputed depth and do not measure M9 or depth inference.
- Standalone Depth Anything V2 real-photo run: 1,695,547,392 B process high-water RSS / 14.402 s total for a 5760×3840 source. This is not a combined-task result.
- Corrected-order 2048×1365 end-to-end task: 1,094,172,672 B high-water RSS / 54.38 s wall.
- Corrected-order 5760×3840 end-to-end task: 2,197,651,456 B high-water RSS / 605.43 s wall; ARNet's first full-resolution pass took 514.45 s. Both used Low Memory, CPUExecutionProvider, one intra-op thread, and all three hash-recorded models. The detailed [unified pipeline report](benchmarks/unified-pipeline-real-photo-2026-10-22.md) records concurrent host load and remaining quality-comparison limits.
- With highlights integrated, the same source processed end-to-end at the explicit 2048×1365 output size in 197.85 s and 1,110,802,432 B process-tree peak RSS. Optical highlight scatter took 140.89 s. This is not a full-resolution highlight run or a Linux cgroup result. The [highlight report](benchmarks/bokeh-highlight-scatter-2026-10-22.md) records stage timings, row-chunk results, Kotlin fixture errors and the real-photo comparison.
- Five-model Low Memory run on a 1080×1620 P3M-10K non-privacy portrait: 788,070,400 B process-tree peak RSS and 112.73 s wall. It ran M9, Depth, ARNet, SAM and ViTMatte on CPUExecutionProvider with one intra-op thread and preserved output dimensions. Three positive clicks yielded binary mask IoU 0.9879 against the dataset mask; SAM's image encoder ran once and the next two point prompts hit the embedding cache. The [subject-protection report](benchmarks/subject-protection-integration-2026-10-22.md) contains model hashes, per-stage times, alpha metrics and phone-reference comparison.
- For the same source, Python subject compositing versus `SubjectProtectionMath.composite` on the physical ADB phone was MAE 0.0368, RMSE 0.5827, p95 absolute channel error 0, and 98.84% exact RGB pixels; maximum channel error was 68 on a small edge region. This fixture isolates final compositing and is not a comparison of SAM or ViTMatte inference.
- Physical phone comparison: Python vs Kotlin M9/Lab on full-size output had PSNR 53.71 dB and 0.215 mean absolute 8-bit error. With the highlight stage enabled, the 2048×1365 bokeh comparison measured PSNR 41.38 dB, MAE 1.297 and p99 absolute channel error 8. This aggregate metric does not isolate highlight quality; the reference comparison limits are documented in the report.
- No 4 GiB or 2 GiB full-pipeline acceptance result exists.

## Known gaps and current blocker

- The unified engine now coordinates M9, Depth, ARNet with highlights, SAM and ViTMatte. Full 5760×3840 processing with highlight scatter and subject protection still needs measurement; current coordinated evidence is a 1080×1620 photo.
- RSS is close to, and slightly above, 2 GiB in decimal/roughly 2.05 GiB in binary units on this macOS run. It is not a Linux cgroup-limited result; no 2 GiB acceptance claim is made.
- GAIC model export, crop scoring/application, and composition-only/combined processing modes are not yet implemented.
- FastAPI tasks, cancellation endpoints, bounded server scheduling, and Web editing flow are not implemented.
- Full-resolution highlight runtime is high on CPU; the 2048×1365 optical scatter stage took about 141 seconds. Native acceleration or a mathematically validated faster execution plan remains a priority.
- The only connected Android device for reference runs is the physical ADB target; do not use the attached emulator. Production image processing remains Python.

## Next action

Complete review and merge `feat/subject-protection-integration` after CI. Immediately branch for GAIC model export, candidate scoring and crop application. Continue optical-scatter CPU profiling and full-resolution/Linux memory measurements alongside GAIC; then deliver FastAPI, Web editing, deployment and low-memory acceptance without treating any single PR as project completion.
