# LeicaShot project status

Updated: 2026-10-22

## Current checkpoint

- Stable integration branch: `develop`, at merge commit `9192f6d` after PR #24.
- Most recent merged PR: [#24, SAM and ViTMatte subject protection](https://github.com/luckylca/LeicaShot/pull/24); all six macOS/Ubuntu/Windows Python 3.11/3.13 CI jobs passed.
- Active branch: `feat/gaic-composition-integration`; this branch adds GAIC export/inference, shared-engine suggestion and crop APIs, CLI, tests, and real-photo/reference evidence. PR is not yet opened.
- Python production path only. Android is used only with `adb -s a4976c80` for reference Kotlin comparisons; the temporary comparison APK has a separate application ID, and the user's installed reference app was left intact.
- The working GAIC checkpoint and ONNX export are local under `/tmp/laika-models/gaic`; neither is in Git.

## Models and verified assets

| Model | Python implementation | Current verification |
|---|---|---|
| M9 series 3.0 epoch11 | `laika.m9.M9ColorEngine` | CPU inference, SHA-256 check, real-photo M9/Lab output and RSS report; local model at `/tmp/laika-models/leica-m9-v3-epoch11.fp16.onnx` |
| Depth Anything V2 Small | `laika.depth.DepthAnythingV2Small` | CPU adapter and full-size real-photo depth report; pinned checkpoint and validated dynamic q64 ONNX are in `/tmp/laika-models` |
| BokehMe ARNet + optical highlights | `laika.bokeh.BokehMeARNet` and `laika.highlights` | CPU multiscale real-photo run with extraction, clipped-light recovery, field-dependent scatter and composite; Python/Kotlin intermediate fixtures numerically compared; local ONNX at `/tmp/laika-models/legendary-1.0-arnet.fp16.onnx` |
| SAM 2.1 Hiera Tiny | `laika.subject.Sam2TinyOnnx` | Hash-verified split ONNX graphs, checkpoint-backed tensor parity, bounded encoder cache, multi-point real-photo pipeline validation |
| ViTMatte Small | `laika.matting.ViTMatteOnnx` | Pinned local checkpoint and dynamic ONNX export, three-shape PyTorch parity, tiled 512-pixel real-photo refinement and mask metrics |
| GAIC MobileNetV2 reddim16 | `laika.composition.GaicCompositionOnnx` | Hash-gated dynamic FP32 ONNX export; three-shape PyTorch parity; 556-candidate real-photo scoring; same-tensor Kotlin parity and actual-photo crop choices match; crop application does not load a model |

Model weights and photographs stay outside Git. Project-use authorization is confirmed by the user; this does not assert public redistribution rights.

## Performance facts

- Host: MacBook Air M1, 8 GiB RAM, CPU-only ONNX Runtime.
- Standalone ARNet renderer: 1024×683 measured 658,980,864 B high-water RSS / 8.31 s; 2048×1365 measured 1,118,208,000 B / 27.74 s. Both reused precomputed depth and do not measure M9 or depth inference.
- Standalone Depth Anything V2 real-photo run: 1,695,547,392 B process high-water RSS / 14.402 s total for a 5760×3840 source. This is not a combined-task result.
- Corrected-order 2048×1365 end-to-end task: 1,094,172,672 B high-water RSS / 54.38 s wall.
- Corrected-order 5760×3840 end-to-end task: 2,197,651,456 B high-water RSS / 605.43 s wall; ARNet's first full-resolution pass took 514.45 s. Both used Low Memory, CPUExecutionProvider, one intra-op thread, and all three hash-recorded models. The detailed [unified pipeline report](benchmarks/unified-pipeline-real-photo-2026-10-22.md) records concurrent host load and remaining quality-comparison limits.
- With highlights integrated, the same source processed end-to-end at the explicit 2048×1365 output size in 197.85 s and 1,110,802,432 B process-tree peak RSS. Optical highlight scatter took 140.89 s. This is not a full-resolution highlight run or a Linux cgroup result. The [highlight report](benchmarks/bokeh-highlight-scatter-2026-10-22.md) records stage timings, row-chunk results, Kotlin fixture errors and the real-photo comparison.
- Five-model Low Memory run on a 1080×1620 P3M-10K non-privacy portrait: 788,070,400 B process-tree peak RSS and 112.73 s wall. It ran M9, Depth, ARNet, SAM and ViTMatte on CPUExecutionProvider with one intra-op thread and preserved output dimensions. Three positive clicks yielded binary mask IoU 0.9879 against the dataset mask; SAM's image encoder ran once and the next two point prompts hit the embedding cache. The [subject-protection report](benchmarks/subject-protection-integration-2026-10-22.md) contains model hashes, per-stage times, alpha metrics and phone-reference comparison.
- GAIC-only Low Memory run on the same 1080×1620 portrait: three fresh workers measured 353,615,872 B median sampled process-tree RSS (range 343,523,328–362,512,384 B) and 1.016 s median worker wall for 556 crop proposals; median ONNX inference was 0.773 s. A model-free selected crop operation took 0.176 s and produced a 918×1224 PNG. Same-tensor Kotlin selected the exact same boxes and scores; on the full photo preprocessing paths the top-four crop boxes matched and score improvements differed by at most 0.0131. See the [GAIC report](benchmarks/gaic-real-photo-2026-10-22.md).
- The selected GAIC crop was also applied inside a real M9+LabFusion `process_file()` request on that portrait: 356,876,288 B sampled process-tree RSS / 20.117 s total, with a 918×1224 result. It matched cropping the full-size M9 result pixel-for-pixel. This is color-only and not a complete depth/bokeh task.
- Same-photo subject-stage tile sweep: 512 / 768 / 1024 gave process-tree RSS 924,696,576 / 923,189,248 / 1,095,548,928 B and helper wall 73.84 / 89.38 / 144.55 s. Alpha IoU stayed between 0.98792 and 0.98804. The helper includes M9, SAM and ViTMatte but not Depth or ARNet; 512 was fastest, while additional scenes remain necessary before generalizing.
- For the same source, Python subject compositing versus `SubjectProtectionMath.composite` on the physical ADB phone was MAE 0.0368, RMSE 0.5827, p95 absolute channel error 0, and 98.84% exact RGB pixels; maximum channel error was 68 on a small edge region. This fixture isolates final compositing and is not a comparison of SAM or ViTMatte inference.
- Physical phone comparison: Python vs Kotlin M9/Lab on full-size output had PSNR 53.71 dB and 0.215 mean absolute 8-bit error. With the highlight stage enabled, the 2048×1365 bokeh comparison measured PSNR 41.38 dB, MAE 1.297 and p99 absolute channel error 8. This aggregate metric does not isolate highlight quality; the reference comparison limits are documented in the report.
- No 4 GiB or 2 GiB full-pipeline acceptance result exists.

## Known gaps and current blocker

- The unified engine now coordinates M9, Depth, ARNet with highlights, SAM and ViTMatte. Full 5760×3840 processing with highlight scatter and subject protection still needs measurement; current coordinated evidence is a 1080×1620 photo.
- RSS is close to, and slightly above, 2 GiB in decimal/roughly 2.05 GiB in binary units on this macOS run. It is not a Linux cgroup-limited result; no 2 GiB acceptance claim is made.
- GAIC suggestions and crop application are implemented through the shared engine and CLI. `ProcessingOptions.crop` applies a previously selected crop after color, bokeh, or combined rendering; API task parameters and Web controls remain unfinished.
- Local verification on the active GAIC branch: 159 pytest cases, Ruff lint and format checks passed; the real M9-plus-crop CPU run and exact crop-output comparison passed.
- FastAPI tasks, cancellation endpoints, bounded server scheduling, and Web editing flow are not implemented.
- Full-resolution highlight runtime is high on CPU; the 2048×1365 optical scatter stage took about 141 seconds. Native acceleration or a mathematically validated faster execution plan remains a priority.
- The only connected Android device for reference runs is the physical ADB target; do not use the attached emulator. Production image processing remains Python.

## Next action

Finish full test and CI checks for `feat/gaic-composition-integration`, open and squash-merge its PR, then immediately begin `feat/fastapi-task-service`. API task requests should pass selected GAIC crops to `ProcessingOptions` and preserve the separate crop-only operation so an already processed image is not sent through AI again. Continue optical-scatter CPU profiling, full-resolution runs, and Linux cgroup memory measurements alongside the API/Web work.
