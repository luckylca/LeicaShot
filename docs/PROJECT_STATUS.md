# LeicaShot project status

Updated: 2026-10-22

## Current checkpoint

- Stable integration branch: `develop`, at merge commit `c7d3e5b` after PR #28.
- Most recent merged PR: [#28, Web subject-protection controls](https://github.com/luckylca/LeicaShot/pull/28); Node 24 Web checks and all six macOS/Ubuntu/Windows Python 3.11/3.13 checks passed.
- Active branch: `perf/linux-cgroup-pipeline`, created from the clean `develop` head. Current task: land cgroup measurements and a reproducible full-pipeline report; then proceed to exact-preserving optical-scatter CPU profiling.
- The Web editor supports enabling SAM subject protection, selecting/removing multiple positive points, and choosing ViTMatte tile size. Four Web tests and the production build pass; a live Chrome run completed a real five-model task through full-size download.
- Python is the only production processing path. Physical Android serial `a4976c80` is reserved for Kotlin reference comparisons; the connected emulator is not used. The user's installed reference app remains untouched.
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
- First real HTTP task on the same 1080×1620 portrait ran M9 + LabFusion, Depth, ARNet, and explicit crop through one uploaded API task. It downloaded a 918×1224 PNG (963,021 bytes); server process-tree sampled peak RSS was 698,236,928 B, server CPU time 69.25 s, and task wall time 90.60 s. This is on macOS, one CPU thread, Low Memory strategy, with no cgroup measurement; see the [API task benchmark](benchmarks/api-task-real-photo-2026-10-22.md).
- Real GAIC HTTP suggestion route on the same 1080×1620 portrait scored 86 candidates in 1.403 s at 416,251,904 B sampled server process-tree peak RSS. CPU provider and model SHA are in the [composition API benchmark](benchmarks/api-composition-real-photo-2026-10-22.md); cgroup memory is unavailable on macOS.
- Five-model subject-protection API task with three positive points and 512-pixel ViTMatte tiles completed on that 1080×1620 portrait in 102.177 s, at 1,029,177,344 B sampled server process-tree RSS. SAM image encoding ran once and two prompts reused cached embeddings; the full-size 1,699,480-byte PNG hash exactly matches the earlier SDK result. This run had 1,618,722,816 B system memory available at startup and no cgroup limit; see the [Web subject-protection report](benchmarks/web-subject-protection-2026-10-22.md).
- Same-photo subject-stage tile sweep: 512 / 768 / 1024 gave process-tree RSS 924,696,576 / 923,189,248 / 1,095,548,928 B and helper wall 73.84 / 89.38 / 144.55 s. Alpha IoU stayed between 0.98792 and 0.98804. The helper includes M9, SAM and ViTMatte but not Depth or ARNet; 512 was fastest, while additional scenes remain necessary before generalizing.
- For the same source, Python subject compositing versus `SubjectProtectionMath.composite` on the physical ADB phone was MAE 0.0368, RMSE 0.5827, p95 absolute channel error 0, and 98.84% exact RGB pixels; maximum channel error was 68 on a small edge region. This fixture isolates final compositing and is not a comparison of SAM or ViTMatte inference.
- Physical phone comparison: Python vs Kotlin M9/Lab on full-size output had PSNR 53.71 dB and 0.215 mean absolute 8-bit error. With the highlight stage enabled, the 2048×1365 bokeh comparison measured PSNR 41.38 dB, MAE 1.297 and p99 absolute channel error 8. This aggregate metric does not isolate highlight quality; the reference comparison limits are documented in the report.
- Linux cgroup results now exist for one 1080×1620 P3M portrait. In a 4 GiB configured guest/3 GiB task cgroup, all five models completed in 988.21 s with 2,097,229,824 B cgroup peak and no OOM. In a 2 GiB configured guest/1.5 GiB task cgroup, the same task completed in 1,196.67 s with 1,564,434,432 B cgroup peak, 1,545,293,824 B process-tree RSS and no OOM. Only about 8 MiB cgroup headroom remained. Linux output SHA was byte-identical across four memory configurations. This is a single-image challenge measurement, not broad 2 GiB service acceptance; see the [full report](benchmarks/linux-cgroup-full-pipeline-2026-10-22.md).

## Active development checkpoint

- PR #28 is merged and its CI passed. Its browser/API run exercised upload, three SAM points, M9, Depth, ViTMatte, ARNet, preview and original-size PNG download.
- Linux validation used Lima 2.2.1, Docker CLI 29.9.2 and a rootful Ubuntu ARM64 VM. Models and photos were mounted read-only from local `/tmp` paths and were not uploaded. The reproducible benchmark image builds from tracked `tools/cgroup/Dockerfile`.
- The cgroup benchmark reports Linux v1/v2 current, peak, limit and event counters alongside process-tree RSS. One 2 GiB guest run completed with only about 8 MiB below the task container's 1.5 GiB limit; system overhead, more images, higher resolutions and concurrency remain unverified.
- A 128/512-row optical scatter comparison yielded byte-identical pixels; the 512-row scatter stage was faster once but used slightly more memory, with severe run-to-run timing noise in other stages. Keep 128 as default until repeated measurements. Next branch: `perf/optical-scatter-cpu`, profile symmetric pupil-kernel reuse, then measure original-resolution tasks.

## Known gaps and current blocker

- The unified engine now coordinates M9, Depth, ARNet with highlights, SAM and ViTMatte. Full 5760×3840 processing with highlight scatter and subject protection still needs measurement; current coordinated evidence is a 1080×1620 photo.
- One 2 GiB configured Linux guest completed a constrained 1080×1620 task, but the app cgroup peaked at 99.5% of its 1.5 GiB cap and the run took about 20 minutes. This is not a supported 2 GiB service tier; broader data, cold-start variance and concurrent API tasks require measurement.
- GAIC suggestions and crop application are implemented through the shared engine and CLI, and `/api/v1/compositions` now routes suggestions through the same bounded API worker pool. `ProcessingOptions.crop` still applies selected crops without rerunning GAIC.
- PR #26 provides the FastAPI task service; PR #27 adds the React editor and `/api/v1/compositions`, with all seven CI checks passed. The API suggestion route scored 86 candidates in 1.403 s at 416,251,904 B sampled process-tree RSS, and the browser-verified editor flow downloaded a 918×1224 PNG.
- The merged editor supports upload, color/bokeh/combined modes, focus clicks, aperture, GAIC crop suggestions, task progress/cancellation, before/after preview, full PNG download, and SAM/ViTMatte subject controls. PR #28's live browser run selected three points, completed all five models, displayed the comparison, and downloaded the full-size PNG. A separate fixed-coordinate API run matched the prior SDK output exactly. Exact-origin API CORS is opt-in; the API has no user account/authentication layer, defaults to loopback, and uses per-task capability tokens.
- Full-resolution highlight runtime is high on CPU; the 2048×1365 optical scatter stage took about 141 seconds. Native acceleration or a mathematically validated faster execution plan remains a priority.
- The only connected Android device for reference runs is the physical ADB target; do not use the attached emulator. Production image processing remains Python.

## Next action

Finish checks and merge the cgroup benchmark harness/report PR. Immediately create `perf/optical-scatter-cpu` and profile exact-preserving pupil-kernel reuse; continue original-resolution memory profiling, repeated cgroup timing, 8 GiB comparison, service concurrency and soft-edge parity work. Do not generalize this one 1080×1620 Linux challenge run into a deployment guarantee.
