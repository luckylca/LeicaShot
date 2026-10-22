# Development log

## 2026-10-22 — GAIC crop in shared processing request

- Added the pinned GAIC MobileNetV2 reddim16 export, CPU ONNX inference, shared-engine crop suggestions, model-free crop operation, and explicit `ProcessingOptions.crop` after color, bokeh, or combined rendering. The crop is passed by the caller after reviewing suggestions; the render request does not rerun GAIC.
- Compared the physical-device Kotlin reference on `adb -s a4976c80`; it selected the same four boxes as Python on the actual photo and exactly matching score deltas on the shared tensor fixture. The Android emulator was not used.
- Ran M9 plus LabFusion through `process_file()` with the selected crop on a real 1080×1620 portrait: 20.117 s total, 356,876,288 B sampled process-tree peak RSS, CPUExecutionProvider, output 918×1224. The output exactly matches cropping the existing full-size M9+LabFusion result (maximum channel error 0).
- Verification after crop integration: all 159 local pytest cases pass; Ruff lint, format, and `git diff --check` pass. GAIC timing, model hashes and comparison limits are in the [real-photo report](benchmarks/gaic-real-photo-2026-10-22.md).

## 2026-10-22 — Depth Anything V2 real-photo CPU smoke

- Ran the pinned dynamic CPU ONNX adapter on the CC0 5760×3840 mountains photo. It used a 1220×813 analysis raster, a 1232×826 padded network tensor, and exported a finite float32 map at the original 5760×3840 size.
- Model inference took 13.915 s and the full engine path took 14.402 s. macOS `/usr/bin/time -l` reported 1,695,547,392 B process high-water RSS. This is one unconstrained depth-only photo run, not an accuracy comparison or a complete-pipeline 4/2 GiB result; see the [real-photo report](benchmarks/m4-depth-real-photo-2026-10-22.md).

## 2026-10-21 — repository bootstrap and M0 baseline

- Inspected the empty worktree and the empty public `luckylca/LeicaShot` repository; created the independent Laika history there without forking or configuring the reference as a remote.
- Added `main`, `develop`, the CI matrix, contribution/PR conventions, package metadata, and independent architecture/research records.
- Reviewed the reference repository at `f012dca8a845b8c37389345676e43a7e99a5ae93`; kept its sparse checkout outside this worktree. No source or weights were imported.
- Added the model-free Pillow image I/O engine, CLI, and process-tree benchmark runner on `feat/python-engine`.
- Local checks: Ruff formatting, Ruff lint, and 8 pytest cases passed on Python 3.11.16.
- GitHub Actions exposed a macOS Python 3.13 `psutil` USS syscall `RuntimeError`; the sampler now treats USS as unavailable while retaining RSS/CPU, with a regression test. Ruff and all 8 tests also pass locally on Python 3.13.16.
- Ran three 2048×1536 synthetic PNG baseline trials. Median worker RSS was 34.94 MiB, engine time 0.241 s, sampled worker CPU 0.272 s. See [the complete report](benchmarks/m0-baseline-2026-10-21.md) for limits and reproducibility details.
- No model inference, HTTP API, real-photo quality comparison, Linux cgroup test, or 8/4/2 GiB full-pipeline result exists yet.

## 2026-10-21 — M1 Depth Anything V2 Small acquisition and conversion baseline

- Added an explicit model manifest and `laika-models` fetch/verify commands. The verified official Small checkpoint remains in `/tmp`, outside Git.
- Strictly loaded the checkpoint from the upstream source pinned in the manifest and ran a CPU-only, single-thread inference using the upstream resize and normalization contract. A seeded 512×384 synthetic input produced a finite 384×512 depth map.
- Added a reproducible fixed-shape ONNX exporter. It verifies the upstream source revision and checkpoint digest, refuses existing outputs, validates with `onnx.checker`, runs ONNX Runtime CPU parity, and writes graph/data hashes with a validation report.
- The 518×686 ONNX check produced max/mean/p99 absolute errors `5.01e-6` / `8.82e-7` / `2.86e-6`. Two same-input comparison trials measured PyTorch median forward 1.185–1.226 s / peak RSS 1,045,364,736–1,082,638,336 bytes and ONNX Runtime median 2.314–3.570 s / peak RSS 563,281,920–570,982,400 bytes. This Mac result shows 45–48% lower worker RSS but 1.95–2.91× slower median inference under ONNX; the slower trial coincided with lower available host memory. It is one shape/runtime configuration, not a general conclusion.
- A separate preprocessing smoke run peaked at 1,007,566,848 bytes RSS (960.9 MiB); cold process wall time including PyTorch imports was 27.530 s. macOS USS and cgroup metrics were unavailable. See [M1 benchmark details](benchmarks/m1-depth-anything-2026-10-21.md).
- M1 remains in progress: dynamic ONNX export is validated only on synthetic tensors; there is no production depth inference adapter, real-photo quality set, Linux cgroup measurement, or full pipeline result.

## 2026-10-21 — dynamic depth export and high-resolution sample

- Added a dynamic height/width export option with shape-aware DINO positional resize and DPT output sizing. The first dynamic attempt specialized positional token length and failed at a second aspect ratio; the corrected graph passes four multiple-of-14 shapes.
- Dynamic DPT adapter output matched the pinned PyTorch model exactly on all four synthetic shapes. ONNX maximum absolute error was `1.67e-5` at most; all result shapes matched inputs and contained finite values.
- A one-warmup, one-run 1400×518 comparison measured PyTorch 1,487,093,760 bytes peak RSS / 4.125 s inference and ONNX Runtime 1,270,792,192 bytes / 6.976 s. This is 14.6% lower sampled RSS and 1.69× slower ONNX on this single M1 case.
- The dynamic graph uses opset 17 and embeds its weights. The future engine must reject dimensions below 518 or not divisible by 14. No dynamic photo quality, Linux cgroup test, query-chunked graph, or full pipeline result exists yet. See [the dynamic export report](benchmarks/m1-dynamic-depth-2026-10-21.md).

## 2026-10-21 — global query chunking for Depth Anything V2

- Added an opt-in TorchScript query loop to the dynamic exporter. Every query chunk still reads every key and value; the exported graph is rejected unless all 12 attention blocks preserve ONNX Loop nodes.
- Tested 32, 64, and 128 query tiles against unmodified PyTorch attention on seeded 518×518, 700×532, and 1400×518 inputs. PyTorch output maximum absolute difference was zero for all sizes. ONNX parity maximum absolute error was at most `1.67e-5` (limit `1e-4`).
- The corrected M1 direct-model sample at 1400×518 measured baseline 1,192,050,688 bytes RSS / 7.397 s; chunks 32 / 64 / 128 measured 457,261,056 / 435,109,888 / 487,211,008 bytes and 8.209 / 7.355 / 6.792 s. The direct graph loop check now reads the exporter sidecar rather than parsing model weights inside the measured worker.
- Two earlier Linux smoke runs passed before that measurement correction; their query-chunk RSS figures included extra graph loading and are superseded. The updated Linux workflow will rerun the fair comparison.
- See [query-chunking benchmark](benchmarks/m1-depth-query-chunking-2026-10-21.md). The exporter experiment is not yet integrated into the production image engine. No real-photo or full-pipeline quality/memory result exists.

## 2026-10-21 — M2 CPU depth inference adapter

- Added `DepthAnythingV2Small`, which checks the dynamic export sidecar and graph hash, loads ONNX Runtime only on explicit construction, uses CPUExecutionProvider, and releases the session through `close()` or context exit.
- Implemented the upstream lower-bound aspect resize, OpenCV bicubic interpolation, RGB normalization, NCHW layout, and full-source-size bilinear `align_corners=True` restoration in bounded row blocks. Image preprocessing converts one source channel at a time to avoid a full-image float32 RGB copy.
- Added the `laika depth` command. It validates actual image format, file size, dimensions, animation, and EXIF orientation; it writes full-resolution float32 `.npy` output atomically. Default source and network caps are 40M and 4M pixels.
- A 700×400 synthetic RGB comparison against the pinned PyTorch model gave preprocessing max absolute error `1.19e-6`, and final depth max/mean/p99 errors `6.15e-5` / `3.07e-6` / `5.01e-6` (limit `1e-4`).
- Three cold CLI workers on the M1 at 1400×518 measured median RSS 882,180,096 B for baseline and 453,623,808 B for query chunk 128; both returned a 1400×518 map. Worker wall medians were 8.186 s and 7.733 s, with variable trial times. macOS USS/cgroup data unavailable; these are synthetic single-stage runs, not a full pipeline result.
- Added opt-in PyTorch comparison and complete image-to-depth measurement to the model smoke workflow. Its first Linux x86_64 run found preprocessing max absolute error `1.58e-4` (above the initial `2e-6` Mac-derived tolerance), while final restored depth max error remained `5.15e-5` (below the unchanged `1e-4` output limit). The validation now uses a `2e-4` cross-platform preprocessing tolerance and records max/mean/p99; the rerun passed. On Linux, one 1400×518 CLI worker sample measured 1,114,427,392 B baseline RSS and 622,305,280 B q128 RSS, with 7.771 s and 7.892 s worker wall time. The runner's cgroup limit was `max`, so this is not a constrained-memory result. See the [M2 benchmark](benchmarks/m2-depth-inference-2026-10-21.md).

## 2026-10-22 — Web editor and GAIC HTTP workflow

- Added `POST /api/v1/compositions`, routed through the shared bounded task pool, image validation, storage admission, Low Memory/Balanced policy, and task cleanup. Added exact-origin opt-in CORS for separately hosted browser clients; wildcard and URL paths are rejected.
- Added the React + TypeScript editor for photo upload, color/bokeh modes, click focus, aperture, GAIC crop selection, progress/cancel, original/result comparison, and full PNG download. Added Vitest coverage, a Node 24 CI build/test job, API docs, and local startup guidance.
- Chrome smoke-tested a real P3M portrait from browser upload through GAIC crop suggestion and M9+Depth+ARNet processing to preview and download; output was 918×1224 PNG, 817,062 bytes. Browser smoke details and SHA-256 are in [`WEB.md`](WEB.md). Separate GAIC API measurement: 86 proposals in 1.403 s, sampled server-tree RSS 416,251,904 B.
- Local checks: 171 Python tests, Ruff lint/format, two Web tests, and Vite production build pass. Remote PR CI is still pending; this work remains on `feat/web-editor` until CI and review.

## 2026-10-21 — M3 high-resolution depth memory study

- Ran three fresh CPU CLI workers per graph on synthetic 3840×2160 RGB input. Both returned a full-size 2160×3840 float32 depth map (33,177,728 bytes); the pinned resize rule used a 518×924 network raster. A q128 run against the PyTorch reference at this full source size had max depth error `6.91e-5` (limit `1e-4`) and a finite full-size map.
- On the 8 GiB M1, median worker RSS was 762.2 MiB for the full-attention graph and 598.4 MiB for q128 (21.5% lower). Median wall times were 4.437 s and 4.399 s, too close and too few trials for a speed conclusion.
- Added these exact 4K source dimensions to the opt-in Linux model-smoke workflow. Its first Ubuntu 24.04 run also returned full-size maps and measured one worker at 731,766,784 B baseline RSS and 556,109,824 B q128 RSS; cgroup `memory.max` was `max`. The experiment is synthetic, has no memory limit, and does not establish full-pipeline or 4/2 GiB behavior. See the [M3 high-resolution report](benchmarks/m3-high-resolution-depth-2026-10-21.md).

## 2026-10-21 — M3 attention chunk-size sweep

- Exported query chunks 16, 32, 64, and 128; each retained 12 global-attention loop nodes, matched unmodified PyTorch attention, and passed four dynamic-shape ONNX checks.
- On the Linux runner, direct-model RSS was 869,384,192–878,256,128 B for chunks 16–128 versus 1,806,860,288 B baseline. Chunk 32 had the lowest direct RSS; chunk 128 had the shortest measured inference. At 4K CLI output, q128 had the lowest of the measured RSS values (510,701,568 B), while q64 had the shortest wall time (4.388 s); q16 was slowest.
- Mac and Linux rankings differ. The Linux cgroup limit remained `max`; keep the chunk configurable until memory-capped and full-pipeline results exist. See the [M3 chunk-sweep report](benchmarks/m3-depth-attention-chunk-sweep-2026-10-21.md).

## 2026-10-21 — reference depth analysis sizing

- Implemented the reference's 1220-pixel maximum analysis edge, no-upscale behavior, Pillow-compatible bicubic resize, right/bottom replicate padding to 14-pixel patch alignment, and crop-before-full-size-restore path. The upstream lower-bound resize remains available explicitly with `--input-size 518`.
- Added geometry, padding, preprocessing, CLI budget, and PyTorch/ONNX parity coverage. The optional model workflow checks both profiles, odd-edge padding, and a 4K source.
- Compared the full-attention graph and q128 graph on three fresh Mac M1 workers each using seeded 3840×2160 RGB inputs and the current reference profile (analysis 1220×686, network 1232×686, full 3840×2160 output). Median RSS was 1,179,484,160 B (1,124.8 MiB) baseline and 742,752,256 B (708.3 MiB) q128, 37.0% lower. Median wall time was 10.479 s and 9.735 s; this small sample is not a CPU speed guarantee.
- q128 output matched pinned PyTorch at the same prepared tensor with max/mean/p99 error `8.11e-6` / `2.84e-6` / `5.01e-6`; full output shape and finiteness were checked. These are synthetic, single-stage measurements on an unconstrained Mac process, not a real-photo or complete-pipeline 4/2 GiB result. See the [reference sizing report](benchmarks/m3-reference-depth-analysis-sizing-2026-10-21.md).

## 2026-10-21 — enforced depth-stage memory limits

- Added a Linux Docker benchmark with cgroup v2 `memory.max`, `memory.peak`, and OOM event capture. It ran three fresh 4K CLI workers per graph under both 4 GiB and 2 GiB caps, with swap disabled and two CPUs.
- All 12 trials completed with full 3840×2160 outputs and zero OOM kills. Median cgroup peak under 2 GiB was 1,428,115,456 B (1,362.0 MiB) for full attention and 733,523,968 B (699.5 MiB) for q128; q128 reduced peak by 48.6% but median inference was 10.9% slower.
- Both tested graphs fit this depth-only 2 GiB container. This does not validate the full Laika pipeline or a 2 GiB minimum server. Per-trial cgroup artifacts and limitations are in the [depth cgroup report](benchmarks/m3-depth-cgroup-memory-limits-2026-10-21.md).

## 2026-10-21 — attention chunk sweep under cgroup limits

- Added q16/q32/q64 to the 4K Docker cgroup sweep. All 30 trials across full attention and q16/q32/q64/q128 under 4 GiB and 2 GiB completed with full finite output and zero OOM kills.
- At 4 GiB, q32 had the lowest median cgroup peak (697,040,896 B / 664.8 MiB) while q128 had the shortest median inference (4.041 s). At 2 GiB, q128 had both the lowest median peak (703,885,312 B / 671.3 MiB) and shortest inference (4.011 s). q16 was slowest under both limits.
- Keep the chunk profile configurable: its memory ranking varied with the limit, and these are synthetic depth-only tests on one runner. See the [cgroup chunk-sweep report](benchmarks/m3-depth-attention-cgroup-sweep-2026-10-21.md).

## 2026-10-21 — SAM 2.1 Tiny provenance and embedding cache adapter

- Verified and pinned the official SAM 2.1 Tiny checkpoint metadata: 156,008,466 bytes, SHA-256 `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`, checkpoint revision `36f406a75c9be63c7f429da63246273f028c6fd4`, and Meta source revision `2b90b9f5ceec907a1c18123530e92e794ad901a4`. The model card and pinned upstream license declare Apache-2.0.
- Added an ONNX split-session inference adapter and byte-bounded LRU for the three SAM encoder feature tensors. It fingerprints the exact prepared input, reuses one image encoding across later prompts, and supports explicit invalidation and release.
- Downloaded the official checkpoint to `/tmp/laika-models` and verified its byte count and SHA-256 through `laika-models verify`; no weights entered Git history.
- Added a reproducible split-graph exporter that imports only the pinned Meta source and validates feature maps, mask logits, and scores against PyTorch. The checkpoint-backed synthetic comparison passed; maximum absolute error was `1.13e-5` for image embeddings and `2.06e-5` for mask logits. The adapter matched direct ONNX outputs exactly.
- Added a fresh-worker benchmark for RSS/USS, CPU time, encoder/decoder latency, and cache reuse. Three Docker workers with swap disabled and a 2 GiB cgroup cap all completed; median cgroup peak was 1,106,763,776 B (1,055.5 MiB), with zero OOM events. Median cold encoder time was 3.351 s and median cached prompt wall time was 0.135 s. This is a synthetic-input SAM-stage result, not a full-pipeline minimum-memory claim; details and environment are in the [SAM cgroup report](benchmarks/m1-sam2-cgroup-2026-10-21.md).
- Added mock-session coverage for cache hits, byte-budget eviction, lifecycle cleanup, prompt validation, and tensor shape checks. Added a labeled GitHub Actions model-smoke job that verifies the pinned checkpoint, exports and compares split ONNX graphs, then repeats the 2 GiB measurement.
- Added the checkpoint to the external asset manifest. Its weights remain outside Git; the reference project's converted ONNX is not copied or used as a Laika validation artifact.

## 2026-10-21 — SAM reference input preparation

- Added subject-analysis HWC RGB `uint8` / `[0,1]` `float32` preprocessing (long edge ≤2048) with bounded temporary chunks, separable triangle antialiasing, the reference's width-then-height square resize, and RGB ImageNet normalization. EXIF orientation and explicit analysis resizing remain caller responsibilities.
- Added mapping for normalized points in the oriented source image to SAM's fixed 1024 coordinate frame; preprocessing preserves the reference's coordinate scaling even when the source aspect ratio is stretched to square.
- Added scalar numerical fixtures for the resampling kernel, output shape/dtype/layout checks, byte-image normalization, analysis-size limits, and prompt boundary validation. Local Ruff and all 50 tests pass. The six Ubuntu/macOS/Windows × Python 3.11/3.13 CI jobs passed.
- Extended the pinned-model benchmark to include the 2048×1536 RGB preparation path and normalized prompt mapping. Three fresh workers under a 2 GiB cgroup limit completed with zero OOM events; median cgroup peak was 1,150,668,800 B (1,097.4 MiB), preprocessing 0.212 s, cold encoder 3.837 s, and warm-prompt wall time 0.214 s. These remain synthetic single-stage measurements; see the [updated SAM cgroup report](benchmarks/m1-sam2-cgroup-2026-10-21.md).

## 2026-10-21 — SAM subject-mask selection and restoration

- Added deterministic selection of the highest-scoring SAM candidate that covers the positive prompt. The chosen logits are bilinearly restored with half-pixel coordinates to the bounded subject-analysis raster and thresholded at zero. Input checks cover score/logit shape, dtype, finite values, point bounds, and the 2048-pixel analysis limit.
- Added scalar resampling parity, prompt-aware selection, mask coverage, failure, and input validation tests. Local Ruff and all 57 tests pass; CI passed on Ubuntu/macOS/Windows with Python 3.11/3.13.
- Extended the 2 GiB benchmark with 2048×1536 mask restoration using a separate synthetic three-candidate mask (not SAM's model output). Three fresh cgroup workers completed with zero OOM; median cgroup peak was 1,113,702,400 B (1,062.1 MiB), mask selection/restoration took 0.050 s, and preprocessing/cold encoding medians were 0.213 s / 3.903 s. The SAM graph's maximum absolute mask-logit error remained `3.72e-5`. This does not establish real-photo segmentation quality or full-pipeline memory; details are in the [SAM cgroup report](benchmarks/m1-sam2-cgroup-2026-10-21.md).

## 2026-10-21 — subject trimap generation

- Implemented the reference's ellipse-based morphology: a pixel is foreground only if all in-image neighborhood samples are foreground, background if none are, and unknown otherwise. Out-of-image samples are neutral. The default radius scales from 5 px at a 1220 long edge to 8 px at 2048.
- Row prefix sums and bounded row chunks avoid scanning each neighborhood pixel-by-pixel. Scalar fixtures cover radius 1/2/5, border behavior, uniform masks, radius scaling, no mutation, and invalid input. Local Ruff and all 63 tests pass.
- Extended the optional 2 GiB model benchmark to exercise mask selection, restoration, and trimap generation with synthetic candidate logits. Three workers passed with no OOM events; median cgroup peak was 1,126,264,832 B (1,074.1 MiB), trimap time 0.481 s, and total worker wall time 6.925 s. The SAM input remained a synthetic image, and the selected mask was a separate synthetic candidate rather than an actual SAM output. See the [updated SAM cgroup report](benchmarks/m1-sam2-cgroup-2026-10-21.md).

## 2026-10-21 — Lab detail fusion and row-streaming study

- Added an independent OpenCV/NumPy implementation of reference-style 8-bit Lab fusion. It accepts caller-provided RGB predictions; it contains no M9 weights or AI color-prediction stage. Row chunks carry 100 pixels of default Gaussian halo and preserve the output dimensions.
- Added synthetic OpenCV-oracle, row-chunk equality, input immutability, validation, and cancellation tests. Full local pytest suite: 94 passed; Ruff checks and package wheel build passed.
- Three fresh workers on a MacBook Air M1 (8 GB) at 3840×2160 found 512-row chunks at 233,750,528 B median sampled RSS / 2.548 s vs full-height 263,372,800 B / 1.897 s, with zero output error: 11.2% lower sampled RSS and 34.4% slower. The separate 2048×1536 trial did not show a consistent streaming benefit. macOS USS and cgroup data were unavailable; the benchmark used procedural arrays, not photos or a model. See the [Lab fusion report](benchmarks/m3-lab-fusion-row-streaming-2026-10-21.md).
- The Adobe Composition-1k license expressly limits trained models to non-commercial use/distribution; ViTMatte Composition-1k remains excluded from Laika's commercial pipeline and artifact manifest absent a separate written grant. This fusion module cannot replace a predictor.

## 2026-10-22 — BokehMe ARNet CPU renderer and model authorization follow-up

- The user confirmed that M9 and all other required models, including ViTMatte, have project-use authorization. Earlier model-use exclusions in this log are historical and superseded; model redistribution permissions remain a separate question. The model artifacts remain outside Git.
- Clarified the runtime boundary: Laika's service, CLI, and Python API are implemented in Python. Android is only used to execute the reference Kotlin engine on a physical phone through ADB for comparison; no Android virtual device or Android production runtime is part of Laika.
- Pinned the BokehMe ARNet checkpoint (11,101,736 bytes, SHA-256 `6c7dba6f32e3afc8e984eb5adfecfea5d13b6f4ebd132a5a890beba384be4752`) and source `neural_renderer.py` at commit `8b3ed556dc146207c9bb39fccd078a113f8d1bd8`. Added a restricted `weights_only=True` exporter, dynamic FP16 ONNX conversion, model report, and runtime hash binding.
- Synthetic FP16 ONNX checks passed at 64×80, 97×129, and 480×481. On a real 5760×3840 CC0 photo explicitly scaled to 1024×683, ONNX versus upstream PyTorch had max float error `0.000742674`, mean `0.0000915710`, 63.63 dB 8-bit PSNR, maximum 8-bit channel error 1, and no errors over 2. The local ONNX SHA-256 is `f0696c16a6eb02099292176f25b47e4dc53863e7e4f3f7748e66470d54cf0173`; the reference ONNX artifact has a different pinned SHA, so byte identity is not claimed.
- Added depth percentile normalization, stable point-focus selection, reference thin-lens CoC math, multi-scale ARNet inference, and the `laika bokeh` CLI. The CLI preserves input dimensions unless both smaller dimensions are explicitly requested; explicit RGB resizing uses OpenCV's reference-compatible bicubic kernel.
- Fresh CPU-only CLI processes on the 8 GiB M1 produced 1024×683 in 8.31 s / 658,980,864 B RSS and 2048×1365 in 27.74 s / 1,118,208,000 B RSS. Both explicitly set their output dimensions and reuse a previously generated depth map; the runs do not measure depth inference, full source-resolution output, highlights, or subject protection. Details are in the [Bokeh benchmark](benchmarks/bokeh-real-photo-2026-10-22.md).

## 2026-10-22 — unified M9, depth, and ARNet Python request

- Added `LaikaProcessingEngine` as a shared Python SDK coordinator and `laika pipeline` CLI command for color-only, bokeh-only, and color-plus-bokeh modes. It owns task-scoped arrays and temporary depth/color intermediates, enforces one request per engine instance, emits progress events, observes cancellation between model runs and during row-wise stages, and supports Low Memory session release or Balanced lazy session reuse.
- Compared the reference `LeicaEngine.kt` order and corrected the coordinator to estimate depth on the oriented original first, then run M9/Lab fusion and ARNet on the color result with that original depth. A regression assertion covers this data flow.
- Executed M9, Depth Anything V2 q64, and ARNet CPU sessions through the same SDK/CLI request on one 5760×3840 CC0 photograph. Explicit 2048×1365 result: 54.38 s wall / 1,094,172,672 B high-water RSS. Original-size result: 605.43 s / 2,197,651,456 B RSS. ARNet's first original-size pass alone took 514.45 s. Host CPU load was shared; the results are observed and not isolated throughput claims. Full model hashes and commands are in the [combined pipeline report](benchmarks/unified-pipeline-real-photo-2026-10-22.md).
- The default-size output stayed 5760×3840 and was visually inspected. Physical Android Kotlin numerical comparisons cover M9/Lab at full resolution and the bokeh chain at 2048. Highlight scatter, SAM/ViTMatte restoration, and GAIC are not yet connected to this Python request, so the ARNet plate is not described as the complete reference finish.
- Built and ran the external reference Kotlin app on physical ADB serial `a4976c80`. Full-size M9/Lab parity against Kotlin measured 53.71 dB PSNR, 0.215 mean absolute channel error, and 97.42% channel values within 1. At 2048×1365, the Python/Kotlin bokeh-image comparison measured 41.82 dB PSNR; the Kotlin result includes highlight scatter (82.4 s reported on the device) absent from this Python renderer. The Android source checkout received temporary local model/hash/input-name adaptations only; none were imported into Laika.
- Local Python suite: 127 tests passed. Ruff formatting/lint and wheel build passed. Linux cgroup validation and GitHub CI remain before this PR can merge.

## 2026-10-22 — optical highlight scatter integrated with ARNet

- Implemented reference-style highlight extraction, clipped-light recovery, nine-blade aperture sampling, 5×5 field-dependent optical scatter, defocused highlight softening, and final radiance compositing in `laika.highlights`; connected stage progress and cancellation checks to `render_bokeh` and the unified engine.
- Executed one real CPU-only M9 + Depth Anything V2 + ARNet `color+bokeh` SDK request from a 5760×3840 source to the explicitly requested 2048×1365 output. The task took 197.846 s and sampled 1,110,802,432 B process-tree peak RSS. The optical scatter stage took 140.888 s. The output is `/tmp/laika-pipeline-results/python-full-highlight-2048.png`; model and output hashes are in the [highlight benchmark](benchmarks/bokeh-highlight-scatter-2026-10-22.md).
- Swept bokeh row chunks 128 and 512 on the real photo. Chunk 512 reduced scatter time from 163.31 s to 137.63 s (15.7%) while sampled renderer RSS rose by 13.6 MB. This is a host-specific measurement; the current default remains 128.
- Compared intermediate buffers to Kotlin execution on physical phone serial `a4976c80`: base plate exact on the real crop; linear split maximum absolute error `1.19e-7`; field optical active-pixel MAE `9.48e-9` and max `8.94e-8`; clipped recovery max `2.86e-6`. Full real-photo Python/Kotlin output comparison measured 41.378 dB PSNR, but is not an isolated highlight-equivalence metric.
- Full 139-test suite passed; Ruff lint and formatting passed. No full-resolution highlight run or Linux cgroup result exists. The next feature after this branch is SAM → trimap → ViTMatte subject protection; optical scatter CPU cost remains a separate performance task.

## 2026-10-22 — GAIC composition scoring and crop application

- After PR #24 merged, created `feat/gaic-composition-integration`. Pinned the authorized GAIC MobileNetV2 `reddim16` checkpoint and upstream MIT implementation revision; exported a dynamic CPU FP32 ONNX graph with a SHA-gated validation report. PyTorch/ONNX parity passed on three shapes with maximum absolute score error ≤`7.16e-7`; model weights stayed outside Git.
- Added original-source-pixel candidate snapping, ≤1220-pixel analysis preprocessing, GAIC suggestions and model-free crop application through `LaikaProcessingEngine`; added `laika composition` and `laika crop`. The crop path preserves ICC, does not upscale, and loads no AI model. Balanced session reuse, Low Memory release, progress, and active GAIC RunOptions cancellation are covered.
- Across three fresh workers, the 1080×1620 portrait task scored 556 candidates in 1.016 s median wall with 353,615,872 B median sampled process-tree RSS (343,523,328–362,512,384 B range). The selected full-resolution 918×1224 crop exported in 0.176 s. No cgroup was visible.
- Compared the same 813×1220 RGB tensor and exported graph on the physical Android phone `a4976c80` through the reference Kotlin `GaicComposition`; top-four crop boxes and score improvements matched exactly. A second comparison ran both platforms' full photo-loading paths: analysis raster MAE was 0.313 and RMSE 0.583, with identical top-four crop boxes and maximum score-improvement difference 0.0131. The Python resize uses OpenCV `INTER_LINEAR_EXACT`, the closest tested resize kernel to the phone output. An isolated test app package left the installed reference app untouched; no emulator was used.
- Added export provenance, real-photo benchmark, model documentation, and SDK/CLI/runtime tests. Broader scene quality, composition in `ProcessingOptions`, the HTTP API, and Web integration remain open.
