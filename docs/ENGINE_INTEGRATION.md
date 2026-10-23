# Engine integration checkpoint

Updated: 2026-10-23

## Intended Python call

```python
from laika import LaikaProcessingEngine, ProcessingOptions

with LaikaProcessingEngine(
    m9_model_path="/local/models/leica-m9-v3-epoch11.fp16.onnx",
    depth_model_path="/local/models/depth-v2-small-dynamic-q64.onnx",
    bokeh_model_path="/local/models/legendary-1.0-arnet.fp16.onnx",
    sam_model_dir="/local/models/sam2-onnx",
    vitmatte_model_path="/local/models/vitmatte-onnx/vitmatte-small-composition-1k.fp32.onnx",
    composition_model_path="/local/models/gaic/gaic-mobilenetv2-reddim16.fp32.onnx",
    strategy="low-memory",
) as engine:
    result = engine.process_file(
        "input.jpg",
        "output.png",
        options=ProcessingOptions(
            mode="color+bokeh",
            focus_x=0.5,
            focus_y=0.5,
            aperture=2.8,
            subject_points=((0.52, 0.30), (0.55, 0.80), (0.80, 0.65)),
            matte_tile_size=512,
        ),
        progress_callback=on_progress,
        cancel_event=cancel_event,
    )
```

The CLI equivalent is `laika pipeline input.jpg output.png --mode color+bokeh ...`. `color` and `bokeh` modes load only the models they use. `output_size=(width, height)` is optional and explicit; otherwise the EXIF-oriented source dimensions are preserved.

## Lifecycle and data ownership

- Low Memory: Depth Anything reads the EXIF-oriented original first, writes its full-size raw depth map to a task-owned temporary `.npy` for color+bokeh requests, then closes the Depth ONNX session. M9 next reads the original and writes a temporary full-size color result, then closes the M9 session. That M9 image is decoded once and shared by SAM, ViTMatte, and ARNet; the saved original-image depth is normalized and resized only to the explicitly requested work size. SAM's encoder runs once for a photo and its byte-bounded feature cache serves later point prompts; SAM closes before ViTMatte loads. ViTMatte processes the refined mask in configured tiles and closes before ARNet. ARNet closes before output export.
- Bokeh-only requests keep the one decoded source RGB array across depth inference and ARNet, while closing the Depth session before ARNet is loaded. Color-only requests do not construct or load depth/ARNet.
- Balanced: M9, depth, ARNet, SAM, ViTMatte, and GAIC sessions are created lazily and reused across serialized calls. `engine.close()` releases all retained sessions.
- Calls on one engine instance are serialized under a lock. A server must still apply a global bounded task queue across engine instances.
- Progress callbacks run at stage boundaries. Cancellation is checked between stages, during M9 row fusion, during ARNet compositing, and during GAIC inference through ONNX Runtime `RunOptions`. Other ONNX runs are not preempted mid-inference.
- PNG output is written to a temporary file beside the destination and atomically replaced only after successful processing.

## Highlight stage integrated on `feat/bokeh-highlight-scatter`

- `render_bokeh` now splits the M9/Lab RGB frame into an 8-bit base and float32 linear highlight surplus. ARNet renders the base plate; reference-style depth-layered highlight scatter is then recovered, softened and radiance-composited in linear light.
- Highlight extraction records compact and diffuse bright regions separately. Clipped-light recovery is restricted to clipped, defocused sources and uses a 4-pixel coarse density table. The optical kernel uses the reference 8×8 coverage sampling and nine-blade pupil, with spatially varying pupil shape interpolated from the 5×5 field grid.
- Highlight stage progress and cancellation checks are wired through the same SDK callback. Buffers are owned by one `render_bokeh` call and released before export; no model or image asset was added to Git.
- In `ProcessingOptions`, `bokeh_row_chunk_height=512` gave a useful CPU/memory tradeoff on the measured 2048×1365 photo. The current default remains 128 to preserve the lower-working-set baseline; deployments can explicitly select another value. Neither setting is asserted to be optimal on other hosts.

## Exact-zero highlight scatter optimization on `perf/highlight-scatter-cpu`

- Merged to `develop` in PR #32 as `a503d0b`; all seven cross-platform Python and Web CI checks passed.
- Field-optical scatter still filters every single-channel CoC-layer mass for depth occlusion. It skips only float64 RGB buffer creation and RGB convolution when the source highlight tile or layer-weighted radiance is exactly zero.
- Real M9/Depth arrays at 1024×683 and 2048×1365 matched the pinned `develop` scatter arrays bit-for-bit. Reversed-order 2048 pairs measured 207.954/112.984 s and 227.922/116.283 s baseline/optimized, with about 46–49% lower scatter time; sampled peak RSS fell from 411–448 MB to 409–413 MB.
- A complete Low Memory `process_file()` request ran M9 + LabFusion, Depth, tiled ARNet and highlights on the 5760×3840 source with an explicit 2048×1365 output. It took 328.207 s, sampled 1,083,146,240 B process-tree peak RSS and exported the same PNG bytes as the pre-optimization engine result. Concurrent Linux cgroup CPU load makes the wall time non-comparable to older runs.
- The pre-optimization 5760×3840 Linux cgroup task completed in 5,658.667 s. Process-tree sampled peak RSS was 2,502,225,920 B; cgroup peak was 2,676,203,520 B under the 3 GiB limit, with all OOM/limit events zero. It exported the requested 5760×3840 PNG (SHA-256 `62bf1f9b29a93e15435d5f38eb9975d5e910ca773fc5eb4e6de43b16a8ca1aef`).
- The paired PR #32 run used the same photo, models, native output size, ARM64 guest, 3 GiB cgroup, 2 CPU quota, one inference thread, Low Memory mode, and 512-row chunks. It completed in 2,112.493 s; scatter fell from 5,323.159 s to 1,739.518 s, process-tree peak RSS was 2,369,044,480 B, and cgroup peak was 2,427,625,472 B, with zero OOM/limit events. The PNG hash and decoded pixels exactly matched the baseline. This is one paired scene and does not establish other images, devices, subject protection, or concurrency. See the [full-resolution cgroup report](benchmarks/full-resolution-highlight-cgroup-2026-10-22.md).

## Subject protection integrated on `feat/subject-protection-integration`

- Set `subject_points` to one or more normalized positive points on the selected subject. Multiple points support disconnected body regions and are deduplicated; each point yields a SAM mask, ViTMatte alpha, and the masks are max-unioned. The current limit is 16 points. Select `matte_tile_size` from 512, 768, or 1024; 512 was used for the measured full request.
- Subject protection is optional and requires a Bokeh mode plus local validated SAM and ViTMatte model paths. It computes alpha at an analysis long edge up to 2048, then restores alpha to the requested output dimensions without reducing the exported photo.
- The subject alpha reduces CoC before ARNet and highlight scatter. A pull-push fill creates a clean background plate under the subject, then the sharp M9/Lab image is composited over the rendered background with row-chunked edge decontamination.
- The P3M-10K 1080×1620 three-point CPU run measured 788,070,400 B process-tree RSS and 112.73 s. Its three points produced mask IoU 0.9879; SAM encoder cache recorded two hits. On the same alpha/source/background arrays, the Python final composite matched the Kotlin phone reference within MAE 0.0368 and RMSE 0.5827 (98.84% of RGB pixels exactly equal). See [subject-protection benchmark](benchmarks/subject-protection-integration-2026-10-22.md).
- PR #28 merged positive point placement/removal and ViTMatte tile-size selection into the Web editor. A real five-model `/api/v1/tasks` request with three points and 512-pixel tiles completed in 102.177 s at 1,029,177,344 B sampled service process-tree RSS. It recorded two SAM embedding-cache hits, retained 1080×1620 output, and matched the earlier SDK output SHA exactly; see the [Web subject API report](benchmarks/web-subject-protection-2026-10-22.md).

## GAIC composition integrated on `feat/gaic-composition-integration`

- `suggest_composition()` runs the validated GAIC ONNX graph through the shared Python engine, applies EXIF orientation and configured file/pixel limits, returns ranked model suggestions, emits progress, and honors cancellation during ONNX scoring.
- The GAIC session follows the engine strategy: Low Memory closes it after one request; Balanced loads lazily and reuses it. `close()` releases retained GAIC state.
- `ProcessingOptions.crop` applies a selected normalized crop after any requested color, bokeh, or combined render. The crop is explicit, so GAIC can run as a preceding suggestion call and does not run again during rendering.
- `apply_crop_file()` remains a model-free path for an existing image, preserves ICC metadata, and avoids rerunning M9, depth, ARNet, SAM, or ViTMatte when the caller already has a rendered result.
- `laika composition`, `laika crop`, and `laika pipeline --crop LEFT,TOP,RIGHT,BOTTOM` call these same SDK operations. Crops use the EXIF-oriented pixel rectangle and do not upscale.
- Across three fresh Low Memory workers on a 1080×1620 P3M-10K image, GAIC scored 556 proposals in a median 1.016 s with 353,615,872 B median process-tree RSS (observed range 343,523,328–362,512,384 B). The physical-phone Kotlin run on `a4976c80` used source dimensions for candidate snapping. Same-tensor score values matched exactly; on the actual phone/Python photo decode paths, the top-four crop boxes matched and score improvements differed by at most 0.0131. See [GAIC real-photo benchmark](benchmarks/gaic-real-photo-2026-10-22.md).
- GAIC suggestions and crop application are available through the shared engine and CLI. FastAPI accepts explicit selected crops for tasks and exposes `/api/v1/compositions`; the React editor can request suggestions and select a crop before processing.

## FastAPI task service merged in PR #26

- `POST /api/v1/tasks` streams uploads into UUID-owned temporary folders, verifies the actual decoded JPEG/PNG/WebP format, rejects animation and enforces byte/pixel limits before queueing the shared `LaikaProcessingEngine`.
- The bounded worker queue supports Low Memory per-task engines and Balanced thread-local engine reuse. Task tokens authorize status, cancellation, source, preview, and download routes; only token digests are stored. Failed and cancelled files are removed, completed tasks expire by TTL, and owned directories are cleaned at service shutdown.
- `/api/v1/tasks/{id}` exposes task stage and progress. Preview and full PNG download routes are available. An explicitly selected normalized crop is passed into the shared engine and does not invoke GAIC again.
- Local end-to-end API benchmark uploaded one 1080×1620 P3M portrait and ran M9, Depth, ARNet and crop in the same task. It returned a 918×1224 PNG, sampled 698,236,928 B server process-tree peak RSS, and took 90.604 s task wall time on an 8 GiB MacBook Air M1 (CPU-only, one thread). See [API task benchmark](benchmarks/api-task-real-photo-2026-10-22.md). This was not cgroup-limited and had no full-frame Kotlin quality comparison.
- The API is a thin adapter and does not change SDK/CLI engine behavior. Its queue/status live in process memory; current deployment is a single Uvicorn process. It binds to loopback by default and uses per-task capability tokens, not user accounts.
- `/api/v1/compositions` accepts a validated upload and returns real GAIC crop proposals. It uses the same bounded worker pool, storage admission, and model lifecycle strategy as render tasks. The 1080×1620 CPU test scored 86 candidates in 1.403 s at 416,251,904 B sampled process-tree peak RSS; see [composition API benchmark](benchmarks/api-composition-real-photo-2026-10-22.md).
- Task options carry normalized subject points and ViTMatte tile size directly into the same engine call. The five-model API run with three prompts returned a 1080×1620 result in 102.177 s at 1,029,177,344 B sampled RSS; SAM ran its encoder once and served two prompts from its embedding cache.
- Linux cgroup v2 now records current, peak, limit and OOM events beside sampled process-tree RSS. A real five-model 1080×1620 task completed in 4 GiB and 2 GiB configured guests. The 2 GiB guest / 1.5 GiB task-cgroup run took 1,196.67 s, peaked at 1,564,434,432 B cgroup memory and 1,545,293,824 B process-tree RSS, and had no OOM events; it left only about 8 MiB of cgroup headroom. This is a single-scene challenge result, not a production memory guarantee. Full configuration, output parity, stage times and cross-platform image error are in the [Linux cgroup benchmark](benchmarks/linux-cgroup-full-pipeline-2026-10-22.md). PR #28's Web editor flow passed a Chrome smoke: upload → GAIC suggestion → combined render with three SAM points → preview → full-size PNG download; `docs/WEB.md` records output hash and dimensions. Cross-origin access is opt-in through exact `LAIKA_API_CORS_ORIGINS` origins.

## Local Docker deployment merged in PR #33

- `Dockerfile` packages the Python API as a non-root user; `.dockerignore` keeps model assets, tests, caches, and reports out of the build context. `web/Dockerfile` builds the existing React editor and serves it from the unprivileged Nginx image.
- `docker-compose.yml` mounts authorized local models read-only, stores tasks in a named volume, uses loopback-only API/Web ports, and applies bounded memory, CPU, task concurrency, temp storage, logs, process count, and Linux capabilities. The web container proxies `/api/v1/` to the API on the private Compose network.
- `docs/DEPLOYMENT.md` and `.env.example` document model placement, resource settings, result retention, network exposure, and local startup. The new CI container job builds both images, starts without model assets, and checks API, Web and same-origin proxy health.
- Local ARM64 API and Web image builds succeeded. Compose validation passed; direct API health, Web health, proxied API health, and the static React entry page all returned successfully. Both containers run non-root with read-only root filesystems; observed idle RSS was 44.14 MiB API and 3.36 MiB Web under 3 GiB / 256 MiB limits.
- A model-enabled HTTP task used the read-only mounted M9, Depth, and ARNet assets on the same 5760×3840 real photo and returned a 1024×683 PNG in 48.808 s. API cgroup peak was 1,402,888,192 B of 3 GiB, with zero OOM/limit events. M9, Depth, ARNet and highlight processing ran through one task; stage timings and hashes are in the [Docker API benchmark](benchmarks/docker-api-real-photo-2026-10-22.md). The downloaded PNG matched direct Python SDK/CLI output in the same Linux image byte-for-byte. The macOS SDK comparison had PSNR 54.97 dB and mean absolute channel error 0.1912.
- Current-source tile telemetry completed a native 5760×3840 CPU M9 + Depth + ARNet + highlight request in a 3 GiB cgroup with 128-row highlight chunks. Cgroup peak was 2,696,331,264 B, sampled process-tree peak was 2,357,911,552 B, wall time was 4,789.789 s, and OOM/limit events were zero. The output PNG SHA-256 matched the earlier 512-row full-resolution result exactly. It emitted 120 ARNet tile start/completion pairs and monotonic whole-task progress. Scatter took 4,473.569 s, versus 1,739.518 s in the older 512-row run; because the runs used different source snapshots, this motivates a same-source control rather than proving chunk size caused the difference. See [tile telemetry report](benchmarks/arnet-tile-telemetry-2026-10-23.md).

## Current integration limits

- M9 low-memory mode decodes its source for model preprocessing and again for exact Lab detail fusion. The combined path then decodes the intermediate M9 image once for shared depth/bokeh input. This preserves existing M9 behavior but is not yet the minimum-decode design.
- Depth inference returns a full-size float32 depth map. Normalization currently sorts a copy to reproduce percentile semantics; future profiling should replace this only if an exact memory-bounded percentile implementation is demonstrated.
- ARNet's first full-scale pass is the dominant Bokeh inference time. Its pinned network uses space-to-depth followed by eight local 3×3 convolutions. Low Memory now tiles even images above 4 MP into 512×512 cores with a 16-pixel halo; Balanced and small or odd-sized inputs retain the full-frame path. The FP32 output matched bitwise against one full-frame inference on a real 1024×682 M9/depth input. Both pre-optimization and PR #32 optimized 5760×3840 Linux cgroup runs completed all ARNet scales and highlights; the optimized result was byte-identical and its scatter stage took 1,739.518 s. No output dimensions are reduced by tiling.
- Composition suggestions remain an explicit model call, while an already selected crop is accepted through `ProcessingOptions.crop` and does not rerun GAIC. Full 5760×3840 requests with highlight scatter and subject protection have not yet been measured.
- Field-dependent optical scatter remains the largest CPU stage. Mirror-kernel reuse reduces kernel builds from 25 to 13 per radius. The exact-zero RGB convolution skip matched real scatter arrays and full 2048×1365 output byte-for-byte, cutting standalone 2048 scatter by 45.7–49.0% in two paired orderings. In a full-size cgroup pair it cut scatter 67.3% and total wall 62.7%, reduced cgroup peak 9.3%, and produced bitwise-identical 5760×3840 output. These single-scene results do not establish a throughput guarantee. See the [kernel reuse report](benchmarks/optical-scatter-kernel-reuse-2026-10-22.md), [sparse-radiance report](benchmarks/highlight-scatter-empty-radiance-2026-10-22.md), and [full-resolution report](benchmarks/full-resolution-highlight-cgroup-2026-10-22.md).

## Real-photo verification

The same 5760×3840 CC0 source was processed in one `process_file()` call and one CLI invocation. The validated order is original-photo depth, M9 plus Lab fusion, then ARNet using the M9 image and that original-photo depth map. No output size was passed in the full-resolution run. On a MacBook Air M1 (8 GiB, macOS 15.7.7), CPUExecutionProvider, one thread, Low Memory: 2048×1365 took 54.38 s wall with 1,094,172,672 B high-water RSS; 5760×3840 took 605.43 s with 2,197,651,456 B RSS. ARNet pass 1 alone took 514.45 s at full resolution. Host CPU load was shared with another active Python task; these are observed run times rather than isolated speed claims. Details, hashes, and commands are in [the benchmark report](benchmarks/unified-pipeline-real-photo-2026-10-22.md).

## Verification checklist for this branch

- [x] Synthetic tests cover mode selection, model laziness, Low Memory release, Balanced reuse, progress, cancellation cleanup, explicit size, and output dimensions.
- [x] Run pinned M9, Depth, and ARNet models in one `process_file()` call on the same real source photograph; Depth consumes the original and ARNet consumes the M9 result.
- [x] Record explicit 2048×1365 and original-size 5760×3840 output timing/RSS after correcting depth/color order.
- [x] Compare fused color and bokeh output to reference Kotlin results using the physical device through `adb -s a4976c80`; the numerical results and local asset/name adaptations are recorded in the pipeline report. The Android emulator was not used.
- [x] Python full suite (141 tests), Ruff lint, and formatting checks pass locally on the subject-protection branch.
- [x] A CPU-only `color+bokeh` request ran all three pinned ONNX models on the same 5760×3840 source, with explicit output 2048×1365.
- [x] Process-tree RSS and per-stage timings recorded for the complete task and standalone 128/512-row bokeh runs.
- [x] Intermediate highlight split, recovery, field scatter and softening were compared numerically against Kotlin fixtures executed on the physical ADB device; detailed limits and metrics are in [the highlight report](benchmarks/bokeh-highlight-scatter-2026-10-22.md).
- [x] Run M9, Depth, ARNet, SAM and ViTMatte together through one Python CLI request on a real 1080×1620 photo; record process-tree RSS, per-stage timing, three-point mask/alpha metrics, and CPU provider/model hashes.
- [x] Compare final subject composite against the reference Kotlin implementation run on the physical phone with an identical input fixture; no emulator was used.
- [x] Export the pinned GAIC checkpoint and verify PyTorch/ONNX score parity on three dynamic tensor shapes.
- [x] Route composition suggestions and model-free crop application through the shared Python engine and CLI.
- [x] Run 556-candidate GAIC scoring on a real photo; compare same-tensor results and both image-preprocessing paths with Kotlin on the physical ADB phone, with no emulator use.
- [x] Apply an explicitly selected GAIC crop after any processing mode through `ProcessingOptions`; suggestions remain an explicit separate model call.
- [x] Add a bounded GAIC suggestion route at `/api/v1/compositions` and connect selected crop support to render tasks.
- [x] Complete Web upload, mode/focus/aperture editing, progress/cancel, GAIC crop selection, preview comparison, and full PNG download in a browser smoke; Vitest and production build pass locally.
- [x] Validate additional curly-hair, long-hair and backlit wispy-hair photos; compare ViTMatte 512/768/1024 tiles and record soft-alpha and boundary metrics.
- [ ] Six-platform GitHub Actions matrix passes before merge.
- [x] Full-resolution M9 + Depth + ARNet + highlight run completed at 5760×3840 in a 3 GiB Linux cgroup with exact baseline pixel parity.
- [x] Run the five-model path at 1080×2123 in a 3 GiB Linux task cgroup and record process RSS, cgroup peak, stage times, and output hash.
- [x] Complete a source-verified native 5760×3840 subject task under a measured cgroup limit. The current-source M9 + Depth + SAM + ViTMatte + ARNet + highlights request completed under 3 GiB with a 2,703,536,128 B cgroup peak, zero OOM/limit events and unchanged output dimensions; see the [2026-10-23 subject cgroup report](benchmarks/full-resolution-subject-protection-cgroup-2026-10-23.md).
- [ ] Add more native-resolution subject quality cases for hair, fine objects and background gaps, then measure API concurrency and repeat on an 8 GiB host. The completed one-point mountain scene verifies end-to-end subject integration and a local detail change, not broad mask quality or service concurrency.

## Full-resolution subject-memory branch (`perf/fullres-subject-protection`)

- Tiled ARNet can write each core directly into the renderer's HWC float32 destination. Full-frame/odd/small input routing remains unchanged. A real `render_bokeh` comparison on the 1080×2123 P3M fixture produced the same RGB SHA-256 `3b9c87f6c337bbd29162eaf9abf76b42443214c9b28d8f86e6b04d53ae0dff79` with zero differing channel values; the direct-output prototype was slower (22.464 s vs 20.823 s), so production routing is limited to frames already using tiled ARNet.
- Subject CoC masking now computes the complement in bounded row chunks. Unit fixtures compare exactly with the former whole-array expression and verify the alpha input is unchanged.
- The direct sink removes construction of one 265,420,800-byte NCHW tensor at 5760×3840; this is structural allocation accounting, not an isolated peak reduction. The current-source full-resolution subject path completed with a 2,703,536,128 B cgroup peak in a 3 GiB cgroup. Historical source-loading errors and partial runs remain documented in the [subject investigation](benchmarks/full-resolution-subject-protection-2026-10-22.md); full measured results are in the [cgroup report](benchmarks/full-resolution-subject-protection-cgroup-2026-10-23.md).
- The pipeline benchmark JSON now records focus, aperture, subject points, explicit output size, and ViTMatte tile size so runs can be reproduced from their result records.

## FastAPI branch verification

- [x] Upload, decoded-format/pixel/byte validation, bounded scheduling, progress, cancellation, token isolation, preview/download, and cleanup tests pass with the shared-engine interface.
- [x] A single uploaded real photograph completed M9 + LabFusion + Depth + ARNet + explicit crop and returned a downloadable PNG through the HTTP API; measured results are in the [API report](benchmarks/api-task-real-photo-2026-10-22.md).
- [x] Active upload reservations account for remaining configured workspace budget, with a regression test for concurrent reservation.
- [x] Local suite: 171 Python tests; Ruff lint and format checks pass.
- [x] GAIC suggestions run through the HTTP route with local authorized model; route timing/RSS and model hash are documented.
- [x] Web editor: upload/GAIC/render/preview/download browser flow passed on `feat/web-editor`; remote Node 24 and six-platform Python CI passed in PR #27.
- [x] Subject controls serialize multi-point prompts and selected tile size, and cap selection at 16 points; four Vitest tests and Vite production build pass locally. Live Chrome completed the full upload → subject prompts → five-model task → preview → PNG download flow. The direct API benchmark records RSS/timing and exact SDK output parity.
- [x] Six-platform remote Python CI and Node 24 Web CI for `feat/web-subject-protection` passed in PR #28.
- [x] Linux cgroup v2 full-pipeline measurement for one real portrait at the documented guest/task limits; broader subject-photo quality, original-resolution tasks, service concurrency, and cross-platform edge parity remain open.
- [x] Mirror-kernel reuse passes bitwise kernel tests, real scatter-array hash comparison, full real-photo renderer pixel parity, and a 4 GiB guest/3 GiB task-cgroup run at explicitly requested 2048×1365 output. The same 5760×3840 source-size task is OOM-killed at this budget; source-resolution memory optimization and repeated controlled timing remain open.

## Per-tile cgroup telemetry (PR #36 merged 2026-10-23)

- The merged PR #35 direct-HWC ARNet sink and row-bounded subject CoC mask have exact tensor/output parity evidence, but the prior full-resolution subject cgroup attempts did not load the mounted source tree. Their OOM results do not verify the claimed HWC/tile-size/glibc variants. See the source-loading audit addendum in the [subject memory report](benchmarks/full-resolution-subject-protection-2026-10-22.md).
- PR #36 (`79148e1`) emits start/complete records for every full-context ARNet tile, carries phase/pass/tile bounds in engine progress, checks cancellation at tile boundaries, and writes per-tile process RSS/cgroup snapshots to the JSONL benchmark record. Unit tests cover complete tile coverage, monotonic progress, and cancellation cleanup.
- Verified local-only M9, Depth, and ARNet artifacts recovered from the physical phone's reference APK have the same SHA-256 values as the pinned project assets. ONNX Runtime CPU inference passed for all three; the Depth graph accepted two dynamic shapes. A source-loaded 5760×3840 no-subject task completed under a 3 GiB cgroup and emitted 120 ARNet tile start/completion pairs. It does not cover SAM or ViTMatte.
- The current-source same-photo 512-row highlight-chunk task completed in 6,000.718 s with 2,597,527,552 B cgroup peak and zero OOM/limit events. Its 5760×3840 PNG hash matches the 128-row run exactly. Scatter was 4,779.098 s at 512 rows and 4,473.569 s at 128 rows; other stage times varied by several-fold between single trials, so neither setting is demonstrated as a stable winner. Keep the 128-row default. The [comparison report](benchmarks/highlight-chunk-size-cgroup-2026-10-23.md) records all stages and limitations. A source-verified native-size subject run has since completed; its result is in the [subject cgroup report](benchmarks/full-resolution-subject-protection-cgroup-2026-10-23.md).

## Source-verified native-resolution subject request (completed)

- Active source branch: `perf/fullres-subject-protection-cgroup`, based on `develop` merge `47712b4` (PR #37). The working tree's Python source is mounted read-only into the test container at `/opt/laika-src`; `PYTHONPATH=/opt/laika-src/src` ensures the current checkout is imported.
- Input: local CC0 mountains photograph, 5760×3840, SHA-256 `4e6661606c195b2462c0f2321a7feefe2ae2fc081499f3462a30fbb108cd7f8e`. The one positive prompt `(0.72, 0.66)` targets the foreground clock tower. This is an integration/performance scene without a pixel-level subject ground truth; it does not independently establish edge quality.
- M9, Depth Anything V2 q64, and ARNet hashes match the previous full-resolution baseline. The SAM checkpoint hash is `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`; split encoder/decoder graph hashes are `37de035410f11fb505c210ce1f8b10b450cb018fbfa7f226ec7e7ab542955c75` and `3a6a96c62fd39c1369641ea626e5b2e4aed10de877f85291e9993a044d5bf981`. The ViTMatte checkpoint hash is `bda9289db1bb6762d978b42d1c62ae3f34daf7497171a347a1d09657efd788cb`; graph hash is `a5c0b49a664867021eaf2b9cadcdc236ea511028d5e709303b418431f20e223b`. Graphs were regenerated and verified against the pinned source/checkpoint; they are in an external local cache, not Git.
- Runtime settings: Lima Linux ARM64 guest, 4 GiB guest memory, task cgroup v2 limit 3,221,225,472 B, two-CPU quota, CPUExecutionProvider, one ONNX intra-op thread, Low Memory strategy, 512-pixel ViTMatte tiles, 128-row highlight chunks, native output size. Temporary benchmark outputs remain outside the repository.
- Completed result: whole-task time 2,955.242 s; sampled process-tree peak RSS 2,558,504,960 B; cgroup peak 2,703,536,128 B of 3 GiB; zero OOM/limit events. Output was a 5760×3840 RGB PNG with SHA-256 `0a6282df408319950f0bd4c8f56bcfce78ed9e7eba95ce9c50b2c70f5555ade2`. The selected clock dial changed locally against the no-subject output. This one scene is not a ground-truth fine-edge acceptance; see the [full report](benchmarks/full-resolution-subject-protection-cgroup-2026-10-23.md).
- Conversion resource note: the SAM exporter was OOM-killed under 3.125 GiB and 3.5 GiB Linux container limits. It completed on the Mac CPU with pinned Torch 2.14.1, TorchVision 0.29.1, ONNX 1.23.2 and ONNX Runtime 1.31.0 in 27.14 s; `/usr/bin/time -l` reported 2,151,366,656 B maximum RSS and 3,212,501,760 B peak memory footprint, with zero swaps. This measures checkpoint conversion, not SAM inference or the production task.

## Optical scatter CoC-weight reuse (tested and rejected)

- Historical experiment branch `perf/native-resolution-scatter-profile`; candidate commit `cd4e6f9` was reverted by `96105d6` and the report merged in PR #39. The candidate computed `source_coc` and `layer_weight` once across the image width per vertical field node, then sliced each horizontal node's source. Field-node accumulation order, kernel, convolution, output dtype and math were unchanged.
- On one real M9/depth photo, the current scatter array was byte-identical to the pre-change implementation at 1024×683 and 2048×1365. Two interleaved 2048 comparisons averaged 96.831 s before and 80.829 s after; mean sampled RSS was 323,198,976 B before and 325,632,000 B after. Timing is from the MacBook Air M1, CPU-only, row chunk 128, aperture 2.8. One 1024 cProfile run found 2,392 `filter2D` calls taking 8.166 s and `_scatter_field_optical` taking 19.935 s total; the candidate addresses the repeated CoC/layer-weight preparation, not the convolution count.
- Same-host parent (`8360f91`) and candidate (`cd4e6f9`) 5760×3840 tasks produced exactly identical RGB outputs (22,118,400/22,118,400 pixels). Candidate wall time was 3,257.537 s vs 3,169.499 s (+2.78%); scatter 2,936.921 s vs 2,858.616 s (+2.74%); total process-tree peak 1,458,585,600 B vs 1,362,968,576 B (+7.01%). Although sampled scatter-stage peak RSS was 5.84% lower, the whole task used more peak RSS and took longer. The candidate was rejected and reverted; see the [benchmark report](benchmarks/optical-scatter-depth-weight-reuse-2026-10-23.md). The 1024/2048 scaled benchmark appeared faster, showing that it did not predict native-size behavior.
- Two Linux candidate runs ended in ARNet before reaching the changed scatter function (3 GiB cgroup memcg OOM; 3.25 GiB cap/global OOM in the 4 GiB guest), so they neither validate nor attribute those failures to the change. The candidate was reverted in PR #39 based on the same-host full-task result; no Linux retest is planned for this rejected implementation.

## API worker concurrency benchmark (2026-10-23)

- `scripts/benchmark_api_task.py --task-count N` now sends N independent uploads to one Uvicorn service process and records each task result, stage times and output hash, alongside service process-tree RSS/USS and cgroup v2 peak. The benchmark rejects counts above the configured worker-plus-queue capacity. `Dockerfile.benchmark` adds the HTTP client dependency to the locally built API image without changing production dependencies.
- One Linux ARM64 1620×1080 five-model request (M9, Depth, SAM, ViTMatte, ARNet/highlights) completed in 81.520 s at 1,867,460,608 B cgroup peak under a 3 GiB limit. Two simultaneous workers completed in 109.291 s at 3,129,069,568 B (97.14% of the limit), with zero OOM events and identical output hashes. This small-photo two-worker run leaves too little headroom for a 3 GiB service tier; retain the default single worker. Full environment, stage results and reproduction notes are in the [benchmark report](benchmarks/api-concurrency-memory-2026-10-23.md).
