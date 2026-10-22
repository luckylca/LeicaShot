# Engine integration checkpoint

Updated: 2026-10-22

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

## Subject protection integrated on `feat/subject-protection-integration`

- Set `subject_points` to one or more normalized positive points on the selected subject. Multiple points support disconnected body regions and are deduplicated; each point yields a SAM mask, ViTMatte alpha, and the masks are max-unioned. The current limit is 16 points. Select `matte_tile_size` from 512, 768, or 1024; 512 was used for the measured full request.
- Subject protection is optional and requires a Bokeh mode plus local validated SAM and ViTMatte model paths. It computes alpha at an analysis long edge up to 2048, then restores alpha to the requested output dimensions without reducing the exported photo.
- The subject alpha reduces CoC before ARNet and highlight scatter. A pull-push fill creates a clean background plate under the subject, then the sharp M9/Lab image is composited over the rendered background with row-chunked edge decontamination.
- The P3M-10K 1080×1620 three-point CPU run measured 788,070,400 B process-tree RSS and 112.73 s. Its three points produced mask IoU 0.9879; SAM encoder cache recorded two hits. On the same alpha/source/background arrays, the Python final composite matched the Kotlin phone reference within MAE 0.0368 and RMSE 0.5827 (98.84% of RGB pixels exactly equal). See [subject-protection benchmark](benchmarks/subject-protection-integration-2026-10-22.md).

## GAIC composition integrated on `feat/gaic-composition-integration`

- `suggest_composition()` runs the validated GAIC ONNX graph through the shared Python engine, applies EXIF orientation and configured file/pixel limits, returns ranked model suggestions, emits progress, and honors cancellation during ONNX scoring.
- The GAIC session follows the engine strategy: Low Memory closes it after one request; Balanced loads lazily and reuses it. `close()` releases retained GAIC state.
- `ProcessingOptions.crop` applies a selected normalized crop after any requested color, bokeh, or combined render. The crop is explicit, so GAIC can run as a preceding suggestion call and does not run again during rendering.
- `apply_crop_file()` remains a model-free path for an existing image, preserves ICC metadata, and avoids rerunning M9, depth, ARNet, SAM, or ViTMatte when the caller already has a rendered result.
- `laika composition`, `laika crop`, and `laika pipeline --crop LEFT,TOP,RIGHT,BOTTOM` call these same SDK operations. Crops use the EXIF-oriented pixel rectangle and do not upscale.
- Across three fresh Low Memory workers on a 1080×1620 P3M-10K image, GAIC scored 556 proposals in a median 1.016 s with 353,615,872 B median process-tree RSS (observed range 343,523,328–362,512,384 B). The physical-phone Kotlin run on `a4976c80` used source dimensions for candidate snapping. Same-tensor score values matched exactly; on the actual phone/Python photo decode paths, the top-four crop boxes matched and score improvements differed by at most 0.0131. See [GAIC real-photo benchmark](benchmarks/gaic-real-photo-2026-10-22.md).
- GAIC suggestions and crop application are available through the shared engine and CLI. FastAPI accepts an explicit selected crop in a task request; a separate HTTP suggestion endpoint and Web crop controls remain pending.

## FastAPI task service on `feat/fastapi-task-service`

- `POST /api/v1/tasks` streams uploads into UUID-owned temporary folders, verifies the actual decoded JPEG/PNG/WebP format, rejects animation and enforces byte/pixel limits before queueing the shared `LaikaProcessingEngine`.
- The bounded worker queue supports Low Memory per-task engines and Balanced thread-local engine reuse. Task tokens authorize status, cancellation, source, preview, and download routes; only token digests are stored. Failed and cancelled files are removed, completed tasks expire by TTL, and owned directories are cleaned at service shutdown.
- `/api/v1/tasks/{id}` exposes task stage and progress. Preview and full PNG download routes are available. An explicitly selected normalized crop is passed into the shared engine and does not invoke GAIC again.
- Local end-to-end API benchmark uploaded one 1080×1620 P3M portrait and ran M9, Depth, ARNet and crop in the same task. It returned a 918×1224 PNG, sampled 698,236,928 B server process-tree peak RSS, and took 90.604 s task wall time on an 8 GiB MacBook Air M1 (CPU-only, one thread). See [API task benchmark](benchmarks/api-task-real-photo-2026-10-22.md). This was not cgroup-limited and had no full-frame Kotlin quality comparison.
- The API is a thin adapter and does not change SDK/CLI engine behavior. Its queue/status live in process memory; current deployment is a single Uvicorn process. It binds to loopback by default and uses per-task capability tokens, not user accounts.
- HTTP composition suggestions, Web editing, Linux cgroup measurements and API real-model CI remain to do.

## Current integration limits

- M9 low-memory mode decodes its source for model preprocessing and again for exact Lab detail fusion. The combined path then decodes the intermediate M9 image once for shared depth/bokeh input. This preserves existing M9 behavior but is not yet the minimum-decode design.
- Depth inference returns a full-size float32 depth map. Normalization currently sorts a copy to reproduce percentile semantics; future profiling should replace this only if an exact memory-bounded percentile implementation is demonstrated.
- ARNet's first full-scale pass is the dominant Bokeh inference time. The current renderer keeps one full-size float32 source and composite and resizes each ARNet pass output back to the requested raster. Tiled full-global equivalence has not been demonstrated.
- Composition suggestions and crop application are available on the shared engine, but composition is not yet an option within `process_file()`. Full 5760×3840 requests with highlight scatter and subject protection have not yet been measured.
- Field-dependent optical scatter is the current CPU bottleneck: 140.89 s on the 2048×1365 real-photo task. It evaluates all required depth layers and the 5×5 pupil field; no reduced-resolution shortcut is used. Exact-preserving CPU optimization and a full-resolution RSS run remain outstanding.

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
- [ ] Add GAIC suggestion and crop selection to the versioned API/Web editing flow.
- [ ] Validate more hair, fine-object and background-gap photos, and measure full-resolution subject protection and Linux cgroup limits.
- [ ] Six-platform GitHub Actions matrix passes before merge.
- [ ] Full-resolution highlight run and Linux cgroup memory acceptance remain outstanding.

## FastAPI branch verification

- [x] Upload, decoded-format/pixel/byte validation, bounded scheduling, progress, cancellation, token isolation, preview/download, and cleanup tests pass with the shared-engine interface.
- [x] A single uploaded real photograph completed M9 + LabFusion + Depth + ARNet + explicit crop and returned a downloadable PNG through the HTTP API; measured results are in the [API report](benchmarks/api-task-real-photo-2026-10-22.md).
- [x] Active upload reservations account for remaining configured workspace budget, with a regression test for concurrent reservation.
- [x] Local suite: 165 tests; Ruff lint and format checks pass.
- [ ] Six-platform remote CI for this branch.
- [ ] Add the GAIC suggestion route; Web editor and Linux cgroup acceptance remain outstanding.
