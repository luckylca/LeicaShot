# Engine integration checkpoint

Updated: 2026-10-22

## Intended Python call

```python
from laika import LaikaProcessingEngine, ProcessingOptions

with LaikaProcessingEngine(
    m9_model_path="/local/models/leica-m9-v3-epoch11.fp16.onnx",
    depth_model_path="/local/models/depth-v2-small-dynamic-q64.onnx",
    bokeh_model_path="/local/models/legendary-1.0-arnet.fp16.onnx",
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
        ),
        progress_callback=on_progress,
        cancel_event=cancel_event,
    )
```

The CLI equivalent is `laika pipeline input.jpg output.png --mode color+bokeh ...`. `color` and `bokeh` modes load only the models they use. `output_size=(width, height)` is optional and explicit; otherwise the EXIF-oriented source dimensions are preserved.

## Lifecycle and data ownership

- Low Memory: Depth Anything reads the EXIF-oriented original first, writes its full-size raw depth map to a task-owned temporary `.npy` for color+bokeh requests, then closes the Depth ONNX session. M9 next reads the original and writes a temporary full-size color result, then closes the M9 session. That M9 image is decoded once and shared by ARNet; the saved original-image depth is normalized and resized only to the explicitly requested work size. ARNet closes before output export.
- Bokeh-only requests keep the one decoded source RGB array across depth inference and ARNet, while closing the Depth session before ARNet is loaded. Color-only requests do not construct or load depth/ARNet.
- Balanced: M9, depth, and ARNet sessions are created lazily and reused across serialized calls. `engine.close()` releases all retained sessions.
- Calls on one engine instance are serialized under a lock. A server must still apply a global bounded task queue across engine instances.
- Progress callbacks run at stage boundaries. Cancellation is checked between stages, during M9 row fusion, and during ARNet compositing; ONNX model runs themselves are not preempted mid-inference.
- PNG output is written to a temporary file beside the destination and atomically replaced only after successful processing.

## Current integration limits

- M9 low-memory mode decodes its source for model preprocessing and again for exact Lab detail fusion. The combined path then decodes the intermediate M9 image once for shared depth/bokeh input. This preserves existing M9 behavior but is not yet the minimum-decode design.
- Depth inference returns a full-size float32 depth map. Normalization currently sorts a copy to reproduce percentile semantics; future profiling should replace this only if an exact memory-bounded percentile implementation is demonstrated.
- ARNet's first full-scale pass is the dominant Bokeh inference time. The current renderer keeps one full-size float32 source and composite and resizes each ARNet pass output back to the requested raster. Tiled full-global equivalence has not been demonstrated.
- The current renderer only makes the depth-guided ARNet plate. Highlight scatter, SAM/ViTMatte foreground protection, and GAIC composition are separate pending stages.

## Real-photo verification

The same 5760×3840 CC0 source was processed in one `process_file()` call and one CLI invocation. The validated order is original-photo depth, M9 plus Lab fusion, then ARNet using the M9 image and that original-photo depth map. No output size was passed in the full-resolution run. On a MacBook Air M1 (8 GiB, macOS 15.7.7), CPUExecutionProvider, one thread, Low Memory: 2048×1365 took 54.38 s wall with 1,094,172,672 B high-water RSS; 5760×3840 took 605.43 s with 2,197,651,456 B RSS. ARNet pass 1 alone took 514.45 s at full resolution. Host CPU load was shared with another active Python task; these are observed run times rather than isolated speed claims. Details, hashes, and commands are in [the benchmark report](benchmarks/unified-pipeline-real-photo-2026-10-22.md).

## Verification checklist for this branch

- [x] Synthetic tests cover mode selection, model laziness, Low Memory release, Balanced reuse, progress, cancellation cleanup, explicit size, and output dimensions.
- [x] Run pinned M9, Depth, and ARNet models in one `process_file()` call on the same real source photograph; Depth consumes the original and ARNet consumes the M9 result.
- [x] Record explicit 2048×1365 and original-size 5760×3840 output timing/RSS after correcting depth/color order.
- [x] Compare fused color and bokeh output to reference Kotlin results using the physical device through `adb -s a4976c80`; the numerical results and local asset/name adaptations are recorded in the pipeline report. The Android emulator was not used.
- [x] Python full suite (127 tests), Ruff formatting/lint, and package wheel build pass locally.
- [ ] Six-platform GitHub Actions matrix passes before merge.
