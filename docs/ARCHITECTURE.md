# Laika architecture

## Current status

This is an independent Python project. M0 established packaging, CI, safe image I/O, and repeatable process-tree measurements. M1 validated Depth Anything V2 Small model acquisition and static/dynamic ONNX export, then tested exact global query chunking. The M9 series 3.0 epoch11 ONNX adapter in `laika.m9` drives reference-style Lab detail fusion through the `laika color` CLI. `LaikaProcessingEngine` coordinates original-photo depth, M9/Lab color, BokehMe ARNet with reference-style optical highlights, and optional SAM 2.1 + ViTMatte subject protection in one SDK call or `laika pipeline` CLI request, with progress/cancellation, Low Memory/Balanced lifecycle strategies, and original-size output. GAIC MobileNetV2 has a pinned local export, dynamic CPU ONNX adapter, shared-engine crop suggestion/application methods, explicit crop support in `ProcessingOptions`, and `laika composition` / `laika crop` commands. Across three fresh workers on one portrait, it scored 556 candidates in a median 1.016 s at 353,615,872 B median process-tree RSS (343,523,328–362,512,384 B range); top crop boxes matched the physical-phone Kotlin reference. The M9-plus-crop request completed in 20.117 s at 356,876,288 B and matched the full-size M9 output crop exactly. PR #26 added bounded FastAPI tasks and PR #27 added the GAIC suggestion endpoint plus React editor. A real browser smoke completed upload, crop selection, combined processing, preview, and full PNG download. The active `feat/web-subject-protection` branch adds subject prompts and tile controls; one real five-model API task with three prompts took 102.177 s and 1,029,177,344 B sampled process-tree peak RSS while producing the exact SDK output hash. At 5760×3840 the three-model CPU path measured 2,197,651,456 B high-water RSS and 605.43 s wall on an 8 GiB MacBook Air M1. See the [unified pipeline report](benchmarks/unified-pipeline-real-photo-2026-10-22.md), [subject integration report](benchmarks/subject-protection-integration-2026-10-22.md), [Web subject report](benchmarks/web-subject-protection-2026-10-22.md), [GAIC report](benchmarks/gaic-real-photo-2026-10-22.md), [API task report](benchmarks/api-task-real-photo-2026-10-22.md), and [Web guide](WEB.md). Android is used only for physical-device ADB comparison runs of the reference Kotlin engine; it is not in Laika's production processing architecture.

## Target boundaries

```text
Clients (Python SDK / CLI / HTTP API)
                 │
                 ▼
       Python processing facade
                 │
       ┌─────────┴─────────┐
       ▼                   ▼
 Image pipeline       Task/resource manager
       │                   │
       ├── color             ├── bounded queue
       ├── depth             ├── model lifecycle
       ├── bokeh             ├── memory budget
       ├── subject           └── cache ownership
       └── composition
                 │
        Native image/model runtimes
        (Pillow/OpenCV/NumPy/ONNX Runtime)
```

The core engine does not depend on FastAPI or a browser. The API validates and stages uploads, creates bounded tasks, calls the same engine interface as the SDK and CLI, exposes stage progress, and cleans temporary data. Model adapters own their session and tensor lifetimes; task state owns intermediate artifacts. The API currently uses an in-process queue and single Uvicorn worker; deployments must size concurrency to their memory budget.

The current depth adapter accepts caller-owned HWC `uint8` RGB arrays and returns a caller-owned float32 depth map at the exact input dimensions. It loads only a dynamic ONNX model whose adjacent validation report and SHA-256 match. Its ONNX session is explicitly closed through a context manager. By default, input sizing caps the analysis long edge at 1220 without upscaling, uses Pillow bicubic resampling, replicate-pads right and bottom edges to a multiple of 14, then applies ImageNet normalization and NCHW layout. Passing `input_size=518` selects the separate upstream lower-bound policy. Final depth resizing implements bilinear `align_corners=True` in bounded row blocks.

The subject path prepares SAM 2.1 inputs, reuses split image-encoder outputs across positive prompt points, refines each selected mask through the hash-verified dynamic ViTMatte CPU graph, unions alpha masks, protects subject CoC, fills the background plate, and composites sharp foreground over the rendered bokeh. Low Memory releases SAM before opening ViTMatte and closes ViTMatte before ARNet. Current real-photo quality and process-tree memory measurements cover one 1080×1620 three-point portrait; broader hair, thin-object, 4K, and Linux cgroup validation remain open. See the [subject integration report](benchmarks/subject-protection-integration-2026-10-22.md) and [ViTMatte study](research/vitmatte-integration.md).

`laika.m9.M9ColorEngine` performs hash-verified CPU M9 inference and feeds its 8-bit RGB output to `laika.color.fuse_lab_detail`. Low Memory closes the ONNX session before the full-resolution source is decoded for fusion; Balanced retains a reusable session. Both keep the oriented source's output dimensions and use 1024 only as the model working long edge. Row-streamed fusion matches full-height output exactly on a 22 MP photo; the [M9 benchmark report](benchmarks/m4-m9-inference-2026-10-22.md) records its CPU timing, process-tree RSS, numerical error, and current cgroup limitation.

## Memory-first execution plan

1. Establish a decode/transform/encode baseline and sample RSS across the complete worker process tree.
2. Add model-by-model and full-pipeline measurements before optimization.
3. Convert pipeline stages to explicit inputs/outputs with one owner and a defined release point. Avoid retaining RGB float copies when byte or tiled representations suffice.
4. Load stages on demand and choose short-lived or resident workers from measured cold-start, hot-run, and peak-memory results.
5. Investigate exact global attention query chunking and overlapping matting tiles only against a numerical baseline. Do not replace global attention with local windows or reduce analysis dimensions silently.
6. Test the complete request under Linux cgroup limits. Report results by machine, runtime, thread count, model hashes, input size, RSS/USS/cgroup peak, and stage timing.

## Model and artifact policy

Every model must have a versioned manifest entry with origin, upstream revision, input/output contract, precision, SHA-256, license evidence, and any conversion recipe. Download and conversion are explicit commands. Hash mismatches fail closed. Weights and converted ONNX files stay outside normal Git history. See [reference-study.md](research/reference-study.md) for the initial mapping and unresolved permissions.

## Image handling invariants

- Validate the decoded format and dimensions, not just a filename extension.
- Set upload-byte and decoded-pixel limits before expensive processing.
- Apply EXIF orientation before interpreting coordinates.
- Keep image-space coordinates consistent across depth, points, masks, crop proposals, and output.
- Never resize the final output unless the request contains an explicit output-size choice.
- Avoid persistent user-image storage by default; isolate temporary data per task and delete it on completion or cancellation.

## Versioning and releases

The package starts at `0.1.0`. `develop` is the integration line; focused changes flow through pull requests and squash merges. Stable milestones move to `main` with a release PR. Keep the public API under `/api/v1/` and use SemVer while the project remains below `1.0.0`.
