# Laika architecture

## Current status

This is an independent Python project. M0 established packaging, CI, safe image I/O, and repeatable process-tree measurements. M1 validated Depth Anything V2 Small model acquisition and static/dynamic ONNX export, then tested exact global query chunking. M2 has a CPU depth adapter in `laika.depth` and a `laika depth` CLI command with the reference max-edge sizing profile and an explicit upstream lower-bound profile. M9 color transfer, neural bokeh, segmentation, matting, GAIC composition, HTTP tasks, and a complete image pipeline are still unimplemented.

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

The core engine will not depend on FastAPI or a browser. The API will validate and stage an upload, create a bounded task, call the same engine interface as the SDK and CLI, emit progress, and clean temporary data. Model adapters own their session and tensor lifetimes; task state owns decoded buffers and intermediate artifacts. The scheduler must serialize work when the configured memory budget cannot support overlap.

The current depth adapter accepts caller-owned HWC `uint8` RGB arrays and returns a caller-owned float32 depth map at the exact input dimensions. It loads only a dynamic ONNX model whose adjacent validation report and SHA-256 match. Its ONNX session is explicitly closed through a context manager. By default, input sizing caps the analysis long edge at 1220 without upscaling, uses Pillow bicubic resampling, replicate-pads right and bottom edges to a multiple of 14, then applies ImageNet normalization and NCHW layout. Passing `input_size=518` selects the separate upstream lower-bound policy. Final depth resizing implements bilinear `align_corners=True` in bounded row blocks.

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
