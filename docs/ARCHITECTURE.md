# Laika architecture

## Current status

This is an independent Python project. M0 establishes packaging, CI, a safe image I/O baseline, and repeatable measurements. The current branch does not claim to provide M9 color transfer, learned depth, neural bokeh, segmentation, matting, or GAIC composition inference.

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
