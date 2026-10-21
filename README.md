# Laika

Laika is a new cross-platform Python image processing engine focused on low peak memory and practical CPU inference. It is an independent project with its own API, implementation, and version history. Python is the production engine and service language. Android is used only to run the reference Kotlin engine on a physical phone through ADB for comparison; no Android runtime or virtual device is part of Laika's processing path.

**Status:** early 0.x development. Laika includes the M9 series 3.0 epoch11 CPU ONNX path connected to Lab detail fusion, with full-resolution PNG/JPEG output and Low Memory/Balanced strategies. A CPU Depth Anything V2 Small adapter, SAM subject-prompt primitives, experimental ViTMatte CPU/tiling support, and a BokehMe ARNet multiscale renderer/CLI are also available. Bokeh highlighting and subject protection, GAIC composition, and the HTTP pipeline remain in progress.

Real-photo CPU, memory, and numerical reports are available for [M9](docs/benchmarks/m4-m9-inference-2026-10-22.md), [depth](docs/benchmarks/m4-depth-real-photo-2026-10-22.md), and [BokehMe ARNet](docs/benchmarks/bokeh-real-photo-2026-10-22.md).

The user confirmed that required model-use authorizations, including M9 and the other planned models, are in place. Model weights remain local user-provided assets unless separate public redistribution rights are established; Laika does not commit model files.

## Project principles

- Keep the image engine usable from a Python SDK, CLI, and versioned HTTP API.
- Preserve requested output dimensions and edge quality; report resource failures instead of silently degrading results.
- Measure process-tree memory and end-to-end time before and after performance changes.
- Load only the models a task needs, with explicit model provenance and license records.
- Do not commit weights, user photos, cache contents, or generated benchmark artifacts.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md), the [architecture](docs/ARCHITECTURE.md), the [model guide](docs/MODELS.md), and the [reference study](docs/research/reference-study.md). Python CLI commands include image I/O, depth inference, M9 color, and BokehMe ARNet background rendering. Model weights and exports remain outside Git.

## License status

The Laika source license has not yet been selected. M9 use in this project is based on the user's confirmation that the required authorization is in place. The model stays in a user-provided local asset path; this project does not claim public model redistribution rights. See the [M9 model record](models/m9-local.json) and [model guide](docs/MODELS.md).
