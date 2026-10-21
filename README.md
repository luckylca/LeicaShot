# Laika

Laika is a new cross-platform Python image processing engine focused on low peak memory and practical CPU inference. It is an independent project with its own API, implementation, and version history. The Android reference project is research material only; its source and model files are not copied into this repository.

**Status:** early 0.x development. A CPU Depth Anything V2 Small ONNX adapter, a reproducible dynamic exporter, and exact global attention query chunking are available on the integration branch. The remaining color, bokeh, subject, composition, and HTTP pipeline are not implemented.

## Project principles

- Keep the image engine usable from a Python SDK, CLI, and versioned HTTP API.
- Preserve requested output dimensions and edge quality; report resource failures instead of silently degrading results.
- Measure process-tree memory and end-to-end time before and after performance changes.
- Load only the models a task needs, with explicit model provenance and license records.
- Do not commit weights, user photos, cache contents, or generated benchmark artifacts.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md), the [architecture](docs/ARCHITECTURE.md), the [model guide](docs/MODELS.md), and the [reference study](docs/research/reference-study.md). The CLI includes model-free image I/O and optional depth inference commands; model weights and exports remain outside Git.

## License status

The Laika source license has not yet been selected. No source or model material from the reference application is included. Model availability and license status are tracked separately; the reference M9 model is explicitly personal non-commercial and is not part of Laika.
