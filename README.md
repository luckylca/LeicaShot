# Laika

Laika is a new cross-platform Python image processing engine focused on low peak memory and practical CPU inference. It is an independent project with its own API, implementation, and version history. The Android reference project is research material only; its source and model files are not copied into this repository.

**Status:** early 0.x development. No AI model inference has been integrated or validated yet. The first milestone establishes the image I/O and measurement baseline before model work begins.

## Project principles

- Keep the image engine usable from a Python SDK, CLI, and versioned HTTP API.
- Preserve requested output dimensions and edge quality; report resource failures instead of silently degrading results.
- Measure process-tree memory and end-to-end time before and after performance changes.
- Load only the models a task needs, with explicit model provenance and license records.
- Do not commit weights, user photos, cache contents, or generated benchmark artifacts.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md), the [architecture](docs/ARCHITECTURE.md), and the [reference study](docs/research/reference-study.md). The baseline CLI and exact benchmark procedure will be added on the first feature branch.

## License status

The Laika source license has not yet been selected. No source or model material from the reference application is included. Model availability and license status are tracked separately; the reference M9 model is explicitly personal non-commercial and is not part of Laika.
