# Laika

Laika is a new cross-platform Python image processing engine focused on low peak memory and practical CPU inference. It is an independent project with its own API, implementation, and version history. Python is the production engine and service language. Android is used only to run the reference Kotlin engine on a physical phone through ADB for comparison; no Android runtime or virtual device is part of Laika's processing path.

**Status:** early 0.x development. The Python engine and CLI run M9 series 3.0 epoch11 color/Lab fusion, Depth Anything V2 Small, BokehMe ARNet with reference-style highlight scatter, and optional SAM 2.1 plus ViTMatte subject protection in one request. It also runs the authorized GAIC MobileNetV2 model for crop suggestions; selected crops apply after any render mode or directly to an existing image without loading AI models. The FastAPI task service is merged in PR #26. One CPU-only HTTP task processed M9, Depth, ARNet and a selected crop in 90.604 s at a sampled 698,236,928 B server process-tree RSS peak. The active `feat/web-editor` branch adds a real GAIC HTTP route and React editor; the connected browser flow uploaded a real photo, selected a GAIC crop, completed combined processing, displayed a preview, and downloaded a verified full-size PNG. The route scored 86 crop candidates in 1.403 s at 416,251,904 B RSS. See the [API task report](docs/benchmarks/api-task-real-photo-2026-10-22.md), [composition API report](docs/benchmarks/api-composition-real-photo-2026-10-22.md), and [Web guide](docs/WEB.md). Existing end-to-end CPU results include a five-model 1080×1620 subject-protection run at 788,070,400 B / 112.73 s and a 5760×3840 three-model run at 2,197,651,456 B / 605.43 s. Linux cgroup acceptance and full-resolution subject/highlight measurements remain outstanding.

Real-photo CPU, memory, and numerical reports are available for [M9](docs/benchmarks/m4-m9-inference-2026-10-22.md), [depth](docs/benchmarks/m4-depth-real-photo-2026-10-22.md), [BokehMe ARNet](docs/benchmarks/bokeh-real-photo-2026-10-22.md), and the [unified full-resolution pipeline](docs/benchmarks/unified-pipeline-real-photo-2026-10-22.md).

The first coordinated M9 + Depth + ARNet + SAM + ViTMatte photo run, matte quality check, and physical-phone Kotlin composite comparison are recorded in the [subject-protection integration report](docs/benchmarks/subject-protection-integration-2026-10-22.md).

The user confirmed that required model-use authorizations, including M9 and the other planned models, are in place. Model weights remain local user-provided assets unless separate public redistribution rights are established; Laika does not commit model files.

## Project principles

- Keep the image engine usable from a Python SDK, CLI, and versioned HTTP API.
- Preserve requested output dimensions and edge quality; report resource failures instead of silently degrading results.
- Measure process-tree memory and end-to-end time before and after performance changes.
- Load only the models a task needs, with explicit model provenance and license records.
- Do not commit weights, user photos, cache contents, or generated benchmark artifacts.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md), the [architecture](docs/ARCHITECTURE.md), the [model guide](docs/MODELS.md), the [engine integration checkpoint](docs/ENGINE_INTEGRATION.md), the [HTTP API guide](docs/API.md), the [Web editor guide](docs/WEB.md), and the [reference study](docs/research/reference-study.md). Python CLI commands include image I/O, depth inference, M9 color, BokehMe ARNet background rendering, coordinated `laika pipeline` processing with an optional selected crop, GAIC `laika composition` suggestions, and model-free `laika crop` application. Model weights and exports remain outside Git.

## License status

The Laika source license has not yet been selected. M9 use in this project is based on the user's confirmation that the required authorization is in place. The model stays in a user-provided local asset path; this project does not claim public model redistribution rights. See the [M9 model record](models/m9-local.json) and [model guide](docs/MODELS.md).
