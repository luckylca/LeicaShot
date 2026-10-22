# LeicaShot development backlog

Updated: 2026-10-22. Priorities reflect user-requested end-to-end delivery order and current integration risk.

| Priority | Work | State | Acceptance |
|---:|---|---|---|
| 1 | Unified M9 → depth → ARNet Python engine, progress/cancel, Low Memory/Balanced lifecycle, shared CLI | Real-photo chain and physical Kotlin comparison pass on `feat/unified-processing-engine`; push/PR/CI/merge remain | One SDK call and one CLI command process a real photo at original dimensions; retain Kotlin reference comparison, quality notes, complete timing/RSS report, and CI evidence |
| 2 | Reference highlight extraction, aperture kernel, scatter, compositing, background repair | Next branch after engine PR: `feat/bokeh-highlight-scatter` | Port from reference math and capture fixtures; compare intermediate maps and final highlights against physical-device reference on real night/point-light photos |
| 3 | Full SAM → trimap → ViTMatte → foreground composite and multi-point management | Partially implemented | Real hair, fine-object, and background-gap photos; SAM embedding reuse; edge and color-decontamination comparison |
| 4 | GAIC model/export, candidate enumeration/scoring, crop application | Not started | Checkpoint-backed score parity, deterministic crop results, apply crop without rerunning prior models |
| 5 | FastAPI `/api/v1` uploads, task queue/status/progress/cancel, previews, download and cleanup | Not started | End-to-end API tests, resource limits, bounded concurrency, per-task data isolation |
| 6 | Minimal React + TypeScript editor | Not started | Upload, mode selection, click focus, aperture, progress, before/after comparison, download |
| 7 | Full-pipeline memory/performance optimization and Linux low-memory acceptance | Not started | Process-tree RSS and cgroup measurements for complete real-photo tasks at 8/4/2 GiB; quality and timing reported together |
| 8 | Packaging, deployment, security, regression and release preparation | Not started | Reproducible Linux/Docker install, user-data cleanup, model manifest, cross-platform CI and documented release process |

## Required per-item evidence

- Separate feature/performance branch and a PR with a bounded acceptance target.
- Unit/API tests plus a real-model smoke when the required local assets are present.
- Peak RSS, stage time, complete-task time, model lifecycle time, and output-quality measurements for every new pipeline stage.
- No silent output-resolution changes and no model weight/photo artifacts in Git.
- After each squash merge to `develop`, update [project status](PROJECT_STATUS.md) and [engine integration](ENGINE_INTEGRATION.md), then start the next unblocked priority immediately. Current checkpoint: push and merge unified engine after CI, then implement high-light scattering.
