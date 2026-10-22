# LeicaShot development backlog

Updated: 2026-10-22. Priorities reflect user-requested end-to-end delivery order and current integration risk. Current feature branch: `feat/subject-protection-integration`.

| Priority | Work | State | Acceptance |
|---:|---|---|---|
| 1 | Unified M9 → depth → ARNet engine and CLI | Merged PR #22 (`371cafb`); modes, progress/cancel, lifecycle, and full-size no-highlight reference run verified | Continue to add production stages through the shared Python engine; real-model and full-size validation for each stage |
| 2 | Reference highlight extraction, clipped-light recovery, nine-blade field optics, softening and composite | Merged PR #23; 2048×1365 three-model run and physical-phone Kotlin intermediate comparison recorded | Keep the 140.89 s optical scatter stage visible; exact-preserving CPU optimization and full-resolution measurement remain open |
| 3 | Full SAM → trimap → ViTMatte → foreground composite and multi-point management | Implemented on `feat/subject-protection-integration`; five-model real-photo pipeline, 512-pixel tiles, mask metrics, cache reuse and Kotlin final-composite comparison recorded; PR pending | Run CI; verify head/hair, fine-object, and background-gap behavior on more photos; compare 768/1024 tiles and document cancel/cache invalidation |
| 4 | GAIC model/export, candidate enumeration/scoring, crop application | Next branch after subject-protection merge: `feat/gaic-composition-integration` | Checkpoint-backed score parity, deterministic crop results, apply crop without rerunning prior models; add composition-only and combined modes |
| 5 | Optical scatter CPU optimization and full-resolution memory measurement | Bottleneck identified; not optimized | Reprofile 63-layer 5×5 field scatter; test exact-preserving execution options first, report approximation error before adoption; measure full 5760×3840 run |
| 6 | FastAPI `/api/v1` uploads, task queue/status/progress/cancel, previews, download and cleanup | Not started | End-to-end API tests, resource limits, bounded concurrency, per-task data isolation |
| 7 | Minimal React + TypeScript editor | Not started | Upload, mode selection, click focus, aperture, progress, before/after comparison, download |
| 8 | Full-pipeline memory/performance optimization and Linux low-memory acceptance | Not started | Process-tree RSS and cgroup measurements for complete real-photo tasks at 8/4/2 GiB; quality and timing reported together |
| 9 | Packaging, deployment, security, regression and release preparation | Not started | Reproducible Linux/Docker install, user-data cleanup, model manifest, cross-platform CI and documented release process |

## Required per-item evidence

- Separate feature/performance branch and a PR with a bounded acceptance target.
- Unit/API tests plus a real-model smoke when the required local assets are present.
- Peak RSS, stage time, complete-task time, model lifecycle time, and output-quality measurements for every new pipeline stage.
- No silent output-resolution changes and no model weight/photo artifacts in Git.
- After each squash merge to `develop`, update [project status](PROJECT_STATUS.md) and [engine integration](ENGINE_INTEGRATION.md), then start the next unblocked priority immediately. Current checkpoint: merge subject protection after CI, then start GAIC composition.
