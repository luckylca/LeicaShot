# LeicaShot development backlog

Updated: 2026-10-22. Priorities reflect user-requested end-to-end delivery order and current integration risk. Current feature branch: `feat/fastapi-task-service` (API implementation in progress).

| Priority | Work | State | Acceptance |
|---:|---|---|---|
| 1 | Unified M9 → depth → ARNet engine and CLI | Merged PR #22 (`371cafb`); modes, progress/cancel, lifecycle, and full-size no-highlight reference run verified | Continue to add production stages through the shared Python engine; real-model and full-size validation for each stage |
| 2 | Reference highlight extraction, clipped-light recovery, nine-blade field optics, softening and composite | Merged PR #23; 2048×1365 three-model run and physical-phone Kotlin intermediate comparison recorded | Keep the 140.89 s optical scatter stage visible; exact-preserving CPU optimization and full-resolution measurement remain open |
| 3 | Full SAM → trimap → ViTMatte → foreground composite and multi-point management | Merged PR #24; real-photo metrics, tile sweep, cache reuse and phone composite comparison recorded | Validate more head/hair, fine-object, and background-gap photos; measure full-resolution and Linux low-memory cases |
| 4 | GAIC model/export, candidate enumeration/scoring, crop application | Merged PR #25; model parity, real-photo CPU benchmark, physical-phone Kotlin comparison, crop-only and combined render crop paths recorded | Expose suggestions and selected crops through API/Web; do not rerun inference when applying an existing crop |
| 5 | FastAPI `/api/v1` uploads, bounded task queue/status/progress/cancel, previews, download and cleanup | Implemented on active `feat/fastapi-task-service`; local API tests and a real M9+Depth+ARNet task passed; PR not yet opened | Run full tests and six-platform CI, review API limits, publish PR, squash merge |
| 6 | Minimal React + TypeScript editor and HTTP GAIC suggestion endpoint | Next immediately after API PR merge | Upload, mode selection, click focus, aperture, progress, before/after preview, crop suggestions and download |
| 7 | Optical scatter CPU optimization and full-resolution memory measurement | Bottleneck identified; not optimized | Reprofile 63-layer 5×5 field scatter; test exact-preserving execution options first, report approximation error before adoption; measure full 5760×3840 run |
| 8 | Full-pipeline memory/performance optimization and Linux low-memory acceptance | Ongoing alongside features; Mac real-photo reports exist, no Linux cgroup acceptance | Process-tree RSS and cgroup measurements for complete real-photo tasks at 8/4/2 GiB; quality and timing reported together |
| 9 | Packaging, deployment, security, regression and release preparation | Partial: API limits and cleanup exist; deployment/security review remains | Reproducible Linux/Docker install, user-data cleanup, model manifest, cross-platform CI and documented release process |

## Required per-item evidence

- Separate feature/performance branch and a PR with a bounded acceptance target.
- Unit/API tests plus a real-model smoke when the required local assets are present.
- Peak RSS, stage time, complete-task time, model lifecycle time, and output-quality measurements for every new pipeline stage.
- No silent output-resolution changes and no model weight/photo artifacts in Git.
- After each squash merge to `develop`, update [project status](PROJECT_STATUS.md) and [engine integration](ENGINE_INTEGRATION.md), then start the next unblocked priority immediately. Current checkpoint: finish and merge the FastAPI task service, then build the minimum usable Web editor and GAIC HTTP suggestion route.
