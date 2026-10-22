# LeicaShot development backlog

Updated: 2026-10-22. Priorities reflect user-requested end-to-end delivery order and current integration risk. Latest merge: PR #28. Current performance branch: `perf/linux-cgroup-pipeline`.

| Priority | Work | State | Acceptance |
|---:|---|---|---|
| 1 | Unified M9 → depth → ARNet engine and CLI | Merged PR #22 (`371cafb`); modes, progress/cancel, lifecycle, and full-size no-highlight reference run verified | Continue to add production stages through the shared Python engine; real-model and full-size validation for each stage |
| 2 | Reference highlight extraction, clipped-light recovery, nine-blade field optics, softening and composite | Merged PR #23; 2048×1365 three-model run and physical-phone Kotlin intermediate comparison recorded | Keep the 140.89 s optical scatter stage visible; exact-preserving CPU optimization and full-resolution measurement remain open |
| 3 | Full SAM → trimap → ViTMatte → foreground composite and multi-point management | Merged PR #24; real-photo metrics, tile sweep, cache reuse and phone composite comparison recorded | Validate more head/hair, fine-object, and background-gap photos; measure full-resolution and Linux low-memory cases |
| 4 | GAIC model/export, candidate enumeration/scoring, crop application | Merged PR #25; model parity, real-photo CPU benchmark, physical-phone Kotlin comparison, crop-only and combined render crop paths recorded | Expose suggestions and selected crops through API/Web; do not rerun inference when applying an existing crop |
| 5 | FastAPI `/api/v1` uploads, bounded task queue/status/progress/cancel, previews, download and cleanup | Merged PR #26; 165 Python tests passed locally, six-platform CI passed, and a real M9+Depth+ARNet HTTP task was recorded | Continue API quality and low-memory verification through real Linux cgroup runs |
| 6 | Minimal React + TypeScript editor and HTTP GAIC suggestion endpoint | Merged PR #27 (`548448c`); browser flow, 171 Python tests, three Web tests, Ruff, Vite build and seven remote CI jobs passed | Keep the editor usable as further task options are connected; browser upload, processing, comparison and verified download work |
| 7 | Subject protection controls in Web editor | Merged PR #28 (`c7d3e5b`); four Web tests, Vite build, browser upload → three points → five-model task → preview → full PNG download and seven remote CI checks passed | Broaden hair/fine-object/background-gap scene quality checks |
| 8 | Optical scatter CPU optimization and full-resolution memory measurement | Next task after Linux cgroup PR; 128/512-row experiment produced byte-identical output, but run-to-run timing was noisy | Profile exact-preserving pupil-kernel reuse and repeat controlled measurements; retain 128-row default until a reproducible gain is demonstrated; measure original-resolution runs |
| 9 | Full-pipeline memory/performance optimization and Linux low-memory acceptance | Linux cgroup v2 harness implemented and Docker build verified; real five-model portrait completed in 4 GiB and 2 GiB guests, including a 1.5 GiB task cgroup | Repeat with controlled CPU contention, add 8 GiB guest and original-resolution tasks, test service concurrency, and broaden image quality; current 2 GiB run is a single slow challenge result with ~8 MiB cgroup headroom |
| 10 | Packaging, deployment, security, regression and release preparation | Partial: API limits and cleanup exist; deployment/security review remains | Reproducible Linux/Docker install, user-data cleanup, model manifest, cross-platform CI and documented release process |

## Required per-item evidence

- Separate feature/performance branch and a PR with a bounded acceptance target.
- Unit/API tests plus a real-model smoke when the required local assets are present.
- Peak RSS, stage time, complete-task time, model lifecycle time, and output-quality measurements for every new pipeline stage.
- No silent output-resolution changes and no model weight/photo artifacts in Git.
- After each squash merge to `develop`, update [project status](PROJECT_STATUS.md) and [engine integration](ENGINE_INTEGRATION.md), then start the next unblocked priority immediately. Current checkpoint: merge the cgroup measurement harness/report, then immediately start exact-preserving optical-scatter CPU profiling and original-resolution resource measurement. See the [Linux cgroup report](benchmarks/linux-cgroup-full-pipeline-2026-10-22.md).
