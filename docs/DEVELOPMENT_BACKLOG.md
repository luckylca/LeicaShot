# LeicaShot development backlog

Updated: 2026-10-22. Priorities reflect user-requested end-to-end delivery order and current integration risk. Latest merge: PR #29. Current performance branch: `perf/optical-scatter-cpu`.

| Priority | Work | State | Acceptance |
|---:|---|---|---|
| 1 | Unified M9 → depth → ARNet engine and CLI | Merged PR #22 (`371cafb`); modes, progress/cancel, lifecycle, and full-size no-highlight reference run verified | Continue to add production stages through the shared Python engine; real-model and full-size validation for each stage |
| 2 | Reference highlight extraction, clipped-light recovery, nine-blade field optics, softening and composite | Merged PR #23; 2048×1365 three-model run and physical-phone Kotlin intermediate comparison recorded | Keep the 140.89 s optical scatter stage visible; exact-preserving CPU optimization and full-resolution measurement remain open |
| 3 | Full SAM → trimap → ViTMatte → foreground composite and multi-point management | Merged PR #24; real-photo metrics, tile sweep, cache reuse and phone composite comparison recorded | Validate more head/hair, fine-object, and background-gap photos; measure full-resolution and Linux low-memory cases |
| 4 | GAIC model/export, candidate enumeration/scoring, crop application | Merged PR #25; model parity, real-photo CPU benchmark, physical-phone Kotlin comparison, crop-only and combined render crop paths recorded | Expose suggestions and selected crops through API/Web; do not rerun inference when applying an existing crop |
| 5 | FastAPI `/api/v1` uploads, bounded task queue/status/progress/cancel, previews, download and cleanup | Merged PR #26; 165 Python tests passed locally, six-platform CI passed, and a real M9+Depth+ARNet HTTP task was recorded | Continue API quality and low-memory verification through real Linux cgroup runs |
| 6 | Minimal React + TypeScript editor and HTTP GAIC suggestion endpoint | Merged PR #27 (`548448c`); browser flow, 171 Python tests, three Web tests, Ruff, Vite build and seven remote CI jobs passed | Keep the editor usable as further task options are connected; browser upload, processing, comparison and verified download work |
| 7 | Subject protection controls in Web editor | Merged PR #28 (`c7d3e5b`); four Web tests, Vite build, browser upload → three points → five-model task → preview → full PNG download and seven remote CI checks passed | Broaden hair/fine-object/background-gap scene quality checks |
| 8 | Optical scatter CPU optimization and full-resolution memory measurement | Active `perf/optical-scatter-cpu`; mirror-kernel reuse preserves the real scatter array and full renderer pixels byte-for-byte; one pair measured 308.55 s to 286.43 s | Repeat paired runs under controlled load, retain 128-row default, and measure original-resolution/cgroup cases before merge |
| 9 | Full-pipeline memory/performance optimization and Linux low-memory acceptance | Cgroup v2 harness/report merged in PR #29; 1080×1620 five-model task completed in 2 GiB and 4 GiB guests; 2048×1365 highlight task completed in a 4 GiB guest/3 GiB cgroup; 5760×3840 highlight task OOM-killed in that 4 GiB guest | Reduce native-resolution live buffers without changing size/quality; repeat under controlled CPU load, add 8 GiB guest and service-concurrency cases, and broaden image quality; 2 GiB portrait run remains a slow challenge result with ~8 MiB cgroup headroom |
| 10 | Packaging, deployment, security, regression and release preparation | Partial: API limits and cleanup exist; deployment/security review remains | Reproducible Linux/Docker install, user-data cleanup, model manifest, cross-platform CI and documented release process |

## Required per-item evidence

- Separate feature/performance branch and a PR with a bounded acceptance target.
- Unit/API tests plus a real-model smoke when the required local assets are present.
- Peak RSS, stage time, complete-task time, model lifecycle time, and output-quality measurements for every new pipeline stage.
- No silent output-resolution changes and no model weight/photo artifacts in Git.
- After each squash merge to `develop`, update [project status](PROJECT_STATUS.md) and [engine integration](ENGINE_INTEGRATION.md), then start the next unblocked priority immediately. Current checkpoint: repeat paired optical-scatter measurements under controlled load and run highlights-enabled cgroup tests; then proceed to original-resolution optical memory measurements. See the [Linux cgroup report](benchmarks/linux-cgroup-full-pipeline-2026-10-22.md) and [kernel reuse report](benchmarks/optical-scatter-kernel-reuse-2026-10-22.md).
