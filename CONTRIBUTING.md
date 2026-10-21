# Contributing to Laika

## Branches and pull requests

- `main` contains stable, reviewed milestones.
- `develop` is the integration branch.
- Start each change from `develop` on a focused branch: `feat/*`, `perf/*`, `fix/*`, `test/*`, `docs/*`, or `release/*`.
- Keep a branch to one reviewable change. Use Conventional Commit messages such as `feat: add image engine` or `perf: reuse decode buffers`.
- Push the branch and open a PR into `develop`. CI and review must pass before squash merge. Delete the feature branch after merge.
- Promote a tested milestone with a `release/*` PR from `develop` into `main`. Use SemVer and keep releases in `0.x` until the full engine is validated.

The first repository bootstrap commit is the only planned direct commit to the empty `main`; all ongoing implementation changes go through PRs.

## Acceptance expectations

- Preserve source dimensions unless the caller explicitly requests an output size. A resource failure must be reported instead of silently reducing quality.
- Add deterministic tests for behavior and numerical algorithms. Keep model-dependent tests opt-in and require exact model provenance.
- For memory or speed changes, record a baseline and the full-device details, input dimensions, thread settings, runtime/model versions, peak memory method, elapsed time, and output error.
- Do not claim whole-pipeline memory improvements from single-model measurements.
- Add or update provenance and license notes before introducing model weights, converted models, or third-party implementation material.
- Keep image handling independent of the API so the Python SDK, CLI, and service call the same engine.

## Local development

Use Python 3.11 or later. Install an editable development environment with `python -m pip install -e ".[dev]"`. The initial baseline will document its concrete CLI and test commands as it lands.
