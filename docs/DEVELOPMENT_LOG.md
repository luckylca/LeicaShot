# Development log

## 2026-10-21 — repository bootstrap and M0 baseline

- Inspected the empty worktree and the empty public `luckylca/LeicaShot` repository; created the independent Laika history there without forking or configuring the reference as a remote.
- Added `main`, `develop`, the CI matrix, contribution/PR conventions, package metadata, and independent architecture/research records.
- Reviewed the reference repository at `f012dca8a845b8c37389345676e43a7e99a5ae93`; kept its sparse checkout outside this worktree. No source or weights were imported.
- Added the model-free Pillow image I/O engine, CLI, and process-tree benchmark runner on `feat/python-engine`.
- Local checks: Ruff formatting, Ruff lint, and 8 pytest cases passed on Python 3.11.16.
- GitHub Actions exposed a macOS Python 3.13 `psutil` USS syscall `RuntimeError`; the sampler now treats USS as unavailable while retaining RSS/CPU, with a regression test. Ruff and all 8 tests also pass locally on Python 3.13.16.
- Ran three 2048×1536 synthetic PNG baseline trials. Median worker RSS was 34.94 MiB, engine time 0.241 s, sampled worker CPU 0.272 s. See [the complete report](benchmarks/m0-baseline-2026-10-21.md) for limits and reproducibility details.
- No model inference, HTTP API, real-photo quality comparison, Linux cgroup test, or 8/4/2 GiB full-pipeline result exists yet.
