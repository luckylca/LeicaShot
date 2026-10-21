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

## 2026-10-21 — M1 Depth Anything V2 Small acquisition and conversion baseline

- Added an explicit model manifest and `laika-models` fetch/verify commands. The verified official Small checkpoint remains in `/tmp`, outside Git.
- Strictly loaded the checkpoint from the upstream source pinned in the manifest and ran a CPU-only, single-thread inference using the upstream resize and normalization contract. A seeded 512×384 synthetic input produced a finite 384×512 depth map.
- Added a reproducible fixed-shape ONNX exporter. It verifies the upstream source revision and checkpoint digest, refuses existing outputs, validates with `onnx.checker`, runs ONNX Runtime CPU parity, and writes graph/data hashes with a validation report.
- The 518×686 ONNX check produced max/mean/p99 absolute errors `5.01e-6` / `8.82e-7` / `2.86e-6`. Two same-input comparison trials measured PyTorch median forward 1.185–1.226 s / peak RSS 1,045,364,736–1,082,638,336 bytes and ONNX Runtime median 2.314–3.570 s / peak RSS 563,281,920–570,982,400 bytes. This Mac result shows 45–48% lower worker RSS but 1.95–2.91× slower median inference under ONNX; the slower trial coincided with lower available host memory. It is one shape/runtime configuration, not a general conclusion.
- A separate preprocessing smoke run peaked at 1,007,566,848 bytes RSS (960.9 MiB); cold process wall time including PyTorch imports was 27.530 s. macOS USS and cgroup metrics were unavailable. See [M1 benchmark details](benchmarks/m1-depth-anything-2026-10-21.md).
- M1 remains in progress: dynamic ONNX export is validated only on synthetic tensors; there is no production depth inference adapter, real-photo quality set, Linux cgroup measurement, or full pipeline result.

## 2026-10-21 — dynamic depth export and high-resolution sample

- Added a dynamic height/width export option with shape-aware DINO positional resize and DPT output sizing. The first dynamic attempt specialized positional token length and failed at a second aspect ratio; the corrected graph passes four multiple-of-14 shapes.
- Dynamic DPT adapter output matched the pinned PyTorch model exactly on all four synthetic shapes. ONNX maximum absolute error was `1.67e-5` at most; all result shapes matched inputs and contained finite values.
- A one-warmup, one-run 1400×518 comparison measured PyTorch 1,487,093,760 bytes peak RSS / 4.125 s inference and ONNX Runtime 1,270,792,192 bytes / 6.976 s. This is 14.6% lower sampled RSS and 1.69× slower ONNX on this single M1 case.
- The dynamic graph uses opset 17 and embeds its weights. The future engine must reject dimensions below 518 or not divisible by 14. No dynamic photo quality, Linux cgroup test, query-chunked graph, or full pipeline result exists yet. See [the dynamic export report](benchmarks/m1-dynamic-depth-2026-10-21.md).

## 2026-10-21 — global query chunking for Depth Anything V2

- Added an opt-in TorchScript query loop to the dynamic exporter. Every query chunk still reads every key and value; the exported graph is rejected unless all 12 attention blocks preserve ONNX Loop nodes.
- Tested 32, 64, and 128 query tiles against unmodified PyTorch attention on seeded 518×518, 700×532, and 1400×518 inputs. PyTorch output maximum absolute difference was zero for all sizes. ONNX parity maximum absolute error was at most `1.67e-5` (limit `1e-4`).
- The corrected M1 direct-model sample at 1400×518 measured baseline 1,192,050,688 bytes RSS / 7.397 s; chunks 32 / 64 / 128 measured 457,261,056 / 435,109,888 / 487,211,008 bytes and 8.209 / 7.355 / 6.792 s. The direct graph loop check now reads the exporter sidecar rather than parsing model weights inside the measured worker.
- Two earlier Linux smoke runs passed before that measurement correction; their query-chunk RSS figures included extra graph loading and are superseded. The updated Linux workflow will rerun the fair comparison.
- See [query-chunking benchmark](benchmarks/m1-depth-query-chunking-2026-10-21.md). The exporter experiment is not yet integrated into the production image engine. No real-photo or full-pipeline quality/memory result exists.

## 2026-10-21 — M2 CPU depth inference adapter

- Added `DepthAnythingV2Small`, which checks the dynamic export sidecar and graph hash, loads ONNX Runtime only on explicit construction, uses CPUExecutionProvider, and releases the session through `close()` or context exit.
- Implemented the upstream lower-bound aspect resize, OpenCV bicubic interpolation, RGB normalization, NCHW layout, and full-source-size bilinear `align_corners=True` restoration in bounded row blocks. Image preprocessing converts one source channel at a time to avoid a full-image float32 RGB copy.
- Added the `laika depth` command. It validates actual image format, file size, dimensions, animation, and EXIF orientation; it writes full-resolution float32 `.npy` output atomically. Default source and network caps are 40M and 4M pixels.
- A 700×400 synthetic RGB comparison against the pinned PyTorch model gave preprocessing max absolute error `1.19e-6`, and final depth max/mean/p99 errors `6.15e-5` / `3.07e-6` / `5.01e-6` (limit `1e-4`).
- Three cold CLI workers on the M1 at 1400×518 measured median RSS 882,180,096 B for baseline and 453,623,808 B for query chunk 128; both returned a 1400×518 map. Worker wall medians were 8.186 s and 7.733 s, with variable trial times. macOS USS/cgroup data unavailable; these are synthetic single-stage runs, not a full pipeline result.
- Added opt-in PyTorch comparison and complete image-to-depth measurement to the model smoke workflow. Its first Linux x86_64 run found preprocessing max absolute error `1.58e-4` (above the initial `2e-6` Mac-derived tolerance), while final restored depth max error remained `5.15e-5` (below the unchanged `1e-4` output limit). The validation now uses a `2e-4` cross-platform preprocessing tolerance and records max/mean/p99; the model smoke will be rerun before merge.
