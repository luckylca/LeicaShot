# Real-photo CPU thread-count A/B — Windows 2026-10-25

**Experiment:** fresh Python process per request; same authentic 2048x1365 CC0 photograph with locally verified M9 + Depth Anything V2 + SAM2.1 Tiny + ViTMatte + ARNet two passes + optical scatter. Low-memory mode, independently verified split ARNet (edge 256), SAM CPU Arena disabled, 512-pixel matte tiles, identical model checkpoints. Thread order 1/2/4 and 4/2/1.

Run locally on Windows using `scripts/benchmark_speed_matrix.py` with the full authorized model flags. Raw JSON/results are under `E:\codex\LeicaShot-local-data\benchmarks\speed-matrix-20261025-edge256` and are not added to Git.

| ONNX CPU threads | Two whole-request times (s) | Median (s) | Max sampled process-tree RSS (MiB) | Max 8-bit RGB channel delta vs thread 1 |
|---|---|---:|---:|---:|
| 1 | 67.90, 69.27 | 68.59 | 867.8 | 0 |
| 2 | 48.66, 52.73 | 50.70 | 862.5 | 1 |
| 4 | 45.76, 45.74 | 45.75 | 873.6 | 1 |

Threads 2 reduced request time by **26.1%** (one real photo, two runs only).

Threads 4 reduced request time by **33.3%** (one real photo, two runs only).

## Stage timings (median across two complete requests, seconds)

| Stage | 1 thread | 2 threads | 4 threads |
|---|---:|---:|---:|
| `bokeh.arnet_pass_1` | 15.89 | 9.26 | 6.77 |
| `bokeh.arnet_pass_2` | 7.77 | 4.59 | 3.64 |
| `bokeh.highlight_scatter` | 9.39 | 9.94 | 9.96 |
| `depth.inference` | 11.14 | 6.79 | 5.69 |
| `subject.sam_encoder` | 3.88 | 2.42 | 1.91 |
| `subject.vitmatte` | 4.76 | 3.07 | 2.56 |
| `color.inference` | 3.25 | 2.02 | 1.61 |

## Interpretation

- Four inference threads were fastest in this controlled Windows test; it is an **opt-in target-hardware recommendation**, not an OS-independent default. Memory varied by tens of MiB and remained below ~874 MiB sampled process-tree RSS in this test.
- More CPU threads greatly accelerated the two ARNet passes and Depth/SAM/ViTMatte neural inference. The CPU optical scatter was NOT faster with four threads; it is the next optimization target.
- RGB output at thread 2 and 4 differed from the single-thread reference by no more than one 8-bit level. The fraction of pixels with any changed channel was under 0.005%. The same-thread repeated PNG hashes match each other.
- The same source/input at larger 20MP resolutions, other CPUs, native Linux cgroups, GPU providers, and concurrent HTTP jobs were **not** validated. Sampled process RSS is not cgroup high-water and excludes unrelated OS allocations.
- No production-default thread change should be made based on one photo and one CPU alone; this data validates manual thread selection and future measured auto-tuning.
