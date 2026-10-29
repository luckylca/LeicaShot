# Full five-model Windows CPU thread sweep, 1–16 threads — 2026-10-25

## Goal / methods

Find fast per-request CPU settings while maintaining original photo quality and minimal peak RSS, without removing any model or image-resolution detail. Tests run on a Windows Intel Core i9-12900H (14 physical CPU cores/20 logical processors), Python 3.11, ONNX Runtime CPUExecutionProvider, one 2048x1365 genuine CC0 portrait. Each of eleven thread counts has two independent fresh-process full M9 + Depth + SAM Tiny + ViTMatte Small + ARNet dual-pass + optical scatter + PNG exports. Within each group, order is ascending on pass 1 and reversed on pass 2.

All tests used 512 matte tiles, verified split ARNet graph core edge 256 and SAM CPU arena disabled; model weights were unchanged and remain local to the user. The chosen thread count was passed consistently to the Python unified engine's model session factories. Entire processing and peak process-tree RSS were measured. Input image and all outputs stayed at the requested full 2048x1365 resolution.

| ONNX CPU threads | Full times (s) | Median (s) | Speedup vs 1 thread | Peak sampled RSS (MiB) |
|---:|---|---:|---:|---:|
| 1 | 67.90, 69.27 | **68.59** | 1.00x | 867.8 |
| 2 | 48.66, 52.73 | **50.70** | 1.35x | 862.5 |
| 4 | 45.76, 45.74 | **45.75** | 1.50x | 873.6 |
| 6 | 41.97, 43.08 | **42.53** | 1.61x | 907.5 |
| 8 | 42.73, 40.61 | **41.67** | 1.65x | 935.9 |
| 9 | 32.83, 33.61 | **33.22** | 2.06x | 853.3 |
| 10 | 32.83, 33.97 | **33.40** | 2.05x | 936.7 |
| 11 | 33.76, 33.93 | **33.84** | 2.03x | 900.8 |
| 12 | 35.47, 36.08 | **35.77** | 1.92x | 936.2 |
| 14 | 34.64, 35.07 | **34.86** | 1.97x | 883.7 |
| 16 | 40.37, 36.82 | **38.59** | 1.78x | 938.9 |

**Observed fastest: 9 ONNX threads**, median 33.22s versus the one-thread baseline 68.59s (full-task duration reduction 51.6%). The difference between 9 and 10 threads is much smaller than run-to-run effects and should not be treated as a universal ranking; 9–10 is the sensible local performance region.

## Why extra threads help

| Stage (median s) | 1 thread | 9 threads | 10 threads |
|---|---:|---:|---:|
| bokeh.arnet_pass_1 | 15.89 | 4.29 | 4.55 |
| bokeh.arnet_pass_2 | 7.77 | 2.33 | 2.49 |
| bokeh.highlight_scatter | 9.39 | 8.18 | 8.09 |
| depth.inference | 11.14 | 3.32 | 3.32 |
| subject.sam_encoder | 3.88 | 1.25 | 1.19 |
| subject.vitmatte | 4.76 | 1.62 | 1.61 |
| color.inference | 3.25 | 1.04 | 1.03 |

Neural inference (ARNet, Depth, SAM and matting) improves strongly with more threads, whereas optical scatter stays relatively expensive. Above roughly 10–12 threads further ONNX workers can contend for cache/memory bandwidth, so 16 threads was slower than 9 or 10.

## Photo quality / memory

Exact decoded RGB output comparisons against the 1-thread photo were performed for every listed high-thread candidate. No candidate changed any RGB channel by more than one 8-bit level, and fewer than 0.007% of output pixels had any differing channel. Pixel-level differences arise from CPU floating-point execution order rather than dropped models, lower output resolution or modified aperture optics.

All these 2.8MP full-engine trials sampled process-tree RSS below approximately 940 MiB. The 9-thread max observed RSS was lower than the 10-thread max, but the two-run sample is too small to establish a robust memory ranking. The 2 GiB budget in the benchmark is a flag/check, NOT a strict Linux cgroup enforcement; native full 20MP and concurrency tests are separate validation items.

## Deployment recommendation (specific to this Windows CPU)

Run your Python API / CLI with LAIKA_API_THREADS=9 (or test 10 in your own workloads) and LAIKA_API_MAX_WORKERS=1 while memory is constrained. Keep low-memory strategy, SAM no-arena, and verified ARNet split assets. The container's CPU quota must permit actual parallel execution; a LAIKA_CPU_LIMIT=2 Docker allocation cannot be expected to reproduce a bare-metal 9-thread result. Do not hardcode 9 globally for Mac, low-core Linux servers, containers, or multiworker deployments; run this matrix once per host/hardware configuration.

## Reproduction and provenance

Tracked research tool: scripts/benchmark_speed_matrix.py. Raw JSON and rendered PNGs are local files under E:\codex\LeicaShot-local-data\benchmarks\speed-matrix-20261025-edge256* and intentionally excluded from Git. No external image/checkpoint redistribution. Real full-photo metrics were recorded by the Python engine, not inferred from individual model FLOPs or synthetic samples.
