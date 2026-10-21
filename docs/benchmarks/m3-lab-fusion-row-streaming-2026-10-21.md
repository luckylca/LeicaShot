# Lab-fusion row-streaming benchmark

**Date:** 2026-10-21. This measures only RGB/Lab detail fusion, without a color
prediction model.

## Method

The benchmark runs every trial/profile in a fresh Python worker, samples the
worker process tree at 5 ms, and compares each streamed output against the
full-height profile for the same deterministic synthetic RGB arrays. Every
profile used one OpenCV thread and produced a byte-identical output. The input
and prediction arrays were generated in the worker; no photo or model weights
were read.

Reproduce the primary 4K run with:

```bash
python scripts/benchmark_lab_fusion.py \
  --width 3840 --height 2160 \
  --row-chunk-heights 128 256 512 1024 \
  --trials 3 --threads 1 \
  --output /tmp/laika-lab-fusion.json
```

Environment: MacBook Air M1, 8 GB RAM, macOS 15.7.7, Python 3.11.16, NumPy
2.4.6, OpenCV 5.0.0, 8 logical CPUs, one OpenCV thread. macOS did not expose
process USS or cgroup v2 memory peak in this run.

## 3840×2160

| Row chunk | Median worker-tree RSS | Median fusion time | Max pixel error vs full frame |
|---:|---:|---:|---:|
| 2160 (full frame) | 263,372,800 B (251.1 MiB) | 1.897 s | 0 |
| 128 | 179,585,024 B (171.2 MiB) | 4.570 s | 0 |
| 256 | 242,728,960 B (231.5 MiB) | 3.217 s | 0 |
| 512 | 233,750,528 B (223.0 MiB) | 2.548 s | 0 |
| 1024 | 279,363,584 B (266.4 MiB) | 2.165 s | 0 |

At this resolution, 512-row chunks reduced median sampled RSS by 11.2% and took
34.4% longer than the full-height profile. 128-row chunks reduced it by 31.8%
and took 2.4× as long. The 1024-row profile did not lower the measured median.

## 2048×1536

A separate three-trial run used the same method and machine:

| Row chunk | Median worker-tree RSS | Median fusion time | Max pixel error vs full frame |
|---:|---:|---:|---:|
| 1536 (full frame) | 143,032,320 B (136.4 MiB) | 0.799 s | 0 |
| 128 | 127,778,816 B (121.8 MiB) | 1.299 s | 0 |
| 256 | 145,293,312 B (138.6 MiB) | 0.939 s | 0 |
| 512 | 149,733,376 B (142.8 MiB) | 0.978 s | 0 |
| 1024 | 161,136,640 B (153.7 MiB) | 0.884 s | 0 |

This smaller raster did not show a stable benefit from streaming. The two
resolution samples disagree on which chunk sizes reduce sampled peak RSS, so
the chunk height remains configurable rather than claimed as a universal win.

## Limits

The sampling interval can miss short allocations. Worker RSS includes Python,
OpenCV, synthetic input allocation, fusion, and `.npy` output saving; fusion
time excludes setup and saving. The benchmark uses synthetic arrays, so it
measures math cost and buffer behavior, not image quality. It does not include
AI inference, image decode, API task overhead, a Linux cgroup cap, or the
complete Laika pipeline. No 8/4/2 GiB service claim follows from this result.
