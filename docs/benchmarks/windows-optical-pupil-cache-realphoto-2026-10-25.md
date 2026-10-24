# Verified real-photo optical pupil-cache experiment — Windows 2026-10-25

## Tested algorithm

Reuse 5x5 field-optical pupil kernels across opposite-signed CoC layers
with matching absolute radii. Cache lifetime is one scatter render.
Unique allocated kernel arrays are charged against a capped 32 MiB budget,
and radius caches are only populated when the radius appears twice.
A private cache-budget=0 comparison arm runs the original uncached code.
Neither model weights nor output image size changes.

## Test setup

Host: Windows 11, Intel Core i9-12900H (14 cores / 20 logical processors),
Python 3.11 and ONNX Runtime 1.31 CPUExecutionProvider.
Real CC0 portrait, 2048x1365, M9 color + Depth Anything V2 Small + SAM2 Tiny
+ ViTMatte Small + two-pass ARNet + actual optics and image export.
Both arms: 4 ONNX threads, verified low-memory ARNet graph partition,
SAM CPU Arena disabled, 512 ViTMatte tiles, existing exact-support
optical scheduling. Two independent fresh workers in order
uncached/cached, then cached/uncached.

The benchmark intercepted the actual laika.highlights.scatter_highlights
binding used by the bokeh code (it is imported within render_bokeh).
The no-cache arm passed _pupil_cache_budget_bytes=0; the cached arm passed
33554432. A traced wrapper also counted _field_pupil_kernels calls
to verify that the new behavior was genuinely selected.

## Real full-request measurements

| Pair order | No-cache full time | Cache full time | No-cache scatter | Cache scatter |
|---|---:|---:|---:|---:|
| no-cache / cache | 42.7335s | 42.4330s | 9.7747s | 9.2136s |
| cache / no-cache | 44.1852s | 43.2825s | 9.5047s | 9.6558s |
| Mean | 43.4594s | 42.8578s | 9.6397s | 9.4347s |

Observed mean full request time reduction was approximately 0.60s (1.4%).
Mean scatter reduction was only approximately 0.20s (2.1%).
The run-to-run scatter noise is comparable to this difference; do not
claim a universally repeatable end-to-end throughput uplift.

Distinct kernel-generation calls: uncached 63, cached 34 in both pairs.
Actual full RGB PNGs are byte-identical across modes, with SHA256
69072c96c9a90f236000264101a62d74efbcbbd2f37f2adfc90332022923a352.
Max channel error zero, zero changed pixels. No resolution downscaling.

Sampled full-worker RSS: uncached 831,614,976 and 911,200,256 bytes;
cached 831,213,568 and 830,795,776 bytes. The no-cache second run
is an outlier; cannot infer a reliably lower complete-request RSS from
these samples, and a 32 MiB budget is a MAXIMUM additional kernel cache,
not necessarily the actual usage.

## What is and is not proven

The GitHub synthetic scatter CPU microbenchmark showed about 1.6x to 1.9x
speedup across Windows, macOS and Ubuntu on small sparse artificial inputs,
all bitwise identical. The matched real-photo test above instead gives
only ~1.4% full-pipeline reduction: kernel creation is now a small fraction
of rendering time, while filtering and neural models dominate.

Future priority is whole-frame convolution work and ONNX CPU thread
configuration. This experiment does not validate native 20 MP photographs,
strict 2 GiB Linux cgroups, parallel FastAPI workers or other photo scenes.

Reproduction assets (not tracked or distributed):
E:\codex\LeicaShot-local-data\benchmark-real-optical-cache.py
E:\codex\LeicaShot-local-data\validation\portrait-cc0-2048.jpg

This performance patch remains independent from the separate
research/full-pipeline-speed-matrix branch until its real-photo
speed/memory and CI merge gates are reviewed.
