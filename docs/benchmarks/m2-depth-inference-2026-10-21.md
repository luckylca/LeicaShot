# M2 Depth Anything V2 Small Python inference adapter

## Implemented path

`laika.depth.DepthAnythingV2Small` accepts an HWC `uint8` RGB array, applies
the pinned Depth Anything V2 lower-bound aspect-preserving resize, OpenCV
bicubic interpolation, ImageNet normalization, and NCHW layout, then runs a
validated dynamic ONNX graph on CPU. It upsamples the depth map to the exact
source width and height using bilinear `align_corners=True` coordinates in
bounded row blocks. Preprocessing converts one source channel to float32 at a
time and writes into a preallocated NCHW tensor, avoiding a full-image
three-channel float32 copy. The session owns no image cache and can be released
with `close()` or a context manager.

`laika depth input.jpg depth.npy --model depth.onnx` validates the decoded
format, file bytes, pixel count, EXIF orientation, and derived network pixel
budget. It writes the raw float32 full-resolution map to an atomic `.npy`
output. The default limits are 50 MiB upload bytes, 40 million source pixels,
and 4 million derived network pixels; requests above a limit fail instead of
silently changing their analysis dimensions.

## Numerical comparison

The pinned upstream PyTorch reference and the Laika CPU adapter were run on the
same seeded 700×400 RGB raster using a query-chunk-128 dynamic ONNX graph:

- The derived network dimensions were 910×518, matching the upstream resize.
- Preprocessed tensor maximum absolute difference: `1.19e-6`.
- Restored 700×400 depth maximum / mean / p99 absolute differences:
  `6.15e-5` / `3.07e-6` / `5.01e-6` (acceptance limit `1e-4`).
- Every output value was finite, and the `.npy` array was float32 with the
  original image dimensions.
- The custom bounded-block align-corners resize matched PyTorch on numeric
  fixtures with maximum absolute error `1.19e-7`.

The GitHub Linux x86_64 model smoke also compared this path with the same pinned
toolchain. It measured preprocessing max absolute error `1.58e-4` and restored
depth max/mean/p99 errors `5.15e-5` / `5.58e-6` / `1.53e-5`. The preprocessor
uses OpenCV bicubic interpolation, whose floating-point results vary slightly
across CPU architectures. The cross-platform preprocessing tolerance is now
`2e-4`; the final depth output limit remains `1e-4`. The Linux depth output
remained inside that unchanged limit.

## Linux model-smoke worker sample

The opt-in GitHub Actions run also measured one fresh CLI worker for each graph
on Linux x86_64 (Ubuntu 24.04, Python 3.11.17, NumPy 2.4.6, ONNX Runtime
1.31.0, OpenCV 5.0.0.93, one thread). The runner sampled process-tree RSS/USS
every 5 ms on the seeded 1400×518 input; cgroup v2 was visible, but
`memory.max` was `max`.

| Graph | Peak worker RSS | Peak worker USS | Inference | Adapter total | Worker wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full attention baseline | 1,114,427,392 B (1,062.4 MiB) | 1,070,276,608 B (1,020.8 MiB) | 6.932 s | 6.943 s | 7.771 s |
| Query chunk 128 | 622,305,280 B (593.3 MiB) | 578,101,248 B (551.2 MiB) | 7.095 s | 7.099 s | 7.892 s |

This single paired sample had 44.2% lower peak RSS for q128 and about 1.6%
longer worker wall time. It is a smoke measurement, not a repeated performance
comparison, and the cgroup was not memory constrained. These results do not
demonstrate the 4 GiB or 2 GiB full-pipeline target.

## Full CLI worker measurements

Three fresh worker trials per graph were measured on an 8 GiB MacBook Air M1
(macOS 15.7.7, arm64, Python 3.11.16, ONNX Runtime 1.31.0, one CPU thread).
The benchmark generated the same 1400×518 RGB PNG before each run, then sampled
the CLI worker process tree every 5 ms. Each worker includes decode, session
load, preprocessing, inference, full-size restoration, and NumPy output write.

| Graph | Median worker RSS | RSS range | Median inference | Median adapter total | Median worker wall | Output |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Full attention baseline | 882,180,096 B (841.3 MiB) | 832.2–883.4 MiB | 7.744 s | 7.749 s | 8.186 s | 1400×518 float32 |
| Query chunk 128 | 453,623,808 B (432.6 MiB) | 410.5–453.8 MiB | 7.229 s | 7.233 s | 7.733 s | 1400×518 float32 |

The chunked median RSS was 48.6% lower in this three-run sample. Worker time
varied between trials; these shared-host measurements do not support a CPU
speed claim. At the same-size 1400×518 input, preprocessing took 2.4–4.1 ms
and restoring the output size took under 1.2 ms. macOS USS and cgroup readings
were unavailable. See the separate
[direct attention experiment](m1-depth-query-chunking-2026-10-21.md) for its
32/64/128 query-size measurements.

## Limits

All inputs are synthetic. This confirms one model stage and its Python image
adapter; it does not measure real-photo depth quality, the color/bokeh/subject
pipeline, HTTP requests, concurrent tasks, or a Linux cgroup memory limit. No
8/4/2 GiB end-to-end target is claimed. The class, CLI, and validation scripts
are now available, but this adapter is not yet part of a complete edit request.
