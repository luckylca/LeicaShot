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
