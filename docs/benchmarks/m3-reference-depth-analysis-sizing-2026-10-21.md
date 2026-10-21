# Reference depth analysis sizing and 4K worker comparison

## Sizing policy

The reference `NativeDepthMath` contract caps the analysis image's long edge at
1220 pixels, keeps smaller images at their source size, and pads the resized
raster on the right and bottom to a multiple of 14. Its resampling notes
describe Pillow-compatible 8-bit bicubic filtering with antialiasing during
downscaling. Laika implements this contract independently and keeps the
upstream model wrapper's lower-bound resize available with `--input-size 518`.
The normal `laika depth` mode does not change the returned image dimensions.

For a 3840×2160 source, the selected analysis raster is 1220×686 and the padded
network tensor is 1232×686. The depth output remains 3840×2160 float32. The
previous 4K reports used the explicit lower-bound mode and a 924×518 network
raster; they are retained as historical comparisons in the
[high-resolution report](m3-high-resolution-depth-2026-10-21.md) and
[chunk-sweep report](m3-depth-attention-chunk-sweep-2026-10-21.md).

## PyTorch parity

The pinned Depth Anything V2 Small implementation and checkpoint were run on
the same prepared NCHW tensor as the q128 ONNX graph. This isolates ONNX
inference and output restoration from differences in the resize profile.

| Input | Analysis | Network | Output | Max / mean / p99 absolute depth error |
| --- | ---: | ---: | ---: | ---: |
| 225×231 | 225×231 | 238×238 | 225×231 | 6.68e-6 / 3.20e-6 / 5.48e-6 |
| 3840×2160 | 1220×686 | 1232×686 | 3840×2160 | 8.11e-6 / 2.84e-6 / 5.01e-6 |

The output was finite and retained the full source dimensions. The comparison
does not by itself establish parity of a production Kotlin build or visual
quality on real photographs. The optional model-smoke workflow now validates
the preprocessing path for small odd dimensions and this 4K profile.

## Full CLI worker measurements

Three fresh CLI processes per graph processed the same seeded synthetic RGB
3840×2160 input on an Apple MacBook Air M1 (8 GiB), macOS 15.7.7, arm64, Python
3.11.16, NumPy 2.4.6, OpenCV 5.0.0.93, ONNX Runtime 1.31.0, CPUExecutionProvider,
one intra-op thread. The model graph and source dimensions were the same across
trials. Process-tree RSS was sampled every 5 ms; USS and cgroup metrics were
unavailable. RSS trial order is the order the workers were run.

| Graph | Peak RSS trials | Median peak RSS | Inference trials | Median inference | Worker wall trials | Median worker wall |
| --- | --- | ---: | --- | ---: | --- | ---: |
| Full attention | 1,318,633,472; 1,179,484,160; 1,162,772,480 B | 1,179,484,160 B (1,124.8 MiB) | 8.570; 9.648; 9.634 s | 9.634 s | 9.428; 10.631; 10.479 s | 10.479 s |
| Query chunk 128 | 633,896,960; 746,569,728; 742,752,256 B | 742,752,256 B (708.3 MiB) | 9.141; 8.865; 8.762 s | 8.865 s | 10.002; 9.735; 9.610 s | 9.735 s |

The q128 median worker used 37.0% less RSS in this sample. Median inference
time was 8.0% lower and median worker wall time 7.1% lower, but these three
trials ran on an unconstrained machine with variable available memory and do
not establish a general CPU speedup. The output was a 33,177,728-byte float32
`.npy` file in every trial. The CLI worker includes image decode, model loading,
preprocessing, inference, full-size restoration, and output writing.

## Reproduction

Keep the validated ONNX graphs and their `.validation.json` files outside the
repository, then run each command three times in sequence:

```bash
PYTHONPATH=src python scripts/benchmark_depth_inference.py \
  --model /path/to/depth-anything-v2-small-dynamic.onnx \
  --width 3840 --height 2160 --analysis-max-edge 1220 --threads 1

PYTHONPATH=src python scripts/benchmark_depth_inference.py \
  --model /path/to/depth-anything-v2-small-dynamic-q128.onnx \
  --width 3840 --height 2160 --analysis-max-edge 1220 --threads 1
```

The Mac measurements used the temporary model environment and graph artifacts
listed in the development log; they were not collected in a memory cgroup. The
optional GitHub model smoke repeats reference-profile measurements on Linux
and records the cgroup state. Its runner currently has no fixed memory cap.

## Limits

These are synthetic, depth-only results. They do not measure image color,
segmentation, matting, bokeh, composition, concurrent requests, or the complete
processing pipeline. They do not establish the 8/4/2 GiB target. The chunk
sweep also shows that the best chunk size depends on machine and input profile;
the inference engine does not silently select q128 for every deployment.
