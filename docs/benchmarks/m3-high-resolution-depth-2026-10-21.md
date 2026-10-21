# 4K source resolution depth worker experiment

## Question

Measure the complete CPU image-to-depth CLI worker on a 3840×2160 input, keep
the returned map at that exact size, and compare the dynamic full-attention
graph with the exact global query-chunk-128 graph.

## Setup

- Host: Apple MacBook Air M1, 8 GiB RAM, macOS 15.7.7, arm64.
- Runtime: Python 3.11.16, NumPy 2.4.6, ONNX Runtime 1.31.0, OpenCV 5.0.0.93.
- Provider and threads: CPUExecutionProvider, one intra-op thread.
- Input: seeded synthetic RGB PNG, 3840×2160, fresh CLI worker for each trial.
- Sampling: process-tree RSS every 5 ms; process-tree USS and cgroup v2 are not
  available on this macOS host.
- Output: full 2160×3840 float32 depth map, 33,177,728-byte `.npy` file.
- Network shape: 518×924 under the pinned upstream aspect-preserving resize
  rule. The network raster is smaller than the source, while the returned depth
  map is restored to full source dimensions.
- Baseline graph SHA-256: `79094f23efa5e1df4c7612d4c476020f526f5837d904787358ac160ecb969fce`.
- Query-chunk-128 graph SHA-256:
  `3e7a177a7e11deb92385bc017355ac14bdffcaf89155ab78fd99851803f3ba99`.

## Results

Three fresh-worker trials per graph:

| Graph | Peak RSS trials | Median peak RSS | Median inference | Median adapter total | Median worker wall |
| --- | --- | ---: | ---: | ---: | ---: |
| Full attention | 762.2, 829.2, 622.8 MiB | 762.2 MiB (799,178,752 B) | 3.559 s | 3.740 s | 4.437 s |
| Query chunk 128 | 673.8, 598.4, 597.8 MiB | 598.4 MiB (627,425,280 B) | 3.661 s | 3.839 s | 4.399 s |

The q128 median worker used 21.5% less RSS in this sample. Its median model
inference was about 2.9% slower, while end-to-end wall time was effectively
unchanged at these run counts. Do not infer a speed improvement from these
measurements.

The q128 output was separately compared with the pinned PyTorch reference on a
seeded 3840×2160 RGB raster. Preprocessing max absolute error was `1.19e-6`;
restored full-resolution depth max/mean/p99 errors were `6.91e-5` /
`1.75e-6` / `4.53e-6` (max limit `1e-4`). The output was finite and had the
original 2160×3840 shape.

## Reproduction

With validated dynamic baseline and q128 graphs available outside the Git
checkout, run:

```bash
PYTHONPATH=src python scripts/benchmark_depth_inference.py \
  --model /path/to/depth-anything-v2-small-dynamic.onnx \
  --width 3840 --height 2160 --threads 1 --sample-interval-ms 5

PYTHONPATH=src python scripts/benchmark_depth_inference.py \
  --model /path/to/depth-anything-v2-small-dynamic-q128.onnx \
  --width 3840 --height 2160 --threads 1 --sample-interval-ms 5
```

The opt-in `model-smoke` GitHub Actions workflow also runs these two worker
measurements on Linux, so future changes can be compared on the same runner
class. Its first 4K run (Ubuntu 24.04 x86_64, Python 3.11.17, NumPy 2.4.6,
ONNX Runtime 1.31.0, OpenCV 5.0.0.93, one thread) returned full-size finite
maps for both graphs. One fresh worker per graph measured:

| Graph | Peak RSS | Inference | Worker wall |
| --- | ---: | ---: | ---: |
| Full attention | 731,766,784 B (697.9 MiB) | 3.454 s | 4.491 s |
| Query chunk 128 | 556,109,824 B (530.3 MiB) | 3.484 s | 4.519 s |

The cgroup v2 limit was `max`; this single pair showed 24.0% lower q128 RSS,
with similar timing. It is not a constrained-memory comparison.

## Limits

The input is synthetic, and the experiment measures only depth estimation. It
does not establish real-photo depth quality, behavior under a fixed Linux
cgroup memory limit, or memory use for the full color/bokeh/subject pipeline.
The Mac had no enforced 4 GiB or 2 GiB process budget. This is one model stage,
not evidence that the complete Laika workflow meets either target.
