# M1 Depth Anything V2 global query chunking

## Goal and method

Depth Anything V2's DINO attention materializes a `heads × queries × keys`
score tensor. This experiment divides only the query axis into fixed chunks.
Every query chunk still attends to the complete key and value tensors, and its
softmax still spans the full key axis; this is global attention, not local
window attention. The outputs are concatenated in their original token order.

The exporter uses a scripted loop and refuses to publish a chunked ONNX graph
unless the graph contains a top-level ONNX `Loop` for each of the 12 attention
blocks. It checks PyTorch against the unmodified attention and checks ONNX
Runtime against PyTorch on deterministic inputs at three dynamic shapes.

## Setup

- Host: Apple MacBook Air M1, 8 GiB RAM, macOS 15.7.7, arm64.
- Runtime: Python 3.11.16, PyTorch 2.14.1, ONNX 1.23.2, ONNX Runtime 1.31.0.
- ONNX Runtime provider: CPUExecutionProvider, one intra-op thread.
- Measurement: fresh worker process per run; 5 ms process-tree RSS sampling;
  one warmup and three measured inferences on a 1400×518 seeded synthetic
  float32 tensor. USS and cgroup readings are unavailable on this host.
- Model: manifest-pinned Depth Anything V2 Small checkpoint and upstream
  source revision. Model files and exports stayed outside the repository.

## Results

| Query chunk | Peak process-tree RSS | Median inference | RSS change vs baseline | Time change vs baseline |
| ---: | ---: | ---: | ---: | ---: |
| Full attention baseline | 1,642,561,536 B (1,566.6 MiB) | 6.560 s | — | — |
| 32 | 656,801,792 B (626.4 MiB) | 7.734 s | −60.0% | +17.9% |
| 64 | 639,533,056 B (609.8 MiB) | 7.152 s | −61.1% | +9.0% |
| 128 | 638,566,400 B (608.9 MiB) | 6.794 s | −61.1% | +3.6% |

All three chunk sizes produced zero maximum absolute difference from the
unmodified PyTorch attention in the tested shapes. The ONNX output's maximum
absolute difference from its PyTorch reference was `1.67e-5` at most, below the
`1e-4` acceptance threshold. The 128-query graph was fastest among the chunked
variants and used essentially the same peak RSS as 64, so 128 is the current
experimental choice.

## Linux opt-in model smoke

The opt-in `model-smoke` workflow compared the two graphs twice on x86_64 Linux
hosted runners (4 logical CPUs, Python 3.11.17, ONNX Runtime 1.31.0, one
inference thread). Each run used a 1400×518 tensor, one warmup, three measured
inferences, and 5 ms process-tree RSS sampling:

| Trial | Graph | Peak process-tree RSS | Median inference |
| ---: | --- | ---: | ---: |
| 1 | Full attention baseline | 1,811,435,520 B (1,727.5 MiB) | 6.079 s |
| 1 | Query chunk 128 | 970,530,816 B (925.6 MiB) | 6.633 s |
| 2 | Full attention baseline | 1,820,291,072 B (1,736.0 MiB) | 5.060 s |
| 2 | Query chunk 128 | 979,382,272 B (934.0 MiB) | 4.809 s |

Across both Linux runs the chunked graph used 46.2–46.4% less RSS. Latency
varied: +9.1% in the first run and −5.0% in the second, so these shared-runner
timings do not establish a speed change. The ONNX max absolute error was
`1.62e-5`, and the graph retained all 12 loops. Neither runner had a cgroup
memory limit; these are cross-platform model smokes, not low-memory constrained
system tests.

## Limits

These are single-stage model tests on one 8 GiB Mac and one hosted Linux runner,
not repeated hardware trials. Host available memory varied between Mac runs.
The tensor is synthetic;
preprocessing, image decoding, output interpolation to original photo size,
other models, and full-pipeline memory are excluded. This experiment does not
establish a 4 GiB or 2 GiB end-to-end target, nor does it measure photo quality.
The dynamic chunked graph is validated, but a production depth engine has not
yet integrated it. The regular model-free CI does not download model weights.
