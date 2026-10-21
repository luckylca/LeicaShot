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
- Direct-model measurement: fresh worker process; 5 ms process-tree RSS
  sampling; one warmup and three measured inferences on a 1400×518 seeded
  synthetic float32 tensor. USS and cgroup readings are unavailable on this host.
- CLI measurement: fresh `laika depth` process per trial, generated 1400×518
  RGB PNG, one thread, original-size float32 `.npy` output. Three trials per
  graph; each includes image decode, ONNX session load, preprocessing,
  inference, full-size restoration, and output writing.
- Model: manifest-pinned Depth Anything V2 Small checkpoint and upstream
  source revision. Model files and exports stayed outside the repository.

## Results

| Query chunk | Peak process-tree RSS | Median inference |
| ---: | ---: | ---: |
| Full attention baseline | 1,192,050,688 B (1,136.8 MiB) | 7.397 s |
| 32 | 457,261,056 B (436.1 MiB) | 8.209 s |
| 64 | 435,109,888 B (415.0 MiB) | 7.355 s |
| 128 | 487,211,008 B (464.6 MiB) | 6.792 s |

All three chunk sizes produced zero maximum absolute difference from the
unmodified PyTorch attention in the tested shapes. The ONNX output's maximum
absolute difference from its PyTorch reference was `1.67e-5` at most, below the
`1e-4` acceptance threshold. In this single direct-model sample, chunk 64 had
the lowest RSS and chunk 128 had the shortest median inference; RSS did not
decrease monotonically as chunk size changed.

## Full image-to-depth CLI trials

The full CLI ran three times per graph on the same 8 GiB Mac with newly started
workers and the same seeded image fixture:

| Graph | Median peak process-tree RSS | RSS range | Median total worker time |
| --- | ---: | ---: | ---: |
| Full attention baseline | 882,180,096 B (841.3 MiB) | 832.2–883.4 MiB | 8.186 s |
| Query chunk 128 | 453,623,808 B (432.6 MiB) | 410.5–453.8 MiB | 7.733 s |

The median RSS fell by 48.6% after changing input conversion to one float32
channel at a time. The CLI writes the full 1400×518 float32 map (2,900,928
bytes); it does not silently reduce the output dimensions. Worker time varied
between trials, so these results do not support a speed conclusion.

## Limits

These are synthetic single-stage tests on one 8 GiB Mac, not a full editing
pipeline or a constrained-memory Linux test. Host available memory varied
between Mac runs. The input is synthetic;
preprocessing, image decoding, output interpolation to original photo size,
other models, and full-pipeline memory are excluded. This experiment does not
establish a 4 GiB or 2 GiB end-to-end target, nor does it measure photo quality.
The dynamic chunked graph is validated, but a production depth engine has not
yet integrated it. The regular model-free CI does not download model weights.
