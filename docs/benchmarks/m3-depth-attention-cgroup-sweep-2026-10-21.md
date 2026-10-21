# Depth attention chunks under enforced Linux memory limits

## Scope and setup

Compare the unchunked Depth Anything V2 Small graph with exact global
attention query chunks 16, 32, 64, and 128. Each query chunk attends across
the complete K/V set. Every graph processed the same seeded 3840×2160 RGB
input through the full CLI, with a 1220×686 analysis raster, 1232×686 padded
network tensor, and 3840×2160 float32 output. For each graph and memory cap,
three fresh Docker containers ran with swap disabled and two CPUs. The two
cgroup v2 caps were 4 GiB and 2 GiB.

Environment: GitHub-hosted Linux x86_64 runner, Linux 6.17.0-1022-azure,
Python 3.11.17, NumPy 2.4.6, ONNX Runtime 1.31.0, OpenCV 5.0.0.93,
CPUExecutionProvider, one intra-op thread. The container also captures
benchmark-harness and temporary-file memory in its cgroup peak; worker RSS is
sampled separately for the fresh CLI subprocess. The run produced 30 JSON
reports, attached to the [model-smoke workflow](https://github.com/luckylca/LeicaShot/actions/runs/39597736919).

## Results

All 30 trials completed, returned full-resolution finite depth maps, and
reported zero cgroup OOM kills. Values below are medians of three trials;
memory values show cgroup `memory.peak` and CLI process-tree RSS.

| Cap | Graph | Cgroup peak | CLI process-tree RSS | Median inference | Median worker wall |
| --- | --- | ---: | ---: | ---: | ---: |
| 4 GiB | Full attention | 1,394,507,776 B (1,329.9 MiB) | 1,355,390,976 B (1,292.6 MiB) | 3.972 s | 4.986 s |
| 4 GiB | q16 | 724,340,736 B (690.8 MiB) | 686,538,752 B (654.7 MiB) | 7.184 s | 8.108 s |
| 4 GiB | q32 | 697,040,896 B (664.8 MiB) | 659,648,512 B (629.1 MiB) | 5.433 s | 6.375 s |
| 4 GiB | q64 | 717,438,976 B (684.2 MiB) | 679,231,488 B (647.8 MiB) | 4.472 s | 5.423 s |
| 4 GiB | q128 | 730,693,632 B (696.8 MiB) | 690,683,904 B (658.7 MiB) | 4.041 s | 5.012 s |
| 2 GiB | Full attention | 1,392,283,648 B (1,327.8 MiB) | 1,348,493,312 B (1,286.0 MiB) | 3.889 s | 4.885 s |
| 2 GiB | q16 | 724,594,688 B (691.0 MiB) | 686,260,224 B (654.5 MiB) | 7.370 s | 8.300 s |
| 2 GiB | q32 | 724,197,376 B (690.6 MiB) | 686,497,792 B (654.7 MiB) | 5.519 s | 6.482 s |
| 2 GiB | q64 | 726,687,744 B (693.0 MiB) | 688,984,064 B (657.1 MiB) | 4.477 s | 5.490 s |
| 2 GiB | q128 | 703,885,312 B (671.3 MiB) | 666,308,608 B (635.4 MiB) | 4.011 s | 4.955 s |

Within this run, q32 had the lowest cgroup peak at 4 GiB (664.8 MiB), while
q128 had the shortest median inference (4.041 s). Under 2 GiB, q128 had both
the lowest cgroup peak (671.3 MiB) and shortest inference (4.011 s). q16 was
the slowest chunk at both limits. All chunked graphs used less than the
baseline's cgroup peak, but their exact memory ranking changed with the limit.

## Accuracy evidence

The exporter checked each chunked graph against the pinned PyTorch model on
four dynamic tensor shapes; the earlier chunk-sweep report records a maximum
ONNX absolute error of `1.67e-5`. The reference-profile 4K PyTorch comparison
for q128 had max/mean/p99 error `8.11e-6` / `2.84e-6` / `5.01e-6`. The cgroup
trials themselves check output shape and finiteness, not per-pixel quality.
No real-photo bokeh or full-pipeline image-quality comparison is included.

## Interpretation and limits

The enforced memory experiment reinforces a configurable chunk profile: q128
was fastest in this Linux 4K setup, q32 used the least memory at 4 GiB, and
q128 had the best measured memory/time pair under 2 GiB. Differences are
runner- and runtime-specific. Keep the chunk explicit until results are
repeated on target CPU classes and with the complete image pipeline. These
depth-only trials do not establish a 2 GiB minimum server configuration.
