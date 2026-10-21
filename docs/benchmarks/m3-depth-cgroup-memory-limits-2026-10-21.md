# Depth worker under enforced Linux memory limits

## Scope

This experiment runs only the image-to-depth CLI on a 3840×2160 seeded
synthetic RGB image. It compares the unchunked full-attention graph with the
exact global query-chunk-128 graph under separate Docker cgroup v2 limits of
4 GiB and 2 GiB. Each graph/limit combination ran in three fresh containers.
Swap was disabled, the container was limited to two CPUs, and each trial
returned a finite 3840×2160 float32 depth map. The experiment does not exercise
color, subject masks, matting, bokeh, composition, concurrent requests, or a
full Laika edit pipeline.

## Environment and artifacts

- GitHub-hosted Linux x86_64 runner, kernel `6.17.0-1022-azure`, glibc 2.36,
  Python 3.11.17.
- Isolated `python:3.11-slim-bookworm` container, two CPU quota, no network,
  read-only workspace/model mounts, and a 512 MiB `/tmp` tmpfs.
- NumPy 2.4.6, ONNX Runtime 1.31.0, OpenCV 5.0.0.93, CPUExecutionProvider,
  one inference thread.
- Source 3840×2160; analysis 1220×686; padded network tensor 1232×686; output
  3840×2160 float32 (33,177,728 bytes).
- Baseline ONNX SHA-256:
  `79094f23efa5e1df4c7612d4c476020f526f5837d904787358ac160ecb969fce`.
- q128 ONNX SHA-256:
  `3e7a177a7e11deb92385bc017355ac14bdffcaf89155ab78fd99851803f3ba99`.
- All 12 JSON trial reports are attached to the [model-smoke run](https://github.com/luckylca/LeicaShot/actions/runs/39595857119).

The cgroup peak includes the benchmark harness, CLI worker, and temporary
input/output pages. Process-tree RSS is sampled separately for the fresh CLI
worker. Cgroup `memory.max`, `memory.peak`, and `memory.events` were recorded;
all twelve runs reported zero OOM kills.

## Results

Values are medians of three fresh containers for each graph and limit. RSS and
cgroup peak are shown in bytes and MiB; elapsed times are seconds.

| Limit | Graph | Runs | Median cgroup peak | Median CLI process-tree RSS | Median inference | Median worker wall | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 4 GiB | Full attention | 3 | 1,428,271,104 B (1,362.1 MiB) | 1,383,940,096 B (1,319.8 MiB) | 3.373 s | 4.123 s | 3/3 complete |
| 4 GiB | q128 | 3 | 726,765,568 B (693.1 MiB) | 684,527,616 B (652.8 MiB) | 3.950 s | 4.782 s | 3/3 complete |
| 2 GiB | Full attention | 3 | 1,428,115,456 B (1,362.0 MiB) | 1,384,034,304 B (1,319.9 MiB) | 3.460 s | 4.227 s | 3/3 complete |
| 2 GiB | q128 | 3 | 733,523,968 B (699.5 MiB) | 691,331,072 B (659.3 MiB) | 3.838 s | 4.620 s | 3/3 complete |

In this runner, q128 reduced median cgroup peak by 49.1% under the 4 GiB cap
and 48.6% under 2 GiB. Median inference was 17.1% slower at 4 GiB and 10.9%
slower at 2 GiB. Wall-time differences include fresh CLI loading, decode,
preprocessing, restore, and output write; they are not a general CPU speed
claim. The result supports keeping attention chunk size configurable.

## Interpretation and limits

This is an initial depth-only feasibility result: both graphs fit the enforced
2 GiB limit on this Linux runner, and q128 uses about half the cgroup peak at
the cost of slower inference. It does not establish that 2 GiB is a supported
minimum server size. The complete pipeline will hold other stages and
intermediates, may use different models, and has not been measured under this
cap. Re-run this workflow when the depth adapter or runtime dependencies
change, then repeat with the complete image pipeline once it exists.
