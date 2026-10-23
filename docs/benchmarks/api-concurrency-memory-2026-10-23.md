# Linux API concurrency and process memory

Date: 2026-10-23. This benchmark compares one and two simultaneous real API
tasks in a 3 GiB Linux cgroup. It exercises the public upload → task status →
download path and records process-tree RSS/USS plus cgroup v2 memory.

## Setup

- Host: MacBook Air M1, 8 GiB RAM. Linux test guest: Ubuntu 26.04 ARM64,
  4,085,092,352 B RAM, 2 CPUs. Each benchmark container had a 3 GiB memory and
  swap limit and a 2 CPU quota.
- Python 3.11.17, ONNX Runtime 1.31.0, CPUExecutionProvider, one intra-op
  thread per task, `low-memory` strategy.
- One Uvicorn process served the API. The one-task run used one processing
  worker and no waiting queue; the two-task run used two processing workers
  and no waiting queue.
- Input fixture: 1620×1080 JPEG, 385,816 bytes, SHA-256
  `85013a47c39a1eaa09b15e004c251f7777ab76d26300d7808e59b708cae35f1e`. It is
  a Lanczos/quality-94 resize of the CC0 5760×3840 mountains photograph used
  in earlier reports (source SHA-256
  `4e6661606c195b2462c0f2321a7feefe2ae2fc081499f3462a30fbb108cd7f8e`). This
  is a concurrency/resource fixture, not a full-resolution acceptance or a
  subject-edge quality test.
- Both runs requested M9 + LabFusion, Depth Anything V2, one SAM 2.1 Tiny
  positive point `(0.72, 0.66)`, ViTMatte 512 tiles, ARNet, and highlight
  rendering at the input's 1620×1080 resolution. Both downloaded the same
  1,135,208-byte PNG, SHA-256
  `9def379187b8a634a38c9b0002cb588bcd5d63e6724ba4bc59d6c50ec9dc0913`.
- Model SHA-256 values: M9
  `8936b8c86656265ab07c30b9b4cbbd7fa195d6e44cb3be1d4fd059bb36781e33`, Depth
  `a8ce30941c85b2c9af29fb26f36fbf7be3c1ba1dcf348187a51f6eaa856a238e`, ARNet
  `f0696c16a6eb02099292176f25b47e4dc53863e7e4f3f7748e66470d54cf0173`, SAM
  encoder `37de035410f11fb505c210ce1f8b10b450cb018fbfa7f226ec7e7ab542955c75`,
  SAM decoder `3a6a96c62fd39c1369641ea626e5b2e4aed10de877f85291e9993a044d5bf981`,
  and ViTMatte `a5c0b49a664867021eaf2b9cadcdc236ea511028d5e709303b418431f20e223b`.

## Results

| Run | Workers / tasks | Wall to all downloads | Server tree peak RSS | Server tree peak USS | cgroup peak / 3 GiB | OOM events |
|---|---:|---:|---:|---:|---:|---:|
| Single | 1 / 1 | 81.520 s | 1,630,384,128 B | 1,615,007,744 B | 1,867,460,608 B (58.0%) | 0 |
| Concurrent | 2 / 2 | 109.291 s | 3,141,713,920 B | 3,128,270,848 B | 3,129,069,568 B (97.14%) | 0 |

Both outputs had the same dimensions and SHA. Each task in the concurrent run
finished in about 109 seconds; highlight scatter took 26.383 s and 27.019 s
for the two tasks, compared with 20.959 s in the single-task run. Two tasks
completed in about 109 seconds versus 163 seconds for two single-task runs,
roughly 1.49× aggregate throughput in this one trial. Concurrent execution
also raised cgroup peak usage by 1,261,608,960 B (67.6%) over the single-task
peak.

The two-worker run left 92,155,904 B (87.9 MiB) below the cgroup limit. It
completed with zero `memory.events` OOM/limit events, but used 97.14% of the
available cgroup budget on a 1.75-megapixel fixture. The current API default
of one worker is appropriate for a 3 GiB service. Do not configure two workers
for this memory tier based on this result; a native-resolution task already
peaks at about 2.70 GiB in the same nominal cgroup, and larger source images or
other concurrent allocations can exhaust the remaining headroom. The two-task
run demonstrates bounded concurrency, not general two-worker safety.

The process-tree sampler runs every 10 ms. RSS and USS sum the server and its
children; this API process uses threads, so model work is in one process.
Cgroup measurements include the benchmark client and container memory beyond
the server tree. Output PNGs and full JSON records remain outside Git at
`/tmp/laika-api-concurrency/` on the Mac.

## Reproduction

Build the production image and the benchmark-only image (which adds the
`httpx` client dependency), then mount the checkout, local model assets and
an input image into the Linux test container. Set `LAIKA_API_MAX_WORKERS=1`
or `2`, `LAIKA_API_QUEUE_CAPACITY=0`, and run:

```sh
python scripts/benchmark_api_task.py /path/to/input.jpg \
  --task-count 2 \
  --options '{"mode":"color+bokeh","strategy":"low-memory","focus_x":0.72,"focus_y":0.66,"aperture":2.8,"subject_points":[[0.72,0.66]],"matte_tile_size":512}' \
  --report /path/to/report.json
```

`--task-count` submits independent uploads to one API process and records each
task result, stage timings and output hash. It rejects a count greater than the
API's configured worker-plus-queue admission capacity. The `Dockerfile.benchmark`
image and this script are used for the measured Linux run.
