# Full-resolution FastAPI request under a 2 GiB cgroup

Date: 2026-10-24. This run exercises one complete API request from multipart
upload through task progress polling, model processing, preview generation and
full PNG download. It uses the native input and output dimensions.

## Configuration

- Source: `develop` commit `3d817b2` plus the benchmark-only status-poll retry
  on `test/api-native-resolution-2g`; the checkout was mounted read-only.
- Host: MacBook Air M1, 8 GiB RAM. Linux guest: Ubuntu 26.04, ARM64,
  4,085,092,352 B memory and 2 vCPUs. Docker cgroup v2 limit:
  2,147,483,648 B memory and the same swap cap (no swap headroom), 2 CPU quota,
  128 PID limit.
- Runtime: Python 3.11.17, NumPy 2.4.6, OpenCV 5.0.0, ONNX Runtime 1.31.0.
  The API ran one Low Memory worker, no queue, one ORT intra-op thread, and
  `CPUExecutionProvider` for every requested model.
- Input: CC0 mountains JPEG, 5760×3840, 3,878,306 B, SHA-256
  `4e6661606c195b2462c0f2321a7feefe2ae2fc081499f3462a30fbb108cd7f8e`.
- Request: `color+bokeh`, focus `(0.5, 0.5)`, aperture `2.8`, foreground
  subject point `(0.72, 0.66)`, ViTMatte tile size 512, no output-size override.
  API Bokeh row chunk retained its default of 256.
- Model hashes verified before execution: M9
  `8936b8c86656265ab07c30b9b4cbbd7fa195d6e44cb3be1d4fd059bb36781e33`;
  Depth Anything V2 Small
  `a8ce30941c85b2c9af29fb26f36fbf7be3c1ba1dcf348187a51f6eaa856a238e`;
  BokehMe ARNet
  `f0696c16a6eb02099292176f25b47e4dc53863e7e4f3f7748e66470d54cf0173`;
  SAM encoder / decoder
  `37de035410f11fb505c210ce1f8b10b450cb018fbfa7f226ec7e7ab542955c75` /
  `3a6a96c62fd39c1369641ea626e5b2e4aed10de877f85291e9993a044d5bf981`;
  ViTMatte Small ONNX
  `a5c0b49a664867021eaf2b9cadcdc236ea511028d5e709303b418431f20e223b`.
  GAIC was not requested and was not loaded.

## Results

| Measurement | Result |
|---|---:|
| API wall time, upload through result and preview download | 3,991.310 s (66.52 min) |
| Engine-reported total | 3,988.833 s |
| API server CPU time | 2,893.310 s |
| Server process-tree peak RSS / USS | 1,863,647,232 / 1,848,631,296 B |
| Exact cgroup `memory.peak` | 1,880,756,224 B (1.752 GiB) |
| Cgroup headroom at peak | 266,727,424 B (0.248 GiB) |
| Maximum 10-second sampled `memory.current` | 1,880,326,144 B |
| Cgroup `max` / `oom` / `oom_kill` events | 0 / 0 / 0 |
| Downloaded full PNG | 16,655,520 B; 5760×3840 |
| Full PNG SHA-256 | `0a6282df408319950f0bd4c8f56bcfce78ed9e7eba95ce9c50b2c70f5555ade2` |
| Downloaded preview | 1,680,414 B; 1600×1067 |
| Preview SHA-256 | `adb3801bcc4811407aacd040303568cb286db53c3a8a0c03040e6f6cdbd0bc4e` |
| Status-poll transport retries | 1 |

The full output hash is byte-identical to the direct engine's verified native-size
result for this image and these processing parameters. The run completed the
real multipart upload, task creation and polling, one-point SAM/ViTMatte subject
protection, M9 + Lab fusion, depth inference, ARNet and optical highlights,
then downloaded both the full image and preview. All model providers in the
task result were `CPUExecutionProvider`.

| Stage | Seconds |
|---|---:|
| M9 inference / Lab fusion | 53.070 / 7.866 |
| Depth inference | 167.726 |
| SAM prepare / encoder / decoder | 67.058 / 62.074 / 0.398 |
| ViTMatte | 51.077 |
| ARNet passes 1–4 | 294.616 / 48.487 / 14.239 / 7.640 |
| Subject background plate / composite | 235.322 / 2.143 |
| Highlight extraction / recovery | 69.286 / 5.482 |
| Highlight scatter | 2,837.063 |
| Highlight softening / composite | 23.351 / 1.806 |
| PNG export | 7.244 |

Highlight scatter accounted for 71.1% of engine-reported task time. This single
scene completed within the 2 GiB no-swap container and retained about 248 MiB
of cgroup headroom; that is not a repeatability or multi-scene service guarantee.

The first attempt in this branch stopped after a status GET returned a connection
reset; Docker reported exit 1 and `OOMKilled=false`, so that attempt did not
establish a processing or memory failure. The benchmark now retries the
idempotent status GET and checks whether the API process exited. In this
successful run one transient status-poll reset was retried, and the task
completed normally.

## Measurement limits

The benchmark client and API server shared the 2 GiB cgroup, so the cgroup peak
includes both. The process-tree RSS/USS values cover the API server only. The
benchmark's `ServerProcessPeak` also samples USS every 10 ms. A late `docker top`
snapshot showed the benchmark parent at 31.0% CPU and the API process at 70.1%
CPU; this may include substantial USS-sampling and polling overhead. The full
request and image result are valid, but the measured wall time should not be
treated as CPU-neutral engine throughput.

The direct-engine comparison used 128-row Bokeh chunks, while this API request
used the API's default 256 rows. Sequential timings also ran on a shared host.
Do not attribute their time difference to the API layer or memory limit. A
follow-up should reduce process-monitor overhead, record cgroup CPU time, and
run a parameter-matched API/engine pair before making throughput claims.

## Artifacts and reproduction

The report, exact full PNG, preview, and 10-second cgroup sample series are
outside Git at
`/Users/lucky/Desktop/project/leicashot-photo-validation/api-native-2g-retry-20261024/`
and in Docker volume `leicashot-api-native-2g-retry-data-20261024`.

Run `scripts/benchmark_api_task.py` against a Linux ARM64 container with
read-only model/input mounts and a separate writable artifact volume. Set API
model paths, `LAIKA_API_MAX_WORKERS=1`, `LAIKA_API_QUEUE_CAPACITY=0`,
`LAIKA_API_THREADS=1`, `LAIKA_API_STRATEGY=low-memory`, and
`LAIKA_API_STORAGE` to the writable volume. Use a 2 GiB cgroup with equal swap
cap, two CPU quota, 128 PID limit, one task, and the options recorded above.
Allow at least 7,200 seconds for the full native-resolution task.
