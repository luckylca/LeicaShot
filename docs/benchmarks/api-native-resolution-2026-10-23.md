# Full-resolution API task under a 3 GiB Linux cgroup

Date: 2026-10-23. This run exercises a native-size photograph through one
real FastAPI upload, task creation, progress polling, full model pipeline, and
PNG download. It verifies the service path around the existing Python engine;
it does not change processing code.

## Configuration

- Source revision: `291b988` (`perf: benchmark API worker concurrency under cgroup`),
  branch `perf/api-native-resolution`. The repository was mounted read-only.
- Host: MacBook Air M1, 8 GiB RAM. Linux guest: Ubuntu 26.04, ARM64, 2 vCPUs,
  4,085,092,352 B guest memory. The benchmark container had a 3 GiB memory cap
  and equal swap cap (no swap headroom), 2 CPU quota, and 128 PID limit.
- Runtime: Python 3.11.17, NumPy 2.4.6, OpenCV 5.0.0, ONNX Runtime 1.31.0;
  CPUExecutionProvider, one ONNX intra-op thread, Low Memory strategy.
- API scheduler: one worker, no queue. One Uvicorn process served the request.
- Input: CC0 mountains JPEG, 5760×3840, 3,878,306 bytes, SHA-256
  `4e6661606c195b2462c0f2321a7feefe2ae2fc081499f3462a30fbb108cd7f8e`.
- Options: `color+bokeh`, focus `(0.5, 0.5)`, aperture `2.8`, one positive
  subject point `(0.72, 0.66)`, ViTMatte tile size 512, no output-size override.
  Export stayed at 5760×3840.
- Model hashes: M9 `8936b8c86656265ab07c30b9b4cbbd7fa195d6e44cb3be1d4fd059bb36781e33`;
  Depth Anything V2 `a8ce30941c85b2c9af29fb26f36fbf7be3c1ba1dcf348187a51f6eaa856a238e`;
  ARNet `f0696c16a6eb02099292176f25b47e4dc53863e7e4f3f7748e66470d54cf0173`;
  SAM encoder `37de035410f11fb505c210ce1f8b10b450cb018fbfa7f226ec7e7ab542955c75`;
  SAM decoder `3a6a96c62fd39c1369641ea626e5b2e4aed10de877f85291e9993a044d5bf981`;
  ViTMatte ONNX `a5c0b49a664867021eaf2b9cadcdc236ea511028d5e709303b418431f20e223b`.

## Results

| Measurement | Result |
|---|---:|
| API task wall time, upload through completed PNG download | 4,167.105 s (69.45 min) |
| Engine-reported total | 4,164.739 s |
| Server CPU time | 4,007.270 s |
| Server process-tree peak RSS | 2,809,442,304 B (2.616 GiB) |
| Server process-tree peak USS | 2,794,074,112 B (2.602 GiB) |
| cgroup v2 peak | 3,075,403,776 B (2.864 GiB; 95.47% of limit) |
| Remaining cgroup budget at peak | 145,821,696 B (139.1 MiB) |
| cgroup OOM / limit events | 0 / 0 |
| Downloaded PNG | 16,655,520 B; 5760×3840 |
| Download SHA-256 | `0a6282df408319950f0bd4c8f56bcfce78ed9e7eba95ce9c50b2c70f5555ade2` |

The download hash is byte-identical to the source-verified full-resolution
subject-protection baseline in the [direct engine report](full-resolution-subject-protection-cgroup-2026-10-23.md).
This confirms that the HTTP service returned the same full-size result for the
same image and processing parameters. The benchmark streamed and hashed the
download; it did not retain the PNG file. API task files were removed when the
benchmark server shut down. The preview endpoint was not separately fetched in
this run; the earlier browser smoke covered preview and comparison.

| Stage | Seconds |
|---|---:|
| M9 model inference | 5.219 |
| Lab detail fusion | 4.774 |
| Depth inference | 20.953 |
| SAM encoder / decoder | 8.157 / 0.142 |
| ViTMatte inference | 9.434 |
| ARNet passes 1–4 | 237.828 / 59.742 / 18.785 / 6.082 |
| Highlight extraction / recovery | 5.760 / 2.167 |
| Highlight scatter | 3,720.483 |
| Highlight softening / composite | 30.966 / 1.433 |
| PNG export | 11.195 |

Highlight scatter accounted for 89.3% of engine-reported time. Compared with
the earlier direct full-resolution run, this API run used 13.75% more cgroup
peak and recorded 40.39% more scatter time. The output is exactly equal, but
these single runs are not a controlled throughput comparison, so the timing
difference is not attributed to the API layer. It reinforces the need to
profile native-size scatter under controlled CPU load.

The task completed under 3 GiB with one worker, but left only 139 MiB cgroup
headroom. Treat that as a demonstrated single-task completion, not a robust
3 GiB service recommendation. Keep one worker and use more memory for reliable
native-size service operation; do not enable concurrent full-resolution jobs
at this limit.

## Reproduction

Mount the checkout read-only, authorized model assets read-only, and the source
photo into a Linux ARM64 container. Set the API model paths, storage path,
`LAIKA_API_MAX_WORKERS=1`, `LAIKA_API_QUEUE_CAPACITY=0`,
`LAIKA_API_THREADS=1`, and `LAIKA_API_STRATEGY=low-memory`, then run:

```sh
python scripts/benchmark_api_task.py /path/to/mountains-5760x3840.jpg \
  --task-count 1 \
  --timeout-seconds 7200 \
  --poll-interval 0.5 \
  --options '{"mode":"color+bokeh","strategy":"low-memory","focus_x":0.5,"focus_y":0.5,"aperture":2.8,"subject_points":[[0.72,0.66]],"matte_tile_size":512}' \
  --report /path/to/api-task.json
```

The raw API JSON and cgroup counters from this run are outside Git at
`/tmp/laika-api-native-20261023/` on the Mac. No model or photo files were
added to the repository.
