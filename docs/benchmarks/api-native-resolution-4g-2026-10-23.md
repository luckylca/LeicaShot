# Full-resolution API task under a 4 GiB Linux cgroup

Date: 2026-10-23. This follow-up repeats the full-resolution HTTP request with
a 4 GiB cgroup and retains both the preview and complete PNG downloads. It is
one CPU-only scene and does not establish a service-wide memory guarantee.

## Configuration

- Source revision: `98ecf38` (PR #41 merged to `develop`); source mounted
  read-only into the Linux ARM64 guest.
- Host: MacBook Air M1, 8 GiB RAM. Linux guest: Ubuntu 26.04, ARM64, 2 vCPUs,
  4,085,092,352 B guest memory. The task container had a 4 GiB cgroup v2
  memory limit, equal swap cap (no swap headroom), 2 CPU quota, and 128 PID
  limit.
- Runtime: Python 3.11.17, ONNX Runtime 1.31.0, CPUExecutionProvider, one
  intra-op thread, Low Memory strategy. FastAPI used one worker and no queue.
- Input: CC0 mountains JPEG, 5760×3840, SHA-256
  `4e6661606c195b2462c0f2321a7feefe2ae2fc081499f3462a30fbb108cd7f8e`.
- Options: `color+bokeh`, focus `(0.5, 0.5)`, aperture `2.8`, positive subject
  point `(0.72, 0.66)`, ViTMatte tile size 512. No output-size override;
  exported dimensions stayed 5760×3840.
- Model hashes: M9
  `8936b8c86656265ab07c30b9b4cbbd7fa195d6e44cb3be1d4fd059bb36781e33`;
  Depth Anything V2 q64
  `a8ce30941c85b2c9af29fb26f36fbf7be3c1ba1dcf348187a51f6eaa856a238e`;
  ARNet
  `f0696c16a6eb02099292176f25b47e4dc53863e7e4f3f7748e66470d54cf0173`;
  SAM encoder/decoder
  `37de035410f11fb505c210ce1f8b10b450cb018fbfa7f226ec7e7ab542955c75` /
  `3a6a96c62fd39c1369641ea626e5b2e4aed10de877f85291e9993a044d5bf981`;
  ViTMatte ONNX
  `a5c0b49a664867021eaf2b9cadcdc236ea511028d5e709303b418431f20e223b`.

## Results

| Measurement | Result |
|---|---:|
| API wall time, upload through result download | 3,562.790 s (59.38 min) |
| Engine-reported total | 3,561.195 s |
| Server CPU time | 3,460.460 s |
| Server process-tree peak RSS / USS | 2,655,535,104 / 2,640,166,912 B |
| cgroup v2 peak | 3,027,701,760 B (2.820 GiB; 70.49% of 4 GiB) |
| cgroup headroom at peak | 1,267,265,536 B (1.180 GiB) |
| cgroup OOM / limit events | 0 / 0 |
| Full PNG | 16,655,520 B; 5760×3840 |
| Full PNG SHA-256 | `0a6282df408319950f0bd4c8f56bcfce78ed9e7eba95ce9c50b2c70f5555ade2` |
| Preview PNG | 1,680,414 B; 1600×1067 |
| Preview SHA-256 | `adb3801bcc4811407aacd040303568cb286db53c3a8a0c03040e6f6cdbd0bc4e` |

M9, Depth, SAM, ViTMatte and ARNet all reported `CPUExecutionProvider`.
The full PNG hash matches both the previous 3 GiB API run and the source-
verified direct-engine subject result exactly. This confirms identical encoded
output for this image and parameter set while also exercising the API preview
route. The benchmark fetched and saved both artifacts, checked their response
types, and verified their PNG metadata and hashes.

| Stage | Seconds |
|---|---:|
| M9 inference / Lab detail fusion | 6.309 / 4.990 |
| Depth inference | 21.076 |
| SAM encoder / decoder | 7.263 / 0.182 |
| ViTMatte inference | 9.714 |
| ARNet passes 1–4 | 249.292 / 62.517 / 17.488 / 7.141 |
| Highlight extraction / recovery | 4.882 / 2.102 |
| Highlight scatter | 3,119.995 |
| Highlight softening / composite | 19.935 / 1.217 |
| PNG export | 6.807 |

Scatter used 87.6% of engine-reported time. It was 600.488 s faster than the
prior 3 GiB API run, while total time was 604.405 s faster. The output and
input match, but these are sequential single trials with no controlled CPU
isolation, so they do not demonstrate a reliable speedup from the larger
cgroup. The 4 GiB run had 1.18 GiB cgroup headroom, a useful single-task result;
concurrency and other scenes remain unmeasured at native resolution.

## Reproduction

Run `scripts/benchmark_api_task.py` against a Linux ARM64 API container with
the same mounted local model assets and input. Use a 4 GiB cgroup, 2 CPU quota,
one API worker, no queue, one ONNX intra-op thread, Low Memory strategy, and
the options recorded above. Fetch and retain both API artifacts:

```sh
python scripts/benchmark_api_task.py /path/to/mountains-5760x3840.jpg \
  --task-count 1 \
  --timeout-seconds 7200 \
  --poll-interval 0.5 \
  --options '{"mode":"color+bokeh","strategy":"low-memory","focus_x":0.5,"focus_y":0.5,"aperture":2.8,"subject_points":[[0.72,0.66]],"matte_tile_size":512}' \
  --save-output /path/to/result.png \
  --save-preview /path/to/preview.png \
  --report /path/to/task.json
```

The retained task report, PNGs, and cgroup counters are outside Git under
`/tmp/laika-api-native-4g-20261023/` on the Mac. The initial attempt lacked a
Depth validation sidecar and failed before model inference; it was corrected
by mounting the complete validated local model directory. No model hash check
was bypassed, and the successful run recorded zero OOM or limit events.
