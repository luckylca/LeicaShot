# Deferred highlight-radiance buffer: full-resolution CPU A/B

Date: 2026-10-24. The Low Memory candidate defers the full-resolution HWC
float32 highlight-radiance buffer until ARNet working frames have been released.
The eager control is the merged `develop` implementation at `acab2d5`; the
candidate was measured from branch `perf/arnet-stage-buffer-lifetimes`, based on
`9f973f0`. Both use the same 256-pixel Low Memory ARNet tiles, input, models,
parameters, guest and cgroup limit.

## Configuration

- Host: MacBook Air M1; Linux 7.0.0-34 ARM64 Lima guest, 4,085,092,352 B RAM,
  2 logical CPUs.
- Container: cgroup v2, `memory.max=2,147,483,648` B,
  `memory.swap.max=0`, 2 CPU quota, 128 PID limit.
- Runtime: Python 3.11.17, NumPy 2.4.6, OpenCV 5.0.0, ONNX Runtime 1.31.0;
  one ORT intra-op thread. All model inference used `CPUExecutionProvider`.
- Input: CC0 Mountains JPEG, 5760×3840, SHA-256
  `4e6661606c195b2462c0f2321a7feefe2ae2fc081499f3462a30fbb108cd7f8e`.
- Request: M9 + LabFusion, Depth Anything V2, BokehMe ARNet, SAM 2.1 Tiny,
  ViTMatte Small, optical highlights, and subject protection. Low Memory mode,
  focus `(0.5, 0.5)`, aperture `2.8`, point `(0.72, 0.66)`, 512-pixel
  ViTMatte tiles, 256-pixel ARNet tiles, 128-row highlight chunks, one thread,
  and no output-size override.
- Candidate source tree SHA-256: `1b733acb5ce55ee64c7b5004817e008d4bb0b8514ecb9ac4b813260ebb07d396`.
- Eager control source tree SHA-256: `00da5d10fe8c22f5467c8f6b5320b7947eaa776dd5f5481622dd3cfaff4c8cfc`.
- Model assets were mounted read-only and stayed outside Git. Their hashes
  matched the five-model baseline: M9
  `8936b8c86656265ab07c30b9b4cbbd7fa195d6e44cb3be1d4fd059bb36781e33`, Depth
  `a8ce30941c85b2c9af29fb26f36fbf7be3c1ba1dcf348187a51f6eaa856a238e`, ARNet
  `f0696c16a6eb02099292176f25b47e4dc53863e7e4f3f7748e66470d54cf0173`, SAM
  encoder/decoder `37de035410f11fb505c210ce1f8b10b450cb018fbfa7f226ec7e7ab542955c75` /
  `3a6a96c62fd39c1369641ea626e5b2e4aed10de877f85291e9993a044d5bf981`, and
  ViTMatte `a5c0b49a664867021eaf2b9cadcdc236ea511028d5e709303b418431f20e223b`.

## Paired 2 GiB results

| Measure | Eager control | Deferred Low Memory | Difference |
|---|---:|---:|---:|
| Complete task wall time | 1,529.660 s | 1,553.704 s | +24.044 s (+1.57%) |
| Process-tree peak RSS | 1,995,259,904 B | 1,855,737,856 B | −139,522,048 B (−6.99%) |
| cgroup `memory.peak` | 2,147,483,648 B (limit) | 2,145,607,680 B | −1,875,968 B |
| Largest 1-second `memory.current` | 2,146,545,664 B | 2,041,085,952 B | −105,459,712 B |
| Anonymous / file memory at largest sample | 1,807,941,632 / 329,895,936 B | 1,696,833,536 / 335,605,760 B | lower anonymous RSS |
| cgroup `memory.events max` | 190 | 0 | −190 |
| cgroup OOM / OOM-kill events | 0 / 0 | 0 / 0 | no OOM |
| Highlight scatter | 1,208.126 s | 1,212.944 s | +4.818 s (+0.40%) |
| Deferred radiance reconstruction | n/a | 1.703 s | additional row pass |
| Output dimensions | 5760×3840 | 5760×3840 | unchanged |
| Output SHA-256 | `0a6282df408319950f0bd4c8f56bcfce78ed9e7eba95ce9c50b2c70f5555ade2` | same | byte-identical |

Both runs completed successfully on CPU and produced byte-identical PNG output.
The eager run hit the cgroup limit and recorded 190 `max` events without an OOM
kill. The deferred run stayed just below the exact cgroup high-water limit and
recorded no `max` event. Its exact `memory.peak` margin was only 1,875,968 B, so
this is not reliable 2 GiB service headroom. The process-tree RSS and largest
one-second cgroup sample show larger reductions than the transient cgroup
high-water. Keep the three measures separate in future acceptance reports.

The single pair does not establish a speed change: the total task was 1.57%
slower and scatter 0.40% slower, while other model stages varied in the
opposite direction. The extra radiance pass took 1.703 s on this run.

## 3 GiB candidate completion

The same Low Memory candidate completed a source-explicit 3 GiB full-resolution
run in 1,484.881 s. Process-tree peak RSS was 1,870,557,184 B, cgroup
`memory.peak` was 2,289,954,816 B of 3,221,225,472 B, and the task had no OOM or
limit events. Highlight scatter took 1,159.545 s and deferred radiance
reconstruction took 0.713 s. The output SHA-256 and dimensions matched the 2 GiB
candidate and eager control.

## Decision and limits

Keep deferred radiance in **Low Memory** mode: the paired native-size outputs
are byte-identical, process-tree peak fell by about 140 MB, and the cgroup
recorded no limit events where the eager control recorded 190. Preserve eager
radiance in **Balanced** mode to avoid its extra pass. Tests verify that both
strategies produce identical pixels.

This is one photo and one paired run. Low Memory completed at the 2 GiB limit
with only about 1.8 MB of exact cgroup headroom; it does not establish a safe
2 GiB deployment tier. Full-resolution optical highlight scatter remains the
dominant CPU stage. No model, output size, or protection stage was reduced.
