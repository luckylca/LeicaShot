# ARNet tile edge and full-pipeline 2 GiB headroom screen

Date: 2026-10-24

## Decision

Use 256-pixel ARNet inference tiles in the **Low Memory** policy and retain
512-pixel tiles in **Balanced**. Both use the same full-context ONNX graph,
16-pixel tile halo, full-resolution output canvas, and model precision. The
selected native-resolution subject task produced byte-identical PNG output.

This is a single successful 256-pixel full-task run and does not establish a
stable 2 GiB service tier. The 512-pixel control completed once, but a second
512-pixel attempt was OOM-killed. More photos and repeat runs are still needed.

## Environment and inputs

- Host: MacBook Air M1; Linux ARM64 Lima guest with 4,085,092,352 B RAM and
  2 vCPUs.
- Container: cgroup v2, 2,147,483,648 B memory limit, no swap, 2 CPU quota,
  128 PID limit. One ONNX Runtime thread.
- Input: 5760×3840 CC0 mountain JPEG, SHA-256
  `4e6661606c195b2462c0f2321a7feefe2ae2fc081499f3462a30fbb108cd7f8e`.
- Request: M9 color + Lab fusion, Depth Anything V2, SAM 2.1 Tiny, ViTMatte
  Small with a 512-pixel matte tile, ARNet, highlight scatter/composite, and
  one subject point `(0.72, 0.66)`. Focus `(0.5, 0.5)`, aperture `2.8`,
  128-row highlight chunks, native output size.
- All neural stages used ONNX Runtime `CPUExecutionProvider`. M9, Depth, SAM,
  ViTMatte and ARNet model hashes match the recorded local assets.

## Measurements

| Run | ARNet tile | Complete task | Process-tree peak RSS | cgroup `memory.peak` | Largest 1 s sample | OOM | Output SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---|
| PR #57 baseline | 512 | 1,475.646 s | 2,067,648,512 B | 2,147,483,648 B (limit) | 2,138,918,912 B | no | `0a6282df408319950f0bd4c8f56bcfce78ed9e7eba95ce9c50b2c70f5555ade2` |
| Low Memory candidate | 256 | 2,098.507 s | 2,004,312,064 B | 2,125,889,536 B | 2,097,598,464 B | no | `0a6282df408319950f0bd4c8f56bcfce78ed9e7eba95ce9c50b2c70f5555ade2` |
| Repeat control | 512 | killed after 277.788 s | at least 1,623,027,712 B at last callback | reached limit | not sampled externally | yes, exit 137 | incomplete |

The completed 256 run lowered overall process-tree peak RSS by 63,336,448 B
and exact cgroup high-water by 21,594,112 B versus the original 512 run. Its
one-second maximum was 41,320,448 B lower. The cgroup recorded zero `max` and
OOM events during the 256 run; the first 512 run recorded 285 `max` events
without OOM. The repeat 512 container was marked `OOMKilled=true` while in
ARNet inference.

Peak memory moved between sub-stages: the recorded process-tree peak for the
ARNet tile callback stage fell from 2,067,648,512 B at 512 tiles to
1,803,034,624 B at 256 tiles, while the 256 run's highest overall RSS was
2,004,312,064 B in the surrounding ARNet stage. This limits the whole-task
gain to about 60 MiB even though the tile-stage peak fell by about 252 MiB.

The 256 task reported 2,098.507 s wall time, versus 1,475.646 s in the first
512 run. However, several unrelated CPU stages also ran slower in the 256
run (for example M9 inference 6.332 s vs 3.816 s; Depth 14.718 s vs 13.943 s),
and the repeat 512 run ended before timing completion. These sequential runs
do not isolate a tile-size latency penalty. In the 1440×960 real M9/depth
intermediate screen, 512/384/256 tiles all had exact full-frame output parity
and took 9.941/9.331/10.171 s respectively.

## Interpretation and follow-up

256-pixel tiles are retained only for Low Memory because the full request
completed with lower process-tree and cgroup peaks, while 512-pixel tiles
reached the cgroup limit in the baseline and OOM-killed in a repeat. Balanced
keeps 512-pixel tiles to avoid the higher tile count in the throughput-oriented
policy. Neither run proves safe 2 GiB service headroom; the successful 256
run left only 21.6 MB between cgroup `memory.peak` and the hard cap.

The next memory task should target the remaining ARNet-stage live arrays: the
256 run's peak shifted outside the tile callback stage. Keep optical scatter
as the separate dominant CPU-time bottleneck. Re-run multiple input scenes and
measure 4 GiB service headroom before defining a supported deployment tier.

Artifacts remain outside Git in Docker volumes
`leicashot-subject-2g-20261024`, `leicashot-arnet-tile-256-2g-20261024`, and
`leicashot-arnet-tile-512-repeat-20261024`.
