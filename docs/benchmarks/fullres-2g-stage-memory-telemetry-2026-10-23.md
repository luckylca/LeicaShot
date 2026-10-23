# Full-resolution 2 GiB OOM with stage memory telemetry

Updated: 2026-10-23

## Goal and environment

Repeat the complete 5760×3840 CPU pipeline under a 2 GiB task memory cap and persist stage-tagged RSS and cgroup memory categories every second, including when the worker is killed.

- Input: CC0 mountains JPEG, 5760×3840, SHA-256 `4e6661606c195b2462c0f2321a7feefe2ae2fc081499f3462a30fbb108cd7f8e`.
- Pipeline: M9 + LabFusion, Depth Anything V2 Small, one SAM 2.1 Tiny subject point, ViTMatte Small, BokehMe ARNet and highlight scatter. Low Memory, CPUExecutionProvider, one thread, 512-pixel matte tiles, 256-row scatter chunks, focus `(0.5, 0.5)`, aperture `2.8`, subject point `(0.72, 0.66)`. No stage was disabled and output dimensions were not changed.
- Linux ARM64 Lima guest: 4 GiB RAM, 2 vCPUs. Container: Python 3.11.17, NumPy 2.4.6, OpenCV 5.0.0, ONNX Runtime 1.31.0.
- Docker used `--memory=2g --memory-swap=2g --cpus=2`. Inspection of the failed container and a same-settings probe confirmed cgroup-v2 `memory.max=2,147,483,648` B, `memory.swap.max=0`, and `memory.swap.current=0`.
- Model hashes: M9 `8936b8c86656265ab07c30b9b4cbbd7fa195d6e44cb3be1d4fd059bb36781e33`; Depth `a8ce30941c85b2c9af29fb26f36fbf7be3c1ba1dcf348187a51f6eaa856a238e`; ARNet `f0696c16a6eb02099292176f25b47e4dc53863e7e4f3f7748e66470d54cf0173`; SAM encoder `37de035410f11fb505c210ce1f8b10b450cb018fbfa7f226ec7e7ab542955c75`; SAM decoder `3a6a96c62fd39c1369641ea626e5b2e4aed10de877f85291e9993a044d5bf981`; ViTMatte `a5c0b49a664867021eaf2b9cadcdc236ea511028d5e709303b418431f20e223b`.

## Outcome

The task was OOM-killed after about four minutes: Docker reported `OOMKilled=true`, Python exit code 137, cgroup `memory.peak=2,147,483,648` B, `oom=1`, `oom_kill=1`, `oom_group_kill=0`. No PNG or normal engine timing report was produced. The final saved engine telemetry was at 246.63 seconds.

The 1-second JSONL logger flushed 499 records before termination: 254 progress records and 245 memory samples. RSS was sampled every 10 ms by the existing process-tree monitor; cgroup snapshots include `memory.current`, `memory.peak`, `memory.events`, swap counters, and `memory.stat` categories.

| Stage | Sampled process-tree RSS max | Max sampled cgroup current | `anon` at that cgroup sample | `file` at that cgroup sample |
|---|---:|---:|---:|---:|
| Depth | 822,681,600 B | 743,211,008 B | 636,153,856 B | 103,747,584 B |
| M9 / color | 1,201,307,648 B | 868,507,648 B | 660,295,680 B | 202,301,440 B |
| SAM + ViTMatte / subject | 1,338,912,768 B | 1,592,057,856 B | 1,225,125,888 B | 358,522,880 B |
| Highlight extraction | 1,431,896,064 B | 1,600,479,232 B | 1,127,968,768 B | 464,404,480 B |
| ARNet tiled passes | 2,089,091,072 B | 2,147,483,648 B | 1,750,659,072 B | 387,514,368 B |
| ARNet pass transition | 2,122,448,896 B | 2,147,463,168 B | 2,067,259,392 B | 73,396,224 B |

For rows after Depth, the two RSS/cgroup maxima can occur at different samples within the stage. The cgroup reached its 2 GiB high-water during tiled ARNet around 151 seconds without an OOM event. File cache later fell from roughly 388 MB to 73 MB as anonymous usage rose. At the last sample, cgroup current was 2,147,463,168 B and sampled process-tree RSS was 2,122,448,896 B; `anon` was 2,067,259,392 B, `file` 73,396,224 B, and `kernel` 6,631,424 B. Swap remained zero.

All 24 tiles of ARNet pass 2 (2880×1920) completed at 244.05 seconds. The engine then reported Bokeh progress 0.75; the process was killed before the next ARNet tile-start event. This locates the failure at the transition into the next pass, but the exact allocation that crossed the limit is unknown. The near-OOM sample is dominated by anonymous memory, so file-cache reclaim alone is not an adequate fix.

## Recovery checkpoint

The 2 GiB full-resolution task remains unsupported. Use the trace to inspect ARNet scale-transition array lifetimes and ONNX Runtime arena residency. Any candidate must preserve all passes and output dimensions, then pass an exact output comparison and repeat this unchanged cgroup task. The JSONL trace, container sidecars, and any temporary output remain outside Git under `/tmp/laika-stage-memory-final-20261023/` and the `leicashot-stage-memory-telemetry-20261023` Docker volume.
