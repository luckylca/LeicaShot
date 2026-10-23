# Native-resolution CPU task at a 2 GiB cgroup limit

Updated: 2026-10-23

## Goal and configuration

Run the complete CPU image task without reducing output size or skipping stages: M9 + LabFusion, Depth Anything V2, a SAM 2.1 Tiny subject point, ViTMatte Small, BokehMe ARNet and highlight scatter. Input was the 5760×3840 CC0 mountains JPEG, SHA-256 `4e6661606c195b2462c0f2321a7feefe2ae2fc081499f3462a30fbb108cd7f8e`. The task used Low Memory, one ONNX/OpenCV thread, focus `(0.5, 0.5)`, aperture `2.8`, one subject point `(0.72, 0.66)`, 512-pixel ViTMatte tiles and 256-row scatter chunks.

The Linux ARM64 Docker container had a 2-vCPU quota, cgroup-v2 `memory.max=2,147,483,648` and `memory.swap.max=2,147,483,648` bytes. These are separate cgroup limits; swap usage at the time of either OOM was not captured, so this experiment does not establish a no-swap condition. Python was 3.11.17, NumPy 2.4.6, OpenCV 5.0.0 and ONNX Runtime 1.31.0. Every model used `CPUExecutionProvider`. Two attempts used the same image, model files and processing options; the second also enabled a flushed JSONL progress log so the last completed stage survived the OOM.

The same mounted model assets were recorded in the 4 GiB full-task reference: M9 `8936b8c86656265ab07c30b9b4cbbd7fa195d6e44cb3be1d4fd059bb36781e33`; Depth Anything V2 `a8ce30941c85b2c9af29fb26f36fbf7be3c1ba1dcf348187a51f6eaa856a238e`; BokehMe ARNet `f0696c16a6eb02099292176f25b47e4dc53863e7e4f3f7748e66470d54cf0173`; SAM 2.1 Tiny encoder `37de035410f11fb505c210ce1f8b10b450cb018fbfa7f226ec7e7ab542955c75` and decoder `3a6a96c62fd39c1369641ea626e5b2e4aed10de877f85291e9993a044d5bf981`; ViTMatte Small ONNX `a5c0b49a664867021eaf2b9cadcdc236ea511028d5e709303b418431f20e223b`.

## Result

Neither 2 GiB task completed. Both exited with code 137 and Docker marked the container `OOMKilled=true`. Each cgroup reached `memory.peak=2,147,483,648` B. The wrapper captured these event counts:

| cgroup event | Attempt 1 | Attempt 2 |
|---|---:|---:|
| `max` | 1,274 | 1,566 |
| `oom` | 1 | 1 |
| `oom_kill` | 1 | 1 |
| `oom_group_kill` | 0 | 0 |

The flushed progress log confirms decode, M9, depth, SAM and ViTMatte stages completed. ARNet pass 1 (5760×3840, 96 tiles) and pass 2 (2880×1920, 24 tiles) completed; the last event was bokeh progress 0.75 at 270.9 seconds. The process was OOM-killed before the next progress event. The precise internal allocation that crossed the cap is unknown. No final PNG or engine timing report was produced. Process-tree peak RSS is therefore unavailable for this failed run; the cgroup peak and OOM counters are exact.

For comparison, the one-run same-photo FP64 full pipeline under 4 GiB completed in 1,305.825 s with 2,629,607,424 B cgroup peak and 2,546,049,024 B process-tree peak RSS. That confirms the 2 GiB limit is below this configuration's measured full-task footprint; it does not establish a universal minimum for other scenes.

## Recovery checkpoint

Treat 2 GiB full-resolution support as not achieved. Keep the normal full-resolution output and all stages. The next work should instrument live per-stage RSS and reduce peak live buffers or model residency around the pass-2-to-pass-3 transition, then repeat the exact 2 GiB cgroup task. Do not report a pass until a complete 5760×3840 result and its cgroup report are present.

The JSONL progress log, failed-run sidecar and container artifacts remain local-only under `/tmp/laika-fullres-2g-20261023/` on the Mac and `/data/fullres-2g/` in the Lima Docker volume. No image, model or cache asset is committed.
