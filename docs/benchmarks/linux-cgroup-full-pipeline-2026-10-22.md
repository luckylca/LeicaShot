# Linux cgroup full-pipeline benchmark — 2026-10-22

## Scope

This report records real CPU-only calls through `LaikaProcessingEngine.process_file()` in a Linux ARM64 VM. It is one P3M-10K portrait, 1080×1620, processed with M9, Depth Anything V2 Small, BokehMe ARNet, SAM 2.1 Tiny and ViTMatte Small. The task used Low Memory mode, one inference thread, three SAM foreground points, 512-pixel ViTMatte tiles, focus `(0.52, 0.30)`, aperture `2.8`, and preserved the input resolution. This is a useful constrained-memory case, not a broad image-quality acceptance set.

The input is `/tmp/laika-p3m-portrait/image.jpg`, SHA-256 `33bcc476421718673730a44b30d136b911dc97c8f086fc58fea55d63666d308b`. The Linux result is 1080×1620 PNG, 1,932,123 bytes, SHA-256 `0d9c27749d21c06ebc0dc79b6721aeca6a18f00480ce8fff9956334f1f41b69c`. The same Linux output hash was produced in every memory configuration below.

## Reproducible environment

- Host: MacBook Air M1, macOS 15.7.7, 8 GiB physical memory.
- Linux VM: Lima 2.2.1, Ubuntu 26.04.1 ARM64, Linux 7.0.0-34, Docker 29.9.2, cgroup v2, no swap; configured with 2, 4, or 6 GiB guest memory as listed below.
- Container: Python 3.11, NumPy 2.4.6, OpenCV 5.0.0.93, ONNX Runtime 1.31.0; the reproducible image is built from [`tools/cgroup/Dockerfile`](../../tools/cgroup/Dockerfile). Model files and photos are mounted read-only from host `/tmp`; neither is copied into the image or repository.
- Execution: `CPUExecutionProvider`, `--threads 1`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `--strategy low-memory`, `--matte-tile-size 512`, `--bokeh-row-chunk-height 128`.
- Models: M9 SHA-256 `8936b8c86656265ab07c30b9b4cbbd7fa195d6e44cb3be1d4fd059bb36781e33`; Depth ONNX `a8ce30941c85b2c9af29fb26f36fbf7be3c1ba1dcf348187a51f6eaa856a238e`; ARNet `f0696c16a6eb02099292176f25b47e4dc53863e7e4f3f7748e66470d54cf0173`; SAM image encoder `37de035410f11fb505c210ce1f8b10b450cb018fbfa7f226ec7e7ab542955c75`, decoder `3a6a96c62fd39c1369641ea626e5b2e4aed10de877f85291e9993a044d5bf981`; ViTMatte ONNX `a5c0b49a664867021eaf2b9cadcdc236ea511028d5e709303b418431f20e223b`.
- The benchmark samples process and descendant RSS every 10 ms and also reads Linux cgroup counters. Docker daemon and guest system processes are outside the task cgroup. A separate already-running Android emulator process was not started or used for these runs.

To run the same task after building the image, mount model and input directories read-only and set the container memory, CPU and thread limits. The benchmark report includes input/model/output hashes, per-stage time, process-tree RSS, cgroup peak/limit/events, and guest memory visible inside the container. `scripts/benchmark_processing_pipeline.py` prints one JSON record on completion.

## Full five-model portrait runs

| Guest RAM configuration | Task cgroup limit | Task wall time | Process-tree peak RSS | cgroup peak | OOM events | Output |
|---|---:|---:|---:|---:|---:|---|
| 6 GiB | 4 GiB | 102.10 s | 1,620,144,128 B | 1,923,289,088 B | 0 | Same hash |
| 6 GiB | 2 GiB | 125.87 s | 1,733,279,744 B | 2,035,470,336 B | 0 | Same hash |
| 4 GiB | 3 GiB | 988.21 s | 1,613,357,056 B | 2,097,229,824 B | 0 | Same hash |
| 2 GiB | 1.5 GiB | 1,196.67 s | 1,545,293,824 B | 1,564,434,432 B | 0 | Same hash |

The 2 GiB guest reported 2,044,289,024 bytes total memory and no swap. Its 1.5 GiB task cgroup peaked at 1,564,434,432 of 1,572,864,000 bytes, leaving 8,429,568 bytes (about 8.0 MiB) below the cgroup limit. `memory.events` reported `oom=0` and `oom_kill=0`; the container exited successfully. The process-tree RSS peak was 1,545,293,824 bytes. Thus this specific complete 1080×1620 five-model task completed in a 2 GiB-configured Linux guest, but with almost no task-cgroup headroom and a 19m57s wall time. It is an experimental pass, not evidence that arbitrary photos, higher resolutions, concurrent tasks, or deployment overhead fit in 2 GiB.

On the 2 GiB guest, the slowest recorded stages were ARNet pass 1 at 527.51 s, ARNet pass 2 at 212.49 s, ViTMatte inference at 212.91 s, and SAM preparation/encoding at 74.76/66.58 s. M9 inference took 45.34 s and depth inference 40.90 s. The 4 GiB guest also completed, but took 16m28s; its ARNet passes took 337.24/241.18 s, SAM encoder/preparation 89.32/94.47 s, and ViTMatte inference 170.99 s. These large timing differences across runs mean the results establish completion and memory behavior, not a stable speed comparison between guest sizes. The 6 GiB guest runs were substantially faster. No thermal warning was recorded; the reason for the timing variation is undetermined.

The 4 GiB guest's `free -b` snapshots during the task showed roughly 2.5–2.9 GB available, and it completed without OOM. A peak total-guest memory sampler was not running, so the process-tree and cgroup figures should not be presented as the guest-wide peak.

## Output quality comparison

All four Linux memory configurations produced a byte-identical output, so the memory limit did not change the result. Against the existing macOS SDK reference PNG (`a50fd8d2dc8615392942c613dca99a7fecb027fe1d184f5d9ffa8064ebb85484`), the Linux output measured MAE 0.7092, RMSE 3.0125, PSNR 38.55 dB, p95 absolute channel error 2, p99 12, maximum 133, and 93.12% of channels within one 8-bit value. Differences concentrate around soft subject edges; within the alpha soft-edge band MAE was 6.147 and 13.97% of pixels had a maximum-channel difference greater than 16. Cross-platform edge parity remains a quality issue to investigate. Same-platform hash equality is not a substitute for that investigation.

## Optical highlight scatter experiment

A separate high-highlight night scene (`/tmp/laika-m9-results/photo/mountains-source.jpg`, SHA-256 `4e6661606c195b2462c0f2321a7feefe2ae2fc081499f3462a30fbb108cd7f8e`) was processed with CPU-only M9 + Depth + ARNet/highlights. The requested output was explicitly 2048×1365 from a 5760×3840 source, so this is not an original-resolution memory result.

| Row chunk | Wall | Scatter stage | Process-tree peak RSS | cgroup peak (4 GiB limit) | Output hash |
|---:|---:|---:|---:|---:|---|
| 128 | 436.56 s | 252.92 s | 2,318,368,768 B | 2,503,704,576 B | `6c3f4d0e51314c07534b7ea76a04f6d1f7b5e896873d4c528f7e01969178839e` |
| 512 | 1,210.09 s | 208.72 s | 2,376,888,320 B | 2,562,228,224 B | Same hash |

The isolated scatter stage was 17.5% shorter in the 512-row run, with byte-identical output and 58.5 MB higher cgroup peak. However, unrelated stages varied sharply between the runs (for example, ARNet pass 1 was 35.85 s in the 128-row run and 786.50 s in the 512-row run). Therefore total wall time is confounded, the scatter timing needs repeated controlled runs, and the default remains 128 rows. Compared with the macOS reference at this output size, the 128-row result measured PSNR 54.47 dB, MAE 0.2186, RMSE 0.4821, p95/p99 absolute error 1, and maximum 5.

## Findings and next work

- The real five-model pipeline can complete in a 2 GiB-configured Linux guest for this 1080×1620 portrait, but needs nearly the entire 1.5 GiB task cgroup and about 20 minutes. Treat 2 GiB as a challenge result, not a supported service tier.
- A 4 GiB-configured guest completed the same task but was also very slow in that run. Collect repeatable, controlled timing samples before drawing speed conclusions.
- The benchmark harness now records cgroup v1/v2 counters and Docker can build the benchmark image from tracked source. No weights or photos are in the image.
- The highlighter scatter stage is a large CPU cost. Next, profile exact-preserving pupil-kernel reuse and other bounded-memory execution changes; retain the current 128-row setting until repeated measurements show a reliable gain.
- Still required: repeat runs with controlled CPU contention; 8 GiB guest comparison; representative portrait/hair/highlight scenes; original-resolution 5760×3840 complete-pipeline memory; total guest and service-process-tree measurements; 4 GiB/2 GiB runs under concurrent/API service conditions; and analysis of Linux/macOS subject-edge differences.
