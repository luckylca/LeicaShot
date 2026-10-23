# ONNX Runtime Low Memory arena policy screening

Updated: 2026-10-23

## Decision

The experiment is rejected and no session-policy code is retained. Disabling
the CPU memory arena and memory-pattern cache for Depth Anything V2, SAM, and
ViTMatte preserved the final PNG bytes, but the measured Depth stage used more
memory and ran more slowly. The complete cgroup peak rose by 125,689,856 B
(5.05%), and process-tree peak RSS rose by 47,353,856 B (1.87%). Keep ONNX
Runtime's existing default allocations for those three adapters. M9 and ARNet
already disable these options in Low Memory mode and were unchanged.

## Controlled full-resolution pair

Both CPU-only Low Memory runs used the 5760×3840 CC0 mountains JPEG, M9 +
LabFusion + Depth Anything V2 + SAM 2.1 Tiny + ViTMatte Small + ARNet and
highlights, focus `(0.5, 0.5)`, aperture `2.8`, one subject point `(0.72,
0.66)`, 512-pixel ViTMatte tiles, 256-row highlight chunks, and one inference
thread. The baseline is the PR #51 channel-wise ARNet blend run. The candidate
changed only Depth, SAM encoder/decoder, and ViTMatte session options to disable
`enable_cpu_mem_arena` and `enable_mem_pattern`; all three candidates reported
those flags as false, while baseline runtime defaults enabled both. Both runs
used Linux ARM64, a 2-vCPU quota, a 4 GiB cgroup-v2 limit, and zero swap. Host
was an M1 Mac with 8 GiB RAM running the Linux benchmark VM. Runtime: Python
3.11.17, NumPy 2.4.6, OpenCV 5.0.0, ONNX Runtime 1.31.0.

Input SHA-256:
`4e6661606c195b2462c0f2321a7feefe2ae2fc081499f3462a30fbb108cd7f8e`.
Model hashes: M9 `8936b8c86656265ab07c30b9b4cbbd7fa195d6e44cb3be1d4fd059bb36781e33`;
Depth `a8ce30941c85b2c9af29fb26f36fbf7be3c1ba1dcf348187a51f6eaa856a238e`;
ARNet `f0696c16a6eb02099292176f25b47e4dc53863e7e4f3f7748e66470d54cf0173`;
SAM encoder `37de035410f11fb505c210ce1f8b10b450cb018fbfa7f226ec7e7ab542955c75`;
SAM decoder `3a6a96c62fd39c1369641ea626e5b2e4aed10de877f85291e9993a044d5bf981`;
ViTMatte `a5c0b49a664867021eaf2b9cadcdc236ea511028d5e709303b418431f20e223b`.

| Measurement | Existing session defaults | Three adapters with arena/pattern disabled | Change |
|---|---:|---:|---:|
| Complete task wall time | 1,345.712 s | 1,337.765 s | −7.947 s (−0.59%) |
| Depth inference | 17.723 s | 23.579 s | +5.855 s (+33.0%) |
| Depth model load | 0.984 s | 1.368 s | +0.384 s |
| SAM image encoder | 5.116 s | 5.091 s | −0.025 s |
| ViTMatte inference | 6.345 s | 6.271 s | −0.074 s |
| Highlight scatter | 1,016.608 s | 1,022.066 s | +5.458 s (+0.54%) |
| Process-tree peak RSS | 2,536,923,136 B | 2,584,276,992 B | +47,353,856 B (+1.87%) |
| cgroup memory peak | 2,489,241,600 B | 2,614,931,456 B | +125,689,856 B (+5.05%) |
| OOM / limit events | 0 / 0 | 0 / 0 | — |
| Output | 5760×3840 PNG, 16,655,520 B | 5760×3840 PNG, 16,655,520 B | — |
| Output SHA-256 | `0a6282df408319950f0bd4c8f56bcfce78ed9e7eba95ce9c50b2c70f5555ade2` | same | byte-identical |

| Stage | Process-tree peak: baseline → candidate | cgroup high-water observed: baseline → candidate |
|---|---:|---:|
| Depth | 822,681,600 → 1,053,683,712 B | 880,480,256 → 1,149,751,296 B |
| Subject (SAM + ViTMatte) | 1,349,091,328 → 1,355,489,280 B | 1,669,685,248 → 1,711,841,280 B |
| ARNet and highlights | 2,536,923,136 → 2,584,276,992 B | 2,489,241,600 → 2,614,931,456 B |

The candidate task finished 0.59% sooner in this one pair, but scatter was
slower and Depth inference was 33% slower. These timing differences are not
evidence of a speed gain. The Depth-stage RSS/cgroup increases and the higher
whole-task peaks make the Low Memory policy change unsuitable. No 2 GiB run
was attempted because the candidate regressed the 4 GiB memory boundary.

Artifacts are outside Git in Docker volume
`leicashot-ort-policy-4g-20261023`; baseline artifacts are in
`leicashot-arnet-channel-stream-4g-20261023`. Neither model files, input photo,
profile, progress trace, nor result PNG are committed.
