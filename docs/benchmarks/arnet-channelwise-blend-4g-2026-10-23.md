# ARNet channel-wise multiscale blend

Updated: 2026-10-23

## Goal and method

The previous renderer enlarged each later ARNet pass into a full-resolution
HWC float32 array, then blended that array into the previous pass. This
experiment resizes one NCHW channel at a time into a reusable H×W float32
scratch array and blends it directly into the existing composite. In Low Memory
mode the scratch is released before the next ARNet pass. Balanced mode keeps
the existing full-frame path. The resize kernel, scale sequence, blend
expression, output dimensions, and model inputs are unchanged.

The full HWC temporary is 265,420,800 bytes at 5760×3840; the channel scratch
is 88,473,600 bytes. The theoretical temporary-buffer reduction is
176,947,200 bytes (168.75 MiB). The test fixture checks bitwise float32 equality
against the previous HWC blend expression.

The controlled pair used the same 5760×3840 CC0 mountains JPEG (SHA-256
`4e6661606c195b2462c0f2321a7feefe2ae2fc081499f3462a30fbb108cd7f8e`),
`color+bokeh`, Low Memory, CPUExecutionProvider, one thread, focus `(0.5, 0.5)`,
aperture `2.8`, subject point `(0.72, 0.66)`, 512-pixel ViTMatte tiles, 256-row
highlight chunks, and no output-size override. Both containers ran on Linux
ARM64 with two CPUs, a 4 GiB cgroup memory limit, and no swap (`memory.swap.max`
was zero). Runtime was Python 3.11.17, NumPy 2.4.6, OpenCV 5.0.0, and ONNX
Runtime 1.31.0.

Model hashes were M9 `8936b8c86656265ab07c30b9b4cbbd7fa195d6e44cb3be1d4fd059bb36781e33`,
Depth Anything V2 `a8ce30941c85b2c9af29fb26f36fbf7be3c1ba1dcf348187a51f6eaa856a238e`,
ARNet `f0696c16a6eb02099292176f25b47e4dc53863e7e4f3f7748e66470d54cf0173`,
SAM encoder `37de035410f11fb505c210ce1f8b10b450cb018fbfa7f226ec7e7ab542955c75`,
SAM decoder `3a6a96c62fd39c1369641ea626e5b2e4aed10de877f85291e9993a044d5bf981`,
and ViTMatte `a5c0b49a664867021eaf2b9cadcdc236ea511028d5e709303b418431f20e223b`.
The baseline used `develop` commit `771602a`; the candidate used the
channel-wise blend implementation on `perf/arnet-channel-stream-blend`.

## Full-resolution results

| Measurement | Baseline | Candidate | Change |
|---|---:|---:|---:|
| Complete task wall time | 1,301.466 s | 1,345.712 s | +44.246 s (+3.40%) |
| Highlight scatter | 988.699 s | 1,016.608 s | +27.909 s (+2.82%) |
| ARNet passes total | 225.002 s | 235.838 s | +10.836 s (+4.82%) |
| Process-tree peak RSS | 2,569,089,024 B | 2,536,923,136 B | −32,165,888 B (−1.25%) |
| cgroup memory peak | 2,559,774,720 B | 2,489,241,600 B | −70,533,120 B (−2.76%) |
| cgroup OOM / limit events | 0 / 0 | 0 / 0 | — |
| Output PNG | 16,655,520 B | 16,655,520 B | — |
| Output SHA-256 | `0a6282df408319950f0bd4c8f56bcfce78ed9e7eba95ce9c50b2c70f5555ade2` | same | byte-identical |

ARNet-stage sampled process-tree RSS fell from 2,569,089,024 B to
2,536,923,136 B. The cgroup current-memory sample maximum in the ARNet stage
fell from 2,529,640,448 B to 2,464,600,064 B. In the tiled-ARNet substage,
sampled process-tree RSS fell by 76,967,936 B and cgroup current memory fell by
73,785,344 B.

This is one same-settings pair rather than a repeated statistical study. The
candidate's full-task and scatter times were slower in this pair; scatter code
was unchanged, so that timing difference is not attributed to this blend.
The smaller scratch produced byte-identical output and measurable, modest
memory savings, but did not provide enough headroom for the 2 GiB task below.

## 2 GiB boundary

Two full 5760×3840 attempts used a 2 GiB cgroup limit with no swap. Both were
OOM-killed with exit 137 after ARNet pass 2 and before pass 3 could begin; the
cgroup reached exactly 2,147,483,648 bytes and neither run produced an output
PNG. The first screening version retained its one-channel scratch across
passes. The final version released the scratch at the end of each pass and
still hit the same boundary. These are failures, not 2 GiB acceptance results.

A separate screen that called glibc `malloc_trim(0)` at pass boundaries also
failed at the same 2 GiB transition and was not retained. The next memory work
should target the persistent anonymous working set and ONNX Runtime allocation
arena, not assume that returning allocator pages or reducing one temporary
buffer is sufficient.

The JSONL traces, model files, and full-resolution images remain outside Git in
the local Docker volumes `leicashot-arnet-channel-stream-4g-20261023`,
`leicashot-arnet-channel-stream-2g-20261023`, and
`leicashot-arnet-channel-baseline-4g-20261023`.
