# File-backed ARNet composite screening

Updated: 2026-10-23

## Decision

The file-backed composite was screened and rejected for production. It moved
about 346 MB from anonymous memory to file-backed pages at the observed cgroup
high-water sample, but increased the complete cgroup peak by 112,414,720 B
(4.52%). Process-tree peak RSS was effectively unchanged. The full-resolution
PNG remained byte-identical. No production code from this experiment is kept.

This confirms that a NumPy memory map is still charged to the Linux cgroup; it
does not make the composite free. Under the measured 4 GiB workload, the kernel
did not reclaim enough mapped file cache to lower the cgroup high-water mark.

## Controlled full-resolution pair

Both runs processed the same 5760×3840 CC0 mountains JPEG with Low Memory
`color+bokeh`, focus `(0.5, 0.5)`, aperture `2.8`, one SAM subject point
`(0.72, 0.66)`, 512-pixel ViTMatte tiles, 256-row highlight chunks, and one
inference thread. The baseline was the channel-wise blend implementation
merged in PR #51; the candidate wrote the first ARNet composite to a temporary
FP32 memmap in the task directory, then used that mapping for later blends.
Both used Linux ARM64, a 2-vCPU quota, a 4 GiB cgroup-v2 memory limit, and zero
swap. Runtime was Python 3.11.17, NumPy 2.4.6, OpenCV 5.0.0, and ONNX Runtime
1.31.0. The host was an M1 Mac with 8 GiB RAM running the Linux benchmark VM.

The input SHA-256 was
`4e6661606c195b2462c0f2321a7feefe2ae2fc081499f3462a30fbb108cd7f8e`. Model
hashes were M9 `8936b8c86656265ab07c30b9b4cbbd7fa195d6e44cb3be1d4fd059bb36781e33`,
Depth Anything V2 `a8ce30941c85b2c9af29fb26f36fbf7be3c1ba1dcf348187a51f6eaa856a238e`,
ARNet `f0696c16a6eb02099292176f25b47e4dc53863e7e4f3f7748e66470d54cf0173`,
SAM encoder `37de035410f11fb505c210ce1f8b10b450cb018fbfa7f226ec7e7ab542955c75`,
SAM decoder `3a6a96c62fd39c1369641ea626e5b2e4aed10de877f85291e9993a044d5bf981`,
and ViTMatte `a5c0b49a664867021eaf2b9cadcdc236ea511028d5e709303b418431f20e223b`.

| Measurement | PR #51 baseline | File-backed candidate | Change |
|---|---:|---:|---:|
| Complete task wall time | 1,345.712 s | 1,318.064 s | −27.648 s (−2.06%) |
| Highlight scatter | 1,016.608 s | 987.253 s | −29.355 s (−2.89%) |
| ARNet passes total | 235.838 s | 241.909 s | +6.071 s (+2.57%) |
| Process-tree peak RSS | 2,536,923,136 B | 2,532,683,776 B | −4,239,360 B (−0.17%) |
| cgroup memory peak | 2,489,241,600 B | 2,601,656,320 B | +112,414,720 B (+4.52%) |
| OOM / limit events | 0 / 0 | 0 / 0 | — |
| Swap limit | 0 B | 0 B | — |
| Output | 5760×3840 PNG, 16,655,520 B | 5760×3840 PNG, 16,655,520 B | — |
| Output SHA-256 | `0a6282df408319950f0bd4c8f56bcfce78ed9e7eba95ce9c50b2c70f5555ade2` | same | byte-identical |

At the first periodic sample that observed the final cgroup high-water value,
the baseline had 1,814,454,272 B anonymous and 98,304 B file-backed memory; the
file-backed candidate had 1,468,866,560 B anonymous and 355,299,328 B
file-backed memory. That sample was in the ARNet stage at about 281 seconds.
This is a periodic sample of memory categories, while `memory.peak` is the
cgroup high-water mark; the two figures should not be treated as an exact
decomposition of the same instant.

The candidate's task and scatter times were lower in this single pair, while
its ARNet passes were slower. The scatter implementation did not change, so
the wall-time differences are not attributed to memmap. They are one-run
measurements with uncontrolled host/VM variation, not a demonstrated speed
gain. The slight process-tree RSS decrease is below a meaningful margin, and
the cgroup peak regressed. Keep the in-memory channel-wise implementation from
PR #51.

Run artifacts are outside Git in Docker volume
`leicashot-arnet-memmap-4g-20261023`; the PR #51 comparison artifacts are in
`leicashot-arnet-channel-stream-4g-20261023`. The model files, photo, progress
trace, profile, and generated PNG are not committed.
