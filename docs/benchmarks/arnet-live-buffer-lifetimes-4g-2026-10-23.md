# ARNet live-buffer lifetime measurement

Updated: 2026-10-23

## Scope and decision

The renderer releases the full-resolution float32 ARNet source, composite,
last-pass resize buffer, and subject-mode CoC plate after producing the final
uint8 composite. The later highlight recovery and scatter stages do not use
those frames. Per-pass resize aliases are also cleared promptly. This is an
exact-preserving lifetime change; the model, scale sequence, input dimensions,
blend math, and output dimensions are unchanged.

Keep the change as a stage-local memory reduction. In the paired full-size
measurement, the highlight-scatter process-tree stage peak fell by about
587 MiB (24.99%) and the sampled cgroup current-memory maximum during scatter
fell by 513,798,144 B (21.60%). The output PNG hash was unchanged. The complete
task cgroup high-water mark did **not** improve: it rose by 147,734,528 B
(5.93%) in this single pair, with the largest high-water point still occurring
in ARNet. Therefore this change is not evidence of lower whole-task peak RSS,
does not solve the 2 GiB OOM, and makes no speed claim. Continue with the
measured ARNet peak itself.

## Paired run

The baseline is the PR #51 implementation (`49eb2c5`), run from Docker volume
`leicashot-arnet-channel-stream-4g-20261023`. The candidate is
`perf/arnet-live-buffer-lifetimes`, from volume
`leicashot-arnet-lifetime-4g-20261023`. Both used the same 5760×3840 CC0
mountains JPEG (SHA-256
`4e6661606c195b2462c0f2321a7feefe2ae2fc081499f3462a30fbb108cd7f8e`), M9,
Depth Anything V2, SAM, ViTMatte, ARNet, and highlight stages, Low Memory
strategy, CPUExecutionProvider, one inference thread, 512-pixel matte tiles,
256-row highlight chunks, one subject point, and native-size output. Both ran
on Linux ARM64 with a 2-vCPU quota, a 4 GiB cgroup v2 limit, and no swap. The
host was an M1 Mac with 8 GiB RAM. Runtime: Python 3.11.17, NumPy 2.4.6,
OpenCV 5.0.0, ONNX Runtime 1.31.0.

Model hashes matched the existing pinned assets: M9
`8936b8c86656265ab07c30b9b4cbbd7fa195d6e44cb3be1d4fd059bb36781e33`, Depth
`a8ce30941c85b2c9af29fb26f36fbf7be3c1ba1dcf348187a51f6eaa856a238e`, ARNet
`f0696c16a6eb02099292176f25b47e4dc53863e7e4f3f7748e66470d54cf0173`, SAM
encoder `37de035410f11fb505c210ce1f8b10b450cb018fbfa7f226ec7e7ab542955c75`,
SAM decoder `3a6a96c62fd39c1369641ea626e5b2e4aed10de877f85291e9993a044d5bf981`,
and ViTMatte `a5c0b49a664867021eaf2b9cadcdc236ea511028d5e709303b418431f20e223b`.

| Measurement | PR #51 baseline | Candidate | Change |
|---|---:|---:|---:|
| Complete task wall | 1,345.712 s | 1,192.392 s | −11.39%; uncontrolled single-run timing |
| ARNet stage process-tree peak | 2,536,923,136 B | 2,561,646,592 B | +24,723,456 B (+0.97%) |
| Highlight-scatter stage process-tree peak | 2,457,657,344 B | 1,842,221,056 B | −615,436,288 B (−25.03%) |
| Complete process-tree peak RSS | 2,536,923,136 B | 2,561,646,592 B | +24,723,456 B (+0.97%) |
| Complete cgroup memory peak | 2,489,241,600 B | 2,636,976,128 B | +147,734,528 B (+5.93%) |
| cgroup OOM / limit events | 0 / 0 | 0 / 0 | — |
| Output | 5760×3840 PNG | 5760×3840 PNG | same dimensions |
| Output SHA-256 | `0a6282df408319950f0bd4c8f56bcfce78ed9e7eba95ce9c50b2c70f5555ade2` | same | byte-identical |

The 513,798,144 B scatter cgroup-current reduction compares the maximum
periodic `memory.current` samples recorded while the stage was active
(2,378,280,960 B baseline and 1,864,482,816 B candidate). The cgroup
`memory.peak` counter is a whole-task high-water mark, so its stage-labelled
telemetry is not a stage-local peak. The candidate's global cgroup peak was
reached during ARNet, before released buffers could reduce the later scatter
working set.

The run used the same output hash as prior full-resolution subject runs. This
is one pair; stage-local memory evidence is strong, while whole-task peak and
timing must not be attributed to this change. Artifacts (images, models,
progress JSONL, and profiles) remain in the two local Docker volumes and are
not committed.
