# ARNet CPU allocator reuse: exact-output speed / RAM trade-off

## Scope (October 25, 2026)

The CPU ONNX Runtime ExecutionProvider already runs the correct original
BokehMe ARNet FP16 graph using an even-aligned tile core of 256x256
with the original 16-pixel receptive field halo. Low-memory strategy
previously disabled ORT CPU memory arena, requiring the allocator to
recreate substantial intermediate convolution features across hundreds
of nearly-identical tiles.

This change adds an opt-in CPU arena in low-memory mode. A controlled
process environment switch enables per-session reuse of ORT-allocated
feature-map memory while preserving the original model, float16
weights, tile shape, complete 8-layer neural convolution, inputs,
outputs and model hash. Memory-pattern optimization remains disabled.

DEFAULT is OFF because full-pipeline memory regression is substantial.
Choose LAIKA_ARNET_CPU_ARENA=1 only for an appropriately provisioned
CPU worker. LAIKA_ARNET_CPU_ARENA=0 keeps the historical no-arena path.

Optional LAIKA_ARNET_ARENA_SHRINK_TILES=32 requests CPU arena shrinkage
after each 32 ARNet tile invocations. The default is 0 (no periodic
shrinkage). This knob is only evaluated when the arena is enabled.
An explicit Python BokehMeARNet(reuse_cpu_arena=bool,
arena_shrink_every_tiles=int) constructor setting overrides the process
environment, facilitating reproducible benchmarks and unit tests.
Bokeh runtime metadata reports the resolved choice.

## Genuine Windows i9-12900H experimental ARNet standalone performance

Native portrait 5472x3648 (19.96 megapixels), original authorized
local ARNet checkpoint, 9 ONNX CPU threads, 330 original tiles,
original geometry and no model conversion. The same genuine color
image and normalized real photograph CoC map were used throughout.

| Stage configuration | ARNet inference | Peak sampled RSS |
|---|---:|---:|
| low-memory, CPU arena OFF | 33.912 sec | 557,146,112 bytes |
| CPU arena ON, pattern OFF | 20.635 sec | 571,441,152 bytes |
| arena and memory pattern ON | 21.589 sec | 581,496,832 bytes |
| arena ON, shrink every 32 tiles | 20.747 sec | 566,882,304 bytes |
| arena ON, shrink every 128 tiles | 21.455 sec | 569,278,464 bytes |
| arena ON, shrink at final tile | 21.903 sec | 570,388,480 bytes |

On this real workload the memory arena improved the neural inference
by approximately 39%, with ~14 MiB extra sampled RSS in the isolated
ARNet process. Every float32 output array had the SAME SHA256:
c6396bd9bc7566f30c7500dcc489613fc41612b2798bb5d6b719354591bb3f9c.
CPU arena shrink experiments also matched the original bit-for-bit.
We are testing more independent reversed-order trials to better
quantify measurement variation.

## Complete native 20MP five-model request

The actual independent Python low-memory CLI loaded and ran all
original M9 color, Depth Anything V2, SAM2 Tiny, ViTMatte Small,
ARNet triple-pass bokeh, full-field optics, highlight fusion and
the original full-resolution output PNG. All model checkpoint hashes,
CPU threads, image and subject selection remained unchanged.

| End-to-end measure | CPU arena OFF | Arena ON | Arena ON + shrink/32 |
|---|---:|---:|---:|
| Total wall seconds | 130.5001 | 116.5998 | 117.0729 |
| ARNet first pass seconds | 33.0264 | 21.2852 | 21.5880 |
| Sampled full-worker RSS bytes | 1,336,766,464 | 1,650,794,496 | 1,616,642,048 |

This represents ~10.7% faster full processing, but approximately
+299.5 MiB sampled peak RSS with arena enabled. Periodic shrinkage
only recovered approximately 32.6 MiB of that additional peak,
and it is not proven to fit strict 2GiB Linux cgroups/4GB hardware.
The candidate should NOT replace the conservative default.

Final genuine 5472x3648 PNGs from all three runs have identical
SHA256, and no different decoded RGB channels/pixels.

## Rejected alternate approaches

- ARNet 256->384 or 512 tile-edge increases reduced one isolated
  20MP pass from 33.9s to 30.3/29.2s, but introduced tiny differences
  in floating output and up to ~74 MiB additional RSS. Not chosen
  over faster exact-output CPU arena reuse.
- Experimental FP16->FP32 graph converted locally: 2048px speed only
  improved about 5% and changed around 7.1% of final 8bit model
  output pixels. Rejected.
- Skip ARNet on zero-defocus tiles is incorrect: for the original
  neural model even zero defocus does NOT give input-exact output.

## Conditions / provenance

This benchmark uses sampled Windows process RSS, not the true Linux
cgroup peak, USS, Windows Job Object hard cap, or multi-worker
HTTP service memory consumption. Do not assume it meets a hard 2 GiB
limit simply because observed RSS is below 2 GiB. Strict workload
memory limits and multiple simultaneous requests remain to be tested.

Source photo, .npy radiance, derived experimental FP32 ONNX model,
and original restricted weights remain ONLY under the local untracked
E:\codex\LeicaShot-local-data folder. Neither original nor
experimental model weights are distributed or committed.

## Fresh-process same-host ABBA measurement

An additional 20MP standalone paired test executed in exact order
arena-OFF / arena-ON / arena-ON / arena-OFF with the same photo,
normalized depth/CoC, original FP16 model, 9 threads and 256 tile edge.
No source image resize or model changes occurred.

| Trial | CPU arena | Inference time (sec) | Peak sampled RSS bytes |
|---|---|---:|---:|
| 1 | OFF | 32.551 | 555,028,480 |
| 2 | ON | 21.945 | 570,425,344 |
| 3 | ON | 22.273 | 571,445,248 |
| 4 | OFF | 34.238 | 556,654,592 |

The median inference time was 33.3945s OFF versus 22.109s ON:
approximately 33.8% faster for this one original 20MP input.
The highest sampled standalone RSS rose 14.1 MiB.
Every trial produced the exact same full float32 output hash
c6396bd9bc7566f30c7500dcc489613fc41612b2798bb5d6b719354591bb3f9c.
The matched full five-model output PNG was also byte identical.

Because the whole pipeline with memory arena grew by approximately
300 MiB sampled RSS, the production low-memory preset intentionally
defaults to NO arena. This PR adds a reversible throughput opt-in,
not a universal low-memory optimization. Both the config and its
on/off state are exposed in Bokeh runtime diagnostics.
