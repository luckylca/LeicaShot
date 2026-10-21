# Depth Anything V2 global-attention query chunk sweep

## Scope

Compare query chunk sizes 16, 32, 64, and 128. Every query still attends to
the complete key/value set. Each graph is exported from the manifest-pinned
Depth Anything V2 Small checkpoint and checked against unmodified PyTorch at
four dynamic input shapes. Query chunking is exact in the tested attention
outputs; ONNX output error must remain at or below `1e-4`.

## Mac direct-model sample

Three measured inferences after one warmup on a 1400×518 float32 input, each in
a fresh worker on an 8 GiB MacBook Air M1, macOS 15.7.7, Python 3.11.16, ONNX
Runtime 1.31.0, one thread. Peak RSS is sampled across the process tree every
5 ms. Chunk 16 was run after the original 32/64/128 trials on the same pinned
toolchain.

| Graph | Peak process-tree RSS | Median inference |
| --- | ---: | ---: |
| Full attention | 1,192,050,688 B (1,136.8 MiB) | 7.397 s |
| 16 | 711,933,952 B (678.9 MiB) | 9.158 s |
| 32 | 457,261,056 B (436.1 MiB) | 8.209 s |
| 64 | 435,109,888 B (415.0 MiB) | 7.355 s |
| 128 | 487,211,008 B (464.6 MiB) | 6.792 s |

Chunk 16 was slower and used more RSS than chunks 32, 64, and 128 in this
sample. Chunk 64 had the lowest RSS; chunk 128 had the shortest inference.

## Linux direct-model sample

The opt-in GitHub model smoke ran one fresh worker per graph, with one warmup
and three inference runs, on Ubuntu 24.04 x86_64, Python 3.11.17, ONNX Runtime
1.31.0, one thread, at 1400×518. Process-tree RSS was sampled every 5 ms.

| Graph | Peak process-tree RSS | Median inference |
| --- | ---: | ---: |
| Full attention | 1,806,860,288 B (1,723.2 MiB) | 6.122 s |
| 16 | 869,384,192 B (829.1 MiB) | 8.117 s |
| 32 | 857,006,080 B (817.3 MiB) | 7.518 s |
| 64 | 871,120,896 B (830.8 MiB) | 6.751 s |
| 128 | 878,256,128 B (837.6 MiB) | 6.575 s |

All chunked graphs reduced direct-model RSS by about 51–53% against the
baseline in this run. Chunk 32 had the lowest measured RSS, while chunk 128
had the shortest inference. The Linux runner reported cgroup v2
`memory.max = max`; this was not a fixed-memory-limit experiment.

## Linux full 4K source CLI sample

One fresh `laika depth` worker per graph processed the same seeded synthetic
3840×2160 RGB input under the earlier explicit `--input-size 518` profile.
All returned a finite 2160×3840 float32 map (33,177,728-byte `.npy` file).
That lower-bound resize mapped it to a 518×924 network input.
The runner sampled process-tree RSS every 5 ms; cgroup `memory.max` remained
`max`.

| Graph | Peak worker RSS | Inference | Worker wall |
| --- | ---: | ---: | ---: |
| Full attention | 692,477,952 B (660.4 MiB) | 3.321 s | 4.231 s |
| 16 | 515,362,816 B (491.5 MiB) | 3.922 s | 4.848 s |
| 32 | 518,017,024 B (494.0 MiB) | 3.634 s | 4.572 s |
| 64 | 519,618,560 B (495.5 MiB) | 3.488 s | 4.388 s |
| 128 | 510,701,568 B (487.0 MiB) | 3.502 s | 4.411 s |

On this single 4K worker sample, q128 had the lowest RSS and q64 the shortest
wall time by about 0.02 s. Smaller query tiles did not consistently lower
RSS, and chunk 16 was slowest. These differences are not enough to establish a
universal setting.

## Decision and limits

Keep chunk size an explicit export/runtime profile choice while more Linux
memory-capped and full-pipeline tests are built. The Mac and Linux rankings
differ, so these measurements do not justify a single preferred chunk size.
They are synthetic, single-stage depth tests and do not establish real-photo
quality or the 4 GiB / 2 GiB complete-edit targets. The 4K output remains at
full source dimensions; the smaller 518×924 network raster follows the pinned
upstream lower-bound resize rule. The current default reference max-edge profile
has a separate [4K report](m3-reference-depth-analysis-sizing-2026-10-21.md).
