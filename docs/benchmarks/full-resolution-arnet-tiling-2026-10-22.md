# Full-resolution ARNet tiling and memory study — 2026-10-22

## Change under evaluation

The pinned BokehMe ARNet applies a 2×2 space-to-depth transform, then a 128-channel convolution stack with eight 3×3 convolutions. At 5760×3840, one FP32 128-channel intermediate at half width and height occupies 2,831,155,200 bytes (2.64 GiB). This single feature map exceeds a 3 GiB task budget once the rest of the image pipeline is resident.

In Low Memory mode, `BokehMeARNet` now tiles even-sized frames above 4,000,000 pixels into 512×512 core regions. Each region receives a 16-pixel halo, derived from the eight convolution layers at half resolution. Tile origins and sizes remain even, preserving the space-to-depth phase; only core output pixels are copied to the full-size NCHW result. Balanced mode and smaller or odd-sized frames continue to use one full-frame invocation. The final result dimensions do not change.

The field-optical scatter path also avoids allocating its unused outer RGB output before dispatching to the field-optical implementation. Its final residual-light composition now runs in row chunks rather than materializing a full-frame coverage complement and RGB multiplication temporary. The arithmetic order for each output pixel is unchanged.

## Real ONNX comparison

Input: the real M9 output and matching Depth Anything V2 map derived from `/tmp/laika-m9-results/photo/mountains-source.jpg`; ONNX Runtime CPUExecutionProvider, one intra-op thread; focus `(0.5, 0.5)`, aperture `2.8`, first-pass defocus `clip(CoC / 10, -1, 1)`.

| Method | Input size | Tiles | Inference time | Raw FP32 SHA-256 |
|---|---:|---:|---:|---|
| Full frame | 1024×682 | 1 | 23.740 s | `09691fc42a0996e1457be840b399b918cc6e666d0129e2fe4ca9b922e41d9744` |
| Low Memory, 512 core / 16 halo | 1024×682 | 4 | 16.481 s | Same |

Maximum and mean absolute FP32 output error were both exactly zero; no output values differed. The runtime figures are one sequential trial on a shared Mac host and are not a stable throughput comparison. This test validates tile seams and border handling on an actual model/image pair, but it does not prove full-frame equality at 5760×3840 because the full-frame inference exceeds the Linux memory budget.

The 2048×1365 check is odd-height, so it correctly stayed on the full-frame path for both sessions; it did not exercise tiling. Large odd-sized photos therefore remain a low-memory gap.

## Full-resolution Linux cgroup runs

Both runs used the same 5760×3840 night photo (`4e6661606c195b2462c0f2321a7feefe2ae2fc081499f3462a30fbb108cd7f8e`), the local hash-validated M9, Depth Anything V2 q64 and ARNet models, CPUExecutionProvider, one inference thread, a two-vCPU Ubuntu ARM64 guest configured with 4 GiB, and a 3 GiB cgroup limit. No output-size override was supplied.

### Without ARNet tiling

- Docker exited 137 and reported `OOMKilled=true` after 222.6 s.
- The last persisted progress record was the start of the first full-scale ARNet call. At that checkpoint cgroup current was 1,202,167,808 B, prior cgroup peak was 1,667,424,256 B, and process-tree RSS was 864,276,480 B.
- The killed cgroup did not preserve its final memory peak. No output was produced.

### Full-resolution tiled run and row chunk experiment

Both full-size attempts use the same 5760×3840 source with no output-size override, the same verified models, a 4 GiB Ubuntu ARM64 Lima guest, two vCPUs, CPUExecutionProvider, one ONNX thread, and a 3 GiB task cgroup.

The first tiled attempt used the existing 128-row scatter default. It completed every ARNet scale and entered optical scatter, but optical scatter remained too slow for this validation window. At elapsed 1,055.6 s, the latest saved record showed scatter progress `0.83414`, cgroup current 2,543,575,040 B, cgroup peak 2,740,789,248 B, process-tree RSS 2,175,488,000 B, and zero OOM events. The run was deliberately stopped to test the previously measured faster 512-row setting. Docker reports exit 137 from the stop timeout and `OOMKilled=false`; this was an incomplete run, not an OOM result, and it produced no output.

The second tiled attempt uses a 512-row scatter chunk. At the latest checkpoint (elapsed 612.1 s), the complete M9, Depth, and all ARNet scales had finished; optical highlight scatter was still running at progress `0.83276`. Cgroup current was 2,340,409,344 B, cgroup peak 2,662,400,000 B of 3,221,225,472 B, process-tree RSS 2,152,103,936 B, and OOM counters were zero. The task had not exported the final PNG, so total task time, final peak and output hash are not yet available.

The 512-row setting follows a real 2048×1365 paired run where scatter was 15.7% faster than 128 rows with byte-identical output and 13.6 MB more sampled process-tree RSS. The full-resolution run is the first cgroup check of that tradeoff at source size. The 5760×3840 ARNet inference path itself completed every scale in this run; the remaining highlighter work is now the run-time bottleneck. Do not treat either incomplete scatter run as a complete-task acceptance result.

## Regression and limits

- `tests/test_bokeh.py` compares a full-frame and four-tile real-shaped fake-session result exactly and checks that odd dimensions retain the validated full path.
- `tests/test_highlights.py` verifies the chunked residual-combination expression is bitwise identical to the previous full-array expression and that field-optical dispatch avoids the unused full RGB allocation.
- Full local suite: 182 passed with one existing Starlette/httpx deprecation warning. Ruff lint, format, script help, and `git diff --check` passed.
- The 512-row full-size tiled task remains active at this report snapshot. Do not mark the 8 GiB complete-function target or any 4 GiB full-system acceptance as achieved from this in-progress run.
- Tiling currently applies only to even dimensions. The full-resolution output canvas is retained, but full 5760×3840 tile-vs-frame image parity cannot be measured under the same memory limit because the reference full-frame model call OOMs.
- The scatter optimization is exact per pixel in unit fixtures. A completed real full-resolution photo comparison is still pending.
