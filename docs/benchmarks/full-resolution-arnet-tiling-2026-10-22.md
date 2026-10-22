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

### With ARNet tiling and row-bounded residual scatter

The real run has completed all ARNet scales and entered optical highlight scatter. At the latest checkpoint it is still running; no final result, complete-task time, or final peak is claimed yet.

- Latest persisted checkpoint: scatter progress `0.8314` of the scatter stage.
- cgroup current: 2,431,787,008 B; cgroup peak so far: 2,740,789,248 B of 3,221,225,472 B; process-tree RSS at the latest progress callback: 2,063,863,808 B.
- OOM counters remained zero. The final output has not yet been exported.

The standalone 2048×1365 scatter stage previously took 286.43 s on macOS and 180.70 s in a separate Linux cgroup pipeline, so full-resolution optical scatter is expected to dominate this run. The run will be reported as a complete pipeline only after PNG export succeeds and its dimensions, hash, stage timings, process-tree peak, and final cgroup events are collected.

## Regression and limits

- `tests/test_bokeh.py` compares a full-frame and four-tile real-shaped fake-session result exactly and checks that odd dimensions retain the validated full path.
- `tests/test_highlights.py` verifies the chunked residual-combination expression is bitwise identical to the previous full-array expression and that field-optical dispatch avoids the unused full RGB allocation.
- Full local suite: 182 passed with one existing Starlette/httpx deprecation warning. Ruff lint, format, script help, and `git diff --check` passed.
- The full-size tiled task remains active and uncommitted at this report snapshot; update this section before merging. Do not mark the 8 GiB complete-function target or any 4 GiB full-system acceptance as achieved from this in-progress run.
- Tiling currently applies only to even dimensions. The full-resolution output canvas is retained, but full 5760×3840 tile-vs-frame image parity cannot be measured under the same memory limit because the reference full-frame model call OOMs.
- The scatter optimization is exact per pixel in unit fixtures. A completed real full-resolution photo comparison is still pending.
