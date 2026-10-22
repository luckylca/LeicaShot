# Empty-radiance skip for CPU highlight scatter — 2026-10-22

## Change under evaluation

Field-optical scattering still computes the single-channel CoC-layer mass for
every active field node because that mass occludes highlights from farther
layers. It now skips allocation of the float64 RGB weighted-light tile when its
source highlight tile is all zero, and skips the RGB `filter2D` call when the
CoC-weighted tile is all zero. This does not change depth, optical kernels,
coverage, occlusion, output resolution, or nonzero convolution arithmetic.

## Paired real-photo measurement

The input is the actual 5760×3840 M9 RGB result and matching Depth Anything V2
map from the night-scene fixture. The benchmark resized those cached arrays to
the listed standalone scatter dimensions, using the same engine resize methods;
it did not run M9, Depth or ARNet in these timing pairs. Each worker ran the
baseline from commit `762ff938c7486f2dc9fa60af0c26a789907b6969` or the current
implementation in a fresh process. The 512-row chunks, focus `(0.5, 0.5)`,
aperture `2.8`, radius scale `3.0`, field optics, and prepared inputs were held
constant. SHA-256 equality below is over the complete float32 scatter array.

Host: MacBook Air M1, macOS 15.7.7, 8 logical CPUs, Python 3.11.16, OpenCV
5.0.0 with 8 reported threads. A two-vCPU Linux cgroup benchmark was running
concurrently, so absolute timings are host-load dependent; the order was
reversed at each size to expose run-position effects. Process-tree RSS was
sampled every 10 ms during scatter. Linux cgroup memory is unavailable for
these macOS workers.

| Size | Order | Baseline time / sampled peak RSS | Optimized time / sampled peak RSS | Time change | Scatter SHA-256 |
|---:|---|---:|---:|---:|---|
| 1024×683 | Baseline first | 25.698 s / 218,546,176 B | 19.230 s / 206,438,400 B | 25.2% faster | `4d28f82e4e2c5ac3eefabf703b2dfbc631831b1c4c6f44b80b0cbe5ea06da99a` |
| 1024×683 | Optimized first | 24.849 s / 226,312,192 B | 12.333 s / 213,794,816 B | 50.4% faster | `4d28f82e4e2c5ac3eefabf703b2dfbc631831b1c4c6f44b80b0cbe5ea06da99a` |
| 2048×1365 | Baseline first | 207.954 s / 411,402,240 B | 112.984 s / 408,813,568 B | 45.7% faster | `849f60f28cf2f1efe4498a2549378ecae5c28e433fb6c275ef7ba2b80de9c731` |
| 2048×1365 | Optimized first | 227.922 s / 447,545,344 B | 116.283 s / 413,335,552 B | 49.0% faster | `849f60f28cf2f1efe4498a2549378ecae5c28e433fb6c275ef7ba2b80de9c731` |

At 1024, highlight extraction found 115 nonzero pixels in the prepared frame;
at 2048×1365 it found 1,903. Both runs at each size produced bitwise-equal
float32 scatter arrays. The baseline-first and optimized-first 2048 trials
measured 45.7% and 49.0% lower scatter time respectively. Run-to-run timing
still varied, especially at 1024, so these are paired host measurements rather
than a throughput guarantee.

Source photo SHA-256: `4e6661606c195b2462c0f2321a7feefe2ae2fc081499f3462a30fbb108cd7f8e`.
Prepared M9 RGB SHA-256: `e631ed1e7885e9ebdff8eefa2631ced973e3744472201fb0d01f3c2803f9edba`.
Raw depth-map SHA-256: `649fb038f693af8a3ef0c5e1ee59d6b3c758f4bc908ea62b049ee3fa04d34332`.

The reproducible comparison tool is
[`scripts/benchmark_highlight_scatter.py`](../../scripts/benchmark_highlight_scatter.py).
It records the baseline revision, input hashes, runtime versions, output hashes,
process-tree RSS, and cgroup counters where available. Full invocation reports
and intermediate arrays are kept under `/tmp/laika-pipeline-results`, outside
Git.

## Rejected exact-cache experiment

An earlier candidate retained all field-pupil kernels for a radius and reused
them between positive and negative CoC levels. Its outputs were bitwise equal,
but two 1024×683 comparisons measured 37.424 s versus 24.166 s (baseline first)
and 47.319 s versus 43.409 s (optimized first). Sampled RSS was inconsistent
between the pairs. The cache was removed because it showed no repeatable speed
or memory benefit.

## Full-resolution Linux task status

The separate source-size task uses the same 5760×3840 photo, the real M9,
Depth Anything V2 and BokehMe ONNX assets, CPUExecutionProvider, one inference
thread, a 4 GiB Ubuntu ARM64 Lima guest and a 3 GiB cgroup. It was started from
the pre-optimization `develop` code. At elapsed 4,374.6 s, highlight scatter
was at 87.74%, process-tree RSS was 2,123,751,424 B, cgroup current was
2,239,565,824 B, and cgroup peak was 2,676,203,520 B of 3,221,225,472 B. All
cgroup OOM and limit events remained zero. It has not exported the PNG; this is
not a completed full-resolution acceptance result. The optimized implementation
still needs its own source-size cgroup run.

## Unified engine output regression

After the standalone pairs, the shared Python processing engine ran the actual
5760×3840 source through M9 + LabFusion, Depth Anything V2, tiled BokehMe ARNet,
and optical highlight processing with Low Memory strategy, one ONNX thread,
CPUExecutionProvider, and an explicitly requested 2048×1365 output. This was a
full coordinated engine request, not separate model commands. It ran while the
Linux 2-vCPU full-resolution cgroup task was active, so wall time and RSS include
shared-host contention and are not compared as a speedup claim.

- Total task wall: 328.207 s; sampled process-tree peak RSS: 1,083,146,240 B.
- Highlight scatter: 117.776 s; sampled stage peak RSS: 476,954,624 B.
- Depth inference: 81.364 s; M9 inference: 13.230 s; Lab fusion: 14.959 s.
- ARNet passes: 40.402 / 6.991 / 7.578 s.
- Output: 2048×1365 RGB PNG, 2,627,627 bytes, SHA-256
  `ec7b8d51001ce7bdbfa73e0ea59c8c046e7b4d9df2d42e3a69e0ab93a04201d1`.
- Model SHA-256: M9 `8936b8c86656265ab07c30b9b4cbbd7fa195d6e44cb3be1d4fd059bb36781e33`,
  Depth `a8ce30941c85b2c9af29fb26f36fbf7be3c1ba1dcf348187a51f6eaa856a238e`,
  ARNet `f0696c16a6eb02099292176f25b47e4dc53863e7e4f3f7748e66470d54cf0173`.
- The output file is byte-for-byte identical to the pre-optimization full
  engine result at the same settings (`cmp` exit status 0).

## Regression checks and limits

- `pytest -q`: 184 passed, with one existing Starlette/httpx deprecation warning.
- Ruff lint, Ruff format check, and `git diff --check` passed.
- The 1024×683 and 2048×1365 real-photo scatter arrays match the pinned baseline
  byte-for-byte. This measures only the standalone highlight scatter stage.
- Unit fixtures exercise positive zero, negative zero, and nonzero radiance;
  the single-channel occlusion mass filter remains active in the zero-radiance
  case.
- The 5760×3840 end-to-end result, final memory peak, stage timings and decoded
  image comparison remain pending. No 4 GiB or 2 GiB full-system claim follows
  from the smaller standalone pairs or this explicitly requested 2048×1365 run.
