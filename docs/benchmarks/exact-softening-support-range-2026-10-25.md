# Exact softening radiance-support query pruning (2026-10-25)

## Universal algorithm

The reference separable highlight-softening method computes two horizontal
and two vertical passes. It computes FP64 prefix sums and queries the
weighted integral for every pixel in each active line, even where weighted
radiance cannot contribute. Earlier PR #83 skipped fully dark input rows.

This improvement preserves the FULL-LENGTH original FP64 prefix sums.
For each active line, determine first and last nonzero weighted radiance
source cells, and query only positions in their bounding range expanded
by ceil(maximum per-pixel CoC integration half-width)+2. For targets outside
this interval, weighted color integrates to exactly zero, and their
original radiance and/or mixing weight cannot produce an update.
Each later orthogonal pass recomputes bounds on the CURRENT output.

The approach is independent of image resolution. An internal reversible
_radiance_support_range=False switch retains the old code path.
No model inference, kernel, float precision, resolution, CoC values,
cancellation checks or number of passes are altered.

## Real native 19.96MP Windows CPU ABBA (same process configuration)

Genuine 5472x3648 CC0 photo scattered radiance and original CoC, old/new/
new/old fresh interpreter order on Intel i9-12900H:

| Order | Variant | Softening seconds | Sampled RSS (bytes) |
|---|---|---:|---:|
| 1 | Existing | 12.7193 | 710,213,632 |
| 2 | Support range | 5.2933 | 751,071,232 |
| 3 | Support range | 5.3158 | 827,023,360 |
| 4 | Existing | 12.6074 | 767,361,024 |

Median 12.6634s old versus 5.3046s new: 58.1% less time in this
stage. All FOUR float32 outputs bitwise identical with SHA256
3bacf0e4665ed21e8dd60423e35656ab5033703a25d3bb969ac6e6ee3a79e6e1.
Sampled RSS across independent short runs varies substantially:
these data prove no consistent memory reduction or increase.

## Complete five-model genuine-photo validation

Windows i9-12900H, 9 ONNX CPU threads, ORIGINAL local M9, Depth,
SAM2 Tiny, ViTMatte Small and ARNet weights, native optical scatter
and original full-size output; conservative ARNet allocator OFF.

| Measure | Recorded merged develop | New branch |
|---|---:|---:|
| 5472x3648 complete wall | 130.5001 s | 125.7652 s |
| Softening stage | 10.1166 s | 5.2444 s |
| Sampled worker RSS | 1,336,766,464 B | 1,319,796,736 B |

Full-resolution final PNG SHA256 and decoded RGB pixels are EXACTLY
equal. This run was approximately 3.6% faster end-to-end.

At 3072x2048 (resized from same authentic source photograph), a separate
earlier baseline run took 46.2971s versus the new 41.7335s (~9.9%
faster, but unmatched-run variance possible). Final PNG bytes are
EXACTLY equal, new softening stage 1.4377s.

## Regression evidence and limits

14 additional exact on/off test combinations for empty/single/two
islands/dense/border/subnormal/source CoC masked light on two input
sizes, plus previous production suite. PR CI across Windows/macOS/
Linux is the release gate.

Performance depends on radiance sparsity, CoC and output scale;
dense illumination can reduce ROI benefit. These are one photograph
at two resolutions, NOT evidence of general percent-speedup across
all scenes. Linux hard cgroup 2GiB and simultaneous HTTP workers are
still outside this Windows evaluation.

No pictures, original checkpoints, converted models or raw arrays
are committed; they remain local only.

## Dense-light safeguard (final follow-up)

A second generality test used three deterministic 512x1024
synthetic light distributions, each matched old / new / new / old:
fully dense RGB radiance, one small island and two separated
islands. All comparisons were SHA256-identical.

A pure sparse-support prepass was approximately 7% slower on
the fully dense case due to redundant scanning. To prevent this
predictable regression, the final code performs a bounded 16x16
spatial sample. When all sample RGB values are strictly positive,
it uses the original full-line integral query without range
pruning. This conservative fallback can never change output;
a false dense classification only forgoes some acceleration.
The original dense softening path remains available.

In post-safeguard test runs, full-dense reference ~0.66-0.77s,
new ~0.75-0.83s with run-to-run noise; small single/double
islands still showed meaningful acceleration. Further dense-scene
profiling is warranted; we do not claim this guarantees a speed
improvement for every content pattern.

One additional strict unit test monkeypatches the sparse scanner
to raise an error for dense input, proving dense fallback really
does use the complete original integral path. Total added tests:
15 parameterized/specialized cases.

Re-running the actual native 20MP radiance after this final
safeguard yielded 5.2604s and the SAME original float32 output
SHA256 3bacf0e4665ed21e8dd60423e35656ab5033703a25d3bb969ac6e6ee3a79e6e1.
