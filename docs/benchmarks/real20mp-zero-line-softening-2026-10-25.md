# Skip empty-radiance lines in exact CoC-weighted softening

## Mechanism
The original highlight-softening stage makes two horizontal+vertical
separable tent passes. Every line creates FP64 weighted-color and
prefix-integral arrays, even when all RGB channels are exactly zero.
With zero source radiance in the currently processed line, the
weighted input and integrated color are exactly zero across that
whole axis pass. Skipping those calculations cannot add or remove light.

The optimized code checks the in-place line after the row weight
check. The next perpendicular pass independently rechecks every
line: energy may have arrived from the previous axis. Progress and
cancellation checks are retained. The private _skip_zero_lines=False
argument executes the original path for exact A/B. No image
dimensions, original ONNX weights, spatial kernels, or number of
horizontal/vertical softening passes are changed.

## Windows genuine native-20MP ABBA (2026-10-25)

Hardware: Intel Core i9-12900H Windows, Python 3.11.
Genuine saved field-scattered radiance + CoC from the original
5472x3648 photo, using full-precision reference math.
Each independent fresh-process run uses the same real inputs.

| Order | Variant | Seconds | Peak sampled process RSS (bytes) |
|---|---|---:|---:|
| 1 | Original | 22.3821 | 766,222,336 |
| 2 | Skip zero lines | 12.5962 | 757,706,752 |
| 3 | Skip zero lines | 12.6200 | 758,231,040 |
| 4 | Original | 22.2364 | 813,387,776 |

Original softening median 22.30925s, candidate median 12.6081s,
equivalent to a 43.5% stage-time reduction. All four float32
softened arrays had IDENTICAL SHA256:
3bacf0e4665ed21e8dd60423e35656ab5033703a25d3bb969ac6e6ee3a79e6e1.

The real input contained 1,026,801 nonzero radiance pixels
(5.14% of 19,961,856 pixels). Before processing,
2,819 of 3,648 rows and 2,089 of 5,472 columns were nonzero.
The optimization costs an extra line scan and may be slower for
fully dense high-radiance scenes.

Five unit test parameterizations include empty, single bright point,
border highlights, dense radiance and vertical radiance.
This stage microbenchmark is not full service end-to-end proof.
Strict Linux 2GiB memory limits and multiworker HTTP loads
remain separate validation requirements. Local model and photo
assets are excluded from Git.
