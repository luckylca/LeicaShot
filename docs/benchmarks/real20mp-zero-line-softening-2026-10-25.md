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

## Complete native 20MP five-model proof

After optical-stage ABBA, the candidate was executed by the actual
standalone Python low-memory CLI with M9, Depth V2 Small, SAM2 Tiny,
ViTMatte Small, three ARNet bokeh inference passes, field optics,
highlight softening and full PNG export. Hardware, 9 CPU threads,
original portrait, model hashes, ARNet split edge 256, SAM arena
setting and all photo-processing options matched merged PR #82.

| Whole-process metric | Merged PR #82 | Candidate |
|---|---:|---:|
| Complete 5472x3648 job | 140.6811 s | 129.9214 s |
| Highlight-softening stage | 18.2830 s | 10.0079 s |
| Optical scattering stage | 43.8047 s | 43.7902 s |
| Sampled whole-worker RSS | 1,306,570,752 B | 1,317,445,632 B |

Full job time was approximately 7.6% shorter on this run, with
approximately 10 MiB higher sampled peak RSS (within the normal
run-to-run memory variation, not a demonstrated systematic memory
increase). Compared with the much older original pre-PR #80 345.06s
run, total latency is approximately 62.4% lower across all combined
changes, but this is not one controlled multi-photo benchmark.

The final PNG SHA256 exactly matched the merged PR #82 photo, as
verified by reading both real output files. 5472x3648 RGB data had
zero different pixels and a maximum 8-bit channel difference of zero.

A production-default merge still requires full Python unit tests and
cross-platform CI. Linux cgroups and multi-API-concurrency are pending.
