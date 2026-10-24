# Compact optical-band accumulation — real 20 MP validation (2026-10-25)

## Method

The merged PR #80 already removes impossible layer/node convolution pairs
via a conservative CoC support interval. However, for every surviving
row band of every depth layer it still allocates two full-width FP64
accumulators: (row_count x image_width) mass and
(row_count x image_width x 3) radiance.

This experiment computes the union of active output x intervals in the
row band and allocates the same accumulators using that width alone.
Node contributions, weights, optical pupils, FFT / OpenCV convolution
selection and summation order are unchanged. All accumulation indices are
translated by the band x origin; receiver depth, current coverage,
output and residual-own layers are updated over the matching x slice.
Outside this support, original mass and radiance contributions were
exactly zero, so no output pixel can change for finite depth inputs.

The fast path is used only when existing conservative CoC interval
pruning is active. Dense and legacy fallback paths retain full-width
buffers. A private _compact_band_output=False switch recreates
the original full-width accumulation, enabling exact A/B tests.
This branch is independent of the optional pupil-cache PR #79 and
global-layer-skip draft PR #81; it starts from merged develop at 2c30760.

## Authentic Windows CPU full-picture evidence

Machine: Intel Core i9-12900H Windows, Python 3.11,
ONNX Runtime CPU provider, 9 inference threads, identical full original
CC0 portrait 5472x3648 (19,961,856 pixels), local original M9 + depth
+ SAM2 Tiny + ViTMatte Small + split ARNet + full optics and PNG export.
Both configurations keep low-memory mode, SAM CPU arena disabled,
verified ARNet split edge 256, and ViTMatte tile 512.

| Outcome | PR #80 merged benchmark | Compact-band candidate |
|---|---:|---:|
| Whole five-model task | 191.0262 seconds | 140.6811 seconds |
| Optical scattering stage | 83.7020 seconds | 43.8047 seconds |
| Sampled full-worker peak RSS | 1,374,720,000 B | 1,306,570,752 B |
| Output resolution | 5472x3648 | 5472x3648 |

Compared to the recorded last full production PR #80 benchmark,
the new complete job ran 26.4% faster and optical scatter 47.7%
faster. Sampled worker RSS was lower by approximately 65 MiB.
Relative to pre-PR #80 original 345.06s, total combined research
optimization is approximately 59.2% shorter whole-job time.
These are individual complete runs taken on different occasions;
use new same-host interleaved A/B results for stronger attribution.

Final full-size PNGs are byte IDENTICAL, SHA256:
1348289adb6c6a05e32e7468af5de38de56cccac715457266ee41194526a4070.
Decoded RGB changed pixel count = 0; max channel difference = 0.

The independently saved real 20MP optical input also produced the
original exact float32 output hash:
c0fe387fd6efbd60ed7836d0f44519c28fdf6802401342a2466ebee57c3465ed.
Initial compact candidate optical-only time was 44.333s, with sampled
process RSS 933,810,176 B, versus past merged-PR #80 measurements
around 75.8s to 81.7s.

## Additional tests and limits

All existing 42 optical regression tests passed after the change.
Twelve more parameterized explicit opt-out-vs-default parity tests cover
sparse light, two disjoint light islands, dense highlights, edge
highlights, mixed signed CoC and row chunk sizes 1, 9 and 32.

This remains one real source photo at native 20MP. No Linux strict
2GiB/4GiB cgroup high-water or multiple simultaneous API jobs have
been validated. Sampled Windows RSS does not prove cgroup fit.
Source portraits, original or derived ONNX weights and .npy intermediates
are excluded from Git; only algorithm source, tests and text evidence
are committed.

## Actual native-20MP optical-stage ABBA, same Windows machine

After the initial full five-model output test, the merged develop branch
and this experimental branch were executed as four fresh Python processes
in reverse-order pairing, all using the EXACT same saved native 20MP
optical scatter input, aperture, row-chunk and local runtime environment.
Python module import was forced from the corresponding checked-out source
tree, not accidentally from an installed package.

| Order | Implementation | Optical seconds | Sampled process RSS (bytes) |
|---|---|---:|---:|
| 1 | develop / full-width accumulator | 75.7617 | 962,715,648 |
| 2 | new / compact accumulator | 43.9242 | 932,036,608 |
| 3 | new / compact accumulator | 42.7789 | 931,151,872 |
| 4 | develop / full-width accumulator | 75.9867 | 961,720,320 |

Median old = 75.8742 seconds; new = 43.35155 seconds.
That is a 42.9% median reduction in native 20MP scatter time.
The highest sampled RSS in this paired optical worker fell from
962,715,648 to 932,036,608 bytes, approximately 29.3 MiB less.

All FOUR runs gave byte-identical float32 scatter output SHA256
c0fe387fd6efbd60ed7836d0f44519c28fdf6802401342a2466ebee57c3465ed.
These paired data strongly support an algorithmic advantage for this real
input beyond warm-up and scheduling noise. They are still not a multi-photo
benchmark, strict cgroup test, or a guarantee for dense-light photographs.

## Additional genuine-photo size: 3072 x 2048 complete pipeline

A separate true 3072x2048 JPEG derived from the same CC0 original
was tested with the actual original and candidate Python CLI engines,
using the same 9 ONNX threads and low-memory model settings.
This is a second size/workload, NOT an independent photographed scene.

| Whole-request metric | Merged develop | Compact bands |
|---|---:|---:|
| Complete five-model + export time | 50.9748 s | 46.2971 s |
| Optical scatter | 11.2435 s | 6.1970 s |
| Sampled full process RSS | 867,704,832 B | 867,323,904 B |
| Output dimension | 3072 x 2048 | 3072 x 2048 |

The entire request was 9.2% faster at 3072x2048; optical scatter was
44.9% faster. The final PNG SHA256 values were exactly identical,
as were the decoded RGB arrays (max channel difference zero).
This reinforces optical-stage gains at a different pixel count;
no material end-to-end memory reduction was measured at this size.
