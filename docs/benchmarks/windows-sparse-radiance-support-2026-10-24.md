# Exact-support RGB highlight radiance convolution experiment

Date: 2026-10-24. Experimental branch perf/win-sparse-radiance-convolution.
Baseline develop e4dcff3 (PR #71, already includes sparse FP64 CoC mass filter).

## Algorithm

For optical highlights, RGB source radiance is sparse after highlight
thresholding. When the weighted three-channel source occupies a compact
region, convolve only its exact nonzero support extended by the unchanged
pupil kernel half-width. The cropped source is explicitly zero-padded.
Fall back to full OpenCV filtering when compact padded support >=55%
of original pixels, or when a result does not overlap the output tile.

This is an exact spatial-support reduction, not a kernel approximation,
image resize, reduced precision or reduced CoC layer count. The RGB
radiance filter still uses the original float32/float64 dtype as selected
by its existing memory profile. Small floating-point differences can
arise because OpenCV chooses different internal convolution sizes.

## Controlled Windows CPU results

Same CC0 portrait/Depth Anything V2 derived CoC, f/2.8, 5x5 optical
field, 128-row chunks. The previously merged mass ROI algorithm
was enabled in **both** arms. The new radiance support optimization was
the only difference.

| Width | Reference (s) | Radiance ROI (s) | Result |
|---|---:|---:|---|
| 512 | 0.912 / 0.780 | 0.835 / 0.818 | 2.3% faster on means |
| 1024 | 4.962 | 4.269 | 14.0% faster |
| 2048 | 37.850 / 37.683 | 34.993 / 35.023 | 7.3% faster on means |

At 2048 the independently sampled process RSS peaks were
307,224,576 and 310,054,912 bytes for baseline and
309,358,592 and 307,642,368 bytes for candidate. Direction is
mixed and these results do not substantiate any whole-process
memory saving. No material memory regression is established either.

Unlike PR #71, the floating RGB scatter arrays have different hashes;
absolute error was max 7.28e-12 at 512 and 1.16e-10 at 1024,
with zero changed 8-bit composite values in those tests.
At 2048 the float32 hash differs; **no** complete 2048 quantized
pixel comparison was made yet. The physically correct kernel weights
and full-size image output are unchanged.

## End-to-end verification

The complete five-model 1024x683 real-photo pipeline used the
optimized default radiance and mass support algorithms and
produced byte-for-byte identical PNG to the already verified
unoptimized run: SHA256
3401f85308a2754771a4a41489facb3d8613f610ddd3b1ed4bfb35e62ffc03a8.
The complete request was 29.07s vs the previous mass-ROI-only
run of 30.06s. Scatter stage was 4.42 vs 4.61s.
These isolated full-request timings fluctuate in other model stages
and are not proof of a systematic whole-app speed increase.

223 Python regression tests passed including dense, random, border
and sparse radiance comparisons, and quantified tolerance to 2e-6.
Ruff lint and formatting passed.

## Limitations

No 20MP or 2/4GiB Linux cgroup result; any production
memory-tier claim still requires its own evidence. Pixel-level
identity for all images is not established by one real sample,
and float-array hashes may differ due to convolution roundoff.
The explicit private opt-out _sparse_radiance_roi=False remains
available for A/B regressions.
