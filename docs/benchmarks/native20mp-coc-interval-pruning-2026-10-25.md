# CoC interval pruning: native 20 MP complete-pipeline A/B, 2026-10-25

## Mechanism

The original field-optical scatter code evaluated every depth layer at every
highlight-output support region, even if the depth values in that region could
not possibly contribute to the layer. The layer weight has exact support:
for integer layer L, weight is zero whenever the scaled CoC is outside
(L-1, L+1).

For each 64-row output band, optical pupil node and sparse output support
span, compute one conservative minimum/maximum depth bound across the region
expanded by the MAXIMUM active pupil halo. Use extra outward rounding margin
to avoid dropping float32 boundary contributions. For each layer, skip a
node/span ONLY when its layer weight is provably zero throughout this source
region. Valid convolution, image resolution, occlusion/coverage behavior,
optical pupil geometry, blend order and model weights do not change.

The prescan stores bounded per-band geometry and two float bounds rather than
new full-resolution image copies. The private switch
_coc_interval_pruning=False is an explicit regression/benchmark opt-out.
Default-enabled fast path applies only with field optics and existing exact
highlight-output support; the old fallback is used otherwise.

This performance improvement does NOT depend on the optional optical pupil
cache experiment in PR #79. The implementation branch originates directly
from develop; only the CoC interval algorithm and its regression tests are
needed.

## Measured Windows native-resolution A/B

Hardware: Intel Core i9-12900H, 14 physical cores, 20 logical processors.
ONNX CPUExecutionProvider, 9 model-inference threads. One original genuine
CC0 photograph at 5472 x 3648 (19,961,856 pixels).
Unmodified M9, Depth V2 Small, SAM2.1 Tiny, ViTMatte Small and ARNet model
assets, low-memory strategy, SAM memory arena disabled, verified split ARNet
edge=256, ViTMatte 512 tiles and optical scatter. Both runs perform the
ENTIRE five-model processing chain including subject-aware bokeh and export.

| Metric | Original code | CoC pruning experimental code |
|---|---:|---:|
| Full end-to-end elapsed | 345.0583 s | 168.3383 s |
| Optical highlight scatter stage | 247.5641 s | 75.3172 s |
| Full-worker sampled RSS peak | 1,379,151,872 bytes | 1,366,110,208 bytes |
| Final output | 5472 x 3648 PNG | 5472 x 3648 PNG |

Single-photo relative reductions: about 51.2% end-to-end duration and
about 69.6% optical scatter duration (about 3.29x scatter throughput).

Final output is BYTE IDENTICAL. Original and pruned SHA256:
1348289adb6c6a05e32e7468af5de38de56cccac715457266ee41194526a4070.
Measured RGB changed pixel count: ZERO; maximum channel difference: ZERO;
mean channel difference: ZERO.

A separate true 2048 x 1365 captured optical input produced byte-identical
float32 scattered radiance arrays and reduced scatter time from 7.97s to
3.05s (about 2.61x) on Windows; saved array SHA256:
a477000897bd26607ffb298f9afdb19d7daef794c49bf4cef72a6ced9bc4821e.

The native scene had 169 distinct CoC layers. A separate 1.03s geometry
prescan estimated 86.3% of weighted layer-region combinations to be
provably noncontributing. This is theoretical work eligibility, NOT the
measured FLOP reduction or acceleration factor.

## Test coverage / caution

The automated test suite includes all original optical scattering tests
plus sparse/dense light, empty and border light, disjoint CoC regions,
reflected halos, small row bands, and difficult float CoC layer boundaries.
Default-on and explicit opt-out produce the same output in these tests.

Numbers are for this one genuine photo and this Windows CPU, not a
guarantee for other images or hardware. Process-tree sampled RSS is NOT a
strict 2GiB Linux cgroup high-water mark. Native 20MP images with extremely
dense highlights, different apertures, concurrent API workers, and
strict 2/4GiB Linux constraints require additional acceptance coverage.

Authorized checkpoints, locally captured per-stage arrays, photos,
and rendered outputs remain only under E:\codex\LeicaShot-local-data and
are deliberately excluded from Git.

## Independent default-enabled production invocation

A second validation used only the three-file standalone branch derived
directly from develop, without the pupil cache and without monkey-patching
any function. The actual Python CLI processing pipeline was executed in
a fresh process with 9 ONNX CPU threads and all five models enabled.

- Full native 5472x3648 job: 191.0262 seconds.
- Optical scatter: 83.7020 seconds.
- Sampled process-tree RSS peak: 1,374,720,000 bytes (1311.04 MiB).
- Output PNG SHA256: 1348289adb6c6a05e32e7468af5de38de56cccac715457266ee41194526a4070.
- Final image SHA256 and decoded RGB arrays EXACTLY equal to original
  unoptimized production output (zero differing pixels/channels).
- Relative to the original 345.0583s run, this independent run was
  44.6% faster end-to-end; optical scatter was about 2.96x as fast.

The difference between the independent run (191.0s) and the first
experimental instrumented run (168.3s) is NOT attributed to the pupil
cache or a specific cause without controlled paired tests. Both are
single-run observations on the same Windows laptop, not promises for
all scenes and CPU loads. The standalone implementation is independent
of the pupil-cache optimization.

Full Python regression suite in the independent worktree: 240 passed.
Ruff lint and format checks passed.
