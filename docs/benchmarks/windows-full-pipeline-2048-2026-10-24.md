# SUPERSEDED / RETRACTED — invalid earlier A/B comparison

**Correction 2026-10-24 (see windows-output-support-scheduling-2026-10-24.md):**
The earlier full-pipeline benchmark script had a monkeypatch wiring error.
render_bokeh locally imports scatter_highlights from laika.highlights, but
that earlier comparison patched laika.bokeh.scatter_highlights, an unused
module attribute. The purported before/after arms did not actually select
different convolution implementations. Consequently the 79.8 MB RSS
reduction and any complete-request speed difference stated below
are **NOT valid optimization evidence**. The same output SHA256 only proves
that the same output was produced, not that the control was changed.

The earlier measurements are retained below as a historical error record,
not authoritative performance evidence. See the newer corrected report
for binding-verified full-pipeline A/B results and output parity.

---
# Windows five-model full-pipeline comparison at 2048 x 1365

Date: 2026-10-24. Source baseline develop 431c1de (merged PR #71 and #72).
This experiment evaluates whether both exact-support convolution changes
help the whole CPU image pipeline, not merely an isolated helper.

## Fixed test

- Windows 10.0.22631, Python 3.11 and ONNX CPUExecutionProvider.
- Input: SHA-verified 5472x3648 real CC0 portrait explicitly resized to
  2048x1365. Output remains this full requested 2048x1365 size.
- M9 + Lab, Depth Anything V2, SAM Tiny + ViTMatte (center foreground
  point), ARNet and optical bokeh, focus center, aperture 2.8.
- Low-memory, 1 ONNX thread, 512 matte tiles, same seven ONNX weights.
- Identical program in separate fresh processes. Control disables only
  private sparse mass ROI and sparse RGB radiance ROI settings. Optimized
  uses the merged defaults.
- Sampled Python process RSS every 150ms; this is not Linux cgroup peak,
  and is not equivalent to multi-process HTTP API resource use.

## Full task results

| Metric | Control | Optimized |
|---|---:|---:|
| Whole engine wall | 80.8304 seconds | 81.5776 seconds |
| Optical scatter | 33.6106 seconds | 33.7090 seconds |
| Sampled process peak RSS | 1,376,432,128 bytes | 1,296,666,624 bytes |
| PNG bytes | 2,110,693 | 2,110,693 |

Both outputs have identical PNG SHA256:
d9816366976bd55aca1d456f367265e8731c6f168bb4b49b57606cd8ac78ac4c

**Interpretation:** this single full-flow pair reduced sampled peak RSS by
79,765,504 bytes (~5.8%) while **not** reducing latency. Whole task was
0.75 seconds (~0.9%) slower and scatter wall time was unchanged. These
single-run variations do not establish robust memory savings or a speed
regression; repeat on more scenes to estimate distributions.

The previous 2048 isolated optical scatter tests using separately derived
CoC/highlight arrays were 17.3% faster with mass ROI alone (two pairs).
Adding RGB ROI to mass ROI improved those isolated runs by about 7.3%.
The actual five-model subject-protected pipeline has different spatial
sparsity and memory allocations; never extrapolate microbenchmark speed
to entire photo-processing time.

## Next algorithmic experiments

1. Record actual production scatter ROI hit-rate, depth-layer counts,
   convolution dimensions, and skipped zero work.
2. Profile ARNet multiscale passes and transition buffer ownership; seek
   mathematically equivalent tile/scale reuse, not downscaled output.
3. Re-run native >=20 MP images and memory-limited Linux 2/4 GiB cgroups
   before asserting low-memory service readiness.
4. Require numerical and decoded-pixel validation for every optimization.

Local reproduction script and photographs are deliberately outside Git,
under E:\codex\LeicaShot-local-data\benchmark-full-2048.py and validation\ .
