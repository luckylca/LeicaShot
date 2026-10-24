# Exact-support spatial scheduling for optical highlight scatter — Windows CPU

Date: 2026-10-24.
Branch: perf/windows-highlight-output-support-scheduling.
Base: develop at 000cb2063da8905fcedf30fb78940679da0ac37d.
This new algorithm makes the field-optical scatter renderer avoid evaluating
CoC-layer convolution over output areas that cannot receive any highlight
light. The other five-model stages, output resolution, pupil kernel, depth
layers, occlusion rule and source images are unchanged.

## Mathematical rationale

For a linear convolution with a spatially finite pupil kernel, an output
pixel cannot receive any highlight radiance when all input highlight
radiance within the kernel's support is zero. The current renderer also
uses convolution of the CoC-layer mass to determine visibility. Occluding
mass is needed *where some highlight may appear*, but computing it elsewhere
cannot alter the output: emitted radiance and local-light accumulation are
zero there. The new scheduler first obtains the **union across all layers**
of possible output highlight support, conservatively expanding input
nonzero highlights by the largest kernel half width (max |CoC level| + 1).
For each short output-row band, it computes nonzero source column ranges,
expands those ranges, then intersects them with each optical field node's
existing spatial support. Unaffected rows/nodes/columns are skipped for
both mass and RGB radiance convolutions; **all original mass and occlusion
computations still run within the affected regions**, including nonluminous
layers that occlude a later luminous layer. The computed image still has
the full native requested output dimensions.

Masks are generated per halo-extended band instead of materializing
a full-frame boolean HWC and HW mask. In this CPU experiment a maximum
64-row band improved the spatial selectivity compared to 128/192/256,
and peak process RSS decreased within the isolated scatter subprocess.
The original dense/full-output scheduler remains available through
private scatter_highlights setting _output_support_roi=False.
No lossy downsample, PSF size reduction, spatial optics simplification or
hidden quality-mode switch was used.

## Provenance and input

Host: Windows laptop, ~15.7 GiB physical memory. Python 3.11, NumPy,
OpenCV and ONNX Runtime CPUExecutionProvider. 1 ONNX inference thread,
Low Memory strategy, ViTMatte 512px tiles. Input: user-authorized
5472x3648 CC0 portrait, SHA256
d237bbf13f886b69a376aad00836403dafdb8a6be0062dc98a50e1a6c7b32a48,
resized bicubic to 2048x1365 for fixed test. The same M9/Depth/ARNet/
SAM Tiny/ViTMatte ONNX weights were used in both arms, and both arms
included the already merged sparse mass and sparse RGB support changes
(PR #71 and #72). Both included complete subject protection,
recover-clipped-highlights, softening and PNG export.

A critical verification: render_bokeh imports scatter_highlights *inside
the function*. The benchmarking wrapper therefore modifies
laika.highlights.scatter_highlights, not an inert laika.bokeh module
attribute. The former is the actual runtime binding used in the local
import. Each arm sets _sparse_mass_roi=True and
_sparse_radiance_roi=True; ONLY the output support option changes.
The optimized arm also automatically chooses max 64-row optical bands.
Both conditions were run in separate fresh Python worker processes.
Execution order was baseline/optimized, then optimized/baseline.

The exact production scatter arrays were captured after all preceding
models and highlight recovery had run: shape 1365x2048, 64 CoC levels,
4,400 spatial pixels with nonzero highlight radiance, largest pupil-kernel
half width 35. Its conservative 128-row support footprint occupied 8.04%
of the full output area, with 8 active bands out of 11.

## True five-model full-pipeline results

| Pair/order | Whole task baseline | Whole task optimized | Scatter baseline | Scatter optimized |
|---|---:|---:|---:|---:|
| Baseline then optimized | 95.0783 s | 71.6768 s | 41.6005 s | 13.9384 s |
| Optimized then baseline | 104.6565 s | 70.6749 s | 41.8686 s | 10.2629 s |
| Mean of the two | 99.8674 s | 71.1759 s | 41.7345 s | 12.1007 s |

Means: full task ~28.7% faster (1.40x throughput per serial worker),
highlight scatter ~71.0% faster (~3.45x) in this input/workload.
These results come from only one photo and two order permutations; they
are not universal guarantees or a performance claim for 20 MP.

| Order | Baseline sampled process RSS peak | Candidate sampled process RSS peak |
|---|---:|---:|
| Baseline then optimized | 1,295,896,576 B | 1,296,748,544 B |
| Optimized then baseline | 1,296,367,616 B | 1,285,763,072 B |

These are sampled worker RSS values, not 2GiB Linux cgroup high-water;
they show no major whole-pipeline memory increase, but also do not
establish a significantly lower minimum RAM requirement.
The isolated scatter worker showed 37.251s baseline / 11.279s candidate,
with sampled RSS ~200,065,024 B / 194,682,880 B. Its float32 arrays had
max absolute difference 1.1641532182693481e-10, from rounding/FFT
tile effects, not a change to the pupil shape. The full real-photo
RGB output is BYTE FOR BYTE identical for every tested full run:
PNG SHA256 D9816366976BD55ACA1D456F367265E8731C6F168BB4B49B57606CD8AC78AC4C.

Additional scatter-only 2048 size candidate band tests:
32 rows 11.171s / 171,114,496B RSS;
48 rows 11.432s / 173,805,568B;
64 rows 10.072s / 178,634,752B;
96 rows 10.660s / 183,894,016B;
192 rows 12.780s / 207,245,312B;
256 rows 13.600s / 220,917,760B.
Samples reflect execution order and warm/cold system effects, not a
perfect multi-run paired sweep; 64-row was chosen for speed and memory
balance, rather than choosing the lowest-RSS 32-row setting.

## Regression / operational limits

The new module tests compare empty, near-boundary, sparse, numerous
and dense radiance and random CoC arrays against the unchanged scheduler,
covering different row band sizes, conservative output support bounds,
float32 tolerances and exact uint8 composite pixels. The full Python suite
and cross-platform CI should gate merging, along with Ruff lint/format.

Still not established: multiple real-photo scenes, full native ~20MP speed,
Linux 2GiB/4GiB cgroup limits, parallel API workers, and deterministic
performance across CPU/OpenCV versions. Do not claim those yet.

### Local reproducibility (excluded from Git)

E:\codex\LeicaShot-local-data\benchmark-full-support-2048.py
E:\codex\LeicaShot-local-data\production-scatter-worker.py
E:\codex\LeicaShot-local-data\capture-production-scatter-2048.py

Source images, model weights and multi-megabyte intermediate NumPy arrays
are not included in Git.
