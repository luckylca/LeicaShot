# In-place weighted highlight scatter on a 2 GiB task

Date: 2026-10-24

## Question

Can the optical highlight scatter avoid allocating a product array for each
weighted mass/radiance ROI without changing the final pixels?

The candidate multiplies the float64 filter-output ROI in place, then adds it
to the accumulator. The experimental non-float64 mass routes retain their
existing expression and promotion behavior. This tests an allocation and CPU
optimization; it does not change filter kernels, border handling, chunking, or
image dimensions.

## Controlled run

- Input: same 5760×3840 JPEG in both runs, SHA-256
  `4e6661606c195b2462c0f2321a7feefe2ae2fc081499f3462a30fbb108cd7f8e`.
- Baseline: `develop` at `bc9cd35` (PR #55); candidate: the same source plus
  the in-place scatter change on `perf/highlight-scatter-cpu`.
- Both: Linux ARM64 Lima guest, 4 GiB host guest memory, 2 vCPUs, 2 GiB
  cgroup v2 memory cap, zero swap, one CPU thread, Low Memory strategy,
  CPUExecutionProvider, identical M9/Depth/ARNet model files and hashes.
- Both produced a 5760×3840 RGB PNG of 16,621,497 bytes.

## Results

| Measurement | Baseline | Candidate | Change |
|---|---:|---:|---:|
| Highlight scatter | 1,108.614 s | 905.330 s | −18.34% |
| Full task wall time | 1,409.220 s | 1,173.245 s | −16.75% |
| Process-tree peak RSS | 1,824,636,928 B | 1,830,719,488 B | +6,082,560 B (+0.33%) |
| Scatter-stage process-tree peak RSS | 1,749,803,008 B | 1,757,118,464 B | +7,315,456 B (+0.42%) |
| cgroup peak | 2,029,903,872 B | 2,015,240,192 B | −14,663,680 B (−0.72%) |
| cgroup OOM events | 0 | 0 | unchanged |

The runner's report hashes and an independent SHA-256 of both output files
match exactly:
`62bf1f9b29a93e15435d5f38eb9975d5e910ca773fc5eb4e6de43b16a8ca1aef`.
The complete output remains at the original 5760×3840 dimensions.

## Interpretation

This single paired run supports retaining the in-place multiplication as an
exact-output CPU optimization. The candidate completed the scatter stage
203.284 s sooner and the full task 235.976 s sooner. Process-tree RSS was
slightly higher in the candidate, so this is not evidence of an RSS reduction.
The measured cgroup high-water was 14.7 MB lower, a small change that should
not be generalized from one pair. The three-model color+bokeh path completed
under 2 GiB with zero OOM events; this is not a five-model subject-protection
result and does not establish a general 2 GiB service guarantee.

Exact hashes establish output parity for this scene, not broad image-quality
coverage. Local validation passed: 207 Python tests, Ruff lint, Ruff format
check, four Web tests, and the Web production build. Remote CI remains required
before merge.
