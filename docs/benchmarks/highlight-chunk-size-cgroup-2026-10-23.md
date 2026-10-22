# Full-resolution highlight chunk comparison under a 3 GiB cgroup

Date: 2026-10-23. Two current-source CPU tasks compared 128-row and 512-row
highlight chunks on the same real photo. Both ran the complete M9 + LabFusion,
Depth Anything V2, ARNet, and optical-highlight path at native resolution.

## Shared configuration

- Input: CC0 Mountains Landscape JPEG, 5760×3840, SHA-256
  `4e6661606c195b2462c0f2321a7feefe2ae2fc081499f3462a30fbb108cd7f8e`.
- Platform: Lima Linux ARM64 guest, 4,085,092,352 B guest memory, 2 logical
  CPUs; Docker cgroup v2 limit 3,221,225,472 B (3 GiB), CPU quota 2.
- Python 3.11, NumPy 2.4.6, OpenCV 5.0.0.93, ONNX Runtime 1.31.0,
  `CPUExecutionProvider`, one ONNX intra-op thread, Low Memory strategy.
- Same mounted source code, container image, model files, photo, mode, focus,
  aperture, and output dimensions. The only requested setting changed was
  `--bokeh-row-chunk-height` (128 versus 512). The ARNet inference tile edge
  stayed 512 for both runs.
- Model SHA-256: M9
  `8936b8c86656265ab07c30b9b4cbbd7fa195d6e44cb3be1d4fd059bb36781e33`;
  Depth `a8ce30941c85b2c9af29fb26f36fbf7be3c1ba1dcf348187a51f6eaa856a238e`;
  ARNet `f0696c16a6eb02099292176f25b47e4dc53863e7e4f3f7748e66470d54cf0173`.

## Results

| Measure | 128 rows | 512 rows | Observed difference |
|---|---:|---:|---:|
| Complete task wall | 4,789.789 s | 6,000.718 s | +25.28% |
| Highlight scatter | 4,473.569 s | 4,779.098 s | +6.83% |
| Process-tree sampled peak RSS | 2,357,911,552 B | 2,405,482,496 B | +2.02% |
| Linux cgroup peak | 2,696,331,264 B | 2,597,527,552 B | −3.66% |
| Cgroup OOM / limit events | 0 / 0 | 0 / 0 | none |
| Output | 5760×3840 RGB PNG, 16,621,497 B | 5760×3840 RGB PNG, 16,621,497 B | same |
| Output SHA-256 | `62bf1f9b29a93e15435d5f38eb9975d5e910ca773fc5eb4e6de43b16a8ca1aef` | same | byte-identical |

Both progress logs contain 433 records, 120 ARNet tile starts and 120
completions across the two tiled scales, monotonic progress from 0.05 to 1.0,
and final completion. The 512-row Docker task exited 0 with `OOMKilled=false`.

The task-level wall-time comparison is confounded by substantial runtime
variation between runs: outside highlight scatter, the 512-row run recorded
Depth inference at 52.953 s versus 13.886 s, M9 inference at 29.179 s versus
3.471 s, and the four ARNet passes at 521.087 / 164.125 / 190.378 / 118.676 s
versus 151.616 / 37.077 / 12.530 / 9.035 s. These differences show that the
guest's effective compute conditions were not stable across the sequential
trials. The scatter-stage difference is smaller, but each setting was still
measured only once and run order was not randomized.

The images are byte-identical, so no output quality or resolution changed.
Neither chunk size demonstrated a reliable overall advantage: the 512-row run
had a lower sampled cgroup peak but a higher process-tree RSS peak, and its
scatter time was 6.83% longer. Keep the 128-row Low Memory default. The small,
contradictory memory differences are not enough evidence to change it; repeat
measurements under controlled CPU conditions would be required before tuning
the default. The 512-row run remains a successful 3 GiB cgroup observation,
not a 4 GiB native-subject acceptance result.

Raw output PNGs, task JSON, and JSONL progress logs remain in
`/tmp/laika-pipeline-results/arnet-tile-telemetry/`; no photo or model is in
Git.

## Reproduction

Run the current benchmark twice in the same ARM64 Linux cgroup and guest, with
the same pinned read-only models and source photo. Use
`--mode color+bokeh --strategy low-memory --threads 1`, do not pass `--width`
or `--height`, and vary only `--bokeh-row-chunk-height 128` and `512`. Set
`PYTHONPATH=/opt/laika-src`, mount the checkout source and scripts read-only,
and preserve the JSON report, progress JSONL, and output PNG outside Git.
