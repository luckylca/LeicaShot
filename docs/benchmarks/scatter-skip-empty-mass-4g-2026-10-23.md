# Exact skip for empty highlight mass filters: 4 GiB A/B

Updated: 2026-10-23

## Question

The native-resolution profile in PR #43 showed that one-channel CoC mass convolution is the dominant CPU cost. This experiment checks whether skipping a `filter2D` call when the complete `layer_weight` input is exactly zero preserves the real-photo result and improves the complete task.

## Method

- One sequential baseline/candidate pair used the same 5760×3840 CC0 mountain JPEG (`4e6661606c195b2462c0f2321a7feefe2ae2fc081499f3462a30fbb108cd7f8e`), model assets, focus `(0.5, 0.5)`, aperture `2.8`, subject point `(0.72, 0.66)`, ViTMatte tile size 512, Low Memory strategy, and CPUExecutionProvider.
- Baseline ran the branch source at `a09575f` with `--disable-empty-mass-skip`; candidate ran the same source with the default exact-zero skip enabled. The same runner timed the full task and each OpenCV `filter2D` call and counted zero-valued 2D filter inputs.
- Both tasks ran in fresh Linux ARM64 containers in a 4 GiB cgroup (`memory.max=4294967296`, swap capped at the same value), with a two-CPU quota, one ONNX Runtime intra-op thread, read-only source and model mounts, and isolated output storage.
- Runtime: Python 3.11.17, ONNX Runtime 1.31.0, NumPy 2.4.6, OpenCV 5.0.0, Linux 7.0.0-34-generic aarch64 / glibc 2.36. Both runs loaded M9, Depth Anything V2, SAM 2.1 Tiny, ViTMatte Small, and BokehMe ARNet from the same hash-verified local assets.

## Results

| Measurement | Baseline: skip disabled | Candidate: skip enabled | Change |
|---|---:|---:|---:|
| Complete task wall time | 2,798.908 s | 1,569.613 s | −43.92% |
| Engine time | 2,798.894 s | 1,569.601 s | −43.92% |
| Highlight scatter stage | 2,490.479 s | 1,272.015 s | −48.92% |
| Process-tree peak RSS | 2,552,340,480 B | 2,551,914,496 B | −425,984 B |
| cgroup peak memory | 2,896,424,960 B | 2,844,807,168 B | −51,617,792 B (−1.78%) |
| All OpenCV filter calls | 55,132 | 22,414 | −32,718 (−59.34%) |
| Single-channel mass filter calls | 53,940 | 21,222 | −32,718 (−60.65%) |
| All-zero single-channel filter inputs | 32,718 | 0 | −32,718 |
| Timed filter calls total | 1,461.788 s | 619.882 s | −57.60% |
| Timed single-channel mass filters | 1,368.865 s | 529.294 s | −61.34% |
| OOM / cgroup limit events | 0 / 0 | 0 / 0 | unchanged |

Both outputs are 5760×3840 PNGs of 16,655,520 bytes with the same SHA-256:

```text
0a6282df408319950f0bd4c8f56bcfce78ed9e7eba95ce9c50b2c70f5555ade2
```

Decoded RGB arrays were also compared and are exactly equal. The focused unit test independently compares the baseline and skipped scatter arrays byte-for-byte.

## Interpretation and limits

On this full-resolution five-model task, the exact-zero check removed 32,718 expensive convolutions and reduced total task time by 20 minutes 29 seconds. Peak memory was effectively unchanged: cgroup high-water fell by about 49.2 MiB while sampled process-tree RSS changed by less than 0.5 MiB. This is a strong speed result and a small memory improvement, not a memory-tier qualification.

This is one paired photo/device run. It establishes exact output parity and a measured improvement for this input under this ARM64/4 GiB configuration; repeated runs, other highlight distributions, other CPUs, and native-size concurrent service tasks remain unmeasured. Do not generalize the percentage to every image or platform.

Raw JSON reports, cProfile files, and both PNGs are retained outside Git under `/tmp/laika-scatter-skip-empty-mass-4g-20261023/` on the Mac. The same source and A/B flag make the procedure reproducible.
