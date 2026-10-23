# Bokeh row-chunk height at native resolution

Updated: 2026-10-23

## Question

Measure whether increasing the field-optical highlight scatter output row chunk from its prior 128-row default improves the full CPU pipeline, then decide whether the new default should change.

## Method

- All runs processed the same 5760×3840 CC0 mountains JPEG (`4e6661606c195b2462c0f2321a7feefe2ae2fc081499f3462a30fbb108cd7f8e`) with M9, Depth Anything V2, SAM 2.1 Tiny, ViTMatte Small and BokehMe ARNet/highlights. Options were Low Memory, CPUExecutionProvider, one ONNX/OpenCV thread, focus `(0.5, 0.5)`, aperture `2.8`, subject point `(0.72, 0.66)`, and 512-pixel ViTMatte tiles.
- The 128-row baseline used the default `ProcessingOptions.bokeh_row_chunk_height`; the two 256-row candidates and one 512-row candidate set this option through `scripts/profile_native_scatter.py --bokeh-row-chunk-height`. All used the exact-zero mass-filter skip and identical model/input mounts.
- Each run used a fresh Linux ARM64 container, 2 vCPU quota, 4 GiB cgroup memory limit (Docker memory and memory+swap both 4 GiB), one task, identical read-only model/input mounts, Python 3.11.17, NumPy 2.4.6, OpenCV 5.0.0 and ONNX Runtime 1.31.0.
- The baseline is the full-resolution one-thread candidate from PR #44; the candidates are source-loaded runs on `perf/scatter-row-chunk-256`. The 256-row case was repeated once; the 128- and 512-row settings each have one full-size run, so host/run-to-run variation and cross-device generalization remain open.

## Results

| Measurement | 128 rows | 256 rows, run 1 | 256 rows, repeat | 512 rows |
|---|---:|---:|---:|---:|
| Complete task wall time | 1,569.613 s | 1,347.697 s | 1,426.347 s | 1,509.538 s |
| Engine time | 1,569.601 s | 1,347.669 s | 1,426.329 s | 1,509.525 s |
| Highlight scatter stage | 1,272.015 s | 1,016.780 s | 1,096.835 s | 1,028.462 s |
| Process CPU time | 1,579.810 s | 1,356.840 s | 1,426.940 s | 1,515.520 s |
| Process-tree peak RSS | 2,551,914,496 B | 2,625,380,352 B | 2,551,332,864 B | 2,622,132,224 B |
| cgroup peak memory | 2,844,807,168 B | 2,802,565,120 B | 2,713,399,296 B | 2,738,757,632 B |
| OpenCV `filter2D` calls | 22,414 | 12,954 | 12,954 | 8,338 |
| Sum of profiled `filter2D` time | 619.882 s | 458.333 s | 473.393 s | 447.748 s |
| OOM / cgroup limit events | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

All four output PNGs are 5760×3840, 16,655,520 bytes, and have the same SHA-256: `0a6282df408319950f0bd4c8f56bcfce78ed9e7eba95ce9c50b2c70f5555ade2`. Their encoded bytes are identical, so decoded pixel error is exactly zero. The cgroup limit was 4,294,967,296 B.

## Interpretation

The two 256-row runs reduced measured scatter time by 20.06% and 13.77% versus 128 rows; their mean was 1,056.807 s (−16.92%). The 512-row run was 19.15% faster than 128 in its single trial and falls within the 256-row scatter-time range. Moving from 256 to 512 rows cut filter calls by another 35.63%, but reduced summed filter time by only 2.31%; one 512-row scatter trial does not separate cleanly from the variation between the repeated 256-row trials. All candidate chunks preserved the exact PNG and completed without cgroup events. The 256-row process RSS ranged from 2,551,332,864 to 2,625,380,352 B, while its cgroup peak ranged from 2,713,399,296 to 2,802,565,120 B. Both 256-row cgroup peaks were below the 128-row baseline peak of 2,844,807,168 B.

Whole-task timing is not a stable basis for ranking these settings: the 512-row run's four ARNet inference pass timers totaled about 180 s more than in the first 256-row run, while these chunk options do not alter ARNet inference inputs. The two 256-row full tasks took 1,347.697 s and 1,426.347 s, both below the 128-row run, but this remains one photo and one device tier. The repeated scatter-stage gain, bitwise output equality, and lower measured 4 GiB cgroup peaks support changing the engine/API/CLI default to 256 rows. Callers can still set 128 explicitly for a lower working-buffer budget. Keep the 512-row size configurable rather than making it the default because its single scatter time is not demonstrably better than the repeated 256-row results. Raw reports, profiles and PNGs are local-only under `/tmp/laika-scatter-row-chunk-256-20261023/` on the Mac and the matching Lima VM directory; no photo or model assets are committed.
