# Optical pupil-kernel mirror reuse — 2026-10-22

## Change

The field-dependent optical renderer evaluates 25 pupil kernels for each CoC radius, one per node in its 5×5 field grid. The warp depends on the outer product of the field direction with itself:

```text
projected = (dx·ux + dy·uy) × extra
warped = (dx, dy) + projected × (ux, uy)
```

Negating both direction components leaves the warped coordinates unchanged. Opposite field nodes therefore have identical kernels. The renderer now canonicalizes opposite directions and constructs 13 distinct kernels per radius (12 mirror pairs plus the center), then passes the same exact arrays into the unchanged node-wise filter and blending path. It still evaluates the full 5×5 spatial field and every original CoC layer; no local-window or approximate blur was introduced. The cache is scoped to one radius and one scatter call.

## Bitwise equivalence

- Directly compared `_optical_pupil_kernel(r, aperture, ux, uy, ratio)` with its opposite-direction kernel over radii 1, 7, 31, 63, 127 and 162; apertures 0.95, 1.8, 2.8 and 5.6; radial ratios 0.82, 0.91 and 1.0; and three field directions. Every coefficient was bitwise equal.
- Unit tests build a real 5×5 grid over a 23×19 image for `(radius=7, aperture=2.8)` and `(radius=63, aperture=5.6)`. Each run calls the kernel builder 13 times, reuses the same array object for opposite corners, and matches all 25 directly generated reference kernels bitwise.
- Real-photo 2048×1365 scatter output from the legacy 25-kernel path and optimized 13-kernel path had the same raw float32 SHA-256: `85d7506380a3f6d7200adf5844c3a48958d3f33d0873749f6b933adf10dd75c4`. Both arrays summed to `71.400718688965` with 899,424 nonzero values. Maximum absolute pixel difference is exactly 0 because the output bytes match.

## Paired real-photo measurement

The benchmark runs both paths in one Python process over the same precomputed M9 RGB and raw Depth Anything arrays. Each path uses the same full scatter implementation, aperture, focus, CoC, chunk size, and OpenCV runtime. The legacy path swaps in the previous 25-independent-kernel builder; the production path uses the mirror-reuse builder. RSS is sampled for the scatter call and its child processes at 10 ms intervals.

- Host: MacBook Air M1, macOS 15.7.7, 8 GiB, Python 3.11.16, NumPy 2.4.6, OpenCV 5.0.0. Observed OpenCV thread count: 8. There is no cgroup on this macOS run.
- M9 RGB file: `/tmp/laika-pipeline-results/python-m9-color-full.png`, SHA-256 `e631ed1e7885e9ebdff8eefa2631ced973e3744472201fb0d01f3c2803f9edba`.
- Raw depth file: `/tmp/laika-m9-results/photo/mountains-depth.npy`, SHA-256 `649fb038f693af8a3ef0c5e1ee59d6b3c758f4bc908ea62b049ee3fa04d34332`.
- Prepared arrays: 2048×1365, focus coordinates `(0.5, 0.5)`, resolved normalized focus `0.0617620`, aperture 2.8, `radius_scale=3`, row chunk 128. Highlight input had 1,903 nonzero pixels and sum `72.38285065`.

| Path | Scatter time | Sampled process-tree RSS peak | Raw result SHA-256 |
|---|---:|---:|---|
| Legacy: 25 kernels per CoC radius | 308.55 s | 387,792,896 B | `85d7506380a3f6d7200adf5844c3a48958d3f33d0873749f6b933adf10dd75c4` |
| Mirror reuse: 13 kernels per CoC radius | 286.43 s | 252,805,120 B | Same |

In this paired trial, kernel reuse took 22.12 s less (7.2%) and sampled RSS was 134,987,776 B lower (34.8%). This is one pair, and the host had concurrent activity; it is not a stable throughput guarantee. Two additional single-path runs on the same host took 351.70 s for legacy and 315.81 s for reuse, with matching array hashes but a different RSS ordering. That spread confirms that more controlled repeated timing is needed. The observed memory change is also larger than the kernel arrays alone imply, so treat the RSS delta as a measured run result rather than a guaranteed allocation reduction.

The benchmark input is a cached M9 output with the corresponding raw depth map, so these figures isolate extraction/preparation and optical scatter rather than ONNX inference. The scatter path itself took the times above; preprocessing took 2.92 s in the paired run. The script reports both RSS peaks, output hashes, cgroup counters where available, prepared input metadata, and runtime versions:

```bash
.venv/bin/python scripts/benchmark_optical_scatter.py \
  /tmp/laika-pipeline-results/python-m9-color-full.png \
  /tmp/laika-m9-results/photo/mountains-depth.npy \
  /tmp/laika-pipeline-results/optical-scatter.npy \
  --long-edge 2048 --focus-x 0.5 --focus-y 0.5 --aperture 2.8 \
  --row-chunk-height 128 --repeat 3
```

The script runs the legacy and optimized implementations consecutively for each repeat, with identical in-memory inputs. It saves the optimized float32 array outside Git. Use the same local photo and depth assets when reproducing these measurements.

## Full renderer output check

Ran `scripts/benchmark_bokeh_highlights.py` on the same cached M9 image and raw depth map with the CPU ARNet model, explicit 2048×1365 working output, focus `(0.5, 0.5)`, aperture 2.8, one ONNX intra-op thread and 128-row chunks. The render completed in 471.38 s; highlight scatter took 267.56 s and sampled process-tree peak RSS was 726,302,720 B. The run was under shared host load, so its elapsed time is not compared to earlier standalone renders.

The new PNG is 2048×1365, 2,625,006 bytes, file SHA-256 `7f6ecd91390e759e858e8824921928f7311c98ecd15b33b0d5df014e8d57cff4`. The existing reference PNG file hash is `ec7b8d51001ce7bdbfa73e0ea59c8c046e7b4d9df2d42e3a69e0ab93a04201d1`; PNG file bytes differ, but decoded RGB pixels are identical: maximum channel error 0, MAE 0, and zero changed channel values. This validates final renderer output equality for the real scene.

## Linux cgroup validation

Ran the complete Python pipeline in the Lima Ubuntu ARM64 guest with 2 vCPUs, CPUExecutionProvider and one ONNX intra-op thread. The local model, image and code mounts were read-only. The 4 GiB guest exposed 4,085,092,352 bytes total RAM and no swap.

| Task | Requested result size | Task cgroup limit | Task wall | Process-tree peak RSS | cgroup peak | Outcome |
|---|---:|---:|---:|---:|---:|---|
| M9 + Depth + ARNet + highlighter, explicit size | 2048×1365 | 3 GiB (3,221,225,472 B) | 769.66 s | 2,365,353,984 B | 2,734,096,384 B | Completed, no OOM |
| Same chain, source-size output | 5760×3840 | 3 GiB (3,221,225,472 B) | OOM at about 6m07s | Last observed `ps` RSS 3,014,320 KiB | Last observed 3,194,138,624 B | Docker exited 137, `OOMKilled=true`; no result |

The successful 2048×1365 run used CPU M9 SHA `8936b8c86656265ab07c30b9b4cbbd7fa195d6e44cb3be1d4fd059bb36781e33`, Depth SHA `a8ce30941c85b2c9af29fb26f36fbf7be3c1ba1dcf348187a51f6eaa856a238e`, and ARNet SHA `f0696c16a6eb02099292176f25b47e4dc53863e7e4f3f7748e66470d54cf0173`. Stages: M9 inference 18.99 s; depth inference 23.77 s; ARNet passes 450.50/31.30/20.47 s; highlight scatter 180.70 s. Output SHA-256 `6c3f4d0e51314c07534b7ea76a04f6d1f7b5e896873d4c528f7e01969178839e`, identical to the earlier Linux highlighter output. cgroup OOM counters were zero at completion; the task retained about 464 MiB below its 3 GiB cap.

The source-size task preserved 5760×3840 output dimensions by leaving `--width` and `--height` unset. Near termination, cgroup current was 3,180,503,040 B and peak was 3,194,138,624 B, about 27 MiB under the 3 GiB cap; the container was subsequently OOM-killed. Because the cgroup disappeared after termination, final `memory.events` could not be collected; `docker inspect` confirmed `OOMKilled=true`. This is a concrete failure of the 4 GiB guest / 3 GiB task budget at source resolution. The task did complete at the explicitly requested smaller working output size, which does not establish source-size support.

## Limits and next work

- The optimized 5760×3840 cgroup task does not fit in a 4 GiB guest with a 3 GiB task cap. The last observed cgroup peak was close to the limit and the process was OOM-killed. Reduce full-resolution live buffers or use measured spill/memory-mapping strategies before claiming this target.
- The optimization does not change Lab fusion, depth, ARNet, subject processing, or image export.
- Keep the production row chunk at 128. This change is independent of chunk height; the earlier 128/512 comparison was noisy and did not justify changing that default.
- Repeat paired runs under controlled CPU load and investigate the original-resolution peak before considering further changes. The 128-row setting and 2048×1365 output still match their references, while native-size 4 GiB acceptance remains unmet.
