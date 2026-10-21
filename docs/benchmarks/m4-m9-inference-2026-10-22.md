# M9 photo pipeline benchmark — 2026-10-22

## Result

The pinned M9 ONNX model ran on the CPU against a real 5760 × 3840 photograph. Laika applied EXIF-aware decode, M9 inference at 1024 × 683, bicubic prediction resize, Lab detail fusion, and full-resolution PNG export. The exported files are 5760 × 3840; the model working size does not reduce output resolution.

The test image is [Mountains Landscape on Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Mountains_Landscape.jpg), a Canon EOS Rebel T6i photograph offered under CC0 1.0. It is used only as a local benchmark input and is not included in this repository. Source SHA-256: `4e6661606c195b2462c0f2321a7feefe2ae2fc081499f3462a30fbb108cd7f8e`.

The local M9 file was 10,053,669 bytes and passed the pinned SHA-256 check:

```text
8936b8c86656265ab07c30b9b4cbbd7fa195d6e44cb3be1d4fd059bb36781e33
```

The ONNX session loaded `input` and `output` as rank-4 float32 NCHW tensors and ran with `CPUExecutionProvider`. No GPU provider was enabled.

## Reference and numerical checks

- All tested Low Memory row heights (64, 128, 256, 512, 1024, and full 3840 rows) matched the full-height result exactly: max absolute pixel error 0, mean error 0, and zero differing RGB values.
- A separate row-bounded OpenCV reconstruction of the reference one-blur Lab formula matched the Low Memory full-height result exactly across 66,355,200 RGB values.
- The four-thread Balanced profile differed from the single-thread Low Memory reference at 93 of 66,355,200 RGB values: max absolute error 3 and mean absolute error `2.064646e-06`. This is a small numerical difference between the runtime profiles; it is recorded rather than treated as exact equivalence.
- The reference project's Kotlin `Fusion.fuse` was also run on the same full-size source RGB and M9 prediction arrays. Python Low Memory versus Kotlin had max absolute error 4, mean absolute error `0.0806453`, and 5,350,207 differing channel values of 66,355,200; only 984 values differed by more than 1 and 41 by more than 2. Balanced versus Kotlin had max error 4, mean `0.0806472`, and 5,350,285 differing values; 1,036 differed by more than 1 and 42 by more than 2. This directly exercises the reference Kotlin fusion implementation and isolates fusion by feeding both implementations identical source and prediction arrays.
- The reference repository's direct JUnit suite passed 19 tests (`LeicaMathTest`). This and the Kotlin fusion run were built from an isolated `/tmp` copy with temporary build-only resources; nothing from that copy was added to Laika. These tests plus the Kotlin fusion call are not a capture of the Android UI's complete import/decode/export path. An Android-device run is being attempted separately.

## Device and runtime

| Item | Value |
|---|---|
| Machine | MacBook Air 10,1, Apple M1 |
| OS / architecture | macOS 15.7.7, arm64 |
| Installed memory | 8 GiB |
| Python | 3.11.16 |
| ONNX Runtime | 1.31.0 |
| NumPy | 2.4.6 |
| OpenCV | 5.0.0 |
| Pillow | 13.0.0 |
| Input | JPEG, 5760 × 3840, 22,118,400 pixels |
| M9 model input | RGB NCHW, float32 interface, 1024 × 683 |
| Output | PNG, 5760 × 3840 |
| RSS sampling | Fresh CLI worker process tree, 5 ms interval |

The RSS report sums the worker and its descendants. No child process was used by this CLI run. macOS did not provide usable USS values in this run. cgroup v2 is unavailable on this host, and Docker was unavailable, so these measurements do not establish that the full service passes a 4 GiB or 2 GiB Linux limit.

## Low Memory row-height scan

Low Memory used one ONNX intra-op thread, disabled the ONNX CPU memory arena and memory pattern, and released the session before full-resolution fusion. The selected rows below are the median of two fresh-process trials per height. RSS values include the entire worker process, not only the Lab buffers.

| Fusion rows | Peak RSS median (range) | Total wall median | M9 inference median | LabFusion median | Error vs full-height |
|---:|---:|---:|---:|---:|---:|
| 128 | 616.1 MiB (580.6–651.6) | 15.68 s | 4.24 s | 9.09 s | 0 |
| 512 | 620.6 MiB (570.4–670.9) | 10.75 s | 3.89 s | 4.52 s | 0 |
| 1024 | 614.4 MiB (602.4–626.3) | 9.95 s | 3.92 s | 3.75 s | 0 |
| 3840 (full height) | 615.9 MiB (582.4–649.4) | 9.84 s | 3.87 s | 3.69 s | 0 |

The repeated Low Memory measurements do not show a reliable peak-RSS reduction from 512-row chunks on this image. They do show the expected extra fusion work at smaller row heights: 128 rows took about 2.5 seconds longer overall than full height. A first six-height scan also exercised 64 and 256 rows; all output pixels matched, but timings varied substantially during that run. The current `row_chunk_height=512` remains configurable; these data do not justify claiming it lowers M9's whole-process memory requirement.

## Balanced profile and row-height tradeoff

Balanced enabled the ONNX CPU arena and memory pattern and used four ONNX/OpenCV threads. Two fresh-process trials per row height gave:

| Fusion rows | Peak RSS median (range) | Total wall median | M9 inference median | LabFusion median | Error vs full-height |
|---:|---:|---:|---:|---:|---:|
| 512 | 914.9 MiB (900.0–929.7) | 4.97 s | 1.20 s | 1.54 s | 0 |
| 3840 (full height) | 1001.1 MiB (920.8–1081.3) | 4.54 s | 1.29 s | 1.22 s | 0 |

On this machine and image, 512 rows reduced median worker RSS by about 86 MiB (8.6%) at a 0.42 second (9.3%) wall-time cost. The ranges overlap, and the full-height RSS varied by about 160 MiB between trials, so this is a useful profile tradeoff rather than a universal guarantee. The separate one-trial sweep also tested 128, 256, and 1024 rows; each matched full-height pixels exactly. Its run-to-run RSS spread was larger, so it is not used to make a stronger memory claim.

## Reproduction

Provide the authorized local model and the same or another legally usable photo, then run:

```bash
./.venv/bin/python scripts/benchmark_m9.py \
  --input /path/to/photo.jpg \
  --model /path/to/leica-m9-v3-epoch11.fp16.onnx \
  --output-dir /tmp/laika-m9-benchmark \
  --strategies low-memory --threads 1 --trials 2 \
  --row-chunk-heights 128 512 1024 0

./.venv/bin/python scripts/benchmark_m9.py \
  --input /path/to/photo.jpg \
  --model /path/to/leica-m9-v3-epoch11.fp16.onnx \
  --output-dir /tmp/laika-m9-balanced \
  --strategies balanced --threads 4 --trials 2 \
  --row-chunk-heights 512 0
```

The original photo and generated PNGs used for this report are local files under `/tmp/laika-m9-results/photo/`; no user image, benchmark output, or model weight is tracked in Git.

The three-panel visual comparison is `/tmp/laika-m9-results/photo/reference-comparison.jpg`. It shows the decoded source, Laika's full-resolution M9 + LabFusion result, and the reference Kotlin fusion result. The Kotlin panel uses the same raw input and M9 prediction as Python; it does not include Android Bitmap decoding or file export.

## Physical Android reference-engine run

The reference engine was also run on the connected physical Android device through ADB. A temporary test-only activity called the repository's real `LeicaEngine.process`; its APK, test activity, model asset, and CC0 test image were kept in `/tmp` and removed from the phone after the run. The M9 model copied into the app-private model cache passed the same SHA-256 check.

| Item | Value |
|---|---|
| Device | `2405CPX3DC`, arm64-v8a |
| Android | 16 |
| Installed memory | 11,443,984 kB reported by `/proc/meminfo` |
| Reference engine output | 2048 × 1365 PNG, the app's default export size |
| M9 inference | 3.535 s |
| Full engine run | 14.342 s |
| ORT provider profile | 4 NNAPI kernel events and 104 CPU kernel events; mixed execution |
| Android output SHA-256 | `81290dc7b7f7368436e60961119488ecc334ed9959b0e99feebecf99eaf935c1` |

This is a real device execution of the reference Kotlin engine, with its M9 model and fusion path. Its mixed NNAPI/CPU execution is not a CPU-only benchmark. The visual file `/tmp/laika-m9-results/photo/android-reference-comparison.jpg` places the original photo, Laika's full-resolution output downsampled for comparison, and the Android engine's default 2048 output side by side. After resizing Laika's result to 2048 × 1365 with Pillow bicubic, max absolute RGB difference was 19, mean absolute difference was `1.15532`, and the 95th percentile was 3 across 8,386,560 channel values. This comparison spans different fusion/output resolutions and image-resampling implementations; the same-input direct Kotlin `Fusion.fuse` comparison above is the tighter numerical check.

A separate stress attempt forced the reference engine's `ORIGINAL` output option for 5760 × 3840. M9 inference completed and the engine entered full-resolution fusion, but the run had not exported an output after more than four minutes. It was stopped; the sampled process high-water RSS was 1,463,144 kB (about 1.40 GiB). This is an incomplete, non-default reference-app experiment, not a successful Laika run or a completed performance comparison.

## Next optimization

These results identify full-resolution Lab fusion and runtime working buffers as the next M9 memory target, while Balanced threading materially reduces CPU time at a higher RSS. The next `perf/m9-memory-optimization` change should measure a fused Lab-to-RGB row output path that removes the full-frame Lab destination allocation, then compare it against this report's full-height and 512-row baselines. It must retain exact output on the same photo before adoption.
