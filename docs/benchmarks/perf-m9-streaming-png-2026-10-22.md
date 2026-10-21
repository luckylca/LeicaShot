# M9 streaming PNG export experiment — 2026-10-22

## Change under test

The pre-optimization engine materialized a full-resolution RGB output array after Lab fusion, then passed that array to Pillow's PNG encoder. The streaming path sends each completed RGB row block directly to a PNG writer. It removes the full-resolution output array while preserving the M9 input size, LabFusion math, and exported dimensions. The writer keeps the input ICC profile and uses PNG filter 0 with zlib compression level 3.

The reference is commit `63643d8` (`feat: add M9 ONNX color pipeline (#18)`). Both variants used the same source image, local model, CPU provider, 512-row fusion blocks, one inference thread, and the same Python environment. Runs alternated baseline/streamed order four times each.

## Result

| Variant | Worker peak RSS median (range) | Engine wall median (range) | PNG size |
|---|---:|---:|---:|
| Full RGB output buffer | 722.5 MiB (645.4–741.7) | 10.06 s (9.86–10.46) | 26,279,400 bytes |
| Streamed PNG rows | 659.5 MiB (627.8–699.1) | 9.58 s (9.53–9.74) | 30,110,710 bytes |

Across these four alternating trials, median worker high-water RSS fell by 63.0 MiB (8.7%) and median engine time fell by 0.48 s (4.8%). The streamed PNG grew by 14.6%. The RSS ranges overlap, so this is an observed result on this host and photo, not a guarantee for other machines or inputs.

Every streamed output was decoded and compared with the baseline PNG in row blocks. Across 66,355,200 RGB values per comparison, maximum absolute error was 0, mean absolute error was 0, and no values differed. PNG output retains the source ICC profile. JPEG export remains on the existing Pillow path and is not part of this memory result.

## Test conditions

| Item | Value |
|---|---|
| Machine | MacBook Air 10,1, Apple M1, 8 GiB |
| OS | macOS 15.7.7, arm64 |
| Python | 3.11.16 |
| ONNX Runtime | 1.31.0, CPUExecutionProvider |
| OpenCV | 5.0.0 |
| Input | CC0 JPEG, 5760 × 3840, 22,118,400 pixels |
| M9 model | 10,053,669 bytes; SHA-256 `8936b8c86656265ab07c30b9b4cbbd7fa195d6e44cb3be1d4fd059bb36781e33` |
| Inference | RGB NCHW float32 interface, 1024 × 683 |
| Fusion | Sigma L 25.0, Sigma AB 20.0, 512 output rows per block |
| Memory metric | `/usr/bin/time -l` maximum resident set size for the CLI process |

The source photo and M9 model are local files outside Git. The full-resolution output remained 5760 × 3840. This was not a cgroup-limited run and does not establish a 4 GiB or 2 GiB deployment target.

## Reproduction

Use the authorized local model and same CC0 photo. Run the baseline at commit `63643d8` and the streaming implementation from this branch, alternating four fresh processes per variant:

```bash
/usr/bin/time -l .venv/bin/python -m laika.cli color \
  /path/to/mountains-source.jpg /tmp/m9-result.png \
  --model /path/to/leica-m9-v3-epoch11.fp16.onnx \
  --strategy low-memory --threads 1 --row-chunk-height 512
```

The CLI prints stage timings as JSON; macOS `time -l` prints the process high-water RSS in bytes. The repository benchmark script also records this high-water metric on macOS alongside 5 ms process-tree RSS sampling:

```bash
./.venv/bin/python scripts/benchmark_m9.py \
  --input /path/to/mountains-source.jpg \
  --model /path/to/leica-m9-v3-epoch11.fp16.onnx \
  --output-dir /tmp/laika-m9-benchmark \
  --strategies low-memory --threads 1 --trials 2 \
  --row-chunk-heights 512 0
```

The paired run's JSON and PNG outputs are under `/tmp/laika-m9-results/perf-rusage-interleaved/`; neither benchmark data nor model weights are tracked.
