# Python optical highlight stage: real-photo and Kotlin comparison

Run date: 2026-10-22
Branch: `feat/bokeh-highlight-scatter`
Base: `develop` at `371cafb` (merged PR #22)

## What was measured

This branch connects highlight extraction, clipped-light recovery, reference nine-blade optical scatter with the 5×5 field pupil, softening, and radiance compositing to `render_bokeh`. A single `LaikaProcessingEngine.process_file()` call ran M9, Depth Anything V2, ARNet, and the highlight stage on one real photograph. The original source was 5760×3840; this validation requested a 2048×1365 output explicitly. No source or final output dimensions were silently changed.

All inference ran on the MacBook Air M1 (arm64, 8 logical CPUs, 8 GiB) with CPUExecutionProvider, one ONNX intra-op thread, Python 3.11.16, NumPy 2.4.6, OpenCV 5.0.0, Pillow 13.0.0, and ONNX Runtime 1.31.0. No other benchmark was intentionally started during the combined run. The report measures process-tree RSS at 10 ms sampling intervals; it is an observed macOS process result, not cgroup memory accounting.

Source photograph: `/tmp/laika-m9-results/photo/mountains-source.jpg`, 5760×3840, SHA-256 `4e6661606c195b2462c0f2321a7feefe2ae2fc081499f3462a30fbb108cd7f8e`. The real photo and outputs are kept outside Git.

Model hashes reported by that request:

| Stage | Local model | SHA-256 |
|---|---|---|
| M9 | `leica-m9-v3-epoch11.fp16.onnx` | `8936b8c86656265ab07c30b9b4cbbd7fa195d6e44cb3be1d4fd059bb36781e33` |
| Depth Anything V2 Small | `depth-v2-small-dynamic-q64.onnx` | `a8ce30941c85b2c9af29fb26f36fbf7be3c1ba1dcf348187a51f6eaa856a238e` |
| BokehMe ARNet | `legendary-1.0-arnet.fp16.onnx` | `f0696c16a6eb02099292176f25b47e4dc53863e7e4f3f7748e66470d54cf0173` |

## Complete Python request

The checked-in `scripts/benchmark_processing_pipeline.py` calls the public Python engine directly and samples its process tree. The 2048×1365 `color+bokeh` task completed in 197.846 s with peak process-tree RSS of 1,110,802,432 B (1.034 GiB). Output was 2048×1365, 2,627,627 bytes, SHA-256 `ec7b8d51001ce7bdbfa73e0ea59c8c046e7b4d9df2d42e3a69e0ab93a04201d1`.

| Stage | Time (s) |
|---|---:|
| Depth model load + preprocess + inference + normalization/restore | 15.02 |
| M9 decode + load + preprocess + inference + Lab fusion + encode | 9.63 |
| ARNet load + preparation + three passes + base composite | 26.68 |
| Highlight extraction | 0.54 |
| Clipped-light recovery | 0.41 |
| Field-dependent optical scatter | 140.89 |
| Highlight softening | 3.20 |
| Highlight radiance composite | 0.18 |
| Depth/frame decode, spool, and task overhead | 1.08 |
| PNG export | 0.21 |
| Full task wall | 197.85 |

The highlight stage adds substantial CPU cost; optical scatter is the main current bottleneck. This is the measured cost of evaluating the full depth-layer and spatial pupil path, not evidence that the engine meets a speed target. The comparable PR #22 run without highlight scatter took 54.38 s and had 1,094,172,672 B high-water RSS at the same explicit output size. That is a before/after observation on this host, not an isolated algorithm speed comparison.

The current output was compared to the physical-device Kotlin reference output `kotlin-reference-bokeh-2048.png` at the same 2048×1365 dimensions. Absolute 8-bit channel metrics: MAE 1.2975, RMSE 2.1759, PSNR 41.378 dB, p95 4, p99 8, max 69, and 85.94% of channel values within 2. This aggregate comparison does not isolate highlight-region quality: Android and macOS decoders/runtimes differ, and the reference/Python color fusion working sizes differ. It is recorded as an end-to-end comparison, not a numerical-equivalence claim.

## Row-chunk experiment

The standalone `scripts/benchmark_bokeh_highlights.py` took the same M9-rendered 5760×3840 color result, resized it explicitly to 2048×1365, and reused the same depth map. Both rows use the CPU-only Low Memory strategy and one thread. Its process-tree peak includes the Python process and loaded ARNet session.

| Bokeh row chunk | Highlight scatter | Renderer total | Process-tree peak RSS |
|---:|---:|---:|---:|
| 128 | 163.31 s | 192.20 s | 1,120,010,240 B (1.043 GiB) |
| 512 | 137.63 s | 167.20 s | 1,133,658,112 B (1.056 GiB) |

In this run, 512 rows reduced scatter time by 15.7% and renderer time by 13.0%, with 13.6 MB more sampled peak RSS. The full SDK run using 512 rows measured 140.89 s scatter and 1,110,802,432 B total-task peak. This sample favors 512 rows for speed on this host, but does not establish a universal optimum or replace a low-memory cgroup sweep. No 256- or 1024-row trial was performed.

## Kotlin intermediate checks

Reference intermediates were captured by running the external Kotlin implementation on the physical Android phone with ADB serial `a4976c80`; no emulator was used. The fixture came from a 256×256 crop of the real night-village photo and a synthetic clipped-point-light variant. All Android app/build artifacts, reference checkout, fixtures and model weights remain outside the Laika repository.

| Intermediate | Comparison |
|---|---|
| Base RGB plate | Exact byte match on the real-photo crop |
| Linear highlight split | Maximum absolute error 1.19e-7 on the clipped-light fixture |
| Uniform layered scatter | Maximum absolute error 1.86e-9 on the point-light fixture |
| Clipped-light recovery | MAE 4.04e-10; active-pixel MAE 2.92e-7; max 2.86e-6 |
| Field-dependent optical scatter | MAE 5.998e-10; active-pixel MAE 9.48e-9; max 8.94e-8 |
| Softened highlight layer | MAE 5.44e-10; active-pixel MAE 7.17e-9; max 8.94e-8 |

These comparisons test specific buffers and fixtures. The full real-photo image metrics above are less exact because the complete Android and Python pipelines have different processing/runtime details.

## Reproduce

```bash
.venv/bin/python scripts/benchmark_processing_pipeline.py \
  /tmp/laika-m9-results/photo/mountains-source.jpg \
  /tmp/laika-pipeline-results/python-full-highlight-2048.png \
  --m9-model /tmp/laika-models/leica-m9-v3-epoch11.fp16.onnx \
  --depth-model /tmp/laika-models/depth-v2-small-dynamic-q64.onnx \
  --bokeh-model /tmp/laika-models/legendary-1.0-arnet.fp16.onnx \
  --mode color+bokeh --strategy low-memory --threads 1 \
  --width 2048 --height 1365 --bokeh-row-chunk-height 512
```

## Limits and next work

- No 5760×3840 run with highlight scatter has been performed yet.
- No 4 GiB or 2 GiB complete-task cgroup test exists for this version.
- This branch does not integrate SAM/ViTMatte foreground protection or GAIC.
- Continue with the subject-protection engine branch after this PR. Keep exact-preserving optical-scatter CPU optimization and the full-resolution RSS run on the active backlog.
