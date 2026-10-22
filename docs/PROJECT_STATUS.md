# LeicaShot project status

Updated: 2026-10-22

## Current checkpoint

- Stable integration branch: `develop`, currently at `371cafb` (PR #22).
- Active work: `feat/bokeh-highlight-scatter` — adds reference-style highlight extraction, clipped-light recovery, field-dependent nine-blade optical scatter, highlight softening, and linear radiance compositing to the Python ARNet renderer.
- Most recent merged PR: [#22, unified M9 + Depth + ARNet Python pipeline](https://github.com/luckylca/LeicaShot/pull/22).
- Existing cross-platform CI passed on macOS, Ubuntu, and Windows with Python 3.11 and 3.13.
- Previously merged model work: M9 + Lab fusion, Depth Anything V2 Small CPU adapter, SAM 2.1 encoder/decoder and bounded feature cache, ViTMatte tiled adapter, and BokehMe ARNet multiscale renderer.

## Models and verified assets

| Model | Python implementation | Current verification |
|---|---|---|
| M9 series 3.0 epoch11 | `laika.m9.M9ColorEngine` | CPU inference, SHA-256 check, real-photo M9/Lab output and RSS report; local model at `/tmp/laika-models/leica-m9-v3-epoch11.fp16.onnx` |
| Depth Anything V2 Small | `laika.depth.DepthAnythingV2Small` | CPU adapter and full-size real-photo depth report; pinned checkpoint and validated dynamic q64 ONNX are in `/tmp/laika-models` |
| BokehMe ARNet + optical highlights | `laika.bokeh.BokehMeARNet` and `laika.highlights` | CPU multiscale real-photo run with extraction, clipped-light recovery, field-dependent scatter and composite; Python/Kotlin intermediate fixtures numerically compared; local ONNX at `/tmp/laika-models/legendary-1.0-arnet.fp16.onnx` |
| SAM 2.1 Hiera Tiny | `laika.subject.Sam2TinyOnnx` | Split model tensor parity and bounded embedding cache; real-photo mask/trimap/edge-quality validation remains incomplete |
| ViTMatte Small | `laika.matting.ViTMatteOnnx` | Tiled CPU adapter and synthetic tests; real-photo edge-quality comparison and coordinated SAM integration remain incomplete |
| GAIC | Not integrated | Checkpoint conversion, scoring, and crop application remain outstanding |

Model weights and photographs stay outside Git. Project-use authorization is confirmed by the user; this does not assert public redistribution rights.

## Performance facts

- Host: MacBook Air M1, 8 GiB RAM, CPU-only ONNX Runtime.
- Standalone ARNet renderer: 1024×683 measured 658,980,864 B high-water RSS / 8.31 s; 2048×1365 measured 1,118,208,000 B / 27.74 s. Both reused precomputed depth and do not measure M9 or depth inference.
- Standalone Depth Anything V2 real-photo run: 1,695,547,392 B process high-water RSS / 14.402 s total for a 5760×3840 source. This is not a combined-task result.
- Corrected-order 2048×1365 end-to-end task: 1,094,172,672 B high-water RSS / 54.38 s wall.
- Corrected-order 5760×3840 end-to-end task: 2,197,651,456 B high-water RSS / 605.43 s wall; ARNet's first full-resolution pass took 514.45 s. Both used Low Memory, CPUExecutionProvider, one intra-op thread, and all three hash-recorded models. The detailed [unified pipeline report](benchmarks/unified-pipeline-real-photo-2026-10-22.md) records concurrent host load and remaining quality-comparison limits.
- With highlights integrated, the same source processed end-to-end at the explicit 2048×1365 output size in 197.85 s and 1,110,802,432 B process-tree peak RSS. Optical highlight scatter took 140.89 s. This is not a full-resolution highlight run or a Linux cgroup result. The [highlight report](benchmarks/bokeh-highlight-scatter-2026-10-22.md) records stage timings, row-chunk results, Kotlin fixture errors and the real-photo comparison.
- Physical phone comparison: Python vs Kotlin M9/Lab on full-size output had PSNR 53.71 dB and 0.215 mean absolute 8-bit error. With the highlight stage enabled, the 2048×1365 bokeh comparison measured PSNR 41.38 dB, MAE 1.297 and p99 absolute channel error 8. This aggregate metric does not isolate highlight quality; the reference comparison limits are documented in the report.
- No 4 GiB or 2 GiB full-pipeline acceptance result exists.

## Known gaps and current blocker

- The unified engine and highlight scatter are integrated. Full 5760×3840 processing with the new highlight stage still needs measurement, and reference comparison remains affected by fusion-size and Android-vs-macOS decode/runtime differences.
- RSS is close to, and slightly above, 2 GiB in decimal/roughly 2.05 GiB in binary units on this macOS run. It is not a Linux cgroup-limited result; no 2 GiB acceptance claim is made.
- SAM, ViTMatte, and GAIC are not yet connected to the same processing request.
- FastAPI tasks, cancellation endpoints, bounded server scheduling, and Web editing flow are not implemented.
- Full-resolution highlight runtime is high on CPU; the 2048×1365 optical scatter stage took about 141 seconds. Native acceleration or a mathematically validated faster execution plan remains a priority.
- The only connected Android device for reference runs is the physical ADB target; do not use the attached emulator. Production image processing remains Python.

## Next action

Finish review and merge the highlight-scatter PR after CI, then immediately start `feat/subject-protection-integration`. Connect SAM, trimap and ViTMatte to the unified request, while keeping optical scatter speed optimization on the backlog. Continue with GAIC, API, Web, full-resolution and Linux low-memory measurements; do not treat the highlight PR as project completion.
