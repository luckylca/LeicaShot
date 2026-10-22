# LeicaShot project status

Updated: 2026-10-22

## Current checkpoint

- Stable integration branch: `develop`, currently based on `61a16be` (PR #21).
- Active work: `feat/unified-processing-engine` — coordinating M9, Depth Anything V2, and BokehMe ARNet through one Python API and CLI invocation. The corrected original-depth → M9 → ARNet order passed 2048×1365 and full 5760×3840 real-photo runs. M9/Lab and bokeh outputs have been compared against the reference Kotlin engine on the physical Android phone.
- Most recent merged PR: [#21, BokehMe ARNet CPU renderer](https://github.com/luckylca/LeicaShot/pull/21).
- Existing cross-platform CI passed on macOS, Ubuntu, and Windows with Python 3.11 and 3.13.
- Previously merged model work: M9 + Lab fusion, Depth Anything V2 Small CPU adapter, SAM 2.1 encoder/decoder and bounded feature cache, ViTMatte tiled adapter, and BokehMe ARNet multiscale renderer.

## Models and verified assets

| Model | Python implementation | Current verification |
|---|---|---|
| M9 series 3.0 epoch11 | `laika.m9.M9ColorEngine` | CPU inference, SHA-256 check, real-photo M9/Lab output and RSS report; local model at `/tmp/laika-models/leica-m9-v3-epoch11.fp16.onnx` |
| Depth Anything V2 Small | `laika.depth.DepthAnythingV2Small` | CPU adapter and full-size real-photo depth report; pinned checkpoint and validated dynamic q64 ONNX are in `/tmp/laika-models` |
| BokehMe ARNet | `laika.bokeh.BokehMeARNet` | Checkpoint-backed synthetic and real-photo parity; CPU multiscale output and RSS report; local ONNX at `/tmp/laika-models/legendary-1.0-arnet.fp16.onnx` |
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
- Physical phone comparison: Python vs Kotlin M9/Lab on full-size output had PSNR 53.71 dB and 0.215 mean absolute 8-bit error. The 2048 bokeh comparison had PSNR 41.82 dB; the reference Kotlin run includes high-light scatter, still missing in Python. The report records device/model adaptations and image metrics.
- No 4 GiB or 2 GiB full-pipeline acceptance result exists.

## Known gaps and current blocker

- The unified engine, real-photo model chain, and first physical Kotlin comparisons are implemented. The bokeh image difference still includes known high-light, fusion-size, and Android-vs-macOS decode/runtime differences.
- RSS is close to, and slightly above, 2 GiB in decimal/roughly 2.05 GiB in binary units on this macOS run. It is not a Linux cgroup-limited result; no 2 GiB acceptance claim is made.
- ARNet output currently omits highlight scattering and subject protection.
- SAM, ViTMatte, and GAIC are not yet connected to the same processing request.
- FastAPI tasks, cancellation endpoints, bounded server scheduling, and Web editing flow are not implemented.
- The only connected Android device for reference runs is the physical ADB target; do not use the attached emulator. Production image processing remains Python.

## Next action

Finish branch review, push and merge the unified-engine PR after CI, then immediately start `feat/bokeh-highlight-scatter`. Continue with subject protection, GAIC, API, Web, and Linux low-memory measurements; do not treat the unified-engine merge as project completion.
