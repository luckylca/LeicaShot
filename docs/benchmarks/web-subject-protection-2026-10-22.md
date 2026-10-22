# Web subject-protection API integration benchmark

Date: 2026-10-22. This verifies the Web task options for multi-point subject
protection against the real FastAPI service and all five CPU model components.

## Request and runtime

- Source: 1080×1620 P3M-10K validation portrait; SHA-256
  `33bcc476421718673730a44b30d136b911dc97c8f086fc58fea55d63666d308b`.
- Host: MacBook Air M1, macOS 15.7.7, 8 GiB RAM; Python 3.11.16; ONNX Runtime
  1.31.0; CPUExecutionProvider; one intra-op thread; Low Memory strategy.
- One uploaded `/api/v1/tasks` request with mode `color+bokeh`, focus `(0.52,
  0.30)`, aperture `2.8`, positive subject points `(0.52, 0.30)`, `(0.55,
  0.80)`, `(0.80, 0.65)`, and ViTMatte tile size 512.
- A direct API benchmark used the same options that the Web editor emits.
  Separately, a live Chrome run exercised the editor controls and completed the
  same five-model API workflow.
- The three points exercised the browser API payload contract. SAM image encoder
  ran once and two later prompts reused the cached embedding.

### Model hashes

| Model | SHA-256 |
|---|---|
| M9 series 3.0 epoch11 ONNX | `8936b8c86656265ab07c30b9b4cbbd7fa195d6e44cb3be1d4fd059bb36781e33` |
| Depth Anything V2 q64 ONNX | `a8ce30941c85b2c9af29fb26f36fbf7be3c1ba1dcf348187a51f6eaa856a238e` |
| BokehMe ARNet ONNX | `f0696c16a6eb02099292176f25b47e4dc53863e7e4f3f7748e66470d54cf0173` |
| SAM 2.1 Tiny image encoder ONNX | `37de035410f11fb505c210ce1f8b10b450cb018fbfa7f226ec7e7ab542955c75` |
| SAM 2.1 Tiny image decoder ONNX | `3a6a96c62fd39c1369641ea626e5b2e4aed10de877f85291e9993a044d5bf981` |
| SAM source checkpoint | `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69` |
| ViTMatte Small ONNX | `a5c0b49a664867021eaf2b9cadcdc236ea511028d5e709303b418431f20e223b` |
| ViTMatte source checkpoint | `bda9289db1bb6762d978b42d1c62ae3f34daf7497171a347a1d09657efd788cb` |

## Measurements

| Measurement | Result |
|---|---:|
| API task wall time | 102.177 s |
| Server process-tree peak RSS | 1,029,177,344 B |
| Server CPU time | 100.079 s |
| Output | 1080×1620 PNG, 1,699,480 B |
| Output SHA-256 | `a50fd8d2dc8615392942c613dca99a7fecb027fe1d184f5d9ffa8064ebb85484` |
| SAM image encoder / decoder prompts | 5.034 s / 0.227 s |
| ViTMatte inference | 54.029 s |
| Depth inference | 13.489 s |
| M9 inference / Lab fusion | 3.500 s / 0.307 s |
| ARNet passes | 12.105 s / 8.530 s |
| M9 / Depth / SAM / ViTMatte / ARNet model load | 0.031 / 0.662 / 0.994 / 0.549 / 0.011 s |
| Linux cgroup measurement | unavailable on macOS |

The downloaded output hash matches the previously validated five-model
`process_file()` result using these image, point, tile, and model settings.
Its mask IoU and alpha metrics are recorded in the
[subject-protection report](subject-protection-integration-2026-10-22.md).
The API run began with 1,618,722,816 B system-available memory; its sampled
process-tree peak is a Mac observation, not a 4 GiB or 2 GiB constrained run.

## Live browser run

Chrome loaded the active React branch through Vite, selected the same P3M
portrait, enabled protection, selected the focus tool and placed focus, then
added three subject prompts and submitted the task. The UI showed all three
numbered markers and the 512 tile setting before submission. The backend
completed M9, Depth, SAM, ViTMatte, ARNet and compositing; the page showed 100%
progress, displayed the before/after preview, and downloaded a full-size PNG.

- Emitted focus: `(0.5191147, 0.2997988)`.
- Emitted subject points: `(0.5191147, 0.2997988)`, `(0.5492958, 0.7994634)`,
  `(0.7997988, 0.6498994)`; matte tile size 512.
- Downloaded output: 1080×1620 PNG, 1,662,940 bytes; SHA-256
  `c8bef8fc7e991ffcde79ee9cf2a6433baa50b5815bc388c233d17c97fd9cd679`.
- Browser screenshots were kept under `/tmp/leicashot-subject-controls.png` and
  `/tmp/leicashot-subject-complete.png`; they are not project artifacts.

The live browser run was not separately instrumented for RSS or stage timing.
Use the process-tree measurements above for the matching model path; the
browser-run coordinates differ slightly from the fixed-decimal benchmark
request because click positions are mapped from rendered image pixels.

All model assets remained in `/tmp/laika-models`. The input, output, server
workspace, and machine report remained outside Git.
