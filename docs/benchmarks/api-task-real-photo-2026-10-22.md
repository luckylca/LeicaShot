# FastAPI real-photo task benchmark

## Configuration

- Source: P3M-10K portrait, 1080×1620; SHA-256 `33bcc476421718673730a44b30d136b911dc97c8f086fc58fea55d63666d308b`
- Host: MacBook Air M1, macOS 15.7.7, 8 GiB RAM
- Python 3.11.16, ONNX Runtime 1.31.0, CPUExecutionProvider, one intra-op thread
- API: one Uvicorn process, `low-memory` strategy, `/api/v1/tasks`
- Options: `color+bokeh`, focus `(0.52, 0.38)`, aperture `2.8`, and explicit GAIC crop `(0.15, 0.0611111111, 1.0, 0.8166666667)`
- Models: M9 SHA-256 `8936b8c86656265ab07c30b9b4cbbd7fa195d6e44cb3be1d4fd059bb36781e33`; Depth ONNX SHA-256 `a8ce30941c85b2c9af29fb26f36fbf7be3c1ba1dcf348187a51f6eaa856a238e`; ARNet SHA-256 `f0696c16a6eb02099292176f25b47e4dc53863e7e4f3f7748e66470d54cf0173`

The request uploaded one image, created one task, polled task state, and downloaded the final PNG through the API. The crop was an explicit request option; this benchmark did not run a separate GAIC suggestion request.

## Result

| Measurement | Observed |
|---|---:|
| Final dimensions | 918×1224 |
| Downloaded PNG | 963,021 bytes |
| Task wall time, excluding API startup | 90.604 s |
| Engine-reported total | 90.224 s |
| API server CPU time | 69.254 s |
| API server process-tree peak RSS | 698,236,928 B |
| API server process-tree peak USS | unavailable on this macOS run |
| Process count at report time | 1 |
| Linux cgroup v2 | unavailable on macOS |
| Output SHA-256 | `d7948d71bdc412498098f7b5d32b0ddd871dd0bff755a52ed069502650181e61` |

## Stage timings

| Stage | Seconds |
|---|---:|
| M9 model load | 0.053 |
| M9 inference | 6.474 |
| Lab fusion | 0.697 |
| Depth model load | 1.195 |
| Depth inference | 29.781 |
| ARNet pass 1 | 31.541 |
| ARNet pass 2 | 19.089 |
| Crop | 0.001 |
| Export | 0.106 |

The process-tree sampler used a 10 ms interval. Startup and the external client are outside the reported API worker peak; the temporary result was downloaded and its temporary task directory removed when the benchmark server exited. System available memory before the request was 1,294,221,312 B. This is a single Mac run without a controlled memory limit, not a 4 GiB/2 GiB acceptance test. The run records pipeline execution and delivery; it does not claim a full-image visual metric against the Kotlin reference.

Reproduce from the repository root with the three local model variables set:

```bash
PYTHONPATH=src \
LAIKA_M9_MODEL=/local/models/leica-m9-v3-epoch11.fp16.onnx \
LAIKA_DEPTH_MODEL=/local/models/depth-v2-small-dynamic-q64.onnx \
LAIKA_BOKEH_MODEL=/local/models/legendary-1.0-arnet.fp16.onnx \
LAIKA_API_THREADS=1 \
python scripts/benchmark_api_task.py /path/to/p3m-portrait.jpg \
  --options '{"mode":"color+bokeh","strategy":"low-memory","focus_x":0.52,"focus_y":0.38,"aperture":2.8,"crop":[0.15,0.06111111111111111,1.0,0.8166666666666667]}' \
  --report /tmp/laika-api-report.json
```

## Remaining work

- Add the HTTP GAIC suggestion endpoint and wire crop selection into the Web editor.
- Repeat the API task under Linux cgroup memory limits and measure USS where supported.
- Run broader photographs and full-resolution tasks, including optional SAM/ViTMatte.
- Compare API-delivered output to reference outputs with explicit image metrics.
- Review external deployment authentication and transport boundaries; task tokens are capability credentials, not user accounts.
