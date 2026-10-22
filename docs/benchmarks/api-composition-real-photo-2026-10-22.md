# FastAPI GAIC composition benchmark

## Configuration

- Source: P3M-10K portrait, 1080×1620; SHA-256 `33bcc476421718673730a44b30d136b911dc97c8f086fc58fea55d63666d308b`
- Host: MacBook Air M1, macOS 15.7.7, 8 GiB RAM
- Python 3.11.16, ONNX Runtime 1.31.0, CPUExecutionProvider, one intra-op thread
- One Uvicorn worker; `low-memory` strategy; target long edge 1200
- GAIC ONNX SHA-256 `b9bd8f7b0af48f8628f6ace41eaaabe96fba7ab6f0b566a9cd30c417bcf81d7b`

## Result

| Measurement | Observed |
|---|---:|
| Candidates scored | 86 |
| Request wall time | 1.403 s |
| API server CPU time | 1.258 s |
| Server process-tree peak RSS | 416,251,904 B |
| Server process-tree peak USS | unavailable on this macOS run |
| Linux cgroup v2 | unavailable on macOS |
| GAIC ONNX inference | 0.400 s |
| Model load | 0.345 s |
| Analysis resize | 0.461 s |
| Decode | 0.167 s |

The top four crop suggestions matched the preceding standalone GAIC CPU run on the same source. The first suggested crop was `(left=0.15, top=0.0611111111, right=1.0, bottom=0.8166666667)`. This measured the real `/api/v1/compositions` multipart route through upload, GAIC inference, and JSON response. RSS was sampled across the API worker process tree at 10 ms intervals. The run is not a Linux memory-limit acceptance result.

## Reproduction

```bash
LAIKA_GAIC_MODEL=/local/models/gaic-mobilenetv2-reddim16.fp32.onnx \
LAIKA_API_THREADS=1 \
python scripts/benchmark_api_composition.py /path/to/p3m-portrait.jpg \
  --strategy low-memory --target-long-edge 1200 \
  --report /tmp/leicashot-api-composition.json
```
