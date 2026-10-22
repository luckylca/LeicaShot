# HTTP API

The FastAPI service is a thin adapter over `LaikaProcessingEngine`; the Python
SDK and HTTP requests use the same `ProcessingOptions` and model lifecycle.

## Install and start

```bash
python -m pip install -e '.[api,inference]'
LAIKA_M9_MODEL=/local/models/leica-m9-v3-epoch11.fp16.onnx \
LAIKA_DEPTH_MODEL=/local/models/depth-v2-small-dynamic-q64.onnx \
LAIKA_BOKEH_MODEL=/local/models/legendary-1.0-arnet.fp16.onnx \
laika-api
```

`laika-api` starts one Uvicorn process on `127.0.0.1:8000`. Set
`LAIKA_API_HOST` and `LAIKA_API_PORT` to change the bind address. The task queue
is process-local, so deploy one service process per task workspace. The OpenAPI
schema is available at `/api/v1/openapi.json` and `/docs`.

Optional model paths are `LAIKA_SAM_MODEL_DIR`, `LAIKA_VITMATTE_MODEL`, and
`LAIKA_GAIC_MODEL`. Subject points require both SAM and ViTMatte assets. A crop
can be supplied in task options after selecting it through the SDK, CLI, or the
GAIC suggestion route.

For a separately hosted browser client, set `LAIKA_API_CORS_ORIGINS` to a
comma-separated list of exact origins, without URL paths or trailing slashes,
for example `https://editor.example`. CORS is disabled by default; the local
Vite proxy keeps development requests same-origin. This setting does not provide
user authentication.

## Limits and lifecycle

| Environment variable | Default | Meaning |
|---|---:|---|
| `LAIKA_API_STORAGE` | system temp `/leicashot-api` | Isolated task directories |
| `LAIKA_API_MAX_UPLOAD_BYTES` | 50 MiB | Per-file upload limit |
| `LAIKA_API_MAX_PIXELS` | 40,000,000 | Decoded source-pixel limit |
| `LAIKA_API_MAX_NETWORK_PIXELS` | 4,000,000 | Model tensor budget forwarded to the engine |
| `LAIKA_API_MAX_WORKERS` | 1 | Simultaneous processing tasks |
| `LAIKA_API_QUEUE_CAPACITY` | 1 | Additional waiting tasks |
| `LAIKA_API_MAX_STORAGE_BYTES` | 10 GiB | Admission budget for retained tasks, active task working sets, and composition uploads |
| `LAIKA_API_RESULT_TTL_SECONDS` | 21,600 | Completed task retention before cleanup |
| `LAIKA_API_STRATEGY` | `low-memory` | Default `low-memory` or `balanced` worker strategy |
| `LAIKA_API_THREADS` | 1 | ONNX Runtime intra-op thread count |

Workers, request bytes and decoded pixels are bounded. Uploads are streamed to
per-task directories after a UUID is generated; the client filename is never
used as a path. The server checks the decoded image format, frame count,
dimensions, and file bytes before queuing inference. A per-task random access
token is returned once and required in the `X-Laika-Task-Token` header for
status, cancellation, preview, source, and result requests. Only a SHA-256
digest of that token is stored.

Completed originals and results are temporary and expire after the configured
TTL. Failed and cancelled tasks delete image files immediately. A normal server
shutdown removes its task directories; after an unclean stop, marked task
directories are removed once they exceed the TTL. The task queue and status
metadata are in memory and are not restored after restart.

Storage admission accounts for the upload and a conservative per-task working
set for every active reservation, so simultaneous uploads cannot each consume
the same remaining disk budget. Real task temporary files also count toward the
workspace limit. This is an admission guard, not a hard OS/container disk quota.
Composition suggestions use the same worker pool, queue limit, upload-byte
limit, storage budget, and Low Memory/Balanced engine policy as image tasks.
The route returns ranked crop coordinates synchronously and deletes the upload
after scoring.

## Create and follow a task

`POST /api/v1/tasks` is multipart form data with a `file` part and an optional
JSON `options` field:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/tasks \
  -F 'file=@portrait.jpg' \
  -F 'options={"mode":"color+bokeh","strategy":"low-memory","focus_x":0.52,"focus_y":0.38,"aperture":2.8,"crop":[0.08,0.04,0.92,0.96]}'
```

The response is `202 Accepted` and contains the task ID, one-time token, and
status/cancel URLs. Poll the status endpoint with the token:

`POST /api/v1/compositions` accepts a multipart `file`, optional `strategy`
(`low-memory` by default), and optional `target_long_edge` (1–8192). It requires
`LAIKA_GAIC_MODEL` and returns source dimensions, candidate count, runtime
metadata, and ranked normalized crop boxes.

```bash
curl http://127.0.0.1:8000/api/v1/tasks/TASK_ID \
  -H 'X-Laika-Task-Token: TASK_TOKEN'
```

Options support the processing modes `color`, `bokeh`, and `color+bokeh`,
`low-memory` or `balanced` strategy, normalized focus coordinates, a supported
aperture, optional `[width,height]` output dimensions, a selected normalized
crop, subject points, ViTMatte tile size, and row chunk sizes. Output dimensions
remain at the oriented source size unless an explicit output size or crop is
requested.

Cancel a queued or running task:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/tasks/TASK_ID/cancel \
  -H 'X-Laika-Task-Token: TASK_TOKEN'
```

When complete, the task response includes links to the original image, a
maximum-1600-pixel preview, and the full-size PNG result. These routes use the
same task token.

The token is a per-task capability, not an account system. Use an authenticated
deployment boundary when serving multiple users over a network.

The first real CPU-only HTTP run processed a 1080×1620 portrait through M9,
Depth, ARNet, and an explicit crop in one task. It returned a 918×1224 PNG in
90.604 seconds with a sampled 698,236,928-byte server process-tree RSS peak.
See the [full benchmark report](benchmarks/api-task-real-photo-2026-10-22.md)
for hardware, stage timing, model hashes, and measurement limits. The real GAIC
route scored 86 candidates in 1.403 seconds at a sampled 416,251,904-byte
server process-tree RSS peak; details are in the
[composition benchmark](benchmarks/api-composition-real-photo-2026-10-22.md).
The Web editor was exercised in a browser against this API: a real image was
uploaded, GAIC suggestions were selected, a M9+Depth+ARNet task completed, the
preview rendered, and the full PNG downloaded. Web interaction and build checks
are included in CI.
