# Local Docker deployment

The Compose deployment runs the Python API and the existing React editor as
separate CPU containers. The API is still the only image-processing engine;
the Web container serves static files and proxies `/api/v1/` to that service.
The images do not contain model weights or user photographs.

## Prepare local model assets

Copy the `.env.example` file to `.env`, then set `LAIKA_MODEL_DIR` to a
directory containing the authorized local model files. The default paths are:

```text
models/leica-m9-v3-epoch11.fp16.onnx
models/depth-v2-small-dynamic-q64.onnx
models/legendary-1.0-arnet.fp16.onnx
```

The directory is mounted read-only at `/models`. The `.onnx` weights remain
ignored by Git and are never copied into either image. Optional SAM, ViTMatte,
and GAIC assets can be enabled by setting `LAIKA_SAM_MODEL_DIR`,
`LAIKA_VITMATTE_MODEL`, and `LAIKA_GAIC_MODEL` to paths inside that same
directory. Follow [the model guide](MODELS.md) for the exact optional file
layout.

## Start the app

```bash
cp .env.example .env
docker compose up --build -d
docker compose ps
```

Open the editor at <http://127.0.0.1:8080>. The API health endpoint is
<http://127.0.0.1:8000/api/v1/health>; the Web service proxies the same endpoint
at <http://127.0.0.1:8080/api/v1/health>. Both ports bind to loopback by default.

Check service logs and stop the containers with:

```bash
docker compose logs -f api web
docker compose down
```

Completed task files live in the named `leicashot-data` volume and expire after
`LAIKA_API_RESULT_TTL_SECONDS` (six hours by default). `docker compose down`
preserves that volume. `docker compose down -v` removes it and all retained task
files. Model files remain in the host directory and are mounted read-only.

## Resource settings

The API defaults to the `low-memory` strategy, one ONNX intra-op thread, one
active processing task, one queued task, a 3 GiB container memory limit, and two
CPU cores. Web has separate 256 MiB and 0.5 CPU limits. Change
`LAIKA_MEMORY_LIMIT`, `LAIKA_CPU_LIMIT`, `LAIKA_API_THREADS`,
`LAIKA_API_MAX_WORKERS`, or `LAIKA_API_QUEUE_CAPACITY` in `.env` to match the
host. More workers multiply image and model working sets. On the 4 GiB Linux
ARM64 guest with a 3 GiB task cgroup, one five-model 1620×1080 request peaked
at 1,867,460,608 B; two concurrent workers reached 3,129,069,568 B (97.14% of
the cap), leaving only 87.9 MiB. Keep one worker for 3 GiB deployments. Raise
concurrency only after measuring the complete model set and largest supported
image sizes under the intended cgroup limit. See the [API concurrency report](benchmarks/api-concurrency-memory-2026-10-23.md).

The 3 GiB default is a starting configuration, not a claim that every mode or
source resolution has passed a 3 GiB cgroup test. See [project status](PROJECT_STATUS.md)
for current end-to-end measurements and remaining full-resolution work.

## Network access

The Web editor uses a same-origin reverse proxy, so it does not require CORS.
The API also binds to loopback for local SDK or CLI clients. If another machine
must connect, set `LAIKA_BIND_ADDRESS` deliberately and use an authenticated
TLS reverse proxy at the network boundary. The API does not provide user
accounts; task tokens isolate task results but are not a substitute for
network-level authentication. For a separately hosted browser client, set
`LAIKA_API_CORS_ORIGINS` to exact trusted origins.

## Build and health checks

GitHub Actions builds both images, validates the Compose file, starts the
containers without model assets, and checks the API and Web health endpoints.
A separate local ARM64 Linux run mounted the real authorized model files and
completed M9 + Depth + ARNet + highlights through one HTTP request; timing,
cgroup peak, hashes and direct SDK parity are in the [Docker API benchmark](benchmarks/docker-api-real-photo-2026-10-22.md).
That preview-size smoke does not certify native-resolution resource targets.
See [project status](PROJECT_STATUS.md) for current low-memory results.
