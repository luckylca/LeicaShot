# Docker API real-photo CPU smoke

Updated: 2026-10-22

## Run configuration

- Source: CC0 [Mountains Landscape](https://commons.wikimedia.org/wiki/File:Mountains_Landscape.jpg), JPEG 5760×3840, 3,878,306 bytes, SHA-256 `4e6661606c195b2462c0f2321a7feefe2ae2fc081499f3462a30fbb108cd7f8e`.
- Path: multipart HTTP upload → one bounded API task → CPU M9 + LabFusion → Depth Anything V2 → BokehMe ARNet and highlight processing → token-protected PNG download.
- Request: `color+bokeh`, Low Memory, focus `(0.5, 0.5)`, aperture `2.8`, 128-row bokeh chunks, explicitly requested 1024×683 output. The output dimensions are a preview-size smoke and do not certify native-size deployment memory.
- Runtime: local ARM64 Linux Docker guest, 2 logical CPUs, 4,085,092,352 B guest memory; API limit 2 CPUs / 3 GiB, one API worker, one ONNX intra-op thread; ONNX Runtime 1.31.0 `CPUExecutionProvider`.
- Model SHA-256: M9 `8936b8c86656265ab07c30b9b4cbbd7fa195d6e44cb3be1d4fd059bb36781e33`; Depth `a8ce30941c85b2c9af29fb26f36fbf7be3c1ba1dcf348187a51f6eaa856a238e`; ARNet `f0696c16a6eb02099292176f25b47e4dc53863e7e4f3f7748e66470d54cf0173`.

## Result

- The API accepted the image, emitted stage progress, completed all three models, and served the result through its download endpoint in 48.808 s wall time. Engine-reported time was 48.428 s.
- Stage times in the API worker: Depth inference 14.280 s; M9 inference 4.671 s; LabFusion 2.972 s; ARNet passes 8.655 / 3.168 s; highlight scatter 7.014 s; all stages and timings are in the task response.
- API container cgroup memory peak was 1,402,888,192 B of 3,221,225,472 B. `memory.events` reported zero OOM and limit events; the container was not OOM-killed. Idle API memory after the task was about 376 MB cgroup usage.
- The returned RGB PNG is 1024×683, 897,962 bytes, SHA-256 `073d9d6dfe92c9e512a88f8da076405ea461818ea36eb604da9e1f171ac7cef1`.
- A direct `laika pipeline` run in the same Linux API image, with the same source, model hashes, options and runtime, produced a pixel-identical PNG: maximum channel error 0, mean error 0, and identical file SHA-256. This checks that HTTP orchestration uses the same Python engine output.
- A same-options Python SDK run on the macOS host took 50.154 s and reached 733,970,432 B maximum process RSS. Its image differed from the Linux API result by mean absolute channel error 0.1912, maximum 8, p95/p99 1, and PSNR 54.97 dB; 81.59% of channel values were exact. The small platform-level numerical variation is recorded rather than hidden.

The API and Web also passed local health checks; Web `/healthz`, the React entry document, direct API health, and the same-origin API proxy returned successfully. The containers ran non-root with read-only root filesystems. This smoke validates deployment packaging and a real CPU inference request, not the 8/4/2 GiB targets for all modes and source resolutions.

## Reproduction

Mount the authorized model directory read-only by setting `LAIKA_MODEL_DIR` to its host path in `.env`, then run `docker compose up --build -d`. Upload the source with `options` set to:

```json
{"mode":"color+bokeh","strategy":"low-memory","focus_x":0.5,"focus_y":0.5,"aperture":2.8,"output_size":[1024,683],"bokeh_row_chunk_height":128}
```

Use `POST /api/v1/tasks`, poll the returned `status_url` with `X-Laika-Task-Token`, then download the returned `download_url`. Model files and image outputs remain outside Git.
