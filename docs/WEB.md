# LeicaShot Web editor

The first Web client is a React + TypeScript editor that calls the versioned
FastAPI service. Image processing remains in the Python engine.

## Run locally

Start the API in one terminal with the models you want to use:

```bash
python -m pip install -e '.[api,inference]'
export LAIKA_M9_MODEL=/local/models/leica-m9-v3-epoch11.fp16.onnx
export LAIKA_DEPTH_MODEL=/local/models/depth-v2-small-dynamic-q64.onnx
export LAIKA_BOKEH_MODEL=/local/models/legendary-1.0-arnet.fp16.onnx
export LAIKA_GAIC_MODEL=/local/models/gaic-mobilenetv2-reddim16.fp32.onnx
# Optional, required only for subject protection:
export LAIKA_SAM_MODEL_DIR=/local/models/sam2-onnx
export LAIKA_VITMATTE_MODEL=/local/models/vitmatte-onnx/vitmatte-small-composition-1k.fp32.onnx
laika-api
```

In another terminal:

```bash
cd web
npm ci
npm run dev
```

Open `http://127.0.0.1:5173`. Vite proxies `/api` to the local Python service
at `http://127.0.0.1:8000`. For another API origin, set
`VITE_API_BASE_URL` before running or building Vite and configure the API's
`LAIKA_API_CORS_ORIGINS` with the exact Web origin.

## Current controls

- Upload or drop JPEG, PNG, and WebP photos.
- Choose M9 color, depth blur, or combined rendering.
- Click the oriented image to set the normalized depth focus point; choose a
  supported aperture value.
- Enable subject protection in a blur mode, switch the image click tool between
  focus and subject prompts, add up to 16 positive points, and remove one or
  clear them all. Select a 512, 768, or 1024 ViTMatte tile size; the labels
  describe the memory/context tradeoff, while per-scene quality still needs
  validation.
- Ask GAIC for ranked crop suggestions and select a crop before rendering.
- Follow stage progress, cancel an active task, compare the original crop with
  the processed preview, and download the full-resolution PNG.

Task tokens stay in page memory and are sent only in the API authorization
header. The browser does not persist uploaded photos or tokens. The API deletes
temporary task files on cancellation/failure or after its configured retention
period.

The Web client currently uses Low Memory mode. Subject protection requires
local SAM and ViTMatte model assets configured on the API. It uses the same
Python engine request as the SDK and CLI; up to 16 selected points are sent in
one task, allowing the engine to reuse SAM image embeddings across prompts.
Interactive freeform cropping, output-format selection, account sign-in, and
production static hosting are future work. The API returns the task preview at
up to 1600 pixels for fast display; the download endpoint returns the full PNG.
The real three-point, five-model HTTP run measured 102.177 s and
1,029,177,344 B sampled server process-tree RSS, reused two SAM embeddings, and
matched the previously validated SDK output hash; see the
[subject-protection report](benchmarks/web-subject-protection-2026-10-22.md).

## Browser smoke record

On 2026-10-22, the editor was exercised in Chrome against the local FastAPI
service with the local authorized M9, Depth, ARNet, and GAIC models. A 1080 ×
1620 P3M-10K portrait uploaded successfully; GAIC returned ranked crops, one was
selected, and a combined color+bokeh task completed. The page showed progress
and the processed preview, and the full PNG download was checked independently:
918 × 1224 pixels, 817,062 bytes, SHA-256
`0007097d9a15b5c12bd45c84b39593eaf86e906634a0464b426de77b0c53c068`.
Desktop and 390 × 844 mobile layouts were visually inspected. This browser
smoke confirms the connected flow; its timing and RSS were not sampled. The
separate [HTTP task benchmark](benchmarks/api-task-real-photo-2026-10-22.md)
contains server memory and stage measurements for the same source and engine.

The subject-protection controls were then exercised in Chrome against the
real-model API. Three click-selected points and tile size 512 completed the
five-model task, preserved 1080 × 1620 output, displayed the before/after
preview and downloaded a 1,662,940-byte PNG. See the
[subject-protection browser/API report](benchmarks/web-subject-protection-2026-10-22.md)
for the emitted point coordinates, output hash, and separate RSS measurement.
