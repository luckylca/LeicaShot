# ARNet scale-transition tiling at a 2 GiB task limit

Updated: 2026-10-23

## Finding

The OOM trace from the prior five-model run shows a sharp memory rise when
ARNet moves from its 2880×1920 tiled pass to the next 1440×960 scale. The
1440×960 frame has 1,382,400 pixels, below the Low Memory 4,000,000-pixel
full-inference threshold, so ONNX Runtime previously processed that scale in a
single frame. The measured cgroup high-water was already at the 2 GiB limit
after the preceding pass; this transition caused the task to be OOM-killed.

Low Memory now tiles any even frame above 1,000,000 pixels into the existing
512×512 cores with the validated 16-pixel receptive-field halo. The graph,
scales, input pixels, and output dimensions remain unchanged. Balanced mode
retains its full-frame policy. Odd dimensions still use the full-frame path.

## Full-model parity at the changed scale

Before changing the default, the exact 1440×960 pass was run twice on the real
M9 output and real Depth Anything V2 map from the CC0 mountains photo. The
first run used one full-frame ARNet call; the second used six 512×512 cores.
Both used the same CPU ONNX session, one thread, and 16-pixel full-context
halo.

| Measurement | Full-frame | Six tiles |
|---|---:|---:|
| Wall time | 8.986 s | 8.649 s |
| Maximum FP32 absolute error | — | 0 |
| Mean FP32 absolute error | — | 0 |
| Differing output values | — | 0 |
| Tensor SHA-256 | `b2fcbe6033a087c34d2a478da8f3717e84d54abde13a5236c8f95f45b6c3a6e4` | same |

These are sequential Mac CPU timings and do not demonstrate a speed change.
The important result is bitwise parity for the real intermediate tensor.

## Full-resolution 2 GiB end-to-end run

The current Python engine processed the 5760×3840 CC0 mountains photo through
M9 + LabFusion, Depth Anything V2, all four ARNet scales, highlight extraction,
clipped-light recovery, optical highlight scatter, and PNG export. It ran on
Linux ARM64 under a 2-vCPU quota and cgroup-v2 `memory.max=2,147,483,648` B
with `memory.swap.max=0`. Runtime: Python 3.11.17, NumPy 2.4.6, OpenCV 5.0.0,
ONNX Runtime 1.31.0, CPUExecutionProvider, one inference thread.

| Measurement | Result |
|---|---:|
| Complete task wall time | 1,409.220 s |
| Process-tree peak RSS | 1,824,636,928 B |
| cgroup `memory.peak` | 2,029,903,872 B (94.52% of limit) |
| cgroup headroom at high-water | 117,581,776 B |
| OOM / limit events | 0 / 0 |
| ARNet pass times | 172.621 / 45.814 / 11.270 / 4.418 s |
| Highlight scatter | 1,108.614 s |
| M9 inference / LabFusion | 4.711 / 3.363 s |
| Depth inference | 15.986 s |
| Output | 5760×3840 RGB PNG, 16,621,497 B |
| Output SHA-256 | `62bf1f9b29a93e15435d5f38eb9975d5e910ca773fc5eb4e6de43b16a8ca1aef` |

The output hash exactly matches the earlier full-resolution no-subject result
for this photograph. All original output pixels were retained. The model
hashes were M9
`8936b8c86656265ab07c30b9b4cbbd7fa195d6e44cb3be1d4fd059bb36781e33`, Depth
`a8ce30941c85b2c9af29fb26f36fbf7be3c1ba1dcf348187a51f6eaa856a238e`, and
ARNet `f0696c16a6eb02099292176f25b47e4dc53863e7e4f3f7748e66470d54cf0173`.

This is a 2 GiB completion for the three-model color + bokeh path with
highlights, without SAM, ViTMatte, or subject protection. It does not establish
2 GiB support for the full five-model subject-enabled task, which previously
OOMed and still requires a separate run. On the completed candidate,
`full_inference_pixel_limit=1,000,000`; the 1440×960 ARNet pass emitted six
tile regions. The output file and detailed 1-second progress trace remain in
the local Docker volume `leicashot-arnet-pass-peak-2g-20261023`.

## Reproduction

Run the real-tensor comparison with the local authorized models and matching
M9 output:

```bash
PYTHONPATH=src python scripts/benchmark_arnet_inference_threshold.py \
  /path/to/mountains.jpg /path/to/mountains-m9.png \
  /models/depth-v2-small-dynamic-q64.onnx \
  /models/legendary-1.0-arnet.fp16.onnx \
  --scale 4 --focus 0.5 --aperture 2.8 \
  --full-threshold 4000000 --tiled-threshold 1000000
```

Run the full-resolution pipeline with the benchmark script and set the
container limits to `--memory=2g --memory-swap=2g --cpus=2` (equal memory and
swap limits leave no swap allowance). Mount the repository at
`/opt/laika-src:ro`, models at `/models:ro`, the source JPEG at
`/input/mountains-5760x3840.jpg:ro`, and a writable results volume at `/out`;
set `PYTHONPATH=/opt/laika-src/src`. The command inside the container was:

```bash
python /opt/laika-src/scripts/benchmark_processing_pipeline.py \
  /input/mountains-5760x3840.jpg /out/result.png \
  --m9-model /models/leica-m9-v3-epoch11.fp16.onnx \
  --depth-model /models/depth-v2-small-dynamic-q64.onnx \
  --bokeh-model /models/legendary-1.0-arnet.fp16.onnx \
  --mode color+bokeh --strategy low-memory --threads 1 \
  --focus-x 0.5 --focus-y 0.5 --aperture 2.8 \
  --progress-log /out/progress.jsonl
```
