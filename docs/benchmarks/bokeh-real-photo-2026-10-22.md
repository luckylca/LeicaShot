# BokehMe ARNet real-photo validation — 2026-10-22

## Model and numerical validation

The BokehMe ARNet checkpoint was downloaded from its pinned upstream commit and verified before conversion:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `arnet.pth` source checkpoint | 11,101,736 | `6c7dba6f32e3afc8e984eb5adfecfea5d13b6f4ebd132a5a890beba384be4752` |
| Laika dynamic FP16 ONNX export | 1,868,251 | `f0696c16a6eb02099292176f25b47e4dc53863e7e4f3f7748e66470d54cf0173` |

The exporter checked the pinned `neural_renderer.py` source hash, loaded the checkpoint with `weights_only=True`, and passed synthetic dynamic shapes 64×80, 97×129, and 480×481. On these cases, the largest FP16-to-PyTorch float error was `0.00103045`; 8-bit PSNR was at least `62.31 dB`, maximum channel error was 1, and no channel error exceeded 2.

The real-photo comparison used the 5760×3840 CC0 [Mountains Landscape photo](https://commons.wikimedia.org/wiki/File:Mountains_Landscape.jpg), a previously measured Depth Anything V2 map, and an explicit 1024×683 validation raster. With focus at `(0.5, 0.5)` and aperture 2.8, FP16 ONNX versus pinned upstream PyTorch ARNet produced:

| Metric | Result |
|---|---:|
| Maximum absolute float error | 0.000742674 |
| Mean absolute float error | 0.0000915710 |
| 8-bit PSNR | 63.63 dB |
| Maximum 8-bit channel error | 1 |
| Channels with error >2 | 0% |

The reference project records a different ONNX SHA-256, `a3aa5fa6a0818f36dbfa4be4dd5471d874cea5981357b7aeb28919954e70d05c`, at 1,867,929 bytes. The generated graph is therefore not byte-identical to the reference ONNX file. The exporter environment and conversion report are recorded; the cause of the serialization difference has not been isolated. The checkpoint-backed numerical tests validate the local graph's outputs against the upstream PyTorch implementation.

## CPU renderer measurement

The Python `laika bokeh` CLI ran the reference CoC/ARNet renderer on CPU. Both runs consumed the raw depth map generated earlier for this photograph; neither includes Depth Anything inference. Output dimensions were explicit, so these experiments do not imply a default size reduction.

| Requested output | Focus / f-number | ARNet scales | CLI measured work | Process wall | High-water RSS | PNG bytes |
|---|---|---|---:|---:|---:|---:|
| 1024×683 | 0.0617418 / 2.8 | 1.0, 1.19626 | 7.922 s | 8.31 s | 658,980,864 B (628.7 MiB) | 847,228 |
| 2048×1365 | 0.0617620 / 2.8 | 1.0, 2.0, 2.39252 | 27.437 s | 27.74 s | 1,118,208,000 B (1,066.6 MiB) | 2,520,612 |

Host for both runs: MacBook Air 10,1, Apple M1, 8 GiB RAM, macOS 15.7.7 arm64, Python 3.11.16, ONNX Runtime 1.31.0, `CPUExecutionProvider`, `low-memory`, one thread. The 1024 run had ARNet pass times 4.055 s and 2.836 s. At 2048, ARNet pass times were 19.228 s, 4.126 s, and 2.987 s; optical math and final composite were 0.0017 s and 0.0499 s. CLI measured work includes depth normalization, explicit resampling, model inference, and PNG export, but excludes image decode and loading the `.npy` header.

RSS came from macOS `/usr/bin/time -l` in fresh Python CLI processes. These are single runs without a cgroup cap, not 8/4/2 GiB acceptance results. They do not include depth inference, M9/LabFusion, highlights, or subject matte. No full 5760×3840 ARNet render was attempted. The photo render outputs remain outside Git under `/tmp/laika-bokeh-results/`, including the 2048×1365 result `mountains-bokeh-final-2048.png`.

## Reproduction

Export with the locally fetched checkpoint and pinned upstream source as documented in [model setup](../MODELS.md#bokehme-arnet). Then run:

```bash
/usr/bin/time -l laika bokeh \
  /path/to/mountains-source.jpg \
  /path/to/mountains-depth.npy \
  /tmp/mountains-bokeh.png \
  --model /path/to/legendary-1.0-arnet.fp16.onnx \
  --width 1024 --height 683 \
  --focus-x 0.5 --focus-y 0.5 --aperture 2.8 \
  --strategy low-memory --threads 1
```

The output retains the requested 1024×683 size. Omit `--width` and `--height` to render at the oriented input dimensions.
