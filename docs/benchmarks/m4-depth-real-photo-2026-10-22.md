# Depth Anything V2 real-photo CPU baseline — 2026-10-22

## Run

The Laika Depth Anything V2 Small CPU adapter processed the same CC0 5760 × 3840 JPEG used for the M9 benchmark. It applied the reference analysis cap of 1220 pixels, padded the model tensor to 1232 × 826, inferred with ONNX Runtime's CPU provider, restored the float32 map to the full 5760 × 3840 source dimensions, and wrote an 88,473,728-byte `.npy` file atomically.

This verifies a real-photo model invocation, output geometry, finite result values, and end-to-end resource use. It is not a depth-accuracy evaluation: the source has no depth ground truth, and the original PyTorch checkpoint was not present in the measured worker for a same-photo comparison.

## Measurements

| Item | Measurement |
|---|---:|
| Device | MacBook Air 10,1, Apple M1, 8 GiB |
| OS / architecture | macOS 15.7.7, arm64 |
| Python / ONNX Runtime | 3.11.16 / 1.31.0 |
| Input | JPEG, 5760 × 3840, 22,118,400 pixels |
| Analysis / network size | 1220 × 813 / 1232 × 826 |
| Output | float32 depth, 5760 × 3840, 88,473,728-byte NPY |
| Preprocessing | 0.099 s |
| Model inference | 13.915 s |
| Restore to source size | 0.387 s |
| Engine total | 14.402 s |
| macOS process high-water RSS | 1,695,547,392 bytes (1,617 MiB) |
| macOS process high-water footprint | 1,819,645,888 bytes (1,735 MiB) |
| `/usr/bin/time -l` wall / user / system | 15.50 s / 12.61 s / 1.87 s |

The depth values were finite and non-constant (min 0, median 0.3422, p99 3.9161, max 5.8553). These are model-relative output values, not calibrated physical distances. The exported ONNX was 98,994,470 bytes, SHA-256 `79094f23efa5e1df4c7612d4c476020f526f5837d904787358ac160ecb969fce`, and its validated sidecar pins the source checkpoint hash and records seeded PyTorch/ONNX parity errors no larger than `1.67e-5`. This graph uses full attention; its sidecar has no query chunk size.

This was one unconstrained Mac run. It does not establish the depth stage's peak under Linux cgroups, the complete pipeline's memory, or the 4 GiB / 2 GiB targets. The full-resolution float32 map alone occupies 88.5 MB on disk and must be owned or streamed deliberately when wired into later bokeh stages.

## Reproduction

The source image is [Mountains Landscape on Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Mountains_Landscape.jpg), offered under CC0 1.0. Its SHA-256 is `4e6661606c195b2462c0f2321a7feefe2ae2fc081499f3462a30fbb108cd7f8e`. The model and output stayed outside Git.

```bash
/usr/bin/time -l .venv/bin/python -m laika.cli depth \
  /path/to/mountains-source.jpg /tmp/mountains-depth.npy \
  --model /path/to/depth-anything-v2-small-dynamic.onnx \
  --threads 1
```
