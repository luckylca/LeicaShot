# M1 dynamic Depth Anything V2 export

## Scope

This experiment adds a dynamic-height/width ONNX export path for the pinned Depth Anything V2 Small model. It tests a seeded tensor set and measures one high-resolution model-only input. It does not include photo decoding, upstream resize/normalization, final interpolation back to source dimensions, or any other Laika stage.

## Export and numerical checks

- Host: MacBook Air `MacBookAir10,1`, Apple M1, arm64, 8 GiB RAM, macOS 15.7.7.
- Python 3.11.16; PyTorch 2.14.1; ONNX 1.23.2; ONNX Runtime 1.31.0; one CPU thread.
- Source revision: `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`.
- Checkpoint SHA-256: `715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378`.
- ONNX opset 17; 98,994,470-byte graph with embedded weights; SHA-256 `79094f23efa5e1df4c7612d4c476020f526f5837d904787358ac160ecb969fce`.
- Export time 3.917 s; model construction and strict weight load 1.525 s; ONNX Runtime session creation 0.282 s; four validation inferences summed to 21.544 s.

An initial dynamic export declared dynamic ONNX inputs but retained a fixed positional-embedding token count. A second aspect ratio failed in ONNX Runtime while broadcasting 1,814 and 1,901 tokens. The implemented adapter computes positional resize scales from runtime dimensions and uses runtime patch sizes for the DPT output. It keeps the upstream resize rule, including the 0.1 offset and base-grid identity. The DPT adapter's PyTorch output matched the upstream forward at all validation shapes with maximum absolute error 0.

| Network input H×W | Output H×W | Max absolute error | Mean absolute error | p99 absolute error |
|---:|---:|---:|---:|---:|
| 518×686 | 518×686 | 5.72e-6 | 1.30e-6 | 3.58e-6 |
| 700×532 | 700×532 | 1.67e-5 | 1.29e-6 | 4.77e-6 |
| 1400×518 | 1400×518 | 5.96e-6 | 2.95e-6 | 4.77e-6 |
| 518×518 | 518×518 | 6.91e-6 | 1.96e-6 | 5.25e-6 |

All four outputs were finite and retained their full input dimensions. The largest error remained below the configured `1e-4` limit. The dimensions are all multiples of 14 and at least 518 pixels; the ONNX graph does not add a runtime assertion for that contract, so the future engine must validate it before inference.

## 1400×518 CPU/RSS comparison

The model benchmark used the same seeded `[1,3,1400,518]` input, one warmup, one measured run, one CPU thread, and separate worker processes. Process-tree RSS and CPU were sampled every 5 ms. The two workers ran sequentially; macOS did not expose USS or cgroup counters. Host available memory before/after was 1,334,018,048 / 2,929,754,112 bytes for ONNX, then 2,387,689,472 / 2,787,246,080 bytes for PyTorch.

| Runtime | Model/session setup | Inference | Worker wall | Peak process-tree RSS | Sampled CPU |
|---|---:|---:|---:|---:|---:|
| PyTorch 2.14.1 | 2.356 s | 4.125 s | 11.716 s | 1,487,093,760 bytes | 10.804 s |
| ONNX Runtime 1.31.0 CPU | 0.588 s | 6.976 s | 14.919 s | 1,270,792,192 bytes | 14.296 s |

At this one large input, ONNX used 14.6% less sampled worker RSS and was 1.69× slower. The sample is not a Linux memory-limit test or a reliable cross-device speed ranking. The input is a synthetic normalized tensor; there is no photo-quality result here.

## Reproduction

```bash
/tmp/laika-depth-env/bin/python scripts/export_depth_anything_v2.py \
  --source-checkout /tmp/laika-depth-anything-v2 \
  --weights "$HOME/.cache/laika/models/depth_anything_v2_vits.pth" \
  --output /tmp/laika-depth-v2-small-dynamic.onnx \
  --height 518 --width 686 --threads 1 --dynamic

/tmp/laika-depth-env/bin/python scripts/benchmark_depth_anything_v2.py \
  --runtime onnx --onnx /tmp/laika-depth-v2-small-dynamic.onnx \
  --height 1400 --width 518 --threads 1 --runs 1
```

The next memory experiment should preserve dynamic global attention while chunking Q and attending to every K/V token. No query-chunked graph or 8/4/2 GiB complete-pipeline result is included in this report.
