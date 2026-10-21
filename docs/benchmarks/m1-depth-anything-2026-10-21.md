# M1 Depth Anything V2 Small baseline

## Scope

This records one CPU-only PyTorch inference on a seeded synthetic image, plus a fixed-shape PyTorch-to-ONNX numerical check. It does not measure a real-photo quality set, a resolution sweep, ONNX peak RSS, or the complete Laika workflow. The exported model has a fixed `[1, 3, 518, 686]` network input and therefore is an export experiment, not a general inference adapter.

## Reproduction context

- Host: MacBook Air `MacBookAir10,1`, Apple M1, arm64, 8 GiB RAM, macOS 15.7.7, 8 CPU cores.
- Python 3.11.16; PyTorch 2.14.1; Torchvision 0.29.1; NumPy 2.4.6; OpenCV 5.0.0; ONNX 1.23.2; ONNX Runtime 1.31.0.
- CPU only, one PyTorch/ORT intra-op thread. PyTorch MPS was not used.
- Depth Anything V2 source revision: `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`.
- Official Small checkpoint: 99,218,434 bytes; SHA-256 `715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378`.
- The weight file, ONNX graph, external ONNX tensor data, and generated image were stored under `/tmp`, outside Git.

The PyTorch run used the upstream RGB conversion, ImageNet mean/std normalization, cubic resize, and aspect-ratio-preserving lower-bound sizing. A deterministic 512×384 RGB fixture became a `[1,3,518,686]` tensor; the output was interpolated back to 512×384 without changing the requested image size. Strict checkpoint loading succeeded and all depth values were finite.

## Preprocessing and model smoke measurement

The sampler observed the worker process tree every 5 ms from process launch through exit. RSS includes interpreter startup and native libraries; USS was unavailable from macOS `psutil` in this environment. The worker wall time includes importing PyTorch, which occurred before the model-stage timer.

| Measurement | Result |
|---|---:|
| Peak process-tree RSS | 1,007,566,848 bytes (960.9 MiB) |
| Peak sampled process-tree CPU time | 3.417 s |
| Cold worker wall time, including imports | 27.530 s |
| Model construction and strict weight load | 0.424 s |
| Preprocess | 0.0159 s |
| Forward pass and resize back to source dimensions | 1.270 s |
| Worker time after imports | 1.713 s |
| Network input / output | 518×686 / 384×512 |
| Output range / mean / standard deviation | 2.028–4.267 / 2.727 / 0.506 |

This is a preprocessing and model smoke baseline on one synthetic input, not evidence that all six model stages fit within a 1 GiB budget. The RSS sampler can miss allocations shorter than 5 ms and is not equivalent to Linux cgroup accounting.

## Fixed-shape ONNX check

`scripts/export_depth_anything_v2.py` exported the same pinned PyTorch model with opset 20 and external tensor data. ONNX Runtime used `CPUExecutionProvider` and one thread. The comparison used the same seeded standard-normal tensor in both runtimes; preprocessing and resize to an original photo size were outside this check.

| Measurement | Result |
|---|---:|
| ONNX graph | 801,345 bytes; SHA-256 `6e02329f90b159376993c9aff6877eaf785cbafe0a084fb04572687926665b7a` |
| External tensor data | 98,838,016 bytes; SHA-256 `73199d96c656d687d4d569b1ccc548bcf3e44eef79d37b2fbb9454f846bf3224` |
| Export time | 3.141 s |
| ORT session creation | 0.203 s |
| One ORT inference | 2.433 s |
| Maximum / mean / p99 absolute error | `5.01e-6` / `8.82e-7` / `2.86e-6` |
| Output | `[1,518,686]`, all finite |

The exporter passed `onnx.checker` and a maximum absolute error threshold of `1e-4`. This establishes fixed-shape numerical parity on one synthetic tensor. The conversion currently exports ordinary full global attention; no query-chunk memory optimization has been applied.

## Repeated PyTorch vs ONNX Runtime comparison

The project benchmark runner gave each runtime the same seeded standard-normal `[1,3,518,686]` tensor, one warmup, three measured runs, one CPU thread, and a separate process. The parent sampled worker-process-tree RSS/USS/CPU every 5 ms. This isolates model forward time from image decode, resize, and final interpolation. The processes were run sequentially. macOS did not expose USS or cgroup counters. The setup time includes runtime import and model/source/weight or ONNX integrity checks, so it is not a pure checkpoint/session initialization measure. Host available memory varied between trials: in trial A it was 1,325,400,064 bytes before and 2,104,967,168 bytes after PyTorch, then 1,963,409,408 before and 2,192,850,944 after ONNX; in trial B it was 1,842,266,112 before and 2,044,166,144 after PyTorch, then 1,710,129,152 before and 1,661,042,688 after ONNX.

| Runtime | Trial A median (range) | Trial B median (range) | Peak RSS A / B | Setup A / B | Worker wall A / B |
|---|---:|---:|---:|---:|---:|
| PyTorch 2.14.1 | 1.185 s (1.168–1.273) | 1.226 s (1.210–1.246) | 1,082,638,336 / 1,045,364,736 bytes | 2.315 / 2.297 s | 7.782 / 7.825 s |
| ONNX Runtime 1.31.0 CPU | 2.314 s (2.303–2.456) | 3.570 s (2.590–4.145) | 563,281,920 / 570,982,400 bytes | 0.280 / 0.247 s | 9.822 / 13.057 s |

For this fixed shape and machine, ONNX Runtime used 45–48% less sampled worker RSS. Its median forward was 1.95× slower in trial A and 2.91× slower in trial B. The slower ONNX trial coincided with lower available host memory and greater timing spread. This is a measured memory/speed tradeoff, not a universal runtime ranking; no other CPU, input size, or cgroup limit has been tested. ONNX has not met a CPU-speed objective in this configuration, while the lower worker RSS is promising for constrained workers.

Reproduce the comparison after exporting the matching 518×686 model:

```bash
/tmp/laika-depth-env/bin/python scripts/benchmark_depth_anything_v2.py \
  --runtime pytorch --source-checkout /tmp/laika-depth-anything-v2 \
  --weights "$HOME/.cache/laika/models/depth_anything_v2_vits.pth" \
  --height 518 --width 686 --threads 1 --runs 3
/tmp/laika-depth-env/bin/python scripts/benchmark_depth_anything_v2.py \
  --runtime onnx --onnx /tmp/laika-depth-v2-small-518x686.onnx \
  --height 518 --width 686 --threads 1 --runs 3
```

## Remaining M1 work

- Exercise multiple portrait, landscape, and extreme-aspect-ratio input sizes without reducing source or output resolution.
- Compare raw depth maps against an approved real-photo fixture set and keep numerical and visual checks separate.
- Export and inspect an exact global-attention query-chunk variant; verify chunk sizes 16/32/64/128 preserve all keys and inspect the ONNX graph for retained chunk structure.
- Repeat the PyTorch/ONNX comparison on Linux with cgroup-v2 limits and additional CPU hardware.
- Do not claim 8/4/2 GiB full-pipeline targets from this model-only experiment.
