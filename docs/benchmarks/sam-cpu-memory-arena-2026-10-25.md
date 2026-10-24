# SAM Tiny CPU memory-arena experiment, Windows 2026-10-25

This is an optional, default-off memory optimization for the low-memory
Python SDK, CLI and Web/API pipeline. To enable:

    $env:LAIKA_SAM_DISABLE_CPU_ARENA = "1"       (PowerShell)
    export LAIKA_SAM_DISABLE_CPU_ARENA=1         (Linux/macOS shell)

It disables ONNX Runtime CPU memory arena allocation for SAM encoder
and decoder sessions only, preserving ORT_ENABLE_ALL, model weights,
SAM preprocessing, float32 arithmetic, and numerical masks.
The balanced strategy is unchanged. Unset the variable for reference mode.

## Standalone SAM paired comparison

2048x1365 reference photo; encoder input 1024x1024; one positive center
point; one CPU inference thread; independent fresh workers:

| Setting | Sampled process RSS peak | Prompt inference time | Mask |
|---|---:|---:|---|
| Normal CPU arena | 1,003,229,184 B | 4.021 s | identical SHA256 |
| Disable CPU arena | 903,708,672 B | 4.328 s | identical SHA256 |

Mask SHA256:
55ef312d6aa67c1ba42c73f5034bc949d67cd5dfb8e0429c4a08d9deed37b257

Disabling memory pattern alone did not lower RSS; disabling arena did.

## Correctly wired full five-model pipeline A/B

The exact same real-photo 2048x1365 full M9+Depth+SAM+ViTMatte+ARNet+
optical-scatter CPU workflow in both arms, two fresh-process pairs
in reversed order. The optional verified ARNet split mode is enabled
in BOTH arms (two split pass calls observed); only SAM arena changes.
Model thread count 1, 512 matte tiles, same model/checkpoint hashes.
Each arm logs the actual SAM encoder and decoder arena setting.

| Pair | Default total | No-arena total | Default RSS B | No-arena RSS B |
|---|---:|---:|---:|---:|
| Default -> disabled | 71.4750s | 71.1704s | 984,813,568 | 828,219,392 |
| Disabled -> default | 71.0061s | 72.2477s | 984,727,552 | 828,043,264 |
| Mean | 71.2406s | 71.7091s | 984,770,560 | 828,131,328 |

Mean sampled peak RSS decreased by **156,639,232 bytes
(about 149.4MiB, 15.9%)**. Mean time was approximately 0.7% slower,
a difference too small to establish stable latency impact here.

The two full RGB PNG images compared pixel-exact: zero differing pixels,
maximum 8-bit channel difference zero. These results compare SAM
allocator configuration with the SAME split-ARNet background in both
arms, not original unsplit ARNet model fidelity.

## Caveats and further research

RSS sampled at 150ms is not Linux cgroup peak memory. No native ~20MP
end-to-end 2GiB cgroup proof, and no multi-worker API concurrency
guarantee. The underlying SAM encoder attentions and intermediate
activations still consume memory; this setting reduces allocator
pooling rather than the mathematical amount of activation data.
Results may vary by CPU or ONNX Runtime revision.

Local, uncommitted reproducibility scripts:
E:\codex\LeicaShot-local-data\sam-ort-options-worker.py
E:\codex\LeicaShot-local-data\benchmark-sam-arena-verified.py
