# CPU throughput / memory speed matrix (experimental)

This research script measures **whole-request** speed, process-tree peak RSS, and
actual PNG pixel differences while varying the CPU inference thread count.
It uses **existing authorized local model files**; it does not download,
retrain, redistribute, or modify a model.

## Windows 2048 × 1365 run

From the `LeicaShot` virtual environment on Windows:

```powershell
& .\.venv\Scripts\python.exe scripts\benchmark_speed_matrix.py `
  --input "E:\codex\LeicaShot-local-data\validation\portrait-cc0-2048.jpg" `
  --m9-model "E:\codex\LeicaShot-local-data\laika-models\leica-m9-v3-epoch11.fp16.onnx" `
  --depth-model "E:\codex\LeicaShot-local-data\laika-models\depth-v2-small-dynamic-q64.onnx" `
  --bokeh-model "E:\codex\LeicaShot-local-data\laika-models\legendary-1.0-arnet.fp16.onnx" `
  --sam-model-dir "E:\codex\LeicaShot-local-data\laika-models\sam2-onnx" `
  --vitmatte-model "E:\codex\LeicaShot-local-data\laika-models\vitmatte-onnx\vitmatte-small-composition-1k.fp32.onnx" `
  --arnet-split-dir "E:\codex\LeicaShot-local-data\laika-models\arnet-split-verified" `
  --arnet-split-edge 384 `
  --threads "1,2,4" --repeat 2 --memory-budget-mib 2048 `
  --output-dir "E:\codex\LeicaShot-local-data\benchmarks\thread-matrix"
```

The script launches a fresh Python process for **each** full five-model job,
samples the process tree, records Laika's per-stage times, and compares each
full-resolution RGB PNG against the first baseline output. Test order reverses
for the second repetition (1, 2, 4; then 4, 2, 1) to reduce warm-up/order bias.

ARNet's optional split-core tile edge may be set to 256, 384 or 512 to
explore the latency-versus-RSS tradeoff. Run each tile edge in separate
matrices, keeping every other setting fixed. The low-memory default
remains 256; the experimental `LAIKA_ARNET_SPLIT_EDGE` is ignored unless
`LAIKA_ARNET_SPLIT_DIR` is also configured. A larger edge might be faster
but could increase peak memory; this requires on-host proof.

The `speed-matrix.json` result includes median times, sampled max RSS,
and pixel-difference metrics. The output retains a single reference PNG
unless `--keep-pngs` is passed. Keep test photos, weights and PNGs **outside
Git**.

## Decision rule

Choose a configuration based on **whole-pipeline** median time, constrained by
sampled peak RSS and output quality. Do not assume more threads means faster:
some layers are memory-bandwidth bound, while others benefit from parallel
compute. Sampled RSS is **not** a Linux cgroup high water mark. Do not describe
this as 2 GiB production readiness or advertise a speedup before real runs.

This research branch is intentionally separate from `develop` until
tested on Windows with the actual model weights. Follow up by running the
same matrix on a native ~20 MP photograph, then Linux under 2/4 GiB cgroup
limits. Concurrency must be tested separately because one worker's RSS is
not an HTTP server's total resource usage.
