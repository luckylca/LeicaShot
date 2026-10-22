# Full-resolution subject protection in a 3 GiB cgroup

Date: 2026-10-23. Executed from the source-verified checkout at commit
`47712b48d36289e90b53f8a55f2e0b2a2ed549cb`, the PR #37 merge on `develop`.
One call to `LaikaProcessingEngine.process_file()` ran the whole request in a
Linux ARM64 container. The repository source was mounted read-only at
`/opt/laika-src` and selected with `PYTHONPATH=/opt/laika-src`; model assets
were mounted read-only outside Git.

## Configuration

- Input: CC0 Mountains Landscape JPEG, 5760×3840, SHA-256
  `4e6661606c195b2462c0f2321a7feefe2ae2fc081499f3462a30fbb108cd7f8e`.
- Request: `color+bokeh`, Low Memory, CPU-only, no output-size override,
  focus `(0.5, 0.5)`, aperture `2.8`, positive subject point `(0.72, 0.66)`.
  The point selects the foreground church clock face. ViTMatte tile size was
  512 pixels; highlight row chunk size was 128. Export remained 5760×3840.
- Platform: Linux 7.0.0 ARM64 guest, 2 logical CPUs and 4,085,092,352 B RAM;
  Docker cgroup v2 cap 3,221,225,472 B, no swap, 2 CPU quota, one ONNX Runtime
  intra-op thread. Python 3.11, NumPy 2.4.6, OpenCV 5.0.0.93 and ONNX Runtime
  1.31.0. Inference used `CPUExecutionProvider` for all models.
- Source revision: `47712b48d36289e90b53f8a55f2e0b2a2ed549cb`.
- Model hashes:
  - M9: `8936b8c86656265ab07c30b9b4cbbd7fa195d6e44cb3be1d4fd059bb36781e33`.
  - Depth Anything V2 Small: `a8ce30941c85b2c9af29fb26f36fbf7be3c1ba1dcf348187a51f6eaa856a238e`.
  - BokehMe ARNet: `f0696c16a6eb02099292176f25b47e4dc53863e7e4f3f7748e66470d54cf0173`.
  - SAM 2.1 Hiera Tiny source checkpoint:
    `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`;
    encoder ONNX `37de035410f11fb505c210ce1f8b10b450cb018fbfa7f226ec7e7ab542955c75`;
    decoder ONNX `3a6a96c62fd39c1369641ea626e5b2e4aed10de877f85291e9993a044d5bf981`.
  - ViTMatte Small checkpoint:
    `bda9289db1bb6762d978b42d1c62ae3f34daf7497171a347a1d09657efd788cb`;
    ONNX `a5c0b49a664867021eaf2b9cadcdc236ea511028d5e709303b418431f20e223b`.

## Result

| Measure | Result |
|---|---:|
| Whole-task wall time | 2,955.242 s (49.25 min) |
| Process-tree sampled peak RSS | 2,558,504,960 B (2.383 GiB) |
| Linux cgroup peak | 2,703,536,128 B (2.518 GiB) of 3 GiB |
| Cgroup OOM / limit events | 0 / 0 |
| M9 inference / LabFusion | 4.897 s / 3.213 s |
| Depth inference | 18.409 s |
| SAM encoder / decoder | 5.702 s / 0.111 s |
| ViTMatte inference | 6.862 s |
| ARNet passes | 157.911 / 40.347 / 11.873 / 4.344 s |
| Optical highlight scatter | 2,650.138 s (89.7% of task wall time) |
| Output | RGB PNG, 5760×3840, 16,655,520 B |
| Output SHA-256 | `0a6282df408319950f0bd4c8f56bcfce78ed9e7eba95ce9c50b2c70f5555ade2` |

The run reached progress 1.0 and exited normally. Full output resolution was
preserved. The task report records all model hashes, stage times, sampled
process-tree peak RSS, cgroup peak and options. Peak process-tree RSS occurred
during ARNet; the largest cgroup usage was 2,703,536,128 B. Model inference
completed on CPU and no GPU provider was enabled.

## Subject result comparison

The result was compared with the same-source, same-photo 5760×3840 no-subject
highlight output from the 512-row run in the [chunk comparison report](highlight-chunk-size-cgroup-2026-10-23.md).
The clock-face crop visibly preserves the selected dial detail while the
surrounding foreground remains blurred. The two full outputs are not identical:
mean absolute 8-bit channel error is 0.01364, the maximum channel difference is
78, and 0.2525% of pixels change. In the measured clock-tower region
`(x=3900..4699, y=2100..3399)`, 5.37% of pixels change (MAE 0.2901); in the
sampled mountain region `(x=800..2999, y=800..2199)`, no pixels change. The
SAM positive-point score was 0.9226. These measurements establish that the
subject path affected the selected local area in this image. They are not a
ground-truth alpha accuracy score or a broad hair/fine-edge quality acceptance.

The optical scatter time is the dominant CPU cost. It differs substantially
from previous full-resolution runs, so this single run does not prove a speedup
or regression. It provides a completed end-to-end subject-enabled memory
measurement and a new scatter timing for further controlled investigation.

No model weights, input photographs, output images or raw progress logs are in
Git. The output and complete task/progress logs are under
`/tmp/laika-pipeline-results/fullres-subject-3g/` on the development host.

## Reproduction

Mount this repository read-only at `/opt/laika-src`, use the hash-verified model
assets recorded above, and invoke `scripts/benchmark_processing_pipeline.py`
with `--mode color+bokeh --strategy low-memory --threads 1
--bokeh-row-chunk-height 128 --matte-tile-size 512 --subject-point 0.72,0.66`.
Do not provide an output-size override. Run in a Linux cgroup v2 with
`--memory=3g --memory-swap=3g --cpus=2` and retain the task report and JSONL
progress log outside the repository.
