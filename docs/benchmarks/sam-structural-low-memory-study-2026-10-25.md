# SAM encoder structural low-memory experiments — 2026-10-25

This research is intentionally NOT approved for production merge.
Branch: perf/sam-verified-two-graph-execution.

## Objective

Lower the complete five-model CPU job peak without changing photo
dimensions, SAM checkpoint values, mask-selection logic or output pixels.
Compare ONNX graph cuts and int8 selective quantization on Windows.

## Quantization experiments — not acceptable for reference-quality SAM

The source encoder is 109,471,976 bytes (SAM Hiera Tiny FP32).
All INT8 variants were generated LOCALLY from authorized original ONNX
weights. No derived checkpoint was committed to Git.
Single point on an actual 2048x1365 portrait; an independent fresh
process for each variant, with ONNX CPU memory arena disabled.

| Candidate | Encoder ONNX bytes | Standalone process RSS | Subject segmentation | Binary mask IoU vs FP32 |
|---|---:|---:|---:|---:|
| Reference FP32 | 109,471,976 | 896,348,160 B | 4.348s | 1.000 |
| Dynamic INT8 all 51 weighted MatMul | 29,438,778 | 836,157,440 B | 2.975s | 0.97566 |
| Dynamic INT8 MLP only (24 MatMul) | 54,617,561 | 857,886,720 B | 3.405s | 0.98081 |
| Dynamic INT8 MLP + projection (39 MatMul) | 46,592,255 | 888,733,696 B | 3.219s | 0.97782 |
| Dynamic INT8 QKV only (12 MatMul) | 92,505,228 | 928,153,600 B | 3.696s | 0.73619 |

Although some INT8 variants improve CPU speed, the mask differs
measurably from reference; QKV-only affects segmentation substantially.
Do not substitute any INT8 derived graph into the main model manifest
or Web API as a transparent optimization.

## Exact-weight graph partitioning

SAM Hiera Tiny's ONNX has 2,720 nodes, including If subgraphs with
outer-scope tensor captures. The stock ONNX graph extractor fails
to preserve these implicit dependencies. The experimental exporter
uses contiguous ordered slices and gathers referenced initializers
recursively, including those captured by If subgraphs.

Two-way cut after trunk block 5 (node 1148) produces original-weight
prefix/suffix graphs with three bridge feature tensors. Execution
in sequence gives the SAME reference binary SAM mask SHA256:
55ef312d6aa67c1ba42c73f5034bc949d67cd5dfb8e0429c4a08d9deed37b257

| SAM encoder execution | Sampled standalone RSS | Subject segmentation |
|---|---:|---:|
| Original single ONNX session, arena disabled | 950,075,392 B | 4.019s |
| Two exact-weight sessions | 831,320,064 B | 4.965s |
| Three sessions | 836,739,072 B | 4.999s |
| Four sessions | 829,972,480 B | 4.997s |

Extra splitting beyond two sessions does not consistently lower RSS,
and adds session management overhead. Mask outputs remain identical
on this one test photo; decoder score differences are below 1e-6.
No general numerical equivalence guarantee is made.

## Full five-model real-photo test

With M9 + Depth + SAM + ViTMatte + optional ARNet global-resize/tiled
convolution + optical bokeh enabled, output 2048x1365:

- Original SAM with CPU arena disabled: ~828.2MB sampled peak in
  preceding #77 experiments, around 71s there.
- Initial two-graph full run: 793,227,264 B sampled peak and 63.09s;
  output PNG exactly matches the previous no-arena original SAM photo.
- Reverse-order comparison: split FIRST 809,971,712 B and 63.62s;
  original SAM SECOND 828,801,024 B and 61.03s. Both PNG outputs exact.

Extra whole-worker memory saving on the direct reverse pair was only
18,829,312 B (~2.3%), with ~2.6s extra runtime. It is substantially
less than the isolated encoder results and does not yet justify
the additional exporter code, dynamic sessions and test surface
in a minimum-config production deployment.

## Decision

Keep the safely verified SAM no-arena optimization from merged PR #77.
Keep two-part SAM exporter and runtime as an unmerged experimental
feature branch; do NOT enable by default and do NOT distribute models.
No native 20MP strict 2GiB Linux cgroup proof was run. Future work
should first target active attention workspace and early model feature
liveness, using multiple real photographs and strict memory controls.

All local derived ONNX graph and mask data remain exclusively under
E:\codex\LeicaShot-local-data\laika-models\experiments and validation.
