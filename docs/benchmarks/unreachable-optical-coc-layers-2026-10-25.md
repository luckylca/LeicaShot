# Unreachable optical depth-layer removal (experimental)

Baseline: develop at 2c30760, containing merged native 20MP CoC
interval pruning from PR #80.

The geometry prescan on the real 5472x3648 photograph found 169
source CoC layers but only 55 eligible layers anywhere in the
existing sparse highlight output-support footprints. This experiment
precomputes a conservative set of globally eligible layers and skips
pupil generation plus all tile loops for each of the other 114 layers.
Progress callbacks and cancellation checks remain one-per-layer.

Pixel parity:
- Real 2048x1365 scatter floating-array SHA256 unchanged:
  a477000897bd26607ffb298f9afdb19d7daef794c49bf4cef72a6ced9bc4821e.
- Real 5472x3648 scatter floating-array SHA256 unchanged:
  c0fe387fd6efbd60ed7836d0f44519c28fdf6802401342a2466ebee57c3465ed.
- Targeted regression checks eliminated radius-12 pupil generation for
  an out-of-range highlighted scene while output and progress remained exact.

Matched-order 2048 scatter A/B (seconds):
- develop: 3.6761, 3.5488; median 3.6125.
- experimental early skip: 2.9379, 2.9913; median 2.9646.
- Observed ~17.9% faster optical scatter at this small native size.

Unpaired native 20MP observations (seconds):
- merged interval baseline: 75.8 in one run, 81.7 in instrumented profiling.
- additional global layer skip: 77.2 in one run.
- NO reliable 20MP speed improvement established yet.
  Full five-model pipeline speedup has NOT been measured for this experiment.

The code may be a useful optional exact-output fast path in other
depth/lighting conditions. It is intentionally NOT in develop until
repeatable native 20MP whole-pipeline benefits are demonstrated.
Model weights and image files remain outside Git.
