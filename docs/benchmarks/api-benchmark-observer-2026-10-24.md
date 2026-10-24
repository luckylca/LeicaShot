# API benchmark observer overhead

Date: 2026-10-24

## Measurement

Compared the previous process-tree sampler (RSS, USS and CPU every 10 ms) with the current sampler (RSS every 100 ms and USS every 1 s). Each run observed the same separate CPU-spinning Python child holding a touched 256 MiB byte array. The benchmark parent process CPU time includes the sampling thread and the waiting harness; it excludes the child process CPU.

| Sampler | Wall time | Observer parent CPU | RSS scans | USS scans |
|---|---:|---:|---:|---:|
| Previous: 10 ms RSS + USS | 20.003 s | 2.06 s | 1,396 | 1,396 |
| Current: 100 ms RSS + 1 s USS | 20.005 s | 0.09 s | 193 | 20 |

The measured observer CPU fell by 95.6% (2.06 s to 0.09 s) for this synthetic target. Target RSS was 273,842,176 B. Environment: Linux ARM64, Python 3.11.17, 2 logical CPUs, 4 GiB Lima guest.

Reproduce from the repository root in an environment with `psutil` installed:

```bash
PYTHONPATH=src:scripts python scripts/benchmark_observer_overhead.py \
  --duration-seconds 20 --target-memory-mib 256
```

This isolates the observer cost; it does not predict API task latency or image-processing speed. The 100 ms RSS and 1 s USS intervals can miss short-lived process peaks. Use Linux cgroup `memory.peak` as the authoritative whole-task memory high-water mark when available, and interpret sampled RSS/USS as diagnostics.

## Implementation

- `ServerProcessPeak` defaults to 100 ms RSS sampling and 1 s USS scans, with both intervals exposed as CLI options.
- Process-tree server CPU is measured at the start and end of the request window rather than on every memory sample.
- The API benchmark reports benchmark-client CPU and cgroup v2 CPU usage deltas so monitoring and service costs can be separated where the cgroup counters are available.
- A test verifies that root and child CPU times are summed and another checks the lower USS scan frequency.

The next validation is a same-host API/direct-engine full-task comparison with identical output size, model settings, Bokeh row chunk, worker count and memory limit. No end-to-end speed improvement is claimed from this microbenchmark.
