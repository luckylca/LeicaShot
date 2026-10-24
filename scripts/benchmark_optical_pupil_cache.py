"""Model-free A/B benchmark for the signed-radius optical pupil cache.

Run:
    python scripts/benchmark_optical_pupil_cache.py --repeat 2

This measures *optical scatter only*, not M9/Depth/SAM/ViTMatte/ARNet or
complete API latency. Full-flow validation requires actual photographs.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np

from laika.highlights import scatter_highlights


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--height", type=int, default=80)
    parser.add_argument("--width", type=int, default=112)
    args = parser.parse_args()
    if not 1 <= args.repeat <= 5:
        parser.error("repeat must be in [1, 5]")
    if not 16 <= args.height <= 512 or not 16 <= args.width <= 512:
        parser.error("image dimensions must be in [16, 512]")

    high = np.zeros((args.height, args.width, 3), dtype=np.float32)
    high[args.height // 2, args.width // 2] = (0.9, 0.5, 0.3)
    high[args.height // 4, args.width // 4] = (0.5, 0.7, 0.2)
    high[-2, -2] = (0.3, 0.1, 0.7)
    signed_levels = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], dtype=np.float32)
    indices = np.arange(args.height, dtype=np.intp) % len(signed_levels)
    coc = np.ascontiguousarray(
        np.broadcast_to(signed_levels[indices, None], (args.height, args.width))
    )

    observations: dict[str, list[float]] = {"uncached": [], "cached": []}
    reference: np.ndarray | None = None
    for repeat in range(args.repeat):
        modes = ("uncached", "cached") if repeat % 2 == 0 else ("cached", "uncached")
        for mode in modes:
            started = time.perf_counter()
            result = scatter_highlights(
                high,
                coc,
                aperture=2.8,
                radius_scale=1.0,
                field_optics=True,
                row_chunk_height=24,
                _pupil_cache_budget_bytes=0 if mode == "uncached" else 32 * 1024 * 1024,
            )
            elapsed = time.perf_counter() - started
            if reference is None:
                reference = result.copy()
            if not np.array_equal(result, reference):
                raise AssertionError(f"optical pupil cache changed pixels in {mode} trial")
            observations[mode].append(round(elapsed, 6))
            print(json.dumps({"trial": repeat, "mode": mode, "seconds": elapsed}), flush=True)

    medians = {mode: statistics.median(values) for mode, values in observations.items()}
    print(
        json.dumps(
            {
                "kind": "synthetic_scatter_only",
                "dimensions": [args.width, args.height],
                "medians": medians,
                "uncached_over_cached_speedup": medians["uncached"] / medians["cached"],
                "float_output_exact": True,
                "full_pipeline_benchmark": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
