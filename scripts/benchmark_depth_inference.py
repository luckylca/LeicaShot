#!/usr/bin/env python3
"""Measure complete image-to-depth CLI RSS and timing in a fresh worker."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import psutil  # noqa: E402
from PIL import Image  # noqa: E402

from laika.benchmark import _cgroup_v2_snapshot, _process_tree_metrics  # noqa: E402
from laika.depth import _sha256  # noqa: E402


def _measure(args: argparse.Namespace) -> dict[str, Any]:
    if min(args.width, args.height, args.threads, args.sample_interval_ms) < 1:
        raise ValueError("width, height, threads, and sample interval must be positive")
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        raise ValueError(
            "benchmark requires the optional inference dependencies"
        ) from exc

    model = args.model.resolve()
    report_path = Path(f"{model}.validation.json")
    validation = json.loads(report_path.read_text(encoding="utf-8"))
    if validation.get("onnx_sha256") != _sha256(model):
        raise ValueError("model does not match its validation report")
    rng = np.random.default_rng(args.seed)
    fixture = rng.integers(0, 256, (args.height, args.width, 3), dtype=np.uint8)
    cgroup_before = _cgroup_v2_snapshot()
    host_before = psutil.virtual_memory()

    with tempfile.TemporaryDirectory(prefix="laika-depth-benchmark-") as directory:
        folder = Path(directory)
        input_path = folder / "seeded-rgb.png"
        output_path = folder / "depth.npy"
        Image.fromarray(fixture, mode="RGB").save(input_path, format="PNG")
        command = [
            sys.executable,
            "-m",
            "laika.cli",
            "depth",
            str(input_path),
            str(output_path),
            "--model",
            str(model),
            "--input-size",
            str(args.input_size),
            "--threads",
            str(args.threads),
        ]
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        started = time.perf_counter()
        interval = args.sample_interval_ms / 1000
        peak_rss = peak_uss = peak_cpu = 0
        uss_observed = False
        process_count = samples = 0
        while process.poll() is None:
            rss, uss, cpu, count = _process_tree_metrics(process.pid)
            peak_rss = max(peak_rss, rss)
            peak_cpu = max(peak_cpu, cpu)
            process_count = max(process_count, count)
            if uss is not None:
                uss_observed = True
                peak_uss = max(peak_uss, uss)
            samples += 1
            time.sleep(interval)
        stdout, stderr = process.communicate()
        wall_seconds = time.perf_counter() - started
        if process.returncode:
            raise RuntimeError(
                stderr.strip() or f"depth worker exited {process.returncode}"
            )
        try:
            engine_result = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("depth worker returned invalid JSON") from exc
        depth = np.load(output_path, allow_pickle=False)
        if depth.shape != (args.height, args.width) or depth.dtype != np.float32:
            raise RuntimeError("depth worker wrote an unexpected output array")
        if not np.isfinite(depth).all():
            raise RuntimeError("depth worker wrote non-finite values")
        output_bytes = output_path.stat().st_size

    return {
        "measurement": "fresh CLI worker process-tree sampling",
        "model_id": validation["model_id"],
        "model_sha256": validation["onnx_sha256"],
        "model_attention_query_chunk_size": validation.get(
            "attention_query_chunk_size"
        ),
        "runtime_versions": {
            "onnxruntime": ort.__version__,
            "numpy": np.__version__,
        },
        "provider": "CPUExecutionProvider",
        "threads": args.threads,
        "seed": args.seed,
        "input_shape": [args.height, args.width, 3],
        "network_shape": [
            engine_result["network_height"],
            engine_result["network_width"],
        ],
        "output_shape": [args.height, args.width],
        "output_dtype": engine_result["depth_dtype"],
        "output_bytes": output_bytes,
        "stage_seconds": engine_result["stage_seconds"],
        "engine_total_seconds": engine_result["total_seconds"],
        "worker_wall_seconds": wall_seconds,
        "process_tree_peak_rss_bytes": peak_rss,
        "process_tree_peak_uss_bytes": peak_uss if uss_observed else None,
        "sampled_process_tree_cpu_seconds": peak_cpu,
        "peak_process_count": process_count,
        "samples": samples,
        "sample_interval_ms": args.sample_interval_ms,
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "python": platform.python_version(),
            "available_memory_before_bytes": host_before.available,
            "available_memory_after_bytes": psutil.virtual_memory().available,
        },
        "cgroup_v2_before": cgroup_before,
        "cgroup_v2_after": _cgroup_v2_snapshot(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=518)
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--sample-interval-ms", type=int, default=5)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        report = _measure(args)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
