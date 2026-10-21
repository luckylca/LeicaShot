#!/usr/bin/env python3
"""Compare pinned Depth Anything V2 PyTorch and ONNX CPU runtimes."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import psutil  # noqa: E402

from laika.benchmark import _cgroup_v2_snapshot, _process_tree_metrics  # noqa: E402
from laika.model_assets import ModelAssetError, load_manifest, verify_artifact  # noqa: E402


def _worker(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np

    input_array = np.random.default_rng(args.seed).standard_normal(
        (1, 3, args.height, args.width), dtype=np.float32
    )
    load_started = time.perf_counter()
    provenance: dict[str, Any] = {"model_id": "depth-anything-v2-small"}
    if args.runtime == "pytorch":
        if args.source_checkout is None or args.weights is None:
            raise ValueError("PyTorch benchmark requires source checkout and weights")
        try:
            import torch
        except ImportError as exc:
            raise ValueError("PyTorch is not installed in this environment") from exc

        from export_depth_anything_v2 import _verify_source_revision

        manifest = json.loads(
            (ROOT / "models/manifest.json").read_text(encoding="utf-8")
        )
        record = next(
            row for row in manifest["models"] if row["id"] == "depth-anything-v2-small"
        )
        artifact = load_manifest()["depth-anything-v2-small"]
        source = args.source_checkout.resolve()
        _verify_source_revision(source, record["source_code_revision"])
        verify_artifact(args.weights, artifact)
        provenance.update(
            {
                "source_code_revision": record["source_code_revision"],
                "weights_sha256": artifact.sha256,
            }
        )
        sys.path.insert(0, str(source))
        from depth_anything_v2.dpt import DepthAnythingV2

        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        model = DepthAnythingV2(
            encoder="vits", features=64, out_channels=[48, 96, 192, 384]
        )
        checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint, strict=True)
        del checkpoint
        model.eval()
        load_seconds = time.perf_counter() - load_started
        tensor = torch.from_numpy(input_array)
        with torch.inference_mode():
            model(tensor)
            elapsed = []
            for _ in range(args.runs):
                started = time.perf_counter()
                output = model(tensor).cpu().numpy()
                elapsed.append(time.perf_counter() - started)
        runtime_versions = {"torch": torch.__version__}
    else:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ValueError(
                "ONNX Runtime is not installed in this environment"
            ) from exc

        if args.onnx is None:
            raise ValueError("ONNX Runtime benchmark requires an ONNX path")
        if not args.onnx.is_file():
            raise ValueError(f"ONNX model is unavailable: {args.onnx}")
        from export_depth_anything_v2 import _sha256

        provenance["onnx_sha256"] = _sha256(args.onnx)
        external_data = Path(f"{args.onnx}.data")
        if external_data.is_file():
            provenance["external_data_sha256"] = _sha256(external_data)
        options = ort.SessionOptions()
        options.intra_op_num_threads = args.threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            str(args.onnx), sess_options=options, providers=["CPUExecutionProvider"]
        )
        load_seconds = time.perf_counter() - load_started
        session.run(["depth"], {"image": input_array})
        elapsed = []
        for _ in range(args.runs):
            started = time.perf_counter()
            output = session.run(["depth"], {"image": input_array})[0]
            elapsed.append(time.perf_counter() - started)
        runtime_versions = {"onnxruntime": ort.__version__}

    if not np.isfinite(output).all():
        raise RuntimeError("inference returned non-finite output")
    return {
        "runtime": args.runtime,
        "seed": args.seed,
        "provenance": provenance,
        "runtime_versions": runtime_versions,
        "provider": "CPU",
        "threads": args.threads,
        "fixed_input_shape": list(input_array.shape),
        "output_shape": list(output.shape),
        "output_min": float(output.min()),
        "output_max": float(output.max()),
        "output_mean": float(output.mean()),
        "model_or_session_load_seconds": load_seconds,
        "warmup_runs": 1,
        "measured_runs_seconds": elapsed,
        "median_inference_seconds": median(elapsed),
        "min_inference_seconds": min(elapsed),
        "max_inference_seconds": max(elapsed),
    }


def _measure(args: argparse.Namespace) -> dict[str, Any]:
    if args.height < 518 or args.width < 518:
        raise ValueError("network input dimensions must be at least 518 pixels")
    if args.height % 14 or args.width % 14:
        raise ValueError("network input dimensions must be multiples of 14")
    if args.threads < 1 or args.runs < 1 or args.sample_interval_ms < 1:
        raise ValueError("threads, runs, and sample interval must be positive")
    if args.worker:
        return _worker(args)

    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
    cgroup_before = _cgroup_v2_snapshot()
    host_before = psutil.virtual_memory()
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    started = time.perf_counter()
    peak_rss = peak_uss = peak_cpu = 0
    uss_observed = False
    peak_process_count = samples = 0
    interval = args.sample_interval_ms / 1000
    while process.poll() is None:
        rss, uss, cpu, process_count = _process_tree_metrics(process.pid)
        peak_rss = max(peak_rss, rss)
        peak_cpu = max(peak_cpu, cpu)
        peak_process_count = max(peak_process_count, process_count)
        if uss is not None:
            uss_observed = True
            peak_uss = max(peak_uss, uss)
        samples += 1
        time.sleep(interval)
    stdout, stderr = process.communicate()
    wall_seconds = time.perf_counter() - started
    if process.returncode:
        raise RuntimeError(stderr.strip() or f"worker exited {process.returncode}")
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("benchmark worker did not return JSON") from exc
    result.update(
        {
            "measurement": "worker process-tree sampling",
            "sample_interval_ms": args.sample_interval_ms,
            "samples": samples,
            "peak_process_count": peak_process_count,
            "process_tree_peak_rss_bytes": peak_rss,
            "process_tree_peak_uss_bytes": peak_uss if uss_observed else None,
            "sampled_process_tree_cpu_seconds": peak_cpu,
            "worker_wall_seconds": wall_seconds,
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
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=("pytorch", "onnx"), required=True)
    parser.add_argument("--source-checkout", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--height", type=int, default=518)
    parser.add_argument("--width", type=int, default=518)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--sample-interval-ms", type=int, default=5)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = _measure(args)
    except (ModelAssetError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
