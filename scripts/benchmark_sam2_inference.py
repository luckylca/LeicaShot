#!/usr/bin/env python3
"""Measure SAM 2.1 Tiny cold worker and repeated-prompt CPU inference."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import psutil  # noqa: E402

from laika.benchmark import (  # noqa: E402
    _cgroup_v2_snapshot,
    _process_tree_metrics,
)


def _worker(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import onnxruntime as ort

    from laika.subject import (
        Sam2TinyOnnx,
        normalized_points_to_sam,
        prepare_sam_input,
        select_subject_mask,
    )

    report_path = args.validation_report.resolve()
    validation = json.loads(report_path.read_text(encoding="utf-8"))
    if validation.get("status") != "passed":
        raise ValueError("SAM ONNX graphs are not marked as validated")
    encoder_path = args.image_encoder.resolve()
    decoder_path = args.image_decoder.resolve()
    if validation.get("image_encoder_sha256") != _sha256(encoder_path):
        raise ValueError("image encoder does not match its validation report")
    if validation.get("image_decoder_sha256") != _sha256(decoder_path):
        raise ValueError("image decoder does not match its validation report")

    rng = np.random.default_rng(args.seed)
    rgb = rng.integers(0, 256, (1536, 2048, 3), dtype=np.uint8)
    preprocess_started = time.perf_counter()
    image = prepare_sam_input(rgb)
    preprocessing_seconds = time.perf_counter() - preprocess_started
    normalized_coords = np.asarray(
        [[0.3125, 0.41015625], [0.595703125, 0.37109375]], dtype=np.float32
    )
    coords = normalized_points_to_sam(normalized_coords)
    labels = np.asarray([[1, 0]], dtype=np.int32)
    synthetic_logits = np.full((1, 3, 256, 256), -1.0, dtype=np.float32)
    yy, xx = np.ogrid[:256, :256]
    synthetic_logits[0, 1, (xx - 128) ** 2 + (yy - 128) ** 2 <= 48**2] = 1.0
    synthetic_scores = np.asarray([[0.7, 0.9, 0.8]], dtype=np.float32)

    with Sam2TinyOnnx(
        encoder_path,
        decoder_path,
        cache_max_bytes=args.cache_max_mib * 1024 * 1024,
        intra_op_num_threads=args.threads,
    ) as engine:
        cold_started = time.perf_counter()
        cold = engine.predict(image, coords, labels)
        cold_total_seconds = time.perf_counter() - cold_started
        warm_runs: list[dict[str, float]] = []
        for run in range(args.runs):
            prompt = coords.copy()
            prompt[0, 0, 0] = min(1023, prompt[0, 0, 0] + run + 1)
            started = time.perf_counter()
            prediction = engine.predict(image, prompt, labels)
            wall_seconds = time.perf_counter() - started
            if not prediction.cache_hit:
                raise RuntimeError(
                    "repeated prompt unexpectedly reran the image encoder"
                )
            warm_runs.append(
                {
                    "decoder_seconds": prediction.decoder_seconds,
                    "wall_seconds": wall_seconds,
                }
            )
        cache_bytes = engine.cache.used_bytes
        cache_hits = engine.cache.hits
        if cold.cache_hit or cold.mask_logits.shape != (1, 3, 256, 256):
            raise RuntimeError("cold SAM inference returned an invalid result")

    mask_started = time.perf_counter()
    selected_mask = select_subject_mask(
        synthetic_logits,
        synthetic_scores,
        [0.5, 0.5],
        width=2048,
        height=1536,
    )
    mask_selection_seconds = time.perf_counter() - mask_started

    return {
        "model_id": validation["model_id"],
        "checkpoint_sha256": validation["checkpoint_sha256"],
        "image_encoder_sha256": validation["image_encoder_sha256"],
        "image_decoder_sha256": validation["image_decoder_sha256"],
        "runtime_versions": {
            "onnxruntime": ort.__version__,
            "numpy": np.__version__,
        },
        "provider": "CPUExecutionProvider",
        "threads": args.threads,
        "seed": args.seed,
        "source_rgb_shape": list(rgb.shape),
        "input_shape": list(image.shape),
        "preprocessing_seconds": preprocessing_seconds,
        "point_count": int(coords.shape[1]),
        "mask_shape": list(cold.mask_logits.shape),
        "synthetic_mask_selection_seconds": mask_selection_seconds,
        "synthetic_selected_candidate_index": selected_mask.candidate_index,
        "synthetic_selected_mask_shape": list(selected_mask.mask.shape),
        "cold_encoder_seconds": cold.encoder_seconds,
        "cold_decoder_seconds": cold.decoder_seconds,
        "cold_predict_wall_seconds": cold_total_seconds,
        "warm_prompt_runs": warm_runs,
        "cache_max_bytes": args.cache_max_mib * 1024 * 1024,
        "cache_used_bytes": cache_bytes,
        "cache_hits": cache_hits,
        "cache_entries_after_close": engine.cache.entry_count,
        "quality_status": "synthetic-image-only",
    }


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _measure(args: argparse.Namespace) -> dict[str, Any]:
    if min(args.threads, args.runs, args.sample_interval_ms, args.cache_max_mib) < 1:
        raise ValueError(
            "threads, runs, cache size, and sample interval must be positive"
        )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--image-encoder",
        str(args.image_encoder.resolve()),
        "--image-decoder",
        str(args.image_decoder.resolve()),
        "--validation-report",
        str(args.validation_report.resolve()),
        "--threads",
        str(args.threads),
        "--runs",
        str(args.runs),
        "--cache-max-mib",
        str(args.cache_max_mib),
        "--seed",
        str(args.seed),
    ]
    cgroup_before = _cgroup_v2_snapshot()
    host_before = psutil.virtual_memory()
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    started = time.perf_counter()
    interval = args.sample_interval_ms / 1000
    peak_rss = peak_uss = peak_cpu = process_count = samples = 0
    uss_observed = False
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
        raise RuntimeError(stderr.strip() or f"SAM worker exited {process.returncode}")
    try:
        benchmark = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("SAM worker returned invalid JSON") from exc

    return {
        "measurement": "fresh CPU worker process-tree sampling",
        "sample_interval_ms": args.sample_interval_ms,
        "samples": samples,
        "process_tree_peak_rss_bytes": peak_rss,
        "process_tree_peak_uss_bytes": peak_uss if uss_observed else None,
        "sampled_process_tree_cpu_seconds": peak_cpu,
        "peak_process_count": process_count,
        "worker_wall_seconds": wall_seconds,
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "python": platform.python_version(),
            "available_memory_before_bytes": host_before.available,
            "available_memory_after_bytes": psutil.virtual_memory().available,
        },
        "cgroup_v2_before": cgroup_before,
        "cgroup_v2_after": _cgroup_v2_snapshot(),
        "benchmark": benchmark,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-encoder", type=Path, required=True)
    parser.add_argument("--image-decoder", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--cache-max-mib", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--sample-interval-ms", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        report = _worker(args) if args.worker else _measure(args)
        if args.output is not None:
            args.output.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
