"""Measure complete Laika CPU throughput vs memory with identical real inputs.

Run from an installed Laika inference environment, for example:
    python scripts/benchmark_speed_matrix.py \
        --input photo.jpg --m9-model m9.onnx --depth-model depth.onnx \
        --bokeh-model arnet.onnx --sam-model-dir sam2-dir \
        --vitmatte-model vitmatte.onnx --subject-point 0.5,0.5 \
        --threads 1,2,4 --repeat 2 --output-dir local-bench

The output directory contains rendered PNGs and a JSON report. Model weights,
test photographs and output PNGs must not be committed or redistributed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil
from PIL import Image


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in (
        "input",
        "m9-model",
        "depth-model",
        "bokeh-model",
        "sam-model-dir",
        "vitmatte-model",
        "output-dir",
    ):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--arnet-split-dir", type=Path)
    parser.add_argument("--subject-point", action="append", default=[])
    parser.add_argument(
        "--matte-tile-size", type=int, choices=(512, 768, 1024), default=512
    )
    parser.add_argument("--threads", default="1,2,4")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--memory-budget-mib", type=int, default=2048)
    parser.add_argument("--max-pixels", type=int, default=40_000_000)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--sample-seconds", type=float, default=0.05)
    parser.add_argument("--keep-pngs", action="store_true")
    return parser


def _parse_threads(text: str) -> list[int]:
    try:
        values = [int(value.strip()) for value in text.split(",")]
    except ValueError as exc:
        raise ValueError("threads must be comma-separated positive integers") from exc
    if (
        not values
        or len(set(values)) != len(values)
        or any(value < 1 or value > 32 for value in values)
    ):
        raise ValueError("threads must be unique integers in [1, 32]")
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode_result(stdout: str) -> dict[str, Any]:
    # The CLI outputs the ProcessingResult JSON as its final stdout line.
    for line in reversed(stdout.splitlines()):
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and "stage_seconds" in item:
            return item
    raise RuntimeError("Laika did not emit a ProcessingResult JSON object")


def _rss_tree(process: psutil.Process) -> int:
    try:
        descendants = process.children(recursive=True)
        return sum(
            proc.memory_info().rss
            for proc in [process, *descendants]
            if proc.is_running()
        )
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0


def _launch(
    argv: list[str],
    environment: dict[str, str],
    *,
    timeout: float,
    interval: float,
) -> tuple[float, int, dict[str, Any]]:
    start = time.perf_counter()
    child = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
    )
    tree = psutil.Process(child.pid)
    peak = 0
    try:
        while child.poll() is None:
            peak = max(peak, _rss_tree(tree))
            if time.perf_counter() - start > timeout:
                child.kill()
                raise TimeoutError(f"Laika processing exceeded {timeout}s")
            time.sleep(interval)
        peak = max(peak, _rss_tree(tree))
        stdout, stderr = child.communicate(timeout=10)
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate()
    elapsed = time.perf_counter() - start
    if child.returncode != 0:
        raise RuntimeError(f"Laika exited {child.returncode}: {stderr[-1600:]}")
    return elapsed, peak, _decode_result(stdout)


def _image_difference(baseline: Path, candidate: Path) -> dict[str, Any]:
    import numpy as np

    with Image.open(baseline) as a, Image.open(candidate) as b:
        if a.size != b.size:
            raise ValueError(f"Output dimensions differ: {a.size}, {b.size}")
        lhs = np.asarray(a.convert("RGB"), dtype=np.int16)
        rhs = np.asarray(b.convert("RGB"), dtype=np.int16)
    delta = np.abs(lhs - rhs)
    return {
        "max_8bit_channel_diff": int(delta.max()),
        "mean_8bit_channel_diff": float(delta.mean()),
        "changed_pixel_fraction": float(np.any(delta > 0, axis=2).mean()),
    }


def main() -> int:
    args = make_parser().parse_args()
    threads = _parse_threads(args.threads)
    if not 1 <= args.repeat <= 5:
        raise ValueError("repeat must be in [1, 5]")
    if args.memory_budget_mib <= 0 or args.timeout_seconds <= 0:
        raise ValueError("memory budget and timeout must be positive")
    if not 0.01 <= args.sample_seconds <= 1.0:
        raise ValueError("sample interval must be between 0.01 and 1 second")
    for name in (
        "input",
        "m9_model",
        "depth_model",
        "bokeh_model",
        "sam_model_dir",
        "vitmatte_model",
    ):
        path = getattr(args, name)
        if not path.exists():
            raise FileNotFoundError(f"{name} missing: {path}")
    if args.arnet_split_dir and not args.arnet_split_dir.is_dir():
        raise FileNotFoundError(args.arnet_split_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["LAIKA_SAM_DISABLE_CPU_ARENA"] = "1"
    if args.arnet_split_dir:
        env["LAIKA_ARNET_SPLIT_DIR"] = str(args.arnet_split_dir.resolve())
    else:
        env.pop("LAIKA_ARNET_SPLIT_DIR", None)
    # A fresh process per trial prevents previously loaded ONNX sessions
    # and native allocators from contaminating the next measurement.
    records: list[dict[str, Any]] = []
    reference: Path | None = None
    for repeat in range(args.repeat):
        order = threads if repeat % 2 == 0 else list(reversed(threads))
        for count in order:
            output = args.output_dir / f"threads-{count}-trial-{repeat}.png"
            cmd = [
                sys.executable,
                "-m",
                "laika.cli",
                "pipeline",
                str(args.input),
                str(output),
                "--mode",
                "color+bokeh",
                "--strategy",
                "low-memory",
                "--threads",
                str(count),
                "--m9-model",
                str(args.m9_model),
                "--depth-model",
                str(args.depth_model),
                "--bokeh-model",
                str(args.bokeh_model),
                "--sam-model-dir",
                str(args.sam_model_dir),
                "--vitmatte-model",
                str(args.vitmatte_model),
                "--matte-tile-size",
                str(args.matte_tile_size),
                "--max-pixels",
                str(args.max_pixels),
            ]
            for point in args.subject_point or ["0.5,0.5"]:
                cmd.extend(("--subject-point", point))
            elapsed, peak, result = _launch(
                cmd,
                env,
                timeout=args.timeout_seconds,
                interval=args.sample_seconds,
            )
            if reference is None:
                reference = output
            metrics = _image_difference(reference, output)
            item = {
                "repeat": repeat,
                "threads": count,
                "wall_seconds": round(elapsed, 4),
                "worker_total_seconds": result["total_seconds"],
                "peak_process_tree_rss": peak,
                "peak_mib": round(peak / (1024 * 1024), 2),
                "stage_seconds": result["stage_seconds"],
                "output_sha256": _sha256(output),
                **metrics,
            }
            print(json.dumps(item, ensure_ascii=False), flush=True)
            records.append(item)
    budget_bytes = args.memory_budget_mib * 1024 * 1024
    ranking = []
    for count in threads:
        trials = [item for item in records if item["threads"] == count]
        ranking.append(
            {
                "threads": count,
                "median_wall_seconds": statistics.median(
                    trial["wall_seconds"] for trial in trials
                ),
                "max_observed_rss": max(
                    trial["peak_process_tree_rss"] for trial in trials
                ),
                "within_sampled_budget": all(
                    trial["peak_process_tree_rss"] <= budget_bytes for trial in trials
                ),
                "max_channel_difference": max(
                    trial["max_8bit_channel_diff"] for trial in trials
                ),
            }
        )
    ranking.sort(key=lambda entry: entry["median_wall_seconds"])
    report = {
        "description": (
            "One input photo; sampled RSS is not Linux cgroup peak. "
            "Thread-count speed and pixel fidelity are environment-specific."
        ),
        "input_path": str(args.input),
        "input_sha256": _sha256(args.input),
        "model_paths": {
            "m9": str(args.m9_model),
            "depth": str(args.depth_model),
            "arnet": str(args.bokeh_model),
            "sam": str(args.sam_model_dir),
            "vitmatte": str(args.vitmatte_model),
        },
        "configuration": {
            "threads": threads,
            "repeats": args.repeat,
            "sam_no_arena": True,
            "arnet_split_dir": str(args.arnet_split_dir)
            if args.arnet_split_dir
            else None,
            "memory_budget_mib": args.memory_budget_mib,
        },
        "ranked_by_median_wall_time": ranking,
        "trials": records,
    }
    (args.output_dir / "speed-matrix.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("REPORT", args.output_dir / "speed-matrix.json")
    if not args.keep_pngs:
        # Keep the reference image for investigation, but avoid retaining
        # repeated high-resolution outputs by default.
        for item in args.output_dir.glob("threads-*-trial-*.png"):
            if item != reference:
                item.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
