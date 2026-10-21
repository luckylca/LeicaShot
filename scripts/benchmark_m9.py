#!/usr/bin/env python3
"""Run real M9+Lab profiles in fresh workers and sample their process-tree RSS."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sysctl_value(name: str) -> str | None:
    try:
        result = subprocess.run(
            ["sysctl", "-n", name],
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError:
        return None
    return result.stdout.strip() or None


def _sample_tree(pid: int) -> tuple[int, int | None, int]:
    import psutil

    try:
        root = psutil.Process(pid)
        processes = [root, *root.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0, None, 0
    rss = uss = count = 0
    uss_available = True
    for process in processes:
        try:
            rss += process.memory_info().rss
            count += 1
            uss += process.memory_full_info().uss
        except (psutil.AccessDenied, psutil.NoSuchProcess, OSError, RuntimeError):
            uss_available = False
    return rss, uss if uss_available and count else None, count


def _run_worker(
    *,
    source: Path,
    model: Path,
    destination: Path,
    strategy: str,
    threads: int,
    row_chunk_height: int,
    sample_interval_ms: int,
) -> dict[str, Any]:
    import psutil

    from laika.benchmark import _cgroup_v2_snapshot

    command = [
        sys.executable,
        "-m",
        "laika.cli",
        "color",
        str(source),
        str(destination),
        "--model",
        str(model),
        "--strategy",
        strategy,
        "--threads",
        str(threads),
        "--row-chunk-height",
        str(row_chunk_height),
    ]
    cgroup_before = _cgroup_v2_snapshot()
    available_before = psutil.virtual_memory().available
    worker = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    started = time.perf_counter()
    peak_rss = peak_uss = process_count = samples = 0
    uss_available = False
    interval = sample_interval_ms / 1000
    while True:
        rss, uss, count = _sample_tree(worker.pid)
        peak_rss = max(peak_rss, rss)
        process_count = max(process_count, count)
        if uss is not None:
            peak_uss = max(peak_uss, uss)
            uss_available = True
        samples += 1
        if worker.poll() is not None:
            break
        time.sleep(interval)
    stdout, stderr = worker.communicate()
    wall_seconds = time.perf_counter() - started
    if worker.returncode:
        raise RuntimeError(stderr.strip() or f"worker exited with {worker.returncode}")
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("M9 worker returned invalid JSON") from exc
    return {
        "strategy": strategy,
        "threads": threads,
        "row_chunk_height": row_chunk_height,
        "sample_interval_ms": sample_interval_ms,
        "samples": samples,
        "peak_process_count": process_count,
        "process_tree_peak_rss_bytes": peak_rss,
        "process_tree_peak_uss_bytes": peak_uss if uss_available else None,
        "wall_seconds": wall_seconds,
        "host_available_memory_before_bytes": available_before,
        "cgroup_v2_before": cgroup_before,
        "cgroup_v2_after": _cgroup_v2_snapshot(),
        "worker": result,
        "output": str(destination),
    }


def _compare_pngs(left_path: Path, right_path: Path) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    maximum = 0
    total = 0
    different = 0
    samples = 0
    with Image.open(left_path) as left, Image.open(right_path) as right:
        if left.size != right.size:
            raise ValueError("profile outputs have different dimensions")
        left_rgb = left.convert("RGB")
        right_rgb = right.convert("RGB")
        try:
            width, height = left_rgb.size
            for y in range(0, height, 128):
                rows = min(128, height - y)
                a = np.asarray(left_rgb.crop((0, y, width, y + rows)), dtype=np.int16)
                b = np.asarray(right_rgb.crop((0, y, width, y + rows)), dtype=np.int16)
                delta = np.abs(a - b)
                maximum = max(maximum, int(delta.max(initial=0)))
                total += int(delta.sum(dtype=np.int64))
                different += int(np.count_nonzero(delta))
                samples += int(delta.size)
                del a, b, delta
        finally:
            left_rgb.close()
            right_rgb.close()
    return {
        "max_absolute_pixel_error": maximum,
        "mean_absolute_pixel_error": total / samples if samples else 0.0,
        "pixels_differing": different,
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    import psutil
    from PIL import Image, ImageOps
    from PIL import __version__ as pillow_version

    from laika.m9 import M9_MODEL_SHA256

    source = Path(args.input)
    model = Path(args.model)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if _sha256(model) != M9_MODEL_SHA256:
        raise ValueError("M9 model SHA-256 does not match the pinned model contract")
    with Image.open(source) as image:
        source_format = image.format
        image.load()
        oriented = ImageOps.exif_transpose(image)
        dimensions = oriented.size
        if oriented is not image:
            oriented.close()
    full_height = dimensions[1]
    chunks = [full_height if chunk == 0 else chunk for chunk in args.row_chunk_heights]
    chunks = list(dict.fromkeys([*chunks, full_height]))
    runs: list[dict[str, Any]] = []
    for trial in range(args.trials):
        for strategy in args.strategies:
            trial_records: list[dict[str, Any]] = []
            for chunk in chunks:
                destination = output_dir / (
                    f"m9-{strategy}-{chunk}-trial{trial + 1}.png"
                )
                record = _run_worker(
                    source=source,
                    model=model,
                    destination=destination,
                    strategy=strategy,
                    threads=args.threads,
                    row_chunk_height=chunk,
                    sample_interval_ms=args.sample_interval_ms,
                )
                record["trial"] = trial + 1
                trial_records.append(record)
            baseline = next(
                run for run in trial_records if run["row_chunk_height"] == full_height
            )
            baseline_path = Path(baseline["output"])
            for record in trial_records:
                record["pixel_error_vs_full_height"] = _compare_pngs(
                    Path(record["output"]), baseline_path
                )
                runs.append(record)
    try:
        import cv2
        import numpy as np
        import onnxruntime as ort

        runtime = {
            "onnxruntime": ort.__version__,
            "opencv": cv2.__version__,
            "opencv_default_threads": cv2.getNumThreads(),
            "numpy": np.__version__,
        }
    except ImportError:
        runtime = {}
    cpu_brand = _sysctl_value("machdep.cpu.brand_string")
    return {
        "measurement": (
            "fresh CLI process; process-tree RSS sampled at the configured interval"
        ),
        "date_local": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "host": {
            "platform": platform.platform(),
            "processor": cpu_brand or platform.processor() or platform.machine(),
            "machine_model": _sysctl_value("hw.model"),
            "cpu_model": cpu_brand or platform.processor() or platform.machine(),
            "architecture": platform.machine(),
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "total_memory_bytes": psutil.virtual_memory().total,
            "available_memory_bytes": psutil.virtual_memory().available,
            "python": platform.python_version(),
            "pillow": pillow_version,
            **runtime,
        },
        "source": {
            "format": source_format,
            "oriented_width": dimensions[0],
            "oriented_height": dimensions[1],
            "sha256": _sha256(source),
        },
        "model": {
            "filename": model.name,
            "bytes": model.stat().st_size,
            "sha256": M9_MODEL_SHA256,
            "provider": "CPUExecutionProvider",
        },
        "runs": runs,
        "limitations": [
            "macOS has no cgroup v2 limit in this run",
            "RSS sampling can miss allocations shorter than the sample interval",
            "Summed RSS can count shared pages more than once",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=("low-memory", "balanced"),
        default=["low-memory"],
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument(
        "--row-chunk-heights",
        nargs="+",
        type=int,
        default=[64, 128, 256, 512, 1024, 0],
        help="test heights; 0 uses the full output height",
    )
    parser.add_argument("--sample-interval-ms", type=int, default=5)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.threads < 1 or args.trials < 1 or args.sample_interval_ms < 1:
        raise SystemExit("threads, trials, and sampling interval must be positive")
    if any(height < 0 for height in args.row_chunk_heights):
        raise SystemExit("row chunk heights must be positive, or 0 for full-frame")
    try:
        report = run_benchmark(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    report_path = Path(args.output_dir) / "m9-benchmark.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps({"report": str(report_path), "runs": len(report["runs"])}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
