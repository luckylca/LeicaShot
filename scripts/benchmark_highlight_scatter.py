"""Compare field-optical scatter to a pinned Git baseline on cached real inputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import platform
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import psutil


class ProcessTreePeak:
    """Sample the benchmark worker's process tree while one scatter runs."""

    def __init__(self, interval_seconds: float = 0.01) -> None:
        self.process = psutil.Process()
        self.interval_seconds = interval_seconds
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _current_bytes(self) -> int:
        total = 0
        for process in [self.process, *self.process.children(recursive=True)]:
            try:
                total += process.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return total

    def _sample(self) -> None:
        while not self._stop.is_set():
            self.peak_bytes = max(self.peak_bytes, self._current_bytes())
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> ProcessTreePeak:
        self._thread.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self._stop.set()
        self._thread.join()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _worker(args: argparse.Namespace) -> int:
    import cv2
    import numpy as np

    from laika.cgroup import memory_stats

    if args.module_path:
        spec = importlib.util.spec_from_file_location(
            "laika_scatter_baseline", args.module_path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import baseline module {args.module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module("laika.highlights")

    high = np.load(args.high, mmap_mode="r", allow_pickle=False)
    coc = np.load(args.coc, mmap_mode="r", allow_pickle=False)
    started = time.perf_counter()
    with ProcessTreePeak() as peak:
        output = module.scatter_highlights(
            high,
            coc,
            args.aperture,
            radius_scale=3.0,
            field_optics=True,
            row_chunk_height=args.row_chunk_height,
        )
    elapsed = time.perf_counter() - started
    output_hash = hashlib.sha256(output.tobytes()).hexdigest()
    np.save(args.output, output, allow_pickle=False)
    print(
        json.dumps(
            {
                "seconds": elapsed,
                "process_tree_peak_rss_bytes": peak.peak_bytes,
                "output_array_sha256": output_hash,
                "output_file_sha256": _sha256(args.output),
                "output_shape": list(output.shape),
                "output_dtype": str(output.dtype),
                "opencv_threads": cv2.getNumThreads(),
                "cgroup_memory": memory_stats(),
            },
            sort_keys=True,
        )
    )
    return 0


def _run_worker(
    *,
    high_path: Path,
    coc_path: Path,
    output_path: Path,
    row_chunk_height: int,
    aperture: float,
    module_path: Path | None,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_worker",
        str(high_path),
        str(coc_path),
        str(output_path),
        "--row-chunk-height",
        str(row_chunk_height),
        "--aperture",
        str(aperture),
    ]
    if module_path is not None:
        command.extend(["--module-path", str(module_path)])
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"scatter worker failed ({completed.returncode}): {completed.stderr}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"scatter worker emitted invalid JSON: {completed.stdout}"
        ) from exc


def _parse_args() -> argparse.Namespace:
    if len(sys.argv) > 1 and sys.argv[1] == "_worker":
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("_worker")
        parser.add_argument("high", type=Path)
        parser.add_argument("coc", type=Path)
        parser.add_argument("output", type=Path)
        parser.add_argument("--row-chunk-height", type=int, required=True)
        parser.add_argument("--aperture", type=float, required=True)
        parser.add_argument("--module-path", type=Path)
        args = parser.parse_args()
        args.worker = True
        return args

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("m9_rgb", type=Path, help="cached M9-rendered RGB image")
    parser.add_argument("depth_npy", type=Path, help="raw Depth Anything float32 map")
    parser.add_argument("report", type=Path)
    parser.add_argument("--baseline-revision", default="762ff93")
    parser.add_argument("--long-edge", type=int, default=2048)
    parser.add_argument("--row-chunk-height", type=int, default=512)
    parser.add_argument("--focus-x", type=float, default=0.5)
    parser.add_argument("--focus-y", type=float, default=0.5)
    parser.add_argument("--aperture", type=float, default=2.8)
    parser.add_argument(
        "--order",
        choices=("baseline-first", "optimized-first"),
        default="baseline-first",
    )
    args = parser.parse_args()
    if min(args.long_edge, args.row_chunk_height) < 1:
        parser.error("long edge and row chunk height must be positive")
    return args


def main() -> int:
    args = _parse_args()
    if getattr(args, "worker", False):
        return _worker(args)

    import cv2
    import numpy as np

    from laika.bokeh import (
        focus_at_point,
        full_frame_coc,
        normalize_depth_map,
        resize_rgb_cubic,
    )
    from laika.cli import _read_rgb
    from laika.depth import resize_bilinear_align_corners
    from laika.highlights import extract_highlights

    image = _read_rgb(args.m9_rgb, max_bytes=512 * 1024 * 1024, max_pixels=80_000_000)
    raw_depth = np.load(args.depth_npy, mmap_mode="r", allow_pickle=False)
    depth = normalize_depth_map(raw_depth)
    del raw_depth
    height, width = image.shape[:2]
    if depth.shape != (height, width):
        depth = resize_bilinear_align_corners(depth, height, width)
    scale = min(1.0, args.long_edge / max(width, height))
    output_width = max(2, int(round(width * scale)))
    output_height = max(2, int(round(height * scale)))
    if (output_width, output_height) != (width, height):
        image = resize_rgb_cubic(image, output_width, output_height)
        depth = resize_bilinear_align_corners(depth, output_height, output_width)
    focus = focus_at_point(depth, args.focus_x, args.focus_y)
    coc = full_frame_coc(depth, focus, args.aperture)
    split = extract_highlights(image)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[1]
    baseline_source = subprocess.run(
        ["git", "show", f"{args.baseline_revision}:src/laika/highlights.py"],
        check=True,
        capture_output=True,
        cwd=repository,
    ).stdout
    baseline_commit = subprocess.run(
        ["git", "rev-parse", args.baseline_revision],
        check=True,
        capture_output=True,
        text=True,
        cwd=repository,
    ).stdout.strip()
    with tempfile.TemporaryDirectory(prefix="laika-scatter-compare-") as temporary:
        temporary_dir = Path(temporary)
        high_path = temporary_dir / "highlight-linear.npy"
        coc_path = temporary_dir / "coc.npy"
        np.save(high_path, split.highlight_linear, allow_pickle=False)
        np.save(coc_path, coc, allow_pickle=False)
        baseline_module = temporary_dir / "baseline-highlights.py"
        baseline_module.write_bytes(baseline_source)
        jobs = {
            "baseline": {
                "output_path": temporary_dir / "baseline-output.npy",
                "module_path": baseline_module,
            },
            "optimized": {
                "output_path": temporary_dir / "optimized-output.npy",
                "module_path": None,
            },
        }
        order = (
            ("baseline", "optimized")
            if args.order == "baseline-first"
            else ("optimized", "baseline")
        )
        results = {}
        for variant in order:
            results[variant] = _run_worker(
                high_path=high_path,
                coc_path=coc_path,
                row_chunk_height=args.row_chunk_height,
                aperture=args.aperture,
                **jobs[variant],
            )
        baseline = results["baseline"]
        optimized = results["optimized"]

    report = {
        "baseline_revision": args.baseline_revision,
        "baseline_commit": baseline_commit,
        "input_rgb": str(args.m9_rgb),
        "input_rgb_sha256": _sha256(args.m9_rgb),
        "raw_depth": str(args.depth_npy),
        "raw_depth_sha256": _sha256(args.depth_npy),
        "prepared_shape": [output_height, output_width, 3],
        "prepared_nonzero_highlight_pixels": int(
            np.count_nonzero(split.highlight_linear)
        ),
        "prepared_highlight_sha256": hashlib.sha256(
            split.highlight_linear.tobytes()
        ).hexdigest(),
        "prepared_coc_sha256": hashlib.sha256(coc.tobytes()).hexdigest(),
        "focus": [args.focus_x, args.focus_y, focus],
        "aperture": args.aperture,
        "radius_scale": 3.0,
        "row_chunk_height": args.row_chunk_height,
        "order": args.order,
        "opencv_version": cv2.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": psutil.cpu_count(logical=True),
        "baseline": baseline,
        "optimized": optimized,
        "outputs_bitwise_equal": baseline["output_array_sha256"]
        == optimized["output_array_sha256"],
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["outputs_bitwise_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
