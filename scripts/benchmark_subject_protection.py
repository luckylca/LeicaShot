#!/usr/bin/env python3
"""Run local M9 → SAM → ViTMatte and report optional portrait-matte metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from PIL import Image

from laika.cli import _read_rgb
from laika.m9 import M9ColorEngine
from laika.matting import ViTMatteOnnx
from laika.protection import (
    prepare_subject_analysis_rgb,
    refine_subject_masks,
    segment_subject_points,
)
from laika.subject import Sam2TinyOnnx


class ProcessTreePeak:
    """Sample this worker and descendants while M9/SAM/ViTMatte are active."""

    def __init__(self, interval_seconds: float = 0.01) -> None:
        self.process = psutil.Process()
        self.interval_seconds = interval_seconds
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.is_set():
            total = 0
            for process in [self.process, *self.process.children(recursive=True)]:
                try:
                    total += process.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            self.peak_bytes = max(self.peak_bytes, total)
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


def _write_gray(path: Path, plane: Any) -> None:
    values = np.asarray(plane, dtype=np.float32)
    pixels = np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(pixels, mode="L").save(path)


def _matte_metrics(prediction: Any, reference: Any) -> dict[str, float]:
    import cv2

    alpha = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(reference, dtype=np.float32)
    if target.ndim == 3:
        target = target[:, :, 0]
    if target.shape != alpha.shape:
        target = cv2.resize(
            target, (alpha.shape[1], alpha.shape[0]), interpolation=cv2.INTER_AREA
        )
    if target.max(initial=0.0) > 1.0:
        target /= np.float32(255.0)
    delta = np.abs(alpha - target)
    pred_binary = alpha >= 0.5
    ref_binary = target >= 0.5
    intersection = int(np.count_nonzero(pred_binary & ref_binary))
    union = int(np.count_nonzero(pred_binary | ref_binary))
    ref_u8 = ref_binary.astype(np.uint8)
    pred_u8 = pred_binary.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ref_edge = cv2.morphologyEx(ref_u8, cv2.MORPH_GRADIENT, kernel) > 0
    pred_edge = cv2.morphologyEx(pred_u8, cv2.MORPH_GRADIENT, kernel) > 0
    edge_tolerance = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    ref_near = cv2.dilate(ref_edge.astype(np.uint8), edge_tolerance) > 0
    pred_near = cv2.dilate(pred_edge.astype(np.uint8), edge_tolerance) > 0
    edge_precision = float(np.count_nonzero(pred_edge & ref_near)) / max(
        1, int(np.count_nonzero(pred_edge))
    )
    edge_recall = float(np.count_nonzero(ref_edge & pred_near)) / max(
        1, int(np.count_nonzero(ref_edge))
    )
    return {
        "soft_alpha_mae": float(delta.mean()),
        "soft_alpha_rmse": float(np.sqrt(np.mean(delta * delta))),
        "soft_alpha_p95_abs_error": float(np.percentile(delta, 95)),
        "binary_iou_at_0_5": intersection / max(1, union),
        "boundary_precision_within_5px": edge_precision,
        "boundary_recall_within_5px": edge_recall,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--m9-model", type=Path, required=True)
    parser.add_argument("--sam-model-dir", type=Path, required=True)
    parser.add_argument("--vitmatte-model", type=Path, required=True)
    parser.add_argument("--ground-truth-alpha", type=Path)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--tile-size", type=int, choices=(512, 768, 1024), default=1024)
    parser.add_argument(
        "--point", action="append", default=[], metavar="X,Y", required=True
    )
    parser.add_argument("--sample-interval-ms", type=int, default=10)
    args = parser.parse_args()
    if args.threads < 1 or args.sample_interval_ms < 1:
        parser.error("threads and sample interval must be positive")
    try:
        points = tuple(
            tuple(float(value) for value in item.split(",", maxsplit=1))
            for item in args.point
        )
    except ValueError:
        parser.error("--point must use normalized X,Y coordinates")
    if any(
        len(point) != 2 or not all(0.0 <= value < 1.0 for value in point)
        for point in points
    ):
        parser.error("point coordinates must be normalized values in [0, 1)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    stages: dict[str, float] = {}
    with ProcessTreePeak(args.sample_interval_ms / 1000.0) as peak:
        with tempfile.TemporaryDirectory(prefix="laika-subject-bench-") as temp_dir:
            colored_path = Path(temp_dir) / "m9-color.png"
            color_started = time.perf_counter()
            with M9ColorEngine(
                args.m9_model,
                strategy="low-memory",
                intra_op_num_threads=args.threads,
            ) as model:
                model.process_file(args.input, colored_path)
            m9_output_sha256 = _sha256(colored_path)
            stages["m9_total"] = time.perf_counter() - color_started
            image = _read_rgb(
                colored_path,
                max_bytes=50 * 1024 * 1024,
                max_pixels=40_000_000,
            )
            analysis = prepare_subject_analysis_rgb(image)

            sam_started = time.perf_counter()
            sam = Sam2TinyOnnx.from_validation_report(
                args.sam_model_dir,
                intra_op_num_threads=args.threads,
            )
            try:
                masks, scores, sam_times, cache_hits = segment_subject_points(
                    analysis, points, sam
                )
            finally:
                sam.close()
            stages["sam_setup_and_prepare"] = time.perf_counter() - sam_started
            stages["sam_encoder"] = sam_times["encoder"]
            stages["sam_decoder"] = sam_times["decoder"]

            matte_started = time.perf_counter()
            matte = ViTMatteOnnx.from_validation_report(
                args.vitmatte_model,
                intra_op_num_threads=args.threads,
            )
            try:
                result = refine_subject_masks(
                    analysis,
                    masks,
                    points,
                    matte,
                    tile_size=args.tile_size,
                    cache_hits=cache_hits,
                )
            finally:
                matte.close()
            stages["vitmatte_total"] = time.perf_counter() - matte_started
            stages.update(
                {
                    f"vitmatte.{key}": value
                    for key, value in result.stage_seconds.items()
                }
            )

            alpha_path = args.output_dir / "alpha.png"
            _write_gray(alpha_path, result.alpha)
            mask_paths = []
            for index, mask in enumerate(masks, start=1):
                mask_path = args.output_dir / f"sam-mask-{index:02d}.png"
                _write_gray(mask_path, mask)
                mask_paths.append(str(mask_path))
            metrics = None
            if args.ground_truth_alpha is not None:
                with Image.open(args.ground_truth_alpha) as opened:
                    truth = np.asarray(opened.convert("L"), dtype=np.float32)
                metrics = _matte_metrics(result.alpha, truth)

    report = {
        "input": str(args.input),
        "input_sha256": _sha256(args.input),
        "m9_output_sha256": m9_output_sha256,
        "width": int(result.alpha.shape[1]),
        "height": int(result.alpha.shape[0]),
        "points": points,
        "subject_point_alpha_scores": result.point_scores,
        "sam_mask_area_fractions": tuple(float(mask.mean()) for mask in masks),
        "sam_cache_hits": cache_hits,
        "sam_checkpoint_sha256": (
            "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"
        ),
        "vitmatte_onnx_sha256": matte.model_sha256,
        "tile_size": args.tile_size,
        "threads": args.threads,
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "memory_total_bytes": psutil.virtual_memory().total,
        },
        "stage_seconds": stages,
        "task_wall_seconds": time.perf_counter() - started,
        "process_tree_peak_rss_bytes": peak.peak_bytes,
        "alpha_output": str(alpha_path),
        "sam_mask_outputs": mask_paths,
        "ground_truth_metrics": metrics,
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
