#!/usr/bin/env python3
"""Compare a Laika M9 result to a row-bounded OpenCV oracle of reference fusion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def compare(args: argparse.Namespace) -> dict[str, Any]:
    import cv2
    import numpy as np
    from PIL import Image

    from laika.engine import ImageLimits
    from laika.m9 import (
        M9_SIGMA_AB,
        M9_SIGMA_L,
        M9Onnx,
        _open_oriented_rgb,
        _resize_prediction,
    )

    source, _ = _open_oriented_rgb(Path(args.input), ImageLimits())
    try:
        original = np.array(source, dtype=np.uint8, order="C", copy=True)
        with M9Onnx(args.model, strategy="low-memory", intra_op_num_threads=1) as m9:
            prediction = m9.predict_pil(source)
    finally:
        source.close()
    height, width = original.shape[:2]
    predicted = _resize_prediction(prediction.rgb, width, height)
    del prediction
    halo = 100
    sigmas = (M9_SIGMA_L, M9_SIGMA_AB, M9_SIGMA_AB)
    max_error = 0
    total_error = 0
    different = 0
    values = 0
    target = Path(args.output)
    with Image.open(target) as result_image:
        if result_image.size != (width, height):
            raise ValueError("Laika output dimensions do not match the oriented source")
        result_rgb = result_image.convert("RGB")
        try:
            for y0 in range(0, height, args.row_chunk_height):
                y1 = min(height, y0 + args.row_chunk_height)
                source_y0 = max(0, y0 - halo)
                source_y1 = min(height, y1 + halo)
                source_lab = cv2.cvtColor(
                    original[source_y0:source_y1], cv2.COLOR_RGB2LAB
                )
                predicted_lab = cv2.cvtColor(
                    predicted[source_y0:source_y1], cv2.COLOR_RGB2LAB
                )
                center0 = y0 - source_y0
                center1 = center0 + y1 - y0
                fused_lab = np.empty((center1 - center0, width, 3), dtype=np.uint8)
                for channel, sigma in enumerate(sigmas):
                    scratch = np.empty(source_lab.shape[:2], dtype=np.float32)
                    np.subtract(
                        predicted_lab[:, :, channel],
                        source_lab[:, :, channel],
                        out=scratch,
                        dtype=np.float32,
                    )
                    cv2.GaussianBlur(
                        scratch,
                        (0, 0),
                        sigmaX=sigma,
                        dst=scratch,
                        borderType=cv2.BORDER_REFLECT_101,
                    )
                    np.add(scratch, source_lab[:, :, channel], out=scratch)
                    np.clip(scratch, 0, 255, out=scratch)
                    fused_lab[:, :, channel] = scratch[center0:center1].astype(np.uint8)
                    del scratch
                reference_rgb = cv2.cvtColor(fused_lab, cv2.COLOR_LAB2RGB)
                actual_rgb = np.asarray(
                    result_rgb.crop((0, y0, width, y1)), dtype=np.uint8
                )
                delta = np.abs(
                    reference_rgb.astype(np.int16) - actual_rgb.astype(np.int16)
                )
                max_error = max(max_error, int(delta.max(initial=0)))
                total_error += int(delta.sum(dtype=np.int64))
                different += int(np.count_nonzero(delta))
                values += int(delta.size)
                del (
                    source_lab,
                    predicted_lab,
                    fused_lab,
                    reference_rgb,
                    actual_rgb,
                    delta,
                )
        finally:
            result_rgb.close()
    return {
        "oracle": "row-bounded OpenCV reconstruction of reference one-blur Lab formula",
        "note": (
            "The reference Kotlin RGB2Lab table is independently documented as "
            "within 1 code value of OpenCV; this oracle is not an Android app capture."
        ),
        "dimensions": {"width": width, "height": height},
        "row_chunk_height": args.row_chunk_height,
        "model_sha256": m9.model_sha256,
        "runtime": {"onnxruntime": m9.runtime_version, "provider": m9.runtime_provider},
        "max_absolute_pixel_error": max_error,
        "mean_absolute_pixel_error": total_error / values if values else 0.0,
        "pixels_differing": different,
        "pixel_values_compared": values,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True, help="Laika M9 PNG/JPEG result")
    parser.add_argument("--model", required=True)
    parser.add_argument("--row-chunk-height", type=int, default=256)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.row_chunk_height < 1:
        raise SystemExit("row chunk height must be positive")
    try:
        report = compare(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
