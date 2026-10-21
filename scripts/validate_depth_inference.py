#!/usr/bin/env python3
"""Compare the Laika depth adapter with the pinned PyTorch reference path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from laika.depth import (  # noqa: E402
    DepthAnythingV2Small,
    DepthInferenceError,
    _preprocess_rgb,
    _sha256,
)
from laika.model_assets import ModelAssetError, load_manifest, verify_artifact  # noqa: E402


def _validate(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import cv2
        import numpy as np
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise DepthInferenceError(
            "depth validation requires the model toolchain and OpenCV"
        ) from exc

    manifest = json.loads((ROOT / "models/manifest.json").read_text(encoding="utf-8"))
    record = next(
        row for row in manifest["models"] if row["id"] == "depth-anything-v2-small"
    )
    artifact = load_manifest()["depth-anything-v2-small"]
    verify_artifact(args.weights, artifact)
    from export_depth_anything_v2 import _verify_source_revision

    _verify_source_revision(args.source_checkout, record["source_code_revision"])
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    rgb = np.random.default_rng(args.seed).integers(
        0, 256, (args.height, args.width, 3), dtype=np.uint8
    )
    input_size = args.input_size
    max_analysis_edge = args.analysis_max_edge
    if input_size is None and max_analysis_edge is None:
        max_analysis_edge = 1220
    actual_tensor, geometry = _preprocess_rgb(
        rgb,
        input_size,
        cv2,
        np,
        4_000_000,
        max_analysis_edge=max_analysis_edge,
    )
    with DepthAnythingV2Small(
        args.onnx,
        input_size=input_size,
        max_analysis_edge=max_analysis_edge,
        intra_op_num_threads=args.threads,
    ) as depth_engine:
        actual = depth_engine.infer_rgb(rgb)

    sys.path.insert(0, str(args.source_checkout.resolve()))
    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(
        encoder="vits", features=64, out_channels=[48, 96, 192, 384]
    )
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint, strict=True)
    del checkpoint
    model.eval()
    preprocess_stats = None
    with torch.inference_mode():
        source_height, source_width = rgb.shape[:2]
        if input_size is not None:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            reference_tensor, _ = model.image2tensor(bgr, input_size)
            reference_tensor = reference_tensor.cpu()
            preprocess_difference = np.abs(reference_tensor.numpy() - actual_tensor)
            preprocess_stats = {
                "max": float(preprocess_difference.max()),
                "mean": float(preprocess_difference.mean()),
                "p99": float(np.quantile(preprocess_difference, 0.99)),
            }
            del preprocess_difference, bgr
        else:
            reference_tensor = torch.from_numpy(actual_tensor)
        network_depth = model(reference_tensor)
        analysis_depth = network_depth[
            :, : geometry.analysis_height, : geometry.analysis_width
        ]
        reference_depth = (
            functional.interpolate(
                analysis_depth[:, None],
                (source_height, source_width),
                mode="bilinear",
                align_corners=True,
            )[0, 0]
            .cpu()
            .numpy()
        )

    error = np.abs(actual.depth - reference_depth)
    report = {
        "model_id": "depth-anything-v2-small",
        "source_code_revision": record["source_code_revision"],
        "weights_sha256": artifact.sha256,
        "onnx_sha256": _sha256(args.onnx),
        "input_rgb_shape": list(rgb.shape),
        "analysis_shape": [geometry.analysis_height, geometry.analysis_width],
        "network_shape": [geometry.network_height, geometry.network_width],
        "output_shape": list(actual.depth.shape),
        "preprocessing_profile": (
            "upstream-lower-bound" if input_size is not None else "max-analysis-edge"
        ),
        "preprocessing_reference": (
            "pinned-upstream-image2tensor"
            if preprocess_stats is not None
            else "identical prepared tensor for PyTorch and ONNX"
        ),
        "depth_max_absolute_error": float(error.max()),
        "depth_mean_absolute_error": float(error.mean()),
        "depth_p99_absolute_error": float(np.quantile(error, 0.99)),
        "max_absolute_error_limit": args.max_abs_error,
        "all_finite": bool(np.isfinite(actual.depth).all()),
    }
    if preprocess_stats is not None:
        report.update(
            {
                "preprocess_max_absolute_error": preprocess_stats["max"],
                "preprocess_mean_absolute_error": preprocess_stats["mean"],
                "preprocess_p99_absolute_error": preprocess_stats["p99"],
            }
        )
    if (
        (
            preprocess_stats is not None
            and report["preprocess_max_absolute_error"] > args.preprocess_error_limit
        )
        or report["depth_max_absolute_error"] > args.max_abs_error
        or not report["all_finite"]
        or tuple(actual.depth.shape) != (args.height, args.width)
        or (actual.analysis_height, actual.analysis_width)
        != (geometry.analysis_height, geometry.analysis_width)
        or (actual.network_height, actual.network_width)
        != (geometry.network_height, geometry.network_width)
    ):
        raise DepthInferenceError(
            "Laika depth adapter failed its PyTorch input/output comparison: "
            + json.dumps(report, sort_keys=True)
        )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--width", type=int, default=700)
    parser.add_argument("--height", type=int, default=400)
    size_group = parser.add_mutually_exclusive_group()
    size_group.add_argument("--analysis-max-edge", type=int)
    size_group.add_argument("--input-size", type=int)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261021)
    # OpenCV's bicubic float path differs slightly between CPU architectures;
    # the final restored depth output retains the tighter 1e-4 acceptance.
    parser.add_argument("--preprocess-error-limit", type=float, default=2e-4)
    parser.add_argument("--max-abs-error", type=float, default=1e-4)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.width < 1 or args.height < 1 or args.threads < 1:
        print(
            json.dumps({"error": "width, height, and threads must be positive"}),
            file=sys.stderr,
        )
        return 2
    try:
        report = _validate(args)
    except (
        DepthInferenceError,
        ModelAssetError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
