#!/usr/bin/env python3
"""Export pinned Depth Anything V2 Small weights to a validated CPU ONNX model.

The upstream Depth Anything V2 checkout is used only as the model definition.
It must be at the exact source revision pinned in models/manifest.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from laika.model_assets import (  # noqa: E402
    ModelAssetError,
    load_manifest,
    verify_artifact,
)

MODEL_ID = "depth-anything-v2-small"
OPSET = 20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_source_revision(source: Path, expected_revision: str) -> None:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ModelAssetError(
            "cannot read the upstream source checkout revision"
        ) from exc
    if result.stdout.strip() != expected_revision:
        raise ModelAssetError(
            "upstream source revision mismatch: "
            f"expected {expected_revision}, got {result.stdout.strip()}"
        )


def _export(args: argparse.Namespace) -> dict[str, Any]:
    artifacts = load_manifest()
    try:
        manifest = json.loads(
            (ROOT / "models/manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelAssetError("cannot read the model manifest") from exc
    try:
        artifact_record = next(
            row for row in manifest["models"] if row["id"] == MODEL_ID
        )
        artifact = artifacts[MODEL_ID]
    except (KeyError, StopIteration) as exc:
        raise ModelAssetError(
            f"model is not pinned in the manifest: {MODEL_ID}"
        ) from exc

    source = args.source_checkout.resolve()
    weights = args.weights.resolve()
    output = args.output.resolve()
    if output.suffix.lower() != ".onnx":
        raise ModelAssetError("output path must end in .onnx")
    if args.height < 518 or args.width < 518:
        raise ModelAssetError("network input dimensions must be at least 518 pixels")
    if args.height % 14 or args.width % 14:
        raise ModelAssetError("network input dimensions must be multiples of 14")
    if args.threads < 1:
        raise ModelAssetError("thread count must be positive")
    if args.max_abs_error <= 0:
        raise ModelAssetError("maximum absolute error must be positive")
    data_output = Path(f"{output}.data")
    report_output = Path(f"{output}.validation.json")
    if any(path.exists() for path in (output, data_output, report_output)):
        raise ModelAssetError("refusing to overwrite an existing ONNX output")

    _verify_source_revision(source, str(artifact_record["source_code_revision"]))
    verify_artifact(weights, artifact)

    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch
    except ImportError as exc:
        raise ModelAssetError(
            "export requires the optional PyTorch, ONNX, ONNX Script, and "
            "ONNX Runtime packages; see docs/MODELS.md"
        ) from exc

    sys.path.insert(0, str(source))

    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except ImportError as exc:
        raise ModelAssetError(
            "upstream model imports failed; install its pinned CPU dependencies"
        ) from exc

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    model = DepthAnythingV2(
        encoder="vits", features=64, out_channels=[48, 96, 192, 384]
    )
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(checkpoint, strict=True)
    del checkpoint
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ModelAssetError("checkpoint does not strictly match the Small model")
    model.eval()
    load_seconds = time.perf_counter() - started

    generator = np.random.default_rng(args.seed)
    input_array = generator.standard_normal(
        (1, 3, args.height, args.width), dtype=np.float32
    )
    input_tensor = torch.from_numpy(input_array)
    with torch.inference_mode():
        reference = model(input_tensor).cpu().numpy()

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".laika-onnx-", dir=output.parent) as tmp:
        temporary_model = Path(tmp) / output.name
        export_started = time.perf_counter()
        torch.onnx.export(
            model,
            (input_tensor,),
            temporary_model,
            input_names=["image"],
            output_names=["depth"],
            opset_version=OPSET,
            dynamo=True,
            external_data=True,
            optimize=True,
        )
        export_seconds = time.perf_counter() - export_started

        onnx.checker.check_model(str(temporary_model))
        options = ort.SessionOptions()
        options.intra_op_num_threads = args.threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_started = time.perf_counter()
        session = ort.InferenceSession(
            str(temporary_model),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        session_create_seconds = time.perf_counter() - session_started
        inference_started = time.perf_counter()
        actual = session.run(["depth"], {"image": input_array})[0]
        inference_seconds = time.perf_counter() - inference_started

        error = np.abs(reference - actual)
        max_error = float(error.max())
        if not np.isfinite(actual).all() or max_error > args.max_abs_error:
            raise ModelAssetError(
                "ONNX output failed the numerical check: "
                f"max_abs_error={max_error:g}, limit={args.max_abs_error:g}"
            )
        temporary_data = Path(f"{temporary_model}.data")
        if not temporary_data.is_file():
            raise ModelAssetError(
                "ONNX exporter did not create the expected weight data"
            )
        report: dict[str, Any] = {
            "model_id": MODEL_ID,
            "source_code_revision": artifact_record["source_code_revision"],
            "weights_sha256": artifact.sha256,
            "torch_version": torch.__version__,
            "onnx_version": onnx.__version__,
            "onnxruntime_version": ort.__version__,
            "provider": session.get_providers(),
            "threads": args.threads,
            "opset": OPSET,
            "fixed_input_shape": list(input_array.shape),
            "output_shape": list(actual.shape),
            "max_absolute_error": max_error,
            "mean_absolute_error": float(error.mean()),
            "p99_absolute_error": float(np.quantile(error, 0.99)),
            "max_absolute_error_limit": args.max_abs_error,
            "all_finite": bool(np.isfinite(actual).all()),
            "model_load_seconds": load_seconds,
            "onnx_export_seconds": export_seconds,
            "ort_session_create_seconds": session_create_seconds,
            "ort_inference_seconds": inference_seconds,
            "onnx_sha256": _sha256(temporary_model),
            "external_data_sha256": _sha256(temporary_data),
            "onnx_bytes": temporary_model.stat().st_size,
            "external_data_bytes": temporary_data.stat().st_size,
            "limitations": [
                "Exported input height and width are static.",
                "Input preprocessing and final source-size interpolation "
                "are not included.",
                "This validates one synthetic tensor, not photo quality or "
                "full-pipeline memory.",
            ],
        }
        temporary_report = Path(tmp) / report_output.name
        temporary_report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        # Publish external tensors and report before the ONNX graph, which is the
        # final marker that the complete artifact set is ready to load.
        os.replace(temporary_data, data_output)
        os.replace(temporary_report, report_output)
        os.replace(temporary_model, output)

    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=518)
    parser.add_argument("--width", type=int, default=518)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--max-abs-error", type=float, default=1e-4)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = _export(args)
    except (ModelAssetError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
