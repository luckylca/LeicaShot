#!/usr/bin/env python3
"""Export pinned ViTMatte Small safetensors to a validated dynamic CPU ONNX graph.

The model file is supplied locally and is never downloaded or checked into Git.
This exporter checks its pinned SHA-256 before loading it and compares every
requested ONNX shape with the same PyTorch weights before publishing artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

MODEL_ID = "vitmatte-small-composition-1k"
MODEL_REPOSITORY = "hustvl/vitmatte-small-composition-1k"
MODEL_REVISION = "6a58ad7646403c1df626fbd746900aec7361ea1d"
CHECKPOINT_SHA256 = "bda9289db1bb6762d978b42d1c62ae3f34daf7497171a347a1d09657efd788cb"
TRANSFORMERS_VERSION = "4.53.1"
ONNX_OPSET = 17
DEFAULT_SHAPES = ((512, 512), (640, 768), (1024, 1024))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _error(reference: Any, actual: Any) -> dict[str, float]:
    delta = np.abs(np.asarray(reference, dtype=np.float32) - actual)
    if not np.isfinite(delta).all():
        raise ValueError("PyTorch/ONNX comparison returned non-finite values")
    return {
        "max_abs": float(delta.max(initial=0.0)),
        "mean_abs": float(delta.mean()),
        "p99_abs": float(np.percentile(delta, 99)),
    }


def export_and_validate(
    *,
    model_dir: Path,
    output_dir: Path,
    threads: int,
    max_abs_tolerance: float,
    shapes: tuple[tuple[int, int], ...] = DEFAULT_SHAPES,
) -> dict[str, Any]:
    import onnx
    import onnxruntime as ort
    import torch
    import transformers
    from transformers import VitMatteForImageMatting

    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise ValueError(
            f"export requires transformers=={TRANSFORMERS_VERSION}, "
            f"got {transformers.__version__}"
        )
    if threads < 1 or max_abs_tolerance <= 0:
        raise ValueError("threads and numerical tolerance must be positive")
    if not shapes or any(
        min(height, width) < 64 or height % 32 or width % 32 for height, width in shapes
    ):
        raise ValueError("all test shapes must be >=64 and divisible by 32")

    weights = model_dir / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"expected local safetensors file: {weights}")
    checkpoint_sha256 = _sha256(weights)
    if checkpoint_sha256 != CHECKPOINT_SHA256:
        raise ValueError(
            "ViTMatte checkpoint SHA-256 mismatch: "
            f"expected {CHECKPOINT_SHA256}, got {checkpoint_sha256}"
        )
    for name in ("config.json", "preprocessor_config.json"):
        if not (model_dir / name).is_file():
            raise FileNotFoundError(
                f"expected pinned model metadata: {model_dir / name}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "vitmatte-small-composition-1k.fp32.onnx"
    report_path = Path(f"{onnx_path}.validation.json")
    if onnx_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite an existing ViTMatte export")

    torch.set_num_threads(threads)
    torch.manual_seed(42)
    load_started = time.perf_counter()
    model = VitMatteForImageMatting.from_pretrained(
        str(model_dir), use_safetensors=True, local_files_only=True
    )
    model.eval()

    class MatteGraph(torch.nn.Module):
        def __init__(self, matte_model: Any) -> None:
            super().__init__()
            self.model = matte_model

        def forward(self, pixel_values: Any) -> Any:
            return self.model(pixel_values=pixel_values, return_dict=True).alphas

    graph = MatteGraph(model).eval()
    load_seconds = time.perf_counter() - load_started
    height, width = shapes[0]
    sample = torch.rand((1, 4, height, width), dtype=torch.float32)
    sample[:, :3].mul_(2.0).sub_(1.0)
    sample[:, 3] = torch.randint(0, 3, (1, height, width)).float() * (128.0 / 255.0)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".vitmatte-export-", suffix=".onnx", dir=output_dir
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        with torch.inference_mode():
            torch.onnx.export(
                graph,
                (sample,),
                str(temporary_path),
                input_names=["pixel_values"],
                output_names=["alpha"],
                dynamic_axes={
                    "pixel_values": {2: "height", 3: "width"},
                    "alpha": {2: "height", 3: "width"},
                },
                opset_version=ONNX_OPSET,
                do_constant_folding=True,
                dynamo=False,
            )
        onnx.checker.check_model(str(temporary_path))
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            str(temporary_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != "pixel_values":
            raise ValueError("ViTMatte ONNX input must be named pixel_values")
        if len(outputs) != 1 or outputs[0].name != "alpha":
            raise ValueError("ViTMatte ONNX output must be named alpha")
        if any(
            not isinstance(dimension, str)
            for dimension in inputs[0].shape[2:] + outputs[0].shape[2:]
        ):
            raise ValueError("ViTMatte ONNX graph did not retain dynamic H/W")

        errors: dict[str, Any] = {}
        with torch.inference_mode():
            for height, width in shapes:
                torch.manual_seed(1000 + height * 31 + width)
                pixels = torch.rand((1, 4, height, width), dtype=torch.float32)
                pixels[:, :3].mul_(2.0).sub_(1.0)
                pixels[:, 3] = torch.randint(0, 3, (1, height, width)).float() * (
                    128.0 / 255.0
                )
                expected = graph(pixels).cpu().numpy()
                actual = session.run(["alpha"], {"pixel_values": pixels.numpy()})[0]
                if actual.shape != (1, 1, height, width):
                    raise ValueError(
                        f"ViTMatte output shape mismatch at {width}×{height}: "
                        f"{actual.shape}"
                    )
                errors[f"{width}x{height}"] = _error(expected, actual)
                if errors[f"{width}x{height}"]["max_abs"] > max_abs_tolerance:
                    raise ValueError(
                        f"PyTorch/ONNX error exceeds {max_abs_tolerance} at "
                        f"{width}×{height}: {errors[f'{width}x{height}']}"
                    )

        graph_sha256 = _sha256(temporary_path)
        temporary_path.replace(onnx_path)
        report = {
            "model_id": MODEL_ID,
            "repository": MODEL_REPOSITORY,
            "repository_revision": MODEL_REVISION,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_bytes": weights.stat().st_size,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "onnx_version": onnx.__version__,
            "onnxruntime_version": ort.__version__,
            "source_precision": "FP32 safetensors",
            "external_precision": "FP32",
            "opset": ONNX_OPSET,
            "dynamic_shapes": True,
            "input_name": "pixel_values",
            "output_name": "alpha",
            "input_layout": "NCHW, RGB[-1,1]+trimap[0,1]",
            "output_layout": "NCHW alpha[0,1]",
            "tested_shapes_width_height": [list((w, h)) for h, w in shapes],
            "numerical_error": errors,
            "max_abs_tolerance": max_abs_tolerance,
            "setup_seconds": load_seconds,
            "onnx_sha256": graph_sha256,
            "onnx_path": onnx_path.name,
            "provider": "CPUExecutionProvider",
            "threads": threads,
            "passed": True,
        }
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-abs-tolerance", type=float, default=1e-3)
    parser.add_argument(
        "--test-shapes",
        type=str,
        nargs="+",
        metavar="WIDTHxHEIGHT",
        help="dynamic ONNX shapes such as 512x512 768x640 1024x1024",
    )
    args = parser.parse_args()
    if args.test_shapes:
        shapes = []
        for item in args.test_shapes:
            try:
                width_text, height_text = item.lower().split("x", 1)
                shapes.append((int(height_text), int(width_text)))
            except (ValueError, TypeError) as exc:
                parser.error(f"invalid test shape {item!r}; use WIDTHxHEIGHT")
                raise AssertionError from exc
        selected_shapes = tuple(shapes)
    else:
        selected_shapes = DEFAULT_SHAPES
    report = export_and_validate(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        threads=args.threads,
        max_abs_tolerance=args.max_abs_tolerance,
        shapes=selected_shapes,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
