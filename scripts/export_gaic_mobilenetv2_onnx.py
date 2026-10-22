#!/usr/bin/env python3
"""Export the authorized GAIC MobileNetV2 checkpoint to dynamic CPU ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

MODEL_ID = "gaic-mobilenetv2-reddim16"
REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPOSITORY = "https://github.com/bo-zhang-cs/GAIC-Pytorch"
SOURCE_CODE_REVISION = "877d6a325943c0a098324dc6d4a975ff4097c3f6"
CHECKPOINT_SHA256 = "eb1d5eed2710c79d3270add448111b9e6fa2475e246694c0ad995a6efb299e8c"
ONNX_OPSET = 17


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _case(height: int, width: int, count: int, torch: Any) -> tuple[Any, Any]:
    image = torch.randn((1, 3, height, width), dtype=torch.float32)
    boxes = []
    for index in range(count):
        left = (index * 17) % max(width // 3, 1)
        top = (index * 11) % max(height // 3, 1)
        right = max(left + 1, width - (index * 13) % max(width // 4, 1))
        bottom = max(top + 1, height - (index * 7) % max(height // 4, 1))
        boxes.append((float(left), float(top), float(right), float(bottom)))
    return image, torch.tensor(boxes, dtype=torch.float32)


def export_and_validate(
    *, checkpoint: Path, output_dir: Path, threads: int, atol: float, rtol: float
) -> dict[str, Any]:
    import onnx
    import onnxruntime as ort
    import torch
    import torchvision

    sys.path.insert(0, str(REPO_ROOT))
    from tools.gaic_mobile_port import GaicMobileNetV2

    if not checkpoint.is_file():
        raise FileNotFoundError(f"GAIC checkpoint is unavailable: {checkpoint}")
    actual_checkpoint_hash = _sha256(checkpoint)
    if actual_checkpoint_hash != CHECKPOINT_SHA256:
        raise ValueError(
            "GAIC checkpoint SHA-256 mismatch: "
            f"expected {CHECKPOINT_SHA256}, got {actual_checkpoint_hash}"
        )
    if threads < 1 or atol <= 0 or rtol < 0:
        raise ValueError("threads and parity tolerances must be positive")
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing export directory: {output_dir}"
        )

    output_dir.mkdir(parents=True)
    onnx_path = output_dir / "gaic-mobilenetv2-reddim16.fp32.onnx"
    report_path = Path(f"{onnx_path}.validation.json")
    torch.set_num_threads(threads)
    torch.manual_seed(42)
    model = GaicMobileNetV2().eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    cases = (
        _case(256, 320, 5, torch),
        _case(288, 256, 3, torch),
        _case(128, 192, 2, torch),
    )
    names = ("image", "boxes")
    started = time.perf_counter()
    report: dict[str, Any] = {
        "model_id": MODEL_ID,
        "source_repository": SOURCE_REPOSITORY,
        "source_code_revision": SOURCE_CODE_REVISION,
        "source_checkpoint_sha256": actual_checkpoint_hash,
        "source_checkpoint_bytes": checkpoint.stat().st_size,
        "opset": ONNX_OPSET,
        "input_names": list(names),
        "output_name": "scores",
        "dynamic_shapes": True,
        "precision": "FP32",
        "provider": "CPUExecutionProvider",
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "validation_cases": [],
        "passed": False,
    }
    try:
        with torch.inference_mode():
            torch.onnx.export(
                model,
                cases[0],
                str(onnx_path),
                input_names=list(names),
                output_names=["scores"],
                dynamic_axes={
                    "image": {2: "height", 3: "width"},
                    "boxes": {0: "candidate_count"},
                    "scores": {0: "candidate_count"},
                },
                opset_version=ONNX_OPSET,
                do_constant_folding=True,
                dynamo=False,
            )
        graph = onnx.load(str(onnx_path))
        onnx.checker.check_model(graph)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError("export parity session did not use CPUExecutionProvider")
        if [value.name for value in session.get_inputs()] != list(names):
            raise RuntimeError("exported GAIC input names are incorrect")
        if [value.name for value in session.get_outputs()] != ["scores"]:
            raise RuntimeError("exported GAIC output name is incorrect")
        for image, boxes in cases:
            with torch.inference_mode():
                expected = model(image, boxes).cpu().numpy()
            actual = session.run(
                ["scores"], {"image": image.cpu().numpy(), "boxes": boxes.numpy()}
            )[0]
            if actual.shape != expected.shape or not np.isfinite(actual).all():
                raise RuntimeError("GAIC ONNX returned an invalid score vector")
            delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
            passed = bool(np.allclose(actual, expected, atol=atol, rtol=rtol))
            case = {
                "image_shape": list(image.shape),
                "box_shape": list(boxes.shape),
                "score_shape": list(actual.shape),
                "max_abs_error": float(delta.max(initial=0.0)),
                "mean_abs_error": float(delta.mean()),
                "passed": passed,
            }
            report["validation_cases"].append(case)
            if not passed:
                raise RuntimeError(f"PyTorch/ONNX parity failed: {case}")
        report["onnx_sha256"] = _sha256(onnx_path)
        report["onnx_bytes"] = onnx_path.stat().st_size
        report["export_and_validation_seconds"] = time.perf_counter() - started
        report["host"] = platform.platform()
        report["passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--atol", type=float, default=0.001)
    parser.add_argument("--rtol", type=float, default=0.001)
    args = parser.parse_args()
    report = export_and_validate(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        threads=args.threads,
        atol=args.atol,
        rtol=args.rtol,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
