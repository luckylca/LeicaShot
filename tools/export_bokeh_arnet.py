#!/usr/bin/env python3
"""Export the pinned BokehMe ARNet checkpoint to a validated dynamic ONNX graph.

This tool reads a local upstream checkout and checkpoint. It does not download
or bundle third-party code or model weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REVISION = "8b3ed556dc146207c9bb39fccd078a113f8d1bd8"
SOURCE_FILE_SHA256 = "cecdec77a9b0bd5b93b0bd2ea7deadf6862f36a86945643521003c3672c8ac2a"
CHECKPOINT_SHA256 = "6c7dba6f32e3afc8e984eb5adfecfea5d13b6f4ebd132a5a890beba384be4752"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_load_checkpoint(torch: Any, np: Any, checkpoint: Path) -> dict[str, Any]:
    """Load the known NumPy metadata through a narrow weights-only allowlist."""

    def numpy_reconstruct(*values: Any) -> Any:
        core = np._core if hasattr(np, "_core") else np.core
        return core.multiarray._reconstruct(*values)

    allowed: list[Any] = [
        (numpy_reconstruct, "numpy.core.multiarray._reconstruct"),
        (np.ndarray, "numpy.ndarray"),
        (np.dtype, "numpy.dtype"),
    ]
    for dtype_name in ("uint32", "int32", "int64", "float64"):
        dtype_type = type(np.dtype(dtype_name))
        allowed.append((dtype_type, f"numpy.dtype[{dtype_name}]"))
    with torch.serialization.safe_globals(allowed):
        result = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(result, dict) or not isinstance(result.get("model"), dict):
        raise ValueError("checkpoint must contain a model state dictionary")
    return result


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream-root",
        required=True,
        type=Path,
        help="BokehMe checkout at the pinned revision",
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="new directory for FP32, FP16, and validation reports",
    )
    parser.add_argument("--validation-photo", type=Path)
    parser.add_argument("--validation-depth", type=Path)
    parser.add_argument("--photo-max-edge", type=int, default=1024)
    parser.add_argument("--focus-x", type=float, default=0.5)
    parser.add_argument("--focus-y", type=float, default=0.5)
    return parser


def _photo_case(
    args: argparse.Namespace, torch: Any, np: Any
) -> tuple[Any, Any, dict[str, Any]]:
    from PIL import Image, ImageOps

    if args.validation_photo is None or args.validation_depth is None:
        raise ValueError(
            "photo validation requires both --validation-photo and --validation-depth"
        )
    if args.photo_max_edge < 2:
        raise ValueError("--photo-max-edge must be at least two pixels")
    from laika.bokeh import (
        focus_at_point,
        full_frame_coc,
        normalize_depth_map,
        resize_rgb_cubic,
    )
    from laika.depth import resize_bilinear_align_corners

    with Image.open(args.validation_photo) as opened:
        oriented = ImageOps.exif_transpose(opened)
        try:
            oriented.load()
            original_size = oriented.size
            converted = oriented.convert("RGB")
            try:
                source_rgb = np.array(converted, dtype=np.uint8, copy=True)
            finally:
                converted.close()
            max_edge = max(original_size)
            if max_edge > args.photo_max_edge:
                scale = args.photo_max_edge / max_edge
                width = max(2, int(math.floor(original_size[0] * scale + 0.5)))
                height = max(2, int(math.floor(original_size[1] * scale + 0.5)))
            else:
                width, height = original_size
            rgb = resize_rgb_cubic(source_rgb, width, height)
            if rgb is not source_rgb:
                del source_rgb
        finally:
            if oriented is not opened:
                oriented.close()

    raw_depth = np.load(args.validation_depth, mmap_mode="r", allow_pickle=False)
    if raw_depth.dtype != np.float32 or raw_depth.ndim != 2:
        raise ValueError("validation depth must be a 2-D float32 .npy array")
    depth = normalize_depth_map(raw_depth)
    if depth.shape != (original_size[1], original_size[0]):
        depth = resize_bilinear_align_corners(depth, original_size[1], original_size[0])
    if depth.shape != rgb.shape[:2]:
        depth = resize_bilinear_align_corners(depth, rgb.shape[0], rgb.shape[1])
    focus = focus_at_point(depth, args.focus_x, args.focus_y)
    coc = full_frame_coc(depth, focus, 2.8)
    defocus = np.empty_like(coc)
    np.divide(coc, np.float32(10.0), out=defocus)
    np.clip(defocus, -1.0, 1.0, out=defocus)
    image_tensor = np.ascontiguousarray(
        np.transpose(rgb.astype(np.float32) / np.float32(255.0), (2, 0, 1))[None]
    )
    defocus_tensor = defocus[None, None]
    return (
        torch.from_numpy(image_tensor),
        torch.from_numpy(defocus_tensor),
        {
            "image_sha256": _sha256(args.validation_photo),
            "depth_sha256": _sha256(args.validation_depth),
            "original_width": original_size[0],
            "original_height": original_size[1],
            "validation_width": int(rgb.shape[1]),
            "validation_height": int(rgb.shape[0]),
            "focus": focus,
            "aperture": 2.8,
            "defocus_range": [float(defocus.min()), float(defocus.max())],
        },
    )


def _check_case(
    model: Any,
    session: Any,
    values: tuple[Any, Any],
    torch: Any,
    np: Any,
    label: str,
) -> dict[str, Any]:
    with torch.inference_mode():
        expected = model(*values).detach().cpu().numpy()
    actual = session.run(
        ["output"],
        {
            "image": values[0].detach().cpu().numpy(),
            "defocus": values[1].detach().cpu().numpy(),
        },
    )[0]
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise RuntimeError(f"invalid ONNX output for {label}")
    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    byte_delta = np.abs(
        (np.clip(actual, 0, 1) * 255).astype(np.uint8).astype(np.int16)
        - (np.clip(expected, 0, 1) * 255).astype(np.uint8).astype(np.int16)
    )
    mse = float(np.mean(byte_delta.astype(np.float64) ** 2))
    report = {
        "case": label,
        "shape": list(actual.shape),
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "psnr_8bit_db": 120.0 if mse == 0 else 10.0 * math.log10(255.0**2 / mse),
        "max_abs_8bit": int(byte_delta.max(initial=0)),
        "fraction_over_2_8bit": float(np.mean(byte_delta > 2)),
    }
    return report


def main() -> int:
    args = _make_parser().parse_args()
    upstream_root = args.upstream_root.resolve()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    if not upstream_root.is_dir():
        raise SystemExit("--upstream-root must be an existing directory")
    if output_dir.exists():
        raise SystemExit("--output-dir must not exist; refusing to overwrite files")
    source_file = upstream_root / "neural_renderer.py"
    if not source_file.is_file() or _sha256(source_file) != SOURCE_FILE_SHA256:
        raise SystemExit(
            "neural_renderer.py does not match the pinned BokehMe source revision"
        )
    if not checkpoint.is_file() or _sha256(checkpoint) != CHECKPOINT_SHA256:
        raise SystemExit("checkpoint SHA-256 does not match the pinned ARNet weights")
    if (args.validation_photo is None) != (args.validation_depth is None):
        raise SystemExit("provide both --validation-photo and --validation-depth")

    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(upstream_root))
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from neural_renderer import ARNet
    from onnxconverter_common import float16
    from torch.nn import functional as F

    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint_data = _safe_load_checkpoint(torch, np, checkpoint)
    network = ARNet()
    network.load_state_dict(checkpoint_data["model"], strict=True)
    network.eval()

    class ExportBokeh(torch.nn.Module):
        def __init__(self, arnet: Any) -> None:
            super().__init__()
            self.net = arnet

        def forward(self, image: Any, defocus: Any) -> Any:
            height, width = image.shape[-2:]
            even_height = height // 2 * 2
            even_width = width // 2 * 2
            values = F.interpolate(
                torch.cat((image, defocus), dim=1),
                size=(even_height, even_width),
                mode="bilinear",
                align_corners=True,
            )
            batch, channels, even_height, even_width = values.shape
            values = (
                values.reshape(
                    batch,
                    channels,
                    even_height // 2,
                    2,
                    even_width // 2,
                    2,
                )
                .permute(0, 1, 3, 5, 2, 4)
                .reshape(batch, channels * 4, even_height // 2, even_width // 2)
            )
            gamma = torch.ones_like(values[:, :1]) * 0.3
            values = torch.cat((values, gamma), dim=1)
            values = self.net.upsample(
                self.net.conv1(self.net.block_stack(self.net.conv0(values)))
            )
            values = F.interpolate(
                values, size=(height, width), mode="bilinear", align_corners=True
            )
            return values[:, :3]

    model = ExportBokeh(network).eval()
    cases = [
        (torch.rand(1, 3, height, width), torch.rand(1, 1, height, width) * 2 - 1)
        for height, width in ((64, 80), (97, 129), (480, 481))
    ]
    with torch.inference_mode():
        for image, defocus in cases:
            torch.testing.assert_close(
                model(image, defocus),
                network(image, defocus, 0.3)[0],
                rtol=0,
                atol=0,
            )

    output_dir.mkdir(parents=True)
    fp32_path = output_dir / "legendary-1.0-arnet.fp32.onnx"
    fp16_path = output_dir / "legendary-1.0-arnet.fp16.onnx"
    report_path = output_dir / "export-report.json"
    cases_for_validation = list(cases)
    photo_metadata = None
    if args.validation_photo is not None:
        photo_case = _photo_case(args, torch, np)
        cases_for_validation.append(photo_case[:2])
        photo_metadata = photo_case[2]

    with torch.inference_mode():
        torch.onnx.export(
            model,
            cases[0],
            str(fp32_path),
            input_names=["image", "defocus"],
            output_names=["output"],
            dynamic_axes={
                "image": {2: "height", 3: "width"},
                "defocus": {2: "height", 3: "width"},
                "output": {2: "height", 3: "width"},
            },
            opset_version=17,
            dynamo=False,
            do_constant_folding=True,
        )
    graph = onnx.load(str(fp32_path))
    onnx.checker.check_model(graph)
    half_graph = float16.convert_float_to_float16(graph, keep_io_types=True)
    onnx.checker.check_model(half_graph)
    onnx.save(half_graph, str(fp16_path))

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    session = ort.InferenceSession(
        str(fp16_path), options, providers=["CPUExecutionProvider"]
    )
    checks = []
    for index, values in enumerate(cases_for_validation):
        label = f"synthetic-{index + 1}" if index < 3 else "real-photo"
        check = _check_case(model, session, values, torch, np, label)
        check["passed"] = (
            check["psnr_8bit_db"] >= 40.0 and check["fraction_over_2_8bit"] <= 0.01
        )
        checks.append(check)
    if not all(check["passed"] for check in checks):
        raise RuntimeError("FP16 ONNX parity validation failed")

    onnx_sha256 = _sha256(fp16_path)
    export_report = {
        "model_id": "bokehme-arnet-fp16",
        "source_revision": SOURCE_REVISION,
        "source_file_sha256": SOURCE_FILE_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "opset": 17,
        "torch_version": torch.__version__,
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "numpy_version": np.__version__,
        "dynamic_shapes": True,
        "provider": "CPUExecutionProvider",
        "photo_validation": photo_metadata,
        "checks": checks,
        "onnx_sha256": onnx_sha256,
        "onnx_bytes": fp16_path.stat().st_size,
        "passed": True,
    }
    report_path.write_text(
        json.dumps(export_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validation_path = Path(f"{fp16_path}.validation.json")
    validation_path.write_text(
        json.dumps(
            {
                "model_id": "bokehme-arnet-fp16",
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "onnx_sha256": onnx_sha256,
                "dynamic_shapes": True,
                "passed": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(export_report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
