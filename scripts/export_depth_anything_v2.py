#!/usr/bin/env python3
"""Export pinned Depth Anything V2 Small weights to a validated CPU ONNX model.

The upstream Depth Anything V2 checkout is used only as the model definition.
It must be at the exact source revision pinned in models/manifest.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from types import MethodType
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


def _install_dynamic_shape_adapters(model: Any, torch: Any) -> Any:
    """Keep positional resize and DPT output sizing tied to runtime dimensions."""
    from torch.nn import functional as functional

    class PositionalResize(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, patch: Any, height: Any, width: Any) -> Any:
            patch_height, patch_width = int(height) // 14, int(width) // 14
            if patch_height == 37 and patch_width == 37:
                return patch
            return functional.interpolate(
                patch,
                scale_factor=((patch_height + 0.1) / 37, (patch_width + 0.1) / 37),
                mode="bicubic",
                align_corners=False,
            )

        @staticmethod
        def symbolic(graph: Any, patch: Any, height: Any, width: Any) -> Any:
            def constant(value: Any, dtype: Any) -> Any:
                return graph.op("Constant", value_t=torch.tensor(value, dtype=dtype))

            patch_height = graph.op("Div", height, constant(14, torch.int64))
            patch_width = graph.op("Div", width, constant(14, torch.int64))
            base_grid = constant(37, torch.int64)
            is_base_grid = graph.op(
                "And",
                graph.op("Equal", patch_height, base_grid),
                graph.op("Equal", patch_width, base_grid),
            )
            offset = graph.op(
                "Where",
                is_base_grid,
                constant(0.0, torch.float32),
                constant(0.1, torch.float32),
            )

            def scale(dimension: Any) -> Any:
                value = graph.op(
                    "Div",
                    graph.op("Add", graph.op("Cast", dimension, to_i=1), offset),
                    constant(37.0, torch.float32),
                )
                return graph.op("Unsqueeze", value, constant([0], torch.int64))

            scales = graph.op(
                "Concat",
                constant([1.0, 1.0], torch.float32),
                scale(patch_height),
                scale(patch_width),
                axis_i=0,
            )
            return graph.op(
                "Resize",
                patch,
                constant([], torch.float32),
                scales,
                mode_s="cubic",
                coordinate_transformation_mode_s="half_pixel",
                cubic_coeff_a_f=-0.75,
            )

    def dynamic_position_resize(
        module: Any, tokens: Any, input_height: Any, input_width: Any
    ) -> Any:
        if not torch.onnx.is_in_onnx_export():
            return original_position_resize(tokens, input_height, input_width)
        position = module.pos_embed.float()
        class_position = position[:, :1]
        patch_position = position[:, 1:].reshape(1, 37, 37, module.embed_dim)
        patch_position = patch_position.permute(0, 3, 1, 2)
        patch_position = PositionalResize.apply(
            patch_position, input_height, input_width
        )
        patch_position = patch_position.permute(0, 2, 3, 1).reshape(
            1, -1, module.embed_dim
        )
        return torch.cat((class_position, patch_position), dim=1).to(tokens.dtype)

    def dynamic_dpt_head(
        head: Any, out_features: Any, patch_height: Any, patch_width: Any
    ) -> Any:
        projected = []
        for index, feature in enumerate(out_features):
            if head.use_clstoken:
                feature, class_token = feature[0], feature[1]
                readout = class_token.unsqueeze(1).expand_as(feature)
                feature = head.readout_projects[index](
                    torch.cat((feature, readout), dim=-1)
                )
            else:
                feature = feature[0]
            feature = feature.permute(0, 2, 1).reshape(
                feature.shape[0], feature.shape[-1], patch_height, patch_width
            )
            projected.append(head.resize_layers[index](head.projects[index](feature)))

        layer_1, layer_2, layer_3, layer_4 = projected
        scratch = head.scratch
        layer_1 = scratch.layer1_rn(layer_1)
        layer_2 = scratch.layer2_rn(layer_2)
        layer_3 = scratch.layer3_rn(layer_3)
        layer_4 = scratch.layer4_rn(layer_4)
        path_4 = scratch.refinenet4(layer_4, size=layer_3.shape[2:])
        path_3 = scratch.refinenet3(path_4, layer_3, size=layer_2.shape[2:])
        path_2 = scratch.refinenet2(path_3, layer_2, size=layer_1.shape[2:])
        path_1 = scratch.refinenet1(path_2, layer_1)
        output = scratch.output_conv1(path_1)
        output = functional.interpolate(
            output,
            size=(patch_height * 14, patch_width * 14),
            mode="bilinear",
            align_corners=True,
        )
        return scratch.output_conv2(output)

    original_position_resize = model.pretrained.interpolate_pos_encoding
    original_head_forward = model.depth_head.forward
    model.pretrained.interpolate_pos_encoding = MethodType(
        dynamic_position_resize, model.pretrained
    )
    model.depth_head.forward = MethodType(dynamic_dpt_head, model.depth_head)
    return original_head_forward


def _install_global_query_chunking(
    model: Any, torch: Any, chunk_size: int
) -> dict[Any, Any]:
    """Compute exact global attention in query chunks while retaining all keys."""
    if chunk_size < 1:
        raise ModelAssetError("attention query chunk size must be positive")

    scripted = torch.jit.CompilationUnit(
        """
def chunked_global_attention(
    q: Tensor, k: Tensor, v: Tensor, scale: float, chunk_size: int
) -> Tensor:
    chunks = torch.jit.annotate(List[Tensor], [])
    start = 0
    token_count = q.size(2)
    while start < token_count:
        end = min(start + chunk_size, token_count)
        query = q[:, :, start:end, :] * scale
        scores = torch.matmul(query, torch.transpose(k, -2, -1))
        probabilities = torch.softmax(scores, dim=-1)
        chunks.append(torch.matmul(probabilities, v))
        start = end
    return torch.cat(chunks, dim=2)
"""
    ).chunked_global_attention

    originals: dict[Any, Any] = {}
    for block in model.pretrained.blocks:
        attention = block.attn
        originals[attention] = attention.forward

        def chunked_forward(
            module: Any, values: Any, *, _attention: Any = attention
        ) -> Any:
            batch, tokens, channels = values.shape
            qkv = _attention.qkv(values).reshape(
                batch,
                tokens,
                3,
                _attention.num_heads,
                channels // _attention.num_heads,
            )
            qkv = qkv.permute(2, 0, 3, 1, 4)
            attended = scripted(qkv[0], qkv[1], qkv[2], _attention.scale, chunk_size)
            attended = attended.transpose(1, 2).reshape(batch, tokens, channels)
            return _attention.proj_drop(_attention.proj(attended))

        attention.forward = MethodType(chunked_forward, attention)
    return originals


def _restore_attention_forwards(originals: dict[Any, Any]) -> None:
    for attention, forward in originals.items():
        attention.forward = forward


def _dynamic_validation_shapes(height: int, width: int) -> list[tuple[int, int]]:
    candidates = [(height, width), (700, 532), (1400, 518), (518, 518)]
    return list(dict.fromkeys(candidates))


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
    if args.query_chunk_size is not None and not args.dynamic:
        raise ModelAssetError("query chunking currently requires --dynamic export")
    if args.query_chunk_size is not None and args.query_chunk_size < 1:
        raise ModelAssetError("attention query chunk size must be positive")
    if not math.isfinite(args.max_abs_error) or args.max_abs_error <= 0:
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

    if args.dynamic:
        original_head_forward = _install_dynamic_shape_adapters(model, torch)
        dynamic_head_forward = model.depth_head.forward
        shapes = _dynamic_validation_shapes(args.height, args.width)
    else:
        original_head_forward = None
        dynamic_head_forward = None
        shapes = [(args.height, args.width)]

    original_attention_forwards = (
        _install_global_query_chunking(model, torch, args.query_chunk_size)
        if args.query_chunk_size is not None
        else None
    )

    inputs: dict[tuple[int, int], Any] = {}
    references: dict[tuple[int, int], Any] = {}
    adapter_errors: list[float] = []
    attention_errors: list[float] = []
    for index, (height, width) in enumerate(shapes):
        if height < 518 or width < 518 or height % 14 or width % 14:
            raise ModelAssetError(
                "validation dimensions must be at least 518 and multiples of 14"
            )
        input_array = np.random.default_rng(args.seed + index).standard_normal(
            (1, 3, height, width), dtype=np.float32
        )
        inputs[(height, width)] = input_array
        input_tensor = torch.from_numpy(input_array)
        with torch.inference_mode():
            if original_attention_forwards is not None:
                chunked_output = model(input_tensor)
                _restore_attention_forwards(original_attention_forwards)
                reference_output = model(input_tensor)
                original_attention_forwards = _install_global_query_chunking(
                    model, torch, args.query_chunk_size
                )
                attention_delta = torch.abs(chunked_output - reference_output)
                attention_error = float(attention_delta.max())
                attention_errors.append(attention_error)
                if attention_error > args.max_abs_error:
                    raise ModelAssetError(
                        "query chunking changed PyTorch output: "
                        f"max_abs_error={attention_error:g}, "
                        f"limit={args.max_abs_error:g}"
                    )
            elif args.dynamic:
                reference_output = model(input_tensor)

            if args.dynamic:
                adapted_output = model(input_tensor)
                model.depth_head.forward = original_head_forward
                adapter_reference = model(input_tensor)
                model.depth_head.forward = dynamic_head_forward
                adapter_delta = torch.abs(adapted_output - adapter_reference)
                adapter_error = float(adapter_delta.max())
                adapter_errors.append(adapter_error)
                if adapter_error > 1e-6:
                    raise ModelAssetError(
                        "dynamic DPT adapter changed PyTorch output: "
                        f"max_abs_error={adapter_error:g}"
                    )
                if original_attention_forwards is not None:
                    reference_output = chunked_output
            elif original_attention_forwards is None:
                reference_output = model(input_tensor)
            references[(height, width)] = reference_output.cpu().numpy()

    export_array = inputs[(args.height, args.width)]
    input_tensor = torch.from_numpy(export_array)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".laika-onnx-", dir=output.parent) as tmp:
        temporary_model = Path(tmp) / output.name
        export_started = time.perf_counter()
        if args.dynamic:
            torch.onnx.export(
                model,
                (input_tensor,),
                temporary_model,
                input_names=["image"],
                output_names=["depth"],
                opset_version=17,
                dynamo=False,
                external_data=True,
                do_constant_folding=True,
                dynamic_axes={
                    "image": {2: "height", 3: "width"},
                    "depth": {1: "height", 2: "width"},
                },
            )
        else:
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
        exported_model = onnx.load(str(temporary_model), load_external_data=False)
        loop_nodes = sum(node.op_type == "Loop" for node in exported_model.graph.node)
        expected_loop_nodes = (
            len(model.pretrained.blocks) if args.query_chunk_size is not None else 0
        )
        if loop_nodes != expected_loop_nodes:
            raise ModelAssetError(
                "ONNX attention loop count mismatch: "
                f"expected {expected_loop_nodes}, found {loop_nodes}"
            )
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
        checks = []
        inference_seconds = 0.0
        inference_started = time.perf_counter()
        for shape in shapes:
            actual = session.run(["depth"], {"image": inputs[shape]})[0]
            inference_seconds += time.perf_counter() - inference_started
            inference_started = time.perf_counter()
            reference = references[shape]
            error = np.abs(reference - actual)
            max_error = float(error.max())
            if (
                actual.shape != reference.shape
                or not np.isfinite(actual).all()
                or max_error > args.max_abs_error
            ):
                raise ModelAssetError(
                    "ONNX output failed the numerical check for "
                    f"{shape}: max_abs_error={max_error:g}, "
                    f"limit={args.max_abs_error:g}"
                )
            checks.append(
                {
                    "input_shape": list(inputs[shape].shape),
                    "output_shape": list(actual.shape),
                    "max_absolute_error": max_error,
                    "mean_absolute_error": float(error.mean()),
                    "p99_absolute_error": float(np.quantile(error, 0.99)),
                    "all_finite": bool(np.isfinite(actual).all()),
                }
            )
        error_max = max(row["max_absolute_error"] for row in checks)
        error_mean = max(row["mean_absolute_error"] for row in checks)
        error_p99 = max(row["p99_absolute_error"] for row in checks)
        temporary_data = Path(f"{temporary_model}.data")
        has_external_data = temporary_data.is_file()
        report: dict[str, Any] = {
            "model_id": MODEL_ID,
            "source_code_revision": artifact_record["source_code_revision"],
            "weights_sha256": artifact.sha256,
            "torch_version": torch.__version__,
            "onnx_version": onnx.__version__,
            "onnxruntime_version": ort.__version__,
            "provider": session.get_providers(),
            "threads": args.threads,
            "opset": 17 if args.dynamic else OPSET,
            "dynamic_shapes": args.dynamic,
            "attention_query_chunk_size": args.query_chunk_size,
            "onnx_top_level_loop_nodes": loop_nodes,
            "expected_attention_loop_nodes": expected_loop_nodes,
            "input_shapes_tested": [list(inputs[shape].shape) for shape in shapes],
            "output_shapes_tested": [row["output_shape"] for row in checks],
            "checks": checks,
            "dynamic_dpt_adapter_max_absolute_error": max(adapter_errors, default=0.0),
            "query_chunking_max_absolute_error": max(attention_errors, default=0.0),
            "max_absolute_error": error_max,
            "mean_absolute_error": error_mean,
            "p99_absolute_error": error_p99,
            "max_absolute_error_limit": args.max_abs_error,
            "all_finite": bool(np.isfinite(actual).all()),
            "model_load_seconds": load_seconds,
            "onnx_export_seconds": export_seconds,
            "ort_session_create_seconds": session_create_seconds,
            "ort_inference_seconds": inference_seconds,
            "onnx_sha256": _sha256(temporary_model),
            "external_data_sha256": (
                _sha256(temporary_data) if has_external_data else None
            ),
            "onnx_bytes": temporary_model.stat().st_size,
            "external_data_bytes": (
                temporary_data.stat().st_size if has_external_data else 0
            ),
            "limitations": [
                (
                    "Exported input height and width are dynamic."
                    if args.dynamic
                    else "Exported input height and width are static."
                ),
                "Input preprocessing and final source-size interpolation "
                "are not included.",
                "This validates seeded synthetic tensors, not photo quality "
                "or full-pipeline memory.",
            ],
        }
        temporary_report = Path(tmp) / report_output.name
        temporary_report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        # Publish external tensors and report before the ONNX graph, which is the
        # final marker that the complete artifact set is ready to load.
        if has_external_data:
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
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="export and validate dynamic height/width input shapes",
    )
    parser.add_argument(
        "--query-chunk-size",
        type=int,
        help="compute full-global attention for this many queries per chunk",
    )
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
