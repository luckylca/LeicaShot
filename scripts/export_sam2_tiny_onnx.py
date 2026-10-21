#!/usr/bin/env python3
"""Export and numerically validate split SAM 2.1 Tiny CPU ONNX graphs.

The script imports the upstream SAM 2 package from the exact source revision
recorded in the Laika model manifest. It keeps image encoding separate from
prompt decoding so a caller can cache image features across repeated prompts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from laika.model_assets import ModelAssetError, load_manifest, verify_artifact
from laika.subject import SAM_FEATURE_OUTPUTS, Sam2TinyOnnx


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_revision(source_checkout: Path) -> str:
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(source_checkout), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.check_output(
            [
                "git",
                "-C",
                str(source_checkout),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if dirty:
            raise ValueError("SAM 2 source checkout has tracked local changes")
        return revision
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            "source checkout must be a clean SAM 2 Git repository"
        ) from exc


def _metric(reference: Any, actual: Any) -> dict[str, float]:
    delta = np.abs(
        np.asarray(reference, dtype=np.float32) - np.asarray(actual, dtype=np.float32)
    )
    if not np.isfinite(delta).all():
        raise ValueError("PyTorch/ONNX comparison contains non-finite values")
    return {
        "max_abs": float(delta.max(initial=0.0)),
        "mean_abs": float(delta.mean()),
        "p99_abs": float(np.percentile(delta, 99)),
    }


def _build_export_wrappers(model: Any, torch: Any) -> tuple[Any, Any]:
    class ImageEncoder(torch.nn.Module):
        def __init__(self, sam_model: Any) -> None:
            super().__init__()
            self.model = sam_model

        def forward(self, image: Any) -> tuple[Any, Any, Any]:
            backbone = self.model.forward_image(image)
            _, features, _, sizes = self.model._prepare_backbone_features(backbone)
            if self.model.directly_add_no_mem_embed:
                features[-1] = features[-1] + self.model.no_mem_embed
            maps = tuple(
                feature.permute(1, 2, 0).reshape(1, -1, height, width)
                for feature, (height, width) in zip(features, sizes, strict=True)
            )
            return maps[0], maps[1], maps[2]

    class PromptDecoder(torch.nn.Module):
        def __init__(self, sam_model: Any) -> None:
            super().__init__()
            self.model = sam_model

        def forward(
            self,
            image_features_0: Any,
            image_features_1: Any,
            image_embeddings: Any,
            point_coords: Any,
            point_labels: Any,
        ) -> tuple[Any, Any]:
            sparse, dense = self.model.sam_prompt_encoder(
                points=(point_coords, point_labels), boxes=None, masks=None
            )
            logits, scores, *_ = self.model.sam_mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=True,
                repeat_image=False,
                high_res_features=[image_features_0, image_features_1],
            )
            return logits, scores

    return ImageEncoder(model).eval(), PromptDecoder(model).eval()


def export_and_validate(
    *,
    source_checkout: Path,
    weights: Path,
    output_dir: Path,
    threads: int,
    max_abs_tolerance: float,
) -> dict[str, Any]:
    import onnx
    import onnxruntime as ort
    import torch

    artifacts = load_manifest()
    artifact = artifacts["sam2.1-hiera-tiny"]
    verify_artifact(weights, artifact)
    source_revision = _source_revision(source_checkout)
    expected_source_revision = artifact.source_code_revision
    if expected_source_revision is None:
        raise ValueError("SAM checkpoint manifest is missing its source code revision")
    if source_revision != expected_source_revision:
        raise ValueError(
            "SAM 2 source revision mismatch: "
            f"expected {expected_source_revision}, "
            f"got {source_revision}"
        )
    sys.path.insert(0, str(source_checkout.resolve()))
    from sam2.build_sam import build_sam2

    if threads < 1 or max_abs_tolerance <= 0:
        raise ValueError("threads and numerical tolerance must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    image_encoder_path = output_dir / "sam2.1_hiera_tiny_image_encoder.onnx"
    image_decoder_path = output_dir / "sam2.1_hiera_tiny_image_decoder.onnx"
    report_path = output_dir / "sam2.1_hiera_tiny.validation.json"
    if any(
        path.exists() for path in (image_encoder_path, image_decoder_path, report_path)
    ):
        raise FileExistsError("refusing to overwrite an existing SAM export artifact")

    torch.set_num_threads(threads)
    torch.manual_seed(42)
    started = time.perf_counter()
    model = build_sam2(
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        str(weights),
        device="cpu",
        mode="eval",
        apply_postprocessing=True,
    )
    model.eval()
    encoder, decoder = _build_export_wrappers(model, torch)
    setup_seconds = time.perf_counter() - started

    # Synthetic but image-like input: RGB values normalized with SAM 2's
    # ImageNet statistics. It is a graph parity fixture, not a photo-quality set.
    rgb = torch.rand((1, 3, 1024, 1024), dtype=torch.float32)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    image = (rgb - mean) / std
    coords = torch.tensor([[[320.0, 420.0], [610.0, 380.0]]], dtype=torch.float32)
    labels = torch.tensor([[1, 0]], dtype=torch.int32)

    with torch.inference_mode():
        framework_features = encoder(image)
        framework_logits, framework_scores = decoder(
            *framework_features, coords, labels
        )

    torch.onnx.export(
        encoder,
        (image,),
        str(image_encoder_path),
        input_names=["image"],
        output_names=list(SAM_FEATURE_OUTPUTS),
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    torch.onnx.export(
        decoder,
        (*framework_features, coords, labels),
        str(image_decoder_path),
        input_names=[
            "image_features_0",
            "image_features_1",
            "image_embeddings",
            "point_coords",
            "point_labels",
        ],
        output_names=["low_res_masks", "iou_predictions"],
        dynamic_axes={
            "point_coords": {1: "point_count"},
            "point_labels": {1: "point_count"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx.checker.check_model(str(image_encoder_path))
    onnx.checker.check_model(str(image_decoder_path))

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    encoder_session = ort.InferenceSession(
        str(image_encoder_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    decoder_session = ort.InferenceSession(
        str(image_decoder_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    ort_image = image.numpy()
    ort_coords = coords.numpy()
    ort_labels = labels.numpy()
    ort_features = encoder_session.run(list(SAM_FEATURE_OUTPUTS), {"image": ort_image})
    ort_logits, ort_scores = decoder_session.run(
        ["low_res_masks", "iou_predictions"],
        {
            "image_features_0": ort_features[0],
            "image_features_1": ort_features[1],
            "image_embeddings": ort_features[2],
            "point_coords": ort_coords,
            "point_labels": ort_labels,
        },
    )

    errors: dict[str, Any] = {
        name: _metric(reference.numpy(), actual)
        for name, reference, actual in zip(
            SAM_FEATURE_OUTPUTS, framework_features, ort_features, strict=True
        )
    }
    errors["low_res_masks"] = _metric(framework_logits.numpy(), ort_logits)
    errors["iou_predictions"] = _metric(framework_scores.numpy(), ort_scores)
    max_error = max(item["max_abs"] for item in errors.values())
    if max_error > max_abs_tolerance:
        raise ValueError(
            f"SAM PyTorch/ONNX max error {max_error:.6g} exceeds "
            f"the limit {max_abs_tolerance:.6g}"
        )

    # Exercise Laika's actual adapter, including its reusable feature cache.
    with Sam2TinyOnnx(
        image_encoder_path,
        image_decoder_path,
        cache_max_bytes=32 * 1024 * 1024,
        intra_op_num_threads=threads,
    ) as laika_model:
        first = laika_model.predict(ort_image, ort_coords, ort_labels)
        second = laika_model.predict(ort_image, ort_coords, ort_labels)
    if first.cache_hit or first.encoder_seconds <= 0:
        raise AssertionError("fresh SAM image unexpectedly hit the embedding cache")
    if not second.cache_hit:
        raise AssertionError("repeated SAM prompt did not reuse image embeddings")
    adapter_errors = {
        "mask_logits": _metric(ort_logits, first.mask_logits),
        "scores": _metric(ort_scores, first.scores),
    }
    if any(metric["max_abs"] != 0.0 for metric in adapter_errors.values()):
        raise AssertionError("Laika adapter output differs from direct ONNX output")

    report = {
        "status": "passed",
        "model_id": artifact.model_id,
        "checkpoint_sha256": artifact.sha256,
        "checkpoint_bytes": artifact.expected_bytes,
        "upstream_source_revision": source_revision,
        "torch_version": torch.__version__,
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "intra_op_num_threads": threads,
        "input_shape": list(ort_image.shape),
        "prompt_count": int(ort_coords.shape[1]),
        "output_shapes": {
            "mask_logits": list(first.mask_logits.shape),
            "scores": list(first.scores.shape),
        },
        "max_abs_tolerance": max_abs_tolerance,
        "numerical_error": errors,
        "adapter_vs_direct_onnx_error": adapter_errors,
        "image_encoder_sha256": _sha256(image_encoder_path),
        "image_decoder_sha256": _sha256(image_decoder_path),
        "encoder_model_setup_seconds": setup_seconds,
        "adapter_encoder_seconds": first.encoder_seconds,
        "adapter_decoder_seconds": first.decoder_seconds,
        "adapter_cache_hit_decoder_seconds": second.decoder_seconds,
        "quality_status": "synthetic-tensor-parity-only",
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-abs-tolerance", type=float, default=1e-3)
    args = parser.parse_args()
    try:
        report = export_and_validate(
            source_checkout=args.source_checkout,
            weights=args.weights,
            output_dir=args.output_dir,
            threads=args.threads,
            max_abs_tolerance=args.max_abs_tolerance,
        )
    except (ModelAssetError, OSError, ValueError, ImportError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
