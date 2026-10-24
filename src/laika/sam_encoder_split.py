"""Manifest-verified optional two-graph SAM encoder inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SAM_SPLIT_CUT_INDEX = 1148
SAM_SPLIT_BRIDGE_NAMES = (
    "/image_encoder/trunk/Transpose_1_output_0",
    "/image_encoder/trunk/Transpose_2_output_0",
    "/image_encoder/trunk/blocks.5/Add_1_output_0",
)
SAM_SPLIT_FILENAMES = ("sam-prefix.onnx", "sam-suffix.onnx")
SAM_SPLIT_OUTPUTS = ("image_features_0", "image_features_1", "image_embeddings")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sam_split_graphs(
    directory: Path, original_encoder_sha256: str
) -> tuple[Path, Path]:
    """Reject absent, stale or altered subgraphs before session creation."""
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("SAM split manifest missing or invalid") from exc
    if (
        manifest.get("schema_version") != 1
        or manifest.get("source_encoder_sha256") != original_encoder_sha256
        or manifest.get("cut_index") != SAM_SPLIT_CUT_INDEX
        or manifest.get("bridge_names") != list(SAM_SPLIT_BRIDGE_NAMES)
    ):
        raise ValueError("SAM split graph source or cut contract mismatch")
    files = manifest.get("graphs")
    if not isinstance(files, dict) or set(files) != set(SAM_SPLIT_FILENAMES):
        raise ValueError("SAM split manifest has invalid graph list")
    paths = tuple(directory / name for name in SAM_SPLIT_FILENAMES)
    for name, path in zip(SAM_SPLIT_FILENAMES, paths, strict=True):
        expected = files[name]
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or sha256_file(path) != expected
        ):
            raise ValueError(f"SAM split graph checksum mismatch: {name}")
    return paths


class TwoPartSamEncoder:
    """ONNX-session facade presenting the original SAM encoder contract."""

    def __init__(self, prefix_session: Any, suffix_session: Any) -> None:
        self.prefix = prefix_session
        self.suffix = suffix_session

    def get_inputs(self) -> Any:
        return self.prefix.get_inputs()

    def get_outputs(self) -> Any:
        return self.suffix.get_outputs()

    def run(self, outputs: list[str], feed: dict[str, Any]) -> list[Any]:
        bridge = self.prefix.run(list(SAM_SPLIT_BRIDGE_NAMES), feed)
        encoded = self.suffix.run(
            list(SAM_SPLIT_OUTPUTS),
            dict(zip(SAM_SPLIT_BRIDGE_NAMES, bridge, strict=True)),
        )
        by_name = dict(zip(SAM_SPLIT_OUTPUTS, encoded, strict=True))
        return [by_name[name] for name in outputs]
