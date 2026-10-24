"""Verified optional split ARNet graph assets; no ONNX dependency at runtime."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SPLIT_GRAPH_NAMES = (
    "arnet-prefix.onnx",
    "arnet-core.onnx",
    "arnet-suffix.onnx",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_split_graph_dir(
    graph_dir: Path, source_model_sha256: str
) -> tuple[Path, Path, Path]:
    """Refuse unpinned or modified split-graph assets."""
    manifest_path = graph_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("ARNet split graph manifest missing or invalid") from exc
    if (
        manifest.get("schema_version") != 1
        or manifest.get("source_sha256") != source_model_sha256
    ):
        raise ValueError("ARNet split graphs do not match verified source ONNX")
    entries = manifest.get("graphs")
    if not isinstance(entries, dict) or set(entries) != set(SPLIT_GRAPH_NAMES):
        raise ValueError("ARNet split graph manifest has invalid entries")
    paths = tuple(graph_dir / name for name in SPLIT_GRAPH_NAMES)
    for name, path in zip(SPLIT_GRAPH_NAMES, paths, strict=True):
        digest = entries[name]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or sha256_file(path) != digest
        ):
            raise ValueError(f"ARNet split graph checksum mismatch: {name}")
    return paths
