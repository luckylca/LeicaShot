"""Explicit, hash-checked acquisition of model artifacts outside Git history."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO
from urllib.request import Request, urlopen

MANIFEST_PATH = Path(__file__).resolve().parent / "data" / "model_manifest.json"
CHUNK_BYTES = 1024 * 1024
USER_AGENT = "LaikaModelAssetManager/0.1"


class ModelAssetError(ValueError):
    """A model manifest, artifact, or integrity check is invalid."""


@dataclass(frozen=True)
class ModelArtifact:
    model_id: str
    name: str
    filename: str
    source_url: str
    expected_bytes: int
    sha256: str
    license: str
    status: str

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> ModelArtifact:
        required = (
            "id",
            "name",
            "artifact",
            "source_url",
            "expected_bytes",
            "sha256",
            "license",
            "status",
        )
        if any(key not in record for key in required):
            raise ModelAssetError("model manifest entry is missing required fields")
        filename = str(record["artifact"])
        if Path(filename).name != filename or filename in {".", ".."}:
            raise ModelAssetError("model artifact must be a plain filename")
        size = record["expected_bytes"]
        digest = str(record["sha256"]).lower()
        if not isinstance(size, int) or size <= 0:
            raise ModelAssetError("expected model size must be a positive integer")
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ModelAssetError(
                "model SHA-256 must contain 64 hexadecimal characters"
            )
        return cls(
            model_id=str(record["id"]),
            name=str(record["name"]),
            filename=filename,
            source_url=str(record["source_url"]),
            expected_bytes=size,
            sha256=digest,
            license=str(record["license"]),
            status=str(record["status"]),
        )


def load_manifest(
    path: str | os.PathLike[str] = MANIFEST_PATH,
) -> dict[str, ModelArtifact]:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelAssetError("cannot read model manifest") from exc
    if document.get("schema_version") != 1 or not isinstance(
        document.get("models"), list
    ):
        raise ModelAssetError("unsupported model manifest schema")
    artifacts = [ModelArtifact.from_record(record) for record in document["models"]]
    result = {artifact.model_id: artifact for artifact in artifacts}
    if len(result) != len(artifacts):
        raise ModelAssetError("model IDs must be unique")
    return result


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact(path: str | os.PathLike[str], artifact: ModelArtifact) -> None:
    candidate = Path(path)
    try:
        size = candidate.stat().st_size
    except OSError as exc:
        raise ModelAssetError(f"model artifact is unavailable: {candidate}") from exc
    if size != artifact.expected_bytes:
        raise ModelAssetError(
            f"model size mismatch for {artifact.model_id}: "
            f"expected {artifact.expected_bytes}, got {size}"
        )
    actual_sha256 = sha256_file(candidate)
    if actual_sha256 != artifact.sha256:
        raise ModelAssetError(
            f"model SHA-256 mismatch for {artifact.model_id}: {actual_sha256}"
        )


def _copy_verified_stream(
    response: BinaryIO,
    temporary_path: Path,
    artifact: ModelArtifact,
) -> None:
    digest = hashlib.sha256()
    total_bytes = 0
    with temporary_path.open("wb") as destination:
        while chunk := response.read(CHUNK_BYTES):
            total_bytes += len(chunk)
            if total_bytes > artifact.expected_bytes:
                raise ModelAssetError("download exceeds the manifest byte count")
            digest.update(chunk)
            destination.write(chunk)
        destination.flush()
        os.fsync(destination.fileno())
    if total_bytes != artifact.expected_bytes:
        raise ModelAssetError(
            "download size mismatch: "
            f"expected {artifact.expected_bytes}, got {total_bytes}"
        )
    if digest.hexdigest() != artifact.sha256:
        raise ModelAssetError("download SHA-256 does not match the model manifest")


def download_artifact(
    artifact: ModelArtifact,
    destination: str | os.PathLike[str],
    *,
    opener: Callable[..., BinaryIO] | None = None,
) -> Path:
    """Download one pinned artifact and publish it only after size/hash checks."""
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        verify_artifact(output, artifact)
        return output

    request = Request(
        artifact.source_url,
        headers={"User-Agent": USER_AGENT},
    )
    stream_opener = opener or urlopen
    temporary_path: Path | None = None
    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output.name}.",
            suffix=".partial",
            dir=output.parent,
        )
        os.close(file_descriptor)
        temporary_path = Path(temporary_name)
        with stream_opener(request, timeout=30) as response:
            _copy_verified_stream(response, temporary_path, artifact)
        # Refuse to replace anything created concurrently after the initial check.
        if output.exists():
            verify_artifact(output, artifact)
            temporary_path.unlink(missing_ok=True)
            return output
        os.replace(temporary_path, output)
        temporary_path = None
        return output
    except ModelAssetError:
        raise
    except Exception as exc:
        raise ModelAssetError(f"failed to download {artifact.model_id}") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _default_cache_dir() -> Path:
    configured = os.environ.get("LAIKA_MODEL_CACHE")
    return (
        Path(configured).expanduser()
        if configured
        else Path.home() / ".cache/laika/models"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="laika-models")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="list pinned model artifact metadata")
    fetch_parser = subparsers.add_parser("fetch", help="download and verify a model")
    fetch_parser.add_argument("model_id")
    fetch_parser.add_argument("--cache-dir", type=Path, default=_default_cache_dir())
    verify_parser = subparsers.add_parser("verify", help="verify a local model file")
    verify_parser.add_argument("model_id")
    verify_parser.add_argument("path", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        artifacts = load_manifest()
        if args.command == "list":
            print(
                json.dumps(
                    [artifact.__dict__ for artifact in artifacts.values()],
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        try:
            artifact = artifacts[args.model_id]
        except KeyError as exc:
            raise ModelAssetError(f"unknown model ID: {args.model_id}") from exc
        if args.command == "fetch":
            path = download_artifact(artifact, args.cache_dir / artifact.filename)
        else:
            verify_artifact(args.path, artifact)
            path = args.path
        print(json.dumps({"model_id": artifact.model_id, "path": str(path)}))
        return 0
    except (ModelAssetError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
