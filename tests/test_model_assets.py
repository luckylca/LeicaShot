from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import pytest

from laika.model_assets import (
    MANIFEST_PATH,
    ModelArtifact,
    ModelAssetError,
    download_artifact,
    load_manifest,
    verify_artifact,
)


def _artifact(payload: bytes) -> ModelArtifact:
    return ModelArtifact(
        model_id="test-model",
        name="Test model",
        filename="weights.bin",
        source_url="https://models.example.invalid/weights.bin",
        expected_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        license="test-only",
        status="test-only",
    )


def test_manifest_pins_depth_small_without_bundling_weights() -> None:
    artifacts = load_manifest()
    artifact = artifacts["depth-anything-v2-small"]
    project_manifest = Path(__file__).parents[1] / "models" / "manifest.json"

    assert project_manifest.read_bytes() == MANIFEST_PATH.read_bytes()
    assert artifact.expected_bytes == 99_218_434
    assert (
        artifact.sha256
        == "715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378"
    )
    assert artifact.license == "Apache-2.0"
    assert not (Path(__file__).parents[1] / "models" / artifact.filename).exists()


def test_verified_download_is_atomic_and_reusable(tmp_path: Path) -> None:
    payload = b"model-bytes-for-test"
    artifact = _artifact(payload)
    destination = tmp_path / artifact.filename
    opened: list[object] = []

    def fake_opener(_request: object, *, timeout: int) -> BytesIO:
        opened.append(timeout)
        return BytesIO(payload)

    downloaded = download_artifact(artifact, destination, opener=fake_opener)
    assert downloaded.read_bytes() == payload
    assert opened == [30]

    # An already-correct cache entry is verified and reused without a request.
    reused = download_artifact(
        artifact,
        destination,
        opener=lambda *_args, **_kwargs: pytest.fail("unexpected network request"),
    )
    assert reused == destination
    verify_artifact(reused, artifact)


@pytest.mark.parametrize("payload", [b"too-short", b"same-size-wrong-hash"])
def test_failed_download_never_publishes_partial_bytes(
    tmp_path: Path, payload: bytes
) -> None:
    expected = b"expected-model-bytes"
    artifact = _artifact(expected)

    with pytest.raises(ModelAssetError):
        download_artifact(
            artifact,
            tmp_path / artifact.filename,
            opener=lambda *_args, **_kwargs: BytesIO(payload),
        )

    assert not (tmp_path / artifact.filename).exists()
    assert list(tmp_path.iterdir()) == []


def test_corrupt_existing_artifact_is_not_overwritten(tmp_path: Path) -> None:
    artifact = _artifact(b"good")
    destination = tmp_path / artifact.filename
    destination.write_bytes(b"bad!")

    with pytest.raises(ModelAssetError, match="SHA-256 mismatch"):
        download_artifact(
            artifact,
            destination,
            opener=lambda *_args, **_kwargs: pytest.fail("must not overwrite"),
        )

    assert destination.read_bytes() == b"bad!"
