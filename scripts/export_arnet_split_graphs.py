"""Export the three independently executed parts of validated ARNet.

Requires optional developer dependency onnx, not the serving runtime.
These assets are derived from the caller's locally authorized model weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from laika.arnet_split import SPLIT_GRAPH_NAMES, sha256_file

SLICES = (
    (["image", "defocus"], ["/Concat_5_output_0"]),
    (["/Concat_5_output_0"], ["/upsample/DepthToSpace_output_0"]),
    (["image", "/upsample/DepthToSpace_output_0"], ["output"]),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import onnx

    source = args.model.resolve(strict=True)
    destination = args.output_dir.resolve()
    if destination.exists() and any(destination.iterdir()):
        parser.error("output-dir must be empty to prevent stale or mixed graphs")
    destination.mkdir(parents=True, exist_ok=True)

    digest = sha256_file(source)
    hashes: dict[str, str] = {}
    for name, (inputs, outputs) in zip(SPLIT_GRAPH_NAMES, SLICES, strict=True):
        target = destination / name
        onnx.utils.extract_model(str(source), str(target), inputs, outputs)
        onnx.checker.check_model(onnx.load(str(target)))
        hashes[name] = sha256_file(target)
    manifest = {
        "schema_version": 1,
        "source_sha256": digest,
        "graphs": hashes,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Extracted {len(hashes)} ARNet split graphs at {destination}")


if __name__ == "__main__":
    main()
