"""Export source-bound SAM 2.1 Tiny encoder halves from authorized ONNX.

The SAM Hiera export uses ONNX If subgraph captures, so the stock ONNX
extractor cannot safely collect its dependencies. Preserve graph order and
collect free initializers recursively instead. No checkpoint is distributed.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from laika.sam_encoder_split import (
    SAM_SPLIT_BRIDGE_NAMES,
    SAM_SPLIT_CUT_INDEX,
    SAM_SPLIT_FILENAMES,
    sha256_file,
)


def used_tensors(node: Any, onnx: Any) -> set[str]:
    used = set(node.input)
    for attr in node.attribute:
        if attr.type == onnx.AttributeProto.GRAPH:
            for child in attr.g.node:
                used.update(used_tensors(child, onnx))
        elif attr.type == onnx.AttributeProto.GRAPHS:
            for graph in attr.graphs:
                for child in graph.node:
                    used.update(used_tensors(child, onnx))
    return used


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import onnx

    source = args.encoder.resolve(strict=True)
    destination = args.output_dir.resolve()
    if destination.exists() and any(destination.iterdir()):
        parser.error("output-dir must be empty")
    model = onnx.load(str(source))
    onnx.checker.check_model(model)
    nodes = model.graph.node
    cut = SAM_SPLIT_CUT_INDEX
    producer = {name: i for i, node in enumerate(nodes) for name in node.output if name}
    uses: dict[str, list[int]] = {}
    for i, node in enumerate(nodes):
        for name in node.input:
            uses.setdefault(name, []).append(i)
    cross = {
        name
        for name, index in producer.items()
        if index <= cut and any(j > cut for j in uses.get(name, ()))
    }
    if cross != set(SAM_SPLIT_BRIDGE_NAMES):
        parser.error("ONNX graph does not match the fixed SAM cut contract")

    destination.mkdir(parents=True, exist_ok=True)
    bridged = [
        onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [None] * 4)
        for name in SAM_SPLIT_BRIDGE_NAMES
    ]
    hashes: dict[str, str] = {}
    parts = (
        (nodes[: cut + 1], model.graph.input, bridged),
        (nodes[cut + 1 :], bridged, model.graph.output),
    )
    for filename, (part_nodes, inputs, outputs) in zip(
        SAM_SPLIT_FILENAMES, parts, strict=True
    ):
        tensor_names = set().union(*(used_tensors(node, onnx) for node in part_nodes))
        initializers = [
            deepcopy(item)
            for item in model.graph.initializer
            if item.name in tensor_names
        ]
        graph = onnx.helper.make_graph(
            list(map(deepcopy, part_nodes)),
            filename,
            list(map(deepcopy, inputs)),
            list(map(deepcopy, outputs)),
            initializer=initializers,
        )
        derived = onnx.helper.make_model(
            graph,
            producer_name="leicashot-local-sam-exporter",
            ir_version=model.ir_version,
            opset_imports=deepcopy(model.opset_import),
        )
        onnx.checker.check_model(derived)
        path = destination / filename
        onnx.save(derived, str(path))
        hashes[filename] = sha256_file(path)
    manifest = {
        "schema_version": 1,
        "source_encoder_sha256": sha256_file(source),
        "cut_index": cut,
        "bridge_names": list(SAM_SPLIT_BRIDGE_NAMES),
        "graphs": hashes,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Verified {len(hashes)} SAM halves at {destination}")


if __name__ == "__main__":
    main()
