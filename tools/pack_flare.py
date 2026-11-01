#!/usr/bin/env python3
"""Pack a trusted beam-5 FLARE export using FP16 storage and FLOAT casts.

FP16 rounding can change recognition. ONNX Runtime may fold casts back to
FP32, so neither lower runtime memory nor faster computation is implied. These
are exported utilities, not newly trained weights or application defaults.
Source hashes check consistency, not authenticity; the source must be trusted.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path
from typing import Any

from benchmark_flare import MODEL_ID, checked_graph
from export_flare import (CACHE_HEAD_DIM, CACHE_HEADS, CACHE_LAYERS, CHECKPOINT_ARTIFACT,
                          CHECKPOINT_REVISION, CHECKPOINT_SHA256, sha256, validate_output)

GRAPHS = (("encoder", "encoder.onnx"), ("cross", "cross.onnx"),
          ("beamDecoder", "decoder-beam.onnx"))
MIN_ELEMENTS = 1024


def validate_source(directory: Path, metadata: dict[str, Any]) -> dict[str, Path]:
    """Validate all metadata and graph hashes before any ONNX graph is parsed."""
    required = {
        "schemaVersion": 1, "modelId": MODEL_ID, "releaseYear": 2026,
        "license": "CC-BY-NC-4.0", "sourceCheckpointRepository": "Anony94/FLARE-checkpoints",
        "sourceCheckpointRevision": CHECKPOINT_REVISION, "sourceCheckpointArtifact": CHECKPOINT_ARTIFACT,
        "sourceCheckpointSha256": CHECKPOINT_SHA256, "decoderMode": "beam", "beamSize": 5,
        "cacheLayers": CACHE_LAYERS, "cacheHeads": CACHE_HEADS, "cacheHeadDim": CACHE_HEAD_DIM,
        "fps": 25, "inputSize": 96, "cropSize": 88, "mean": 0.421, "std": 0.165,
        "inputGeometry": [96, 96], "crop": [4, 92, 4, 92], "cropGeometry": [88, 88],
        "normalization": {"scale": 255.0, "mean": 0.421, "std": 0.165},
        "maxFrames": 200, "maxPrefix": 128, "featureSize": 512, "vocabularySize": 1000,
        "bosId": 0, "padId": 1, "eosId": 2, "unkId": 3,
    }
    if any(metadata.get(key) != value or type(metadata.get(key)) is not type(value)
           for key, value in required.items()):
        raise ValueError("Unsupported model, normalization, cache, or provenance metadata")
    symbols = metadata.get("symbols")
    if not isinstance(symbols, list) or len(symbols) != 1000 or not all(isinstance(s, str) for s in symbols):
        raise ValueError("Invalid vocabulary")
    entries = metadata.get("files")
    if not isinstance(entries, list) or len(entries) != len(GRAPHS):
        raise ValueError("Invalid graph file list")
    for key, filename in GRAPHS:
        matches = [entry for entry in entries if isinstance(entry, dict) and entry.get("name") == filename]
        size, digest = metadata.get(key + "Bytes"), metadata.get(key + "Sha256")
        if (type(size) is not int or size <= 0 or not isinstance(digest, str)
                or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest)
                or len(matches) != 1 or matches[0].get("sizeBytes") != size
                or matches[0].get("sha256") != digest):
            raise ValueError(f"Invalid graph metadata: {filename}")
    return {key: checked_graph(directory, key, metadata, filename) for key, filename in GRAPHS}


def reject_external_data(message: Any) -> None:
    """Reject external tensors anywhere in the model, including nested graphs."""
    import onnx
    from google.protobuf.descriptor import FieldDescriptor

    if isinstance(message, onnx.TensorProto):
        if message.data_location == onnx.TensorProto.EXTERNAL or message.external_data:
            raise ValueError("External ONNX tensor data is unsupported")
        return
    for field, value in message.ListFields():
        if field.type == FieldDescriptor.TYPE_MESSAGE:
            if field.is_repeated:
                for item in value:
                    reject_external_data(item)
            else:
                reject_external_data(value)


def graph_names(graph: Any) -> set[str]:
    names = {item.name for item in (*graph.input, *graph.output, *graph.value_info, *graph.initializer)}
    names.update(item.values.name for item in graph.sparse_initializer)
    for node in graph.node:
        names.update(name for name in (*node.input, *node.output) if name)
        for attribute in node.attribute:
            if attribute.HasField("g"):
                names.update(graph_names(attribute.g))
            for nested in attribute.graphs:
                names.update(graph_names(nested))
    return names


def transform(source: Path, output: Path) -> dict[str, int]:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(str(source), load_external_data=False)
    reject_external_data(model)
    graph = model.graph
    used = graph_names(graph)
    inputs = {value.name for value in graph.input}
    minimum = float(np.nextafter(np.float16(0), np.float16(1)))
    maximum = float(np.finfo(np.float16).max)
    counts = {key: 0 for key in ("converted", "small", "non_float32", "graph_input",
                                  "non_finite", "overflow", "underflow")}
    initializers = []
    casts = []
    for tensor in graph.initializer:
        reason = None
        if tensor.data_type != TensorProto.FLOAT:
            reason = "non_float32"
        elif math.prod(tensor.dims) < MIN_ELEMENTS:
            reason = "small"
        elif tensor.name in inputs:
            reason = "graph_input"
        else:
            array = numpy_helper.to_array(tensor)
            if not np.isfinite(array).all():
                reason = "non_finite"
            else:
                absolute = np.abs(array)
                if np.any(absolute > maximum):
                    reason = "overflow"
                elif np.any((absolute != 0) & (absolute < minimum)):
                    reason = "underflow"
        if reason is not None:
            counts[reason] += 1
            initializers.append(tensor)
            continue
        name = tensor.name + "__fp16_storage"
        while name in used:
            name += "_"
        used.add(name)
        initializers.append(numpy_helper.from_array(array.astype(np.float16), name=name))
        casts.append(helper.make_node("Cast", [name], [tensor.name], to=TensorProto.FLOAT))
        counts["converted"] += 1
    nodes = list(graph.node)
    del graph.initializer[:]
    graph.initializer.extend(initializers)
    del graph.node[:]
    graph.node.extend(casts)
    graph.node.extend(nodes)
    with output.open("xb") as stream:
        stream.write(model.SerializeToString())
    del model, graph, initializers, casts, nodes
    gc.collect()
    onnx.checker.check_model(str(output))
    return counts


def pack(model_dir: Path, output: Path) -> dict[str, Any]:
    output = validate_output(output)
    if output.exists():
        raise ValueError("Refusing to overwrite an existing output directory")
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Invalid export metadata")
    originals = validate_source(model_dir, metadata)
    # Delay creation until every source hash has passed.
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    records = []
    for key, filename in GRAPHS:
        source_entry = {"name": filename, "sizeBytes": metadata[key + "Bytes"],
                        "sha256": metadata[key + "Sha256"]}
        target = output / filename
        counts = transform(originals[key], target)
        packed = {"name": filename, "sizeBytes": target.stat().st_size, "sha256": sha256(target)}
        records.append({"source": source_entry, "output": packed, "initializerCounts": counts})
        metadata[key + "Bytes"], metadata[key + "Sha256"] = packed["sizeBytes"], packed["sha256"]
    metadata["files"] = [record["output"] for record in records]
    metadata["storageTransformation"] = {
        "description": "FP16-stored FLOAT initializers restored by FLOAT Cast for graph computation. "
                       "Rounding can change recognition; ORT may fold casts back to FP32. "
                       "No runtime memory or speed improvement is implied. Not newly trained weights.",
        "minimumElements": MIN_ELEMENTS, "minimumNonzeroMagnitude": 2 ** -24,
        "maximumMagnitude": 65504.0, "sourceGraphs": [record["source"] for record in records],
    }
    metadata_path = output / "metadata.json"
    created = False
    try:
        with metadata_path.open("x", encoding="utf-8") as stream:
            created = True
            json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    except Exception:
        if created:
            metadata_path.unlink(missing_ok=True)
        raise
    return {"output": str(output), "graphs": records,
            "note": metadata["storageTransformation"]["description"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="Trusted beam-5 export directory.")
    parser.add_argument("--output", type=Path, required=True, help="New directory outside any Git working tree.")
    parser.add_argument("--accept-license", action="store_true", required=True, help="Accept CC-BY-NC-4.0 model terms.")
    args = parser.parse_args(argv)
    try:
        result = pack(args.model_dir, args.output)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
