#!/usr/bin/env python3
"""Export weightless ONNX cache utilities for the FLARE decoder layout.

These utilities contain no checkpoint weights. They do not alter the terms of
the underlying checkpoint or require a checkpoint download.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto, helper

from export_flare import CACHE_HEAD_DIM, CACHE_HEADS, CACHE_LAYERS, sha256, validate_output

SUFFIXES = tuple(f"{layer}_{kind}" for layer in range(CACHE_LAYERS) for kind in ("key", "value"))
GRAPH_NAMES = ("cache-gather.onnx", "cache-merge.onnx")


def cache_value(name: str, rows: str, length: str) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, [rows, CACHE_HEADS, length, CACHE_HEAD_DIM])


def make_model(nodes: list[onnx.NodeProto], inputs: list[onnx.ValueInfoProto],
               outputs: list[onnx.ValueInfoProto], name: str) -> onnx.ModelProto:
    graph = helper.make_graph(nodes, name, inputs, outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    onnx.checker.check_model(model)
    return model


def gather_model() -> onnx.ModelProto:
    inputs = [helper.make_tensor_value_info("indices", TensorProto.INT64, ["N"])]
    inputs.extend(cache_value(f"past_{suffix}", "B", "P") for suffix in SUFFIXES)
    outputs = [cache_value(f"selected_{suffix}", "N", "P") for suffix in SUFFIXES]
    nodes = [helper.make_node("Gather", [f"past_{suffix}", "indices"], [f"selected_{suffix}"], axis=0)
             for suffix in SUFFIXES]
    return make_model(nodes, inputs, outputs, "cache_gather")


def merge_model(chunks: int) -> onnx.ModelProto:
    inputs = [helper.make_tensor_value_info("indices", TensorProto.INT64, ["N"])]
    inputs.extend(cache_value(f"chunk_{chunk}_{suffix}", f"C{chunk}", "L")
                  for chunk in range(chunks) for suffix in SUFFIXES)
    outputs = [cache_value(f"past_{suffix}", "N", "L") for suffix in SUFFIXES]
    outputs.append(helper.make_tensor_value_info("cache_finite", TensorProto.BOOL, []))
    nodes = []
    counts = []
    for suffix in SUFFIXES:
        combined = f"combined_{suffix}"
        nodes.extend([
            helper.make_node("Concat", [f"chunk_{chunk}_{suffix}" for chunk in range(chunks)], [combined], axis=0),
            helper.make_node("Gather", [combined, "indices"], [f"past_{suffix}"], axis=0),
            helper.make_node("IsNaN", [combined], [f"nan_{suffix}"]),
            helper.make_node("IsInf", [combined], [f"inf_{suffix}"]),
            helper.make_node("Or", [f"nan_{suffix}", f"inf_{suffix}"], [f"invalid_{suffix}"]),
            helper.make_node("Cast", [f"invalid_{suffix}"], [f"invalid_int_{suffix}"], to=TensorProto.INT64),
            helper.make_node("ReduceSum", [f"invalid_int_{suffix}"], [f"count_{suffix}"], keepdims=0),
        ])
        counts.append(f"count_{suffix}")
    total = counts[0]
    for index, count in enumerate(counts[1:], start=1):
        next_total = "invalid_count" if index == len(counts) - 1 else f"partial_count_{index}"
        nodes.append(helper.make_node("Add", [total, count], [next_total]))
        total = next_total
    nodes.append(helper.make_node("Constant", [], ["zero"],
                                  value=helper.make_tensor("zero_value", TensorProto.INT64, [], [0])))
    nodes.append(helper.make_node("Equal", ["invalid_count", "zero"], ["cache_finite"]))
    return make_model(nodes, inputs, outputs, "cache_merge")


def export(beam_size: int, output: Path) -> dict[str, Any]:
    if type(beam_size) is not int or beam_size not in (5, 50):
        raise ValueError("beam size must be 5 or 50")
    if os.path.lexists(output):
        raise ValueError(f"output directory must be new: {output}")
    output = validate_output(output)
    models = (gather_model(), merge_model(beam_size // 5))
    output.mkdir(parents=True, exist_ok=False)
    for name, model in zip(GRAPH_NAMES, models):
        with (output / name).open("xb") as stream:
            stream.write(model.SerializeToString())
    files = [{"name": name, "sizeBytes": (output / name).stat().st_size, "sha256": sha256(output / name)}
             for name in GRAPH_NAMES]
    metadata = {
        "schemaVersion": 1, "modelId": "flare-base-transformer-2026-research", "beamSize": beam_size,
        "maximumNativeRows": 5, "cacheLayers": CACHE_LAYERS, "cacheHeads": CACHE_HEADS,
        "cacheHeadDim": CACHE_HEAD_DIM, "cacheChunks": beam_size // 5,
        "cacheGatherBytes": files[0]["sizeBytes"], "cacheGatherSha256": files[0]["sha256"],
        "cacheMergeBytes": files[1]["sizeBytes"], "cacheMergeSha256": files[1]["sha256"],
        "files": files,
    }
    with (output / "cache.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beam-size", required=True, type=int, choices=(5, 50))
    parser.add_argument("--output", required=True, type=Path, help="New directory outside any Git worktree.")
    args = parser.parse_args(argv)
    try:
        result = export(args.beam_size, args.output)
    except (ValueError, OSError, onnx.checker.ValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
