#!/usr/bin/env python3
"""Run trusted research FLARE exports and prepare an app-private Android fixture.

Requires NumPy and ONNX Runtime, not Torch or fairseq. This is an ungated,
inference-only diagnostic; it must not replace the application's recognizer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from export_flare import CACHE_HEAD_DIM, CACHE_HEADS, CACHE_LAYERS, validate_output

if TYPE_CHECKING:
    import numpy as np
    import onnxruntime as ort

MODEL_ID = "flare-base-transformer-2026-research"


def checked_graph(directory: Path, kind: str, metadata: dict, filename: str | None = None) -> Path:
    """Check export consistency, not authenticity: the directory must be trusted."""
    path = directory / (filename or f"{kind}.onnx")
    if path.stat().st_size != metadata[f"{kind}Bytes"]:
        raise ValueError(f"{kind} graph size does not match metadata")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != metadata[f"{kind}Sha256"]:
        raise ValueError(f"{kind} graph SHA-256 does not match metadata")
    return path


def cached_greedy_decode(
    session: ort.InferenceSession, encoded: np.ndarray, mask: np.ndarray,
    cross_values: dict[str, np.ndarray],
) -> list[int]:
    import numpy as np

    names = [(i, kind) for i in range(CACHE_LAYERS) for kind in ("key", "value")]
    past_names = [f"past_{i}_{kind}" for i, kind in names]
    present_names = [f"present_{i}_{kind}" for i, kind in names]
    input_names = {node.name for node in session.get_inputs()}
    required = {"tokens", "encoder_padding_mask", *past_names, *cross_values}
    if not required <= input_names or input_names - (required | {"encoder_out"}):
        raise ValueError("Unsupported cached decoder inputs")
    cache = {name: np.zeros((1, CACHE_HEADS, 0, CACHE_HEAD_DIM), dtype=np.float32)
             for name in past_names}
    tokens = [2]
    for step in range(128):
        candidates = {"tokens": np.asarray([tokens], dtype=np.int64), "encoder_out": encoded,
                      "encoder_padding_mask": mask, **cache, **cross_values}
        values = session.run(["next_logits", *present_names],
                             {key: value for key, value in candidates.items() if key in input_names})
        if len(values) != 1 + len(past_names):
            raise ValueError("Invalid cached decoder output count")
        logits = values[0]
        if logits.shape != (1, 1000) or not np.isfinite(logits).all():
            raise ValueError("Invalid or non-finite decoder output")
        expected_shape = (1, CACHE_HEADS, len(tokens), CACHE_HEAD_DIM)
        if any(value.shape != expected_shape or not np.isfinite(value).all() for value in values[1:]):
            raise ValueError("Invalid or non-finite self cache output")
        cache = dict(zip(past_names, values[1:]))
        logits[0, 1] = -np.inf
        if step == 0:
            logits[0, 2] = -np.inf
        token = int(logits[0].argmax())
        tokens.append(token)
        if token == 2:
            return tokens
    raise ValueError("Decoder did not generate EOS within 128 iterations")


def greedy_decode(
    session: ort.InferenceSession, encoded: np.ndarray, mask: np.ndarray,
) -> list[int]:
    import numpy as np

    tokens = [2]
    for step in range(128):
        logits = session.run(["next_logits"], {
            "tokens": np.asarray([tokens], dtype=np.int64),
            "encoder_out": encoded,
            "encoder_padding_mask": mask,
        })[0]
        if logits.shape != (1, 1000) or not np.isfinite(logits).all():
            raise ValueError("Invalid or non-finite decoder output")
        logits[0, 1] = -np.inf
        if step == 0:
            logits[0, 2] = -np.inf
        token = int(logits[0].argmax())
        tokens.append(token)
        if token == 2:
            return tokens
    raise ValueError("Decoder did not generate EOS within 128 iterations")


def run(model_dir: Path, sample: Path, output: Path) -> dict:
    import numpy as np
    import onnxruntime as ort

    output = validate_output(output)
    if output.exists():
        raise ValueError("Refusing to overwrite an existing fixture directory")
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    required = {
        "schemaVersion": 1, "modelId": MODEL_ID, "releaseYear": 2026,
        "license": "CC-BY-NC-4.0", "maxFrames": 200, "maxPrefix": 128,
        "featureSize": 512, "vocabularySize": 1000, "eosId": 2, "padId": 1,
        "inputSize": 96, "cropSize": 88, "mean": 0.421, "std": 0.165,
    }
    if any(metadata.get(key) != value for key, value in required.items()):
        raise ValueError("Unsupported FLARE export metadata")
    mode = metadata.get("decoderMode", "full")
    if mode not in ("full", "cached"):
        raise ValueError("Unsupported FLARE decoder mode")
    if mode == "cached" and any(metadata.get(key) != value for key, value in {
        "cacheLayers": CACHE_LAYERS, "cacheHeads": CACHE_HEADS, "cacheHeadDim": CACHE_HEAD_DIM,
    }.items()):
        raise ValueError("Unsupported FLARE cache geometry")
    graph_specs = [("encoder", "encoder.onnx", "encoder")]
    graph_specs += ([("decoder", "decoder.onnx", "decoder")] if mode == "full" else [
        ("cross", "cross.onnx", "cross"), ("cachedDecoder", "decoder-cache.onnx", "cachedDecoder")])
    entries = metadata.get("files")
    if not isinstance(entries, list) or len(entries) != len(graph_specs):
        raise ValueError("Invalid graph file list")
    for _, filename, key in graph_specs:
        matches = [entry for entry in entries if isinstance(entry, dict) and entry.get("name") == filename]
        if (len(matches) != 1 or matches[0].get("sizeBytes") != metadata.get(f"{key}Bytes")
                or matches[0].get("sha256") != metadata.get(f"{key}Sha256")):
            raise ValueError("Graph file list does not match metadata")
    graphs = {name: checked_graph(model_dir, key, metadata, filename)
              for name, filename, key in graph_specs}
    with np.load(sample, allow_pickle=False) as data:
        if set(data.files) != {"face", "lip"}:
            raise ValueError("Sample must contain only face and lip arrays")
        face, lip = data["face"], data["lip"]
    if (face.dtype != np.uint8 or lip.dtype != np.uint8 or face.shape != lip.shape
            or face.ndim != 3 or face.shape[1:] != (96, 96) or not 1 <= len(face) <= 200):
        raise ValueError("Expected matching uint8 face/lip arrays with shape [1..200,96,96]")
    frames = len(face)
    face = np.ascontiguousarray((face[:, 4:92, 4:92].astype(np.float32) / 255.0 - 0.421) / 0.165)
    lip = np.ascontiguousarray((lip[:, 4:92, 4:92].astype(np.float32) / 255.0 - 0.421) / 0.165)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    sessions: dict[str, ort.InferenceSession] = {}
    load_ms: dict[str, float] = {}
    for kind, path in graphs.items():
        started = time.perf_counter()
        sessions[kind] = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        load_ms[kind] = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    encoded, mask = sessions["encoder"].run(["encoder_out", "encoder_padding_mask"], {
        "face": face[None, None], "lip": lip[None, None],
        "face_mask": np.ones(1, dtype=bool), "lip_mask": np.ones(1, dtype=bool),
        "padding_mask": np.zeros((1, frames), dtype=bool),
    })
    encoder_ms = (time.perf_counter() - started) * 1000
    if (encoded.shape != (frames, 1, 512) or mask.shape != (1, frames)
            or not np.isfinite(encoded).all()):
        raise ValueError("Invalid or non-finite encoder output")
    summary = {
        "modelId": MODEL_ID, "provider": "CPUExecutionProvider", "frames": frames,
        "loadMs": load_ms, "encoderMs": encoder_ms,
        "excludes": "camera, preprocessing, graph hashing; not a phone benchmark",
    }
    if mode == "full":
        started = time.perf_counter()
        tokens = greedy_decode(sessions["decoder"], encoded, mask)
        summary["fullPrefixDecoderMs"] = (time.perf_counter() - started) * 1000
    else:
        cross_names = [f"cross_{i}_{kind}" for i in range(CACHE_LAYERS) for kind in ("key", "value")]
        started = time.perf_counter()
        values = sessions["cross"].run(cross_names, {"encoder_out": encoded})
        summary["crossMs"] = (time.perf_counter() - started) * 1000
        expected_shape = (1, CACHE_HEADS, frames, CACHE_HEAD_DIM)
        if (len(values) != len(cross_names) or any(
                value.shape != expected_shape or not np.isfinite(value).all() for value in values)):
            raise ValueError("Invalid or non-finite cross cache output")
        started = time.perf_counter()
        tokens = cached_greedy_decode(sessions["cachedDecoder"], encoded, mask, dict(zip(cross_names, values)))
        summary["cachedDecoderMs"] = (time.perf_counter() - started) * 1000
    summary["generatedTokenCount"] = len(tokens) - 1
    write_fixture(output, face, lip, metadata, tokens, summary)
    return summary


def write_fixture(
    output: Path, face: np.ndarray, lip: np.ndarray, metadata: dict,
    tokens: list[int], summary: dict,
) -> None:
    output = validate_output(output)
    output.mkdir(parents=True, mode=0o700)
    for kind, values in (("face", face), ("lip", lip)):
        with (output / f"{kind}.f32").open("xb") as stream:
            stream.write(values.astype("<f4").tobytes())
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump({**metadata, "frames": len(face), "expectedTokenIds": tokens}, stream, indent=2)
    with (output / "results.json").open("x", encoding="utf-8") as stream:
        json.dump({**summary, "generatedTokenIds": tokens}, stream, indent=2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path, help="Trusted export_flare.py output")
    parser.add_argument("--sample", required=True, type=Path, help="Reference-free uint8 face/lip NPZ")
    parser.add_argument("--output", required=True, type=Path, help="New fixture directory outside Git")
    args = parser.parse_args(argv)
    try:
        summary = run(args.model_dir, args.sample, args.output)
    except Exception as error:  # CLI boundary includes native ORT failures; never report partial success.
        print(f"error: {str(error)[:1000]}", file=sys.stderr)
        return 2
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
