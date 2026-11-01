#!/usr/bin/env python3
"""Prepare pinned FLARE-LLM research artifacts locally; no downloads or inference service.

The 2026 visual adaptation uses an older Qwen2.5-7B base and is subject to
CC-BY-NC-4.0 terms. These utilities do not establish paper parity or new training.
The Base encoder export directory must be trusted: its metadata is not a signature.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from benchmark_flare import MODEL_ID, checked_graph
from export_flare import load_sample, sha256, validate_output

CHECKPOINT_SIZE = 615_953_103
CHECKPOINT_SHA256 = "d36cf01efd81090ddf0ad814e966bb50a5b42d141c715beb83462f625d2fd3c4"
PROJECTION_SIZE = 33_606_088
PROJECTION_SHA256 = "8e6775a4e23cf6723c89464f13c7b88527449f5bccaa3f8cbd43b4b5bc2316bf"
PROJECTION_SHAPES = {
    "0.weight": (2048, 512), "0.bias": (2048,), "2.weight": (3584, 2048),
    "2.bias": (3584,), "3.weight": (3584,), "3.bias": (3584,),
}
FAMILIES = {
    "self_attn.q_proj": (3584, 3584), "self_attn.k_proj": (3584, 512),
    "self_attn.v_proj": (3584, 512), "self_attn.o_proj": (3584, 3584),
    "mlp.gate_proj": (3584, 18944), "mlp.up_proj": (3584, 18944),
    "mlp.down_proj": (18944, 3584),
}


class InertDictionary:
    def __setstate__(self, state: Any) -> None:
        pass


class InertSentencepieceBPE:
    def __setstate__(self, state: Any) -> None:
        pass


class InertSentencePieceProcessor:
    def __setstate__(self, state: Any) -> None:
        pass


def adapter_shapes() -> dict[str, tuple[int, ...]]:
    bases = {f"base_model.model.model.layers.{layer}.{family}": shape
             for layer in range(28) for family, shape in FAMILIES.items()}
    bases["base_model.model.lm_head"] = (3584, 152064)
    return {f"{base}.lora_{side}.weight": shape
            for base, (inputs, outputs) in bases.items()
            for side, shape in (("A", (16, inputs)), ("B", (outputs, 16)))}


def check_schema(tensors: Mapping[str, Any], shapes: Mapping[str, tuple[int, ...]], dtype: Any) -> None:
    if set(tensors) != set(shapes):
        raise ValueError("unexpected tensor keys")
    for name, shape in shapes.items():
        tensor = tensors[name]
        if tensor.dtype != dtype or tuple(tensor.shape) != shape:
            raise ValueError(f"unexpected tensor dtype or shape: {name}")


def new_output(path: Path) -> Path:
    if os.path.lexists(path):
        raise ValueError("refusing an existing output path")
    result = validate_output(path)
    if os.path.lexists(result):
        raise ValueError("refusing an existing output path")
    return result


def file_info(path: Path) -> dict[str, Any]:
    return {"sizeBytes": path.stat().st_size, "sha256": sha256(path)}


def export(checkpoint: Path, output: Path) -> dict[str, Any]:
    output = new_output(output)
    if checkpoint.stat().st_size != CHECKPOINT_SIZE:
        raise ValueError("checkpoint size does not match the pinned stage-3 release")
    digest = sha256(checkpoint)
    if digest != CHECKPOINT_SHA256:
        raise ValueError("checkpoint SHA-256 does not match the pinned stage-3 release")

    import torch
    from safetensors.torch import load_file, save_file

    allowed = [(InertDictionary, "fairseq.data.dictionary.Dictionary"),
               (InertSentencepieceBPE, "fairseq.data.encoders.sentencepiece_bpe.SentencepieceBPE"),
               (InertSentencePieceProcessor, "sentencepiece.SentencePieceProcessor")]
    with torch.serialization.safe_globals(allowed):
        checkpoint_data = torch.load(checkpoint, weights_only=True, mmap=True, map_location="cpu")
    if not isinstance(checkpoint_data, dict) or not isinstance(checkpoint_data.get("model"), dict):
        raise ValueError("checkpoint has no model state")
    state = checkpoint_data["model"]
    expected_adapter = adapter_shapes()
    source_shapes = {f"llm.{name.replace('.weight', '.default.weight')}": shape
                     for name, shape in expected_adapter.items()}
    source_shapes.update({f"modality_projection.{name}": shape for name, shape in PROJECTION_SHAPES.items()})
    if set(state) != set(source_shapes) or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("unexpected checkpoint model tensors")
    check_schema(state, source_shapes, torch.float32)
    if any(not torch.isfinite(tensor).all().item() for tensor in state.values()):
        raise ValueError("checkpoint contains nonfinite tensors")
    adapter = {name.removeprefix("llm.").replace(".default.weight", ".weight"): tensor.contiguous()
               for name, tensor in state.items() if name.startswith("llm.")}
    projection = {name.removeprefix("modality_projection."): tensor.contiguous()
                  for name, tensor in state.items() if name.startswith("modality_projection.")}
    check_schema(adapter, expected_adapter, torch.float32)
    check_schema(projection, PROJECTION_SHAPES, torch.float32)

    output.mkdir(parents=True, exist_ok=False)
    peft = output / "peft"
    peft.mkdir()
    adapter_path, projection_path = peft / "adapter_model.safetensors", output / "projection.safetensors"
    save_file(adapter, adapter_path)
    save_file(projection, projection_path)
    config = {"base_model_name_or_path": "Qwen/Qwen2.5-7B", "peft_type": "LORA",
              "task_type": "CAUSAL_LM", "r": 16, "lora_alpha": 32, "lora_dropout": 0.1,
              "bias": "none", "inference_mode": True,
              "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"]}
    config_path = peft / "adapter_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    for path, original in ((adapter_path, adapter), (projection_path, projection)):
        restored = load_file(path, device="cpu")
        if restored.keys() != original.keys() or any(
            not torch.equal(restored[name].view(torch.int32), value.view(torch.int32))
            for name, value in original.items()
        ):
            raise ValueError(f"safetensors round-trip differs: {path.name}")
    files = {str(path.relative_to(output)): file_info(path) for path in (adapter_path, config_path, projection_path)}
    metadata = {"schemaVersion": 1, "releaseYear": 2026, "license": "CC-BY-NC-4.0",
                "baseModel": "Qwen/Qwen2.5-7B", "checkpoint": {"sizeBytes": CHECKPOINT_SIZE, "sha256": digest},
                "adapterTensors": 394, "projectionTensors": 6, "rank": 16, "alpha": 32,
                "files": files, "note": "2026 visual adaptation over an older base; exported utility, not newly trained weights or a paper-parity claim."}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {"output": str(output), "files": files, "metadata": "metadata.json"}


def project(model_dir: Path, projection_path: Path, sample: Path, output: Path) -> dict[str, Any]:
    output = new_output(output)
    start = time.monotonic()
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    required = {"schemaVersion": 1, "modelId": MODEL_ID, "releaseYear": 2026,
                "license": "CC-BY-NC-4.0", "maxFrames": 200, "maxPrefix": 128,
                "featureSize": 512, "vocabularySize": 1000, "eosId": 2, "padId": 1,
                "inputSize": 96, "cropSize": 88, "mean": 0.421, "std": 0.165}
    if not isinstance(metadata, dict) or any(metadata.get(k) != v for k, v in required.items()):
        raise ValueError("unsupported FLARE encoder metadata")
    entries = metadata.get("files")
    size, digest = metadata.get("encoderBytes"), metadata.get("encoderSha256")
    if (not isinstance(entries, list) or type(size) is not int or size <= 0
            or not isinstance(digest, str) or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)):
        raise ValueError("invalid encoder file metadata")
    matches = [entry for entry in entries if isinstance(entry, dict) and entry.get("name") == "encoder.onnx"]
    if (len(matches) != 1 or matches[0].get("sizeBytes") != size
            or matches[0].get("sha256") != digest):
        raise ValueError("encoder file list does not match metadata")
    encoder = checked_graph(model_dir, "encoder", metadata)
    projection_info = file_info(projection_path)
    if projection_info != {"sizeBytes": PROJECTION_SIZE, "sha256": PROJECTION_SHA256}:
        raise ValueError("projection does not match the pinned export")
    sample_hash = sha256(sample)
    face, lip = load_sample(sample)

    import numpy as np
    import onnxruntime as ort
    import torch
    import torch.nn.functional as functional
    from safetensors.torch import load_file

    weights = load_file(projection_path, device="cpu")
    check_schema(weights, PROJECTION_SHAPES, torch.float32)
    if any(not torch.isfinite(value).all().item() for value in weights.values()):
        raise ValueError("projection contains nonfinite tensors")
    torch.set_num_threads(4)
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
    session = ort.InferenceSession(str(encoder), sess_options=options, providers=["CPUExecutionProvider"])
    loaded = time.monotonic()
    encoded, padding = session.run(None, {
        "face": face[None, None], "lip": lip[None, None],
        "face_mask": np.array([True]), "lip_mask": np.array([True]),
        "padding_mask": np.zeros((1, len(face)), dtype=np.bool_),
    })
    encoded_at = time.monotonic()
    if (encoded.ndim != 3 or encoded.shape[1:] != (1, 512) or not 1 <= encoded.shape[0] <= 200
            or encoded.dtype != np.float32 or not np.isfinite(encoded).all()
            or padding.shape != (1, encoded.shape[0]) or padding.dtype != np.bool_):
        raise ValueError("invalid encoder output or padding mask")
    length = int((~padding[0]).sum())
    if not 1 <= length <= 200 or padding[0, :length].any() or not padding[0, length:].all():
        raise ValueError("invalid encoder padding mask")
    with torch.inference_mode():
        value = torch.from_numpy(encoded[:length, 0, :].copy())
        value = functional.linear(value, weights["0.weight"], weights["0.bias"])
        value = functional.gelu(value)
        value = functional.linear(value, weights["2.weight"], weights["2.bias"])
        value = functional.layer_norm(value, (3584,), weights["3.weight"], weights["3.bias"], 1e-5)
        data = value.to(torch.bfloat16).to(torch.float32).contiguous().numpy()
    if data.shape != (length, 3584) or not np.isfinite(data).all():
        raise ValueError("invalid or nonfinite projected features")
    projected_at = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    visual = output / "visual.bin"
    with visual.open("xb") as stream:
        stream.write(struct.pack("<II", length, 3584))
        stream.write(data.astype("<f4", copy=False).tobytes())
    result = {"schemaVersion": 1, "sampleSha256": sample_hash, "encoderSha256": metadata["encoderSha256"],
              "projectionSha256": projection_info["sha256"], "inputFrames": len(face), "shape": [length, 3584],
              "visual": file_info(visual), "timingsSeconds": {"load": loaded - start,
              "encoder": encoded_at - loaded, "projection": projected_at - encoded_at,
              "total": time.monotonic() - start},
              "note": "Published Base encoder used as an approximate substitute; exact stage-3 encoder weight identity is unverified. Not a paper-parity claim."}
    (output / "metadata.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return {"output": str(output), "visual": "visual.bin", "metadata": "metadata.json", "shape": result["shape"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export_parser = commands.add_parser("export", help="Export the pinned stage-3 adapter and projection.")
    export_parser.add_argument("--checkpoint", type=Path, required=True)
    project_parser = commands.add_parser("project", help="Project a reference-free sample on CPU.")
    project_parser.add_argument("--model-dir", type=Path, required=True, help="Trusted full Base encoder export.")
    project_parser.add_argument("--projection", type=Path, required=True)
    project_parser.add_argument("--sample", type=Path, required=True)
    for command in (export_parser, project_parser):
        command.add_argument("--output", type=Path, required=True, help="New directory outside any Git working tree.")
        command.add_argument("--accept-license", action="store_true", required=True, help="Accept CC-BY-NC-4.0 terms.")
    args = parser.parse_args(argv)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = (export(args.checkpoint, args.output) if args.command == "export" else
                      project(args.model_dir, args.projection, args.sample, args.output))
    except (ValueError, RuntimeError, OSError, ImportError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
