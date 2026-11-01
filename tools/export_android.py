#!/usr/bin/env python3
"""Export the visual encoder and CTC head from the original USR2 checkpoint.

Run from the repository root in the existing Stilltype Python runtime:
  python tools/export_android.py --checkpoint /path/baseplus.pth --output /path/output
This writes model weights outside the source tree; do not commit them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
import inference  # noqa: E402
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E  # noqa: E402

CHECKPOINT_SHA = "4ab1f3f230aa6a724bcc698db8a17f2f54919136d2885c1a07a6b69247012773"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class VisualCTC(torch.nn.Module):
    def __init__(self, model: E2E):
        super().__init__()
        self.encoder = model.encoder
        self.ctc = model.ctc_v

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.ctc.log_softmax(self.encoder(xs_v=video))


def load_model(checkpoint: Path) -> VisualCTC:
    if sha256(checkpoint) != CHECKPOINT_SHA:
        raise ValueError("Checkpoint SHA256 does not match the pinned USR2 Base+ release")
    model = E2E(len(inference.token_list()), SimpleNamespace(**inference.BASEPLUS))
    raw = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    state = {}
    for key, value in raw.items():
        key = key.removeprefix("_orig_mod.")
        if key.startswith("model.backbone."):
            state[key.removeprefix("model.backbone.")] = value
    model.load_state_dict(state, strict=True)
    del raw, state
    model.eval()
    return VisualCTC(model).eval()


def export(checkpoint: Path, output: Path) -> None:
    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=True)
    model = load_model(checkpoint)
    video = torch.zeros((1, 25, 88, 88), dtype=torch.float32)
    with torch.inference_mode():
        torch.onnx.export(
            model, (video,), str(output / "usr2-ctc-fp32.onnx"),
            input_names=["video"], output_names=["logits"],
            dynamic_axes={"video": {1: "frames"}, "logits": {1: "frames"}},
            opset_version=17, dynamo=False, external_data=False,
        )


def quantize(output: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        str(output / "usr2-ctc-fp32.onnx"), str(output / "usr2-ctc-int8.onnx"),
        op_types_to_quantize=["MatMul", "Gemm"], weight_type=QuantType.QInt8,
        extra_options={"MatMulConstBOnly": True},
    )


def manifest(output: Path) -> None:
    tokens = inference.token_list()
    if len(tokens) != 1049:
        raise ValueError(f"Expected 1049 tokens, got {len(tokens)}")
    (output / "tokens.json").write_text(json.dumps(tokens, ensure_ascii=False), encoding="utf-8")
    (output / "mean-face.json").write_text(json.dumps(inference.stable_reference().tolist()), encoding="utf-8")
    shutil.copyfile(ROOT / "public/vision/face_detection_yunet_2026may.onnx", output / "yunet-2026may.onnx")
    names = ["usr2-ctc-int8.onnx", "tokens.json", "yunet-2026may.onnx", "mean-face.json"]
    files = [{"name": name, "sizeBytes": (output / name).stat().st_size, "sha256": sha256(output / name)} for name in names]
    data = {"schemaVersion": 1, "modelId": "usr2-baseplus-2026-ctc-int8", "releaseYear": 2026,
            "exportKind": "ctc-int8", "sourceCheckpointSha256": CHECKPOINT_SHA,
            "inputName": "video", "outputName": "logits", "fps": 25, "cropSize": 88,
            "maxFrames": 200, "blankId": 0, "eosId": 1048, "files": files}
    (output / "manifest.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stage", choices=("export", "quantize", "manifest"), default="export")
    args = parser.parse_args()
    if args.stage == "export":
        export(args.checkpoint, args.output)
    elif args.stage == "quantize":
        quantize(args.output)
    else:
        manifest(args.output)


if __name__ == "__main__":
    main()
