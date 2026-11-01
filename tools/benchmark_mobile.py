#!/usr/bin/env python3
"""Compare CPU CTC logits on a real aligned clip, without claiming accuracy.

Run stages in order, in separate processes to avoid keeping model copies alive:
  python tools/benchmark_mobile.py prepare --output DIR --clip CLIP_DIRECTORY
  python tools/benchmark_mobile.py torch --output DIR --checkpoint CHECKPOINT
  python tools/benchmark_mobile.py ort --output DIR
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
import inference  # noqa: E402


def decode(logits: np.ndarray, tokens: list[str]) -> str:
    ids = np.argmax(logits[0], axis=-1)
    previous = -1
    out = []
    for index in ids:
        index = int(index)
        if index != previous and index not in (0, len(tokens) - 1):
            out.append(tokens[index])
        previous = index
    return "".join(out).replace("▁", " ").strip()


def prepare(output: Path, clip: Path) -> None:
    records, _ = inference.load_clip_records(str(clip))
    start = records[0]["timestamp"]
    records = [r for r in records if r["timestamp"] - start < 8000]
    fps = inference.fps_from_records(records)
    if (error := inference.check_clip(len(records), sum(r["landmarks"] is not None for r in records), fps)):
        raise ValueError(f"Invalid benchmark clip: {error}")
    frames, landmarks = inference.decode_records(records)
    rois = inference.align_mouths(frames, landmarks)
    if rois is None:
        raise ValueError("No face in benchmark clip")
    video = inference.USR2Visual.to_input(rois, fps).numpy()[:, :200].copy()
    if video.shape[1] < 26:
        raise ValueError("Benchmark clip too short for dynamic-shape checks")
    video.astype("<f4").tofile(output / "benchmark-input.f32")
    (output / "benchmark-input.json").write_text(json.dumps({
        "shape": list(video.shape), "dtype": "float32-le", "sourceFps": fps,
        "sourceFrames": len(records), "truncatedAtMs": 8000,
        "note": "Public demo clip; runtime smoke fixture, not an accuracy benchmark."
    }, indent=2) + "\n", encoding="utf-8")


def input_video(output: Path) -> np.ndarray:
    shape = json.loads((output / "benchmark-input.json").read_text())["shape"]
    return np.fromfile(output / "benchmark-input.f32", dtype="<f4").reshape(shape)


def run_torch(output: Path, checkpoint: Path) -> None:
    import torch
    from export_android import load_model

    torch.set_num_threads(4)
    video = input_video(output)
    started = time.perf_counter()
    model = load_model(checkpoint)
    load_ms = (time.perf_counter() - started) * 1000
    tokens = inference.token_list()
    results = {}
    with torch.inference_mode():
        for length in (25, video.shape[1]):
            x = torch.from_numpy(video[:, :length])
            model(x)
            started = time.perf_counter()
            logits = model(x).numpy()
            elapsed = (time.perf_counter() - started) * 1000
            np.save(output / f"torch-logits-{length}.npy", logits)
            results[str(length)] = {"warmMs": elapsed, "text": decode(logits, tokens), "shape": list(logits.shape)}
    (output / "torch-results.json").write_text(json.dumps({"loadMs": load_ms, "lengths": results}, indent=2))


def run_ort(output: Path) -> None:
    import onnx
    import onnxruntime as ort

    video = input_video(output)
    tokens = inference.token_list()
    all_results = {}
    for variant in ("fp32", "int8"):
        path = output / f"usr2-ctc-{variant}.onnx"
        onnx.checker.check_model(str(path))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        started = time.perf_counter()
        session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        load_ms = (time.perf_counter() - started) * 1000
        results = {}
        for length in (25, video.shape[1]):
            feeds = {"video": video[:, :length].copy()}
            session.run(["logits"], feeds)
            started = time.perf_counter()
            logits = session.run(["logits"], feeds)[0]
            elapsed = (time.perf_counter() - started) * 1000
            expected = np.load(output / f"torch-logits-{length}.npy")
            if logits.shape != expected.shape:
                raise ValueError(f"{variant} shape {logits.shape} != PyTorch {expected.shape}")
            difference = np.abs(logits - expected)
            results[str(length)] = {"warmMs": elapsed, "text": decode(logits, tokens),
                                    "maxAbsLogitDifference": float(difference.max()),
                                    "meanAbsLogitDifference": float(difference.mean()),
                                    "argmaxAgreement": float(np.mean(logits.argmax(-1) == expected.argmax(-1))),
                                    "shape": list(logits.shape)}
            if variant == "fp32" and not np.allclose(logits, expected, atol=2e-3, rtol=2e-3):
                raise ValueError(f"FP32 parity failed at T={length}: max difference {difference.max()}")
        all_results[variant] = {"bytes": path.stat().st_size, "loadMs": load_ms, "lengths": results}
        del session
    (output / "benchmark-results.json").write_text(json.dumps({"host": "desktop CPU, four ORT threads", "variants": all_results}, indent=2) + "\n")
    print(json.dumps(all_results, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "torch", "ort"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--clip", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.stage == "prepare" and args.clip:
        prepare(args.output, args.clip)
    elif args.stage == "torch" and args.checkpoint:
        run_torch(args.output, args.checkpoint)
    elif args.stage == "ort":
        run_ort(args.output)
    else:
        parser.error("prepare requires --clip; torch requires --checkpoint")


if __name__ == "__main__":
    main()
