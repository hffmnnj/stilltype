#!/usr/bin/env python3
"""Prepare independent desktop pixel references for the native aligner test."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from export_flare import validate_output

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from inference import (  # noqa: E402
    align_mouths,
    decode_records,
    interpolate_landmarks,
    load_clip_records,
    stable_reference,
)


def align_face_and_lip(frames: np.ndarray, landmarks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Compute the face region and independently check the lip region."""
    filled = interpolate_landmarks(landmarks)
    if filled is None:
        raise ValueError("no usable four-point landmarks")
    reference = stable_reference()
    faces = np.empty((len(frames), 96, 96), dtype=np.uint8)
    lips = np.empty_like(faces)
    previous = None
    for i, frame in enumerate(frames):
        margin = min(6, i, len(frames) - 1 - i)
        smooth = np.mean(filled[i - margin : i + margin + 1], axis=0)
        smooth += filled[i].mean(0) - smooth.mean(0)
        warp, _ = cv2.estimateAffinePartial2D(smooth, reference, method=cv2.LMEDS)
        if warp is None:
            warp = previous if previous is not None else cv2.estimateAffinePartial2D(smooth, reference)[0]
        if warp is None:
            raise ValueError("unable to estimate an alignment transform")
        previous = warp
        aligned = cv2.warpAffine(
            frame, warp, (256, 256), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        mouth = smooth[3] @ warp[:, :2].T + warp[:, 2]
        x0 = int(round(float(np.clip(float(mouth[0]) - 48, 0, 160))))
        y0 = int(round(float(np.clip(float(mouth[1]) - 48, 0, 160))))
        lips[i] = aligned[y0 : y0 + 96, x0 : x0 + 96]
        faces[i] = cv2.resize(aligned[32:224, 32:224], (96, 96))
    if not np.array_equal(lips, align_mouths(frames, landmarks)):
        raise ValueError("lip pixels differ from desktop align_mouths")
    return faces, lips


def prepare(clip_dir: Path, output_path: Path) -> None:
    if os.path.lexists(output_path):
        raise ValueError("output directory must be new")
    output = validate_output(output_path)
    records, _ = load_clip_records(str(clip_dir))
    if len(records) > 200:
        raise ValueError("fixture input must contain at most 200 frames")
    frames, landmarks = decode_records(records)
    if len(frames) != len(records):
        raise ValueError("undecodable frames")

    runs: list[tuple[int, int]] = []
    start = None
    for i in range(len(landmarks) + 1):
        if i < len(landmarks) and landmarks[i] is not None:
            if start is None:
                start = i
        elif start is not None:
            runs.append((start, i))
            start = None
    if not runs:
        raise ValueError("fixture needs at least 30 consecutive detected faces")
    start, end = max(runs, key=lambda run: run[1] - run[0])
    frames = frames[start:end]
    selected = np.asarray(landmarks[start:end], dtype="<f4")
    if len(frames) < 30:
        raise ValueError("fixture needs at least 30 consecutive detected faces")

    faces, lips = align_face_and_lip(frames, list(selected))
    split = len(frames) // 2
    face_a, lip_a = align_face_and_lip(frames[:split], list(selected[:split]))
    face_b, lip_b = align_face_and_lip(frames[split:], list(selected[split:]))
    arrays = {
        "frames.gray": frames,
        "landmarks.f32": selected,
        "reference.f32": stable_reference().astype("<f4"),
        "faces.gray": faces,
        "lips.gray": lips,
        "gap-faces.gray": np.concatenate((face_a, face_b)),
        "gap-lips.gray": np.concatenate((lip_a, lip_b)),
    }
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    files = {}
    for name, array in arrays.items():
        data = array.tobytes(order="C")
        (output / name).write_bytes(data)
        files[name] = {
            "shape": list(array.shape), "dtype": str(array.dtype),
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
        }
    manifest = {
        "source": "prepared clip, longest contiguous detected-face segment",
        "source_indices": [start, end], "frames": len(frames), "height": frames.shape[1],
        "width": frames.shape[2], "split": split,
        "geometry": "independent full-clip desktop align_mouths; separate face region", "files": files,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "frames": len(frames), "shape": list(frames.shape), "split": split}))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare native alignment fixture pixels. Keep fixture data out of Git."
    )
    parser.add_argument("--clip-dir", type=Path, required=True, help="Prepared clip directory")
    parser.add_argument("--output", type=Path, required=True, help="New directory outside any Git worktree")
    args = parser.parse_args()
    prepare(args.clip_dir, args.output)


if __name__ == "__main__":
    main()
