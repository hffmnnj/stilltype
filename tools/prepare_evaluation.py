#!/usr/bin/env python3
"""Prepare a local video segment as a Stilltype frames.jsonl capture.

Host capture-equivalent preparation for offline accuracy work: decodes a
local video file, contain-fits each sampled frame into the production
640x480 capture buffer, runs the YuNet face detector exactly like the
browser/Android capture path, and streams one JSON record per kept frame:

    {"jpeg": base64, "timestamp": ms, "landmarks": [[x, y] x 4] | null}

Landmark order matches capture: image-left eye, image-right eye, nose
base, mouth centre (mouth-corner midpoint), in 640x480 capture pixels.
Frames without a face are kept with null landmarks; the production
face-ratio gate owns rejection. No reference text is read, stored, or
inferred. No network, no downloads, no manifest generation.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

ENCODE_WIDTH = 640
ENCODE_HEIGHT = 480
DET_W = 320
DET_H = 256
DET_STRIDES = (8, 16, 32)
FACE_SCORE_THRESHOLD = 0.75
JPEG_QUALITY = 75
MIN_SEGMENT_MS = 500
MAX_SEGMENT_MS = 120_000

NOTE = (
    "Host capture-equivalent preparation from a local file, "
    "not a physical camera timing benchmark."
)


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clamp01(value: float) -> float:
    if isinstance(value, bool) or not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, float(value)))


def round1(value: float) -> float:
    """Round to 0.1 with half away from -inf, matching Math.round(v * 10)."""
    return math.floor(float(value) * 10.0 + 0.5) / 10.0


def contain_fit(frame: Any) -> Any:
    """Fit an unmirrored BGR frame into a black 640x480 buffer, keeping aspect."""
    import cv2

    if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
        raise ValueError("contain_fit: expected a decoded image array")
    height, width = int(frame.shape[0]), int(frame.shape[1])
    if width <= 0 or height <= 0:
        raise ValueError("contain_fit: frame has no pixels")
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    scale = min(ENCODE_WIDTH / width, ENCODE_HEIGHT / height)
    draw_w = max(1, int(round(width * scale)))
    draw_h = max(1, int(round(height * scale)))
    resized = cv2.resize(frame, (draw_w, draw_h), interpolation=cv2.INTER_LINEAR)
    import numpy as np

    canvas = np.zeros((ENCODE_HEIGHT, ENCODE_WIDTH, 3), dtype=np.uint8)
    dx = (ENCODE_WIDTH - draw_w) // 2
    dy = (ENCODE_HEIGHT - draw_h) // 2
    canvas[dy : dy + draw_h, dx : dx + draw_w] = resized
    return canvas


def detector_input(canvas: Any) -> Any:
    """640x480 capture buffer -> YuNet input: (1, 3, 256, 320) BGR float32."""
    import cv2
    import numpy as np

    small = cv2.resize(canvas, (DET_W, DET_H), interpolation=cv2.INTER_LINEAR)
    out = np.empty((1, 3, DET_H, DET_W), dtype=np.float32)
    out[0, 0] = small[:, :, 0]
    out[0, 1] = small[:, :, 1]
    out[0, 2] = small[:, :, 2]
    return out


def _head(outputs: Mapping[str, Any], name: str, expected: int) -> Any:
    import numpy as np

    if name not in outputs:
        available = sorted(str(k) for k in outputs.keys())
        raise ValueError(
            f"corrupt YuNet output: missing {name!r} (available: {available})"
        )
    try:
        arr = np.asarray(outputs[name], dtype=np.float64).ravel()
    except (ValueError, TypeError) as exc:
        raise ValueError(f"corrupt YuNet output: {name!r} is not numeric") from exc
    if arr.size < expected:
        raise ValueError(
            f"corrupt YuNet output: {name!r} has {arr.size} values, want {expected}"
        )
    return arr


def decode_yunet(outputs: Mapping[str, Any]) -> list[list[float]] | None:
    """Best valid face from the 12 YuNet heads, in 640x480 capture pixels.

    Score is sqrt(clamped cls * obj) with threshold >= 0.75; the highest
    valid face wins. Returns the four capture-pixel pairs (image-left eye,
    image-right eye, nose, mouth midpoint) or None when no face is valid.
    Missing or short heads raise ValueError; invalid detections are
    skipped, never fabricated.
    """
    best: tuple[float, tuple[float, float], tuple[float, float],
                 tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
    for stride in DET_STRIDES:
        cols = DET_W // stride
        rows = DET_H // stride
        count = cols * rows
        cls = _head(outputs, f"cls_{stride}", count)
        obj = _head(outputs, f"obj_{stride}", count)
        bbox = _head(outputs, f"bbox_{stride}", count * 4)
        kps = _head(outputs, f"kps_{stride}", count * 10)
        for i in range(count):
            score = math.sqrt(clamp01(cls[i]) * clamp01(obj[i]))
            if not (score >= FACE_SCORE_THRESHOLD):
                continue
            if best is not None and score <= best[0]:
                continue
            try:
                bw = math.exp(bbox[i * 4 + 2]) * stride
                bh = math.exp(bbox[i * 4 + 3]) * stride
            except OverflowError:
                continue
            if not math.isfinite(bw) or not math.isfinite(bh) or bw <= 0 or bh <= 0:
                continue
            row, col = divmod(i, cols)
            pts: list[tuple[float, float]] = []
            valid = True
            for j in range(5):
                x = (col + float(kps[i * 10 + j * 2])) * stride
                y = (row + float(kps[i * 10 + j * 2 + 1])) * stride
                if (
                    not math.isfinite(x)
                    or not math.isfinite(y)
                    or x < -4
                    or y < -4
                    or x > DET_W + 4
                    or y > DET_H + 4
                ):
                    valid = False
                    break
                pts.append((x, y))
            if not valid:
                continue
            eye_dist = math.hypot(pts[0][0] - pts[1][0], pts[0][1] - pts[1][1])
            if not math.isfinite(eye_dist) or not (eye_dist > 4):
                continue
            best = (score, pts[0], pts[1], pts[2], pts[3], pts[4])
    if best is None:
        return None
    _, re, le, nose, mr, ml = best
    left, right = (re, le) if re[0] <= le[0] else (le, re)
    mouth = ((mr[0] + ml[0]) / 2.0, (mr[1] + ml[1]) / 2.0)
    scale_x = ENCODE_WIDTH / DET_W
    scale_y = ENCODE_HEIGHT / DET_H
    return [
        [round1(left[0] * scale_x), round1(left[1] * scale_y)],
        [round1(right[0] * scale_x), round1(right[1] * scale_y)],
        [round1(nose[0] * scale_x), round1(nose[1] * scale_y)],
        [round1(mouth[0] * scale_x), round1(mouth[1] * scale_y)],
    ]


def create_detector_session(detector_path: str) -> Any:
    """One ORT CPU YuNet session: 4 intra-op threads, 1 inter-op thread."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        detector_path, options, providers=["CPUExecutionProvider"]
    )


def run_detector(session: Any, det_input: Any) -> dict[str, Any]:
    names = [str(o.name) for o in session.get_outputs()]
    feed = {"input": det_input}
    raw = session.run(names, feed)
    return dict(zip(names, raw))


def ensure_output_dir(path: Path) -> Path:
    """Create a new clip directory; refuse repo-internal or existing paths."""
    resolved = path.resolve()
    repo = REPO_ROOT.resolve()
    if resolved == repo or repo in resolved.parents:
        raise ValueError(f"refusing to write inside the repository: {path}")
    if os.path.lexists(path) or resolved.exists() or resolved.is_symlink():
        raise ValueError(f"refusing to overwrite existing output: {path}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    try:
        resolved.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise ValueError(f"refusing to overwrite existing output: {path}") from exc
    return resolved


def prepare_clip(
    *,
    video: str,
    detector: str,
    output: str,
    start_ms: int = 0,
    end_ms: int | None = None,
    requested_fps: float = 25.0,
    session: Any = None,
) -> dict[str, Any]:
    """Convert [start_ms, end_ms) of a video file into a frames.jsonl clip."""
    import cv2

    if not os.path.isfile(video):
        raise ValueError(f"missing video file: {video}")
    if not os.path.isfile(detector):
        raise ValueError(f"missing detector file: {detector}")
    if not math.isfinite(requested_fps) or not 1.0 <= requested_fps <= 25.0:
        raise ValueError(f"--fps must be within 1..25, got {requested_fps!r}")
    if start_ms < 0:
        raise ValueError(f"--start-ms must be >= 0, got {start_ms}")
    if end_ms is not None and end_ms <= start_ms:
        raise ValueError("--end-ms must be after --start-ms")

    cap = cv2.VideoCapture(video)
    try:
        if not cap.isOpened():
            raise ValueError(f"cannot open video: {video}")
        src_fps = float(cap.get(cv2.CAP_PROP_FPS))
        total = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(src_fps) or src_fps <= 0 or not math.isfinite(total) or total <= 0:
            raise ValueError(f"cannot read frame timing from video: {video}")
        duration_ms = total / src_fps * 1000.0
        effective_end = min(float(end_ms) if end_ms is not None else duration_ms, duration_ms)
        if start_ms >= duration_ms:
            raise ValueError("--start-ms is past the end of the video")
        segment_ms = effective_end - start_ms
        if segment_ms < MIN_SEGMENT_MS or segment_ms > MAX_SEGMENT_MS:
            raise ValueError(
                f"selected segment is {segment_ms / 1000.0:.2f}s; want 0.5..120s"
            )

        out_dir = ensure_output_dir(Path(output))
        owned = session is None
        if owned:
            session = create_detector_session(detector)
        try:
            return _stream_frames(
                cap, session, out_dir, video, detector,
                start_ms, effective_end, src_fps, float(requested_fps),
            )
        finally:
            if owned:
                del session
    finally:
        cap.release()


def _stream_frames(
    cap: Any,
    session: Any,
    out_dir: Path,
    video: str,
    detector: str,
    start_ms: float,
    effective_end: float,
    src_fps: float,
    requested_fps: float,
) -> dict[str, Any]:
    import cv2

    interval_ms = 1000.0 / requested_fps
    start_frame = int(math.ceil(start_ms / 1000.0 * src_fps - 1e-9))
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    index = start_frame
    kept = 0
    faces = 0
    next_sample_ms = float(start_ms)
    prev_ts: int | None = None
    first_ts: int | None = None
    last_ts: int | None = None
    frames_path = out_dir / "frames.jsonl"
    with open(frames_path, "w", encoding="utf-8") as sink:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            src_ms = index / src_fps * 1000.0
            index += 1
            if src_ms >= effective_end:
                break
            if src_ms < start_ms - 1e-6:
                continue
            if src_ms < next_sample_ms - 1e-6:
                continue
            # Follow the target clock, not the last chosen source frame. The
            # latter accidentally turns a 30 fps source into 15 fps at a 25 fps target.
            next_sample_ms = start_ms + (math.floor((src_ms - start_ms) / interval_ms + 1e-6) + 1) * interval_ms
            canvas = contain_fit(frame)
            landmarks = decode_yunet(run_detector(session, detector_input(canvas)))
            if landmarks is not None:
                faces += 1
            ts = int(round(src_ms - start_ms))
            if prev_ts is not None and ts <= prev_ts:
                ts = prev_ts + 1
            prev_ts = ts
            if first_ts is None:
                first_ts = ts
            last_ts = ts
            ok_enc, buf = cv2.imencode(
                ".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )
            if not ok_enc:
                raise RuntimeError("JPEG encoding failed")
            record = {
                "jpeg": base64.b64encode(buf.tobytes()).decode("ascii"),
                "timestamp": ts,
                "landmarks": landmarks,
            }
            sink.write(json.dumps(record) + "\n")
            kept += 1
    if kept == 0:
        raise ValueError("no frames selected in the requested range")
    span_s = (last_ts - first_ts) / 1000.0 if kept >= 2 else 0.0
    actual_fps = (kept - 1) / span_s if kept >= 2 and span_s > 0 else 0.0
    metadata = {
        "sourceVideoSha256": sha256_file(video),
        "detectorSha256": sha256_file(detector),
        "startMs": int(start_ms),
        "endMs": int(effective_end),
        "requestedFps": float(requested_fps),
        "actualFps": float(actual_fps),
        "frameCount": kept,
        "faceCount": faces,
        "framesSha256": sha256_file(frames_path),
        "note": NOTE,
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as sink:
        json.dump(metadata, sink, indent=2)
        sink.write("\n")
    return {"output": str(out_dir), **metadata}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a local video segment into a Stilltype frames.jsonl capture."
    )
    parser.add_argument("--video", required=True, help="Path to a local video file.")
    parser.add_argument("--detector", required=True, help="Path to the YuNet .onnx file.")
    parser.add_argument("--output", required=True, help="New clip directory (outside the repo).")
    parser.add_argument("--start-ms", type=int, default=0, help="Inclusive segment start (default 0).")
    parser.add_argument("--end-ms", type=int, default=None, help="Exclusive segment end (default video end).")
    parser.add_argument("--fps", type=float, default=25.0, help="Output rate 1..25 (default 25).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = prepare_clip(
            video=args.video,
            detector=args.detector,
            output=args.output,
            start_ms=args.start_ms,
            end_ms=args.end_ms,
            requested_fps=args.fps,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
