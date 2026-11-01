#!/usr/bin/env python3
"""Small reproducible accuracy evaluator for Stilltype raw visual recognition.

Benchmark tooling only: scores raw hypotheses (never cleaned text) against a
frozen JSON manifest. Model functions receive clip paths and settings, never
reference text; scoring runs afterward with vocabulary and cleanup disabled.

CTC decoders (ctc-fp32/ctc-int8) are explicitly a host export benchmark, not
an Android device score. Host crops use desktop temporal alignment
(inference.align_mouths); mobile on-device cropping may differ.

Manifest contract (docs/RECOGNITION.md): JSON with schemaVersion=1, datasetId and
items[]. Each item has id, split (dev/test), clipDir, reference, sourceUrl,
sourceSha256 (64 lowercase hex), license, speakerGroup, speechMode
(spoken/silent) and trainingOverlap (unknown/known-unseen/known-seen).
Paths resolve relative to the manifest. Source videos and references stay
outside Git. Each run selects one split and one decoder and writes JSON
outside Git. WER is reported unclamped; it is a measurement, not a
guarantee of accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "engine"))

SCHEMA_VERSION = 1
TARGET_WER = 0.10
MAX_WINDOW_FRAMES = 200

REQUIRED_ITEM_FIELDS = (
    "id",
    "split",
    "clipDir",
    "reference",
    "sourceUrl",
    "sourceSha256",
    "license",
    "speakerGroup",
    "speechMode",
    "trainingOverlap",
)
ALLOWED_TOP_FIELDS = {"schemaVersion", "datasetId", "items"}
VALID_SPLITS = ("dev", "test")
VALID_SPEECH_MODES = ("spoken", "silent")
VALID_TRAINING_OVERLAP = ("unknown", "known-unseen", "known-seen")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
# Alphanumeric (unicode letters/digits, no underscore) with internal ASCII
# apostrophes. Other punctuation is dropped; hyphens split words.
WORD_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)*", re.UNICODE)
# Curly apostrophes normalise to ASCII before tokenising.
CURLY_APOSTROPHES = ("\u2018", "\u2019", "\u201b", "\u02bc")

CTC_DISCLOSURE = (
    "Host export benchmark, not an Android device score: ORT CPU session on "
    "desktop with desktop temporal alignment (inference.align_mouths). "
    "On-device cropping, timing and power behaviour may differ."
)
DESKTOP_DISCLOSURE = (
    "Desktop visual-only inference (inference.transcribe_clip); raw "
    "hypotheses scored, cleaned text ignored."
)
SCORE_ONLY_DISCLOSURE = "Score-only run: no inference performed in this process."


def normalize_text(text: str) -> str:
    """Normalise for scoring: NFKC, casefold, ASCII apostrophes, words only."""
    if not isinstance(text, str):
        return ""
    folded = unicodedata.normalize("NFKC", text).casefold()
    for curly in CURLY_APOSTROPHES:
        folded = folded.replace(curly, "'")
    return " ".join(WORD_RE.findall(folded))


def edit_counts(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """Word-level (substitutions, deletions, insertions) via Levenshtein.

    Deterministic tie-breaking on backtrace: diagonal (match/substitution)
    first, then deletion, then insertion. Totals are minimal; the tie-break
    only fixes the S/D/I split reproducibly.
    """
    n, m = len(ref), len(hyp)
    costs = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        costs[i][0] = i
    for j in range(1, m + 1):
        costs[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub_cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            costs[i][j] = min(
                costs[i - 1][j - 1] + sub_cost,
                costs[i - 1][j] + 1,
                costs[i][j - 1] + 1,
            )
    subs = dels = inss = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and costs[i][j] == costs[i - 1][j - 1] + (
            0 if ref[i - 1] == hyp[j - 1] else 1
        ):
            if ref[i - 1] != hyp[j - 1]:
                subs += 1
            i -= 1
            j -= 1
        elif i > 0 and costs[i][j] == costs[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            inss += 1
            j -= 1
    return subs, dels, inss


@dataclass
class ManifestItem:
    id: str
    split: str
    clipDir: str
    reference: str
    sourceUrl: str
    sourceSha256: str
    license: str
    speakerGroup: str
    speechMode: str
    trainingOverlap: str


@dataclass
class LoadedManifest:
    datasetId: str
    items: list[ManifestItem]
    sha256: str
    manifest_dir: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_nonempty_str(value: object, field_name: str, item_id: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Manifest item {item_id!r}: field {field_name!r} is empty")
    return value


def load_manifest(manifest_path: Path) -> LoadedManifest:
    """Load and validate a manifest. Raises ValueError on any contract breach."""
    raw_bytes = manifest_path.read_bytes()
    try:
        data = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"Manifest is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Manifest top level must be an object")
    unknown_top = set(data.keys()) - ALLOWED_TOP_FIELDS
    if unknown_top:
        raise ValueError(f"Manifest has unsupported fields: {sorted(unknown_top)}")
    if data.get("schemaVersion") != SCHEMA_VERSION:
        raise ValueError("Manifest schemaVersion must be 1")
    dataset_id = data.get("datasetId")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("Manifest datasetId is empty")
    raw_items = data.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("Manifest items must be a non-empty array")
    items: list[ManifestItem] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_items):
        label = f"index {index}"
        if not isinstance(raw, dict):
            raise ValueError(f"Manifest item {label} must be an object")
        unknown = set(raw.keys()) - set(REQUIRED_ITEM_FIELDS)
        if unknown:
            raise ValueError(f"Manifest item {label} has unsupported fields: {sorted(unknown)}")
        missing = [k for k in REQUIRED_ITEM_FIELDS if k not in raw]
        if missing:
            raise ValueError(f"Manifest item {label} is missing fields: {missing}")
        item_id = raw.get("id")
        if not isinstance(item_id, str) or not item_id.strip():
            raise ValueError(f"Manifest item {label} has an empty id")
        if item_id in seen:
            raise ValueError(f"Manifest has a duplicate id: {item_id!r}")
        seen.add(item_id)
        for key in REQUIRED_ITEM_FIELDS:
            _require_nonempty_str(raw.get(key), key, item_id)
        if raw["split"] not in VALID_SPLITS:
            raise ValueError(f"Manifest item {item_id!r} has unsupported split {raw['split']!r}")
        if raw["speechMode"] not in VALID_SPEECH_MODES:
            raise ValueError(
                f"Manifest item {item_id!r} has unsupported speechMode {raw['speechMode']!r}"
            )
        if raw["trainingOverlap"] not in VALID_TRAINING_OVERLAP:
            raise ValueError(
                f"Manifest item {item_id!r} has unsupported trainingOverlap "
                f"{raw['trainingOverlap']!r}"
            )
        if not SHA256_RE.match(raw["sourceSha256"]):
            raise ValueError(
                f"Manifest item {item_id!r} has an invalid sourceSha256 "
                "(want 64 lowercase hex)"
            )
        if not normalize_text(raw["reference"]):
            raise ValueError(f"Manifest item {item_id!r} has an empty normalized reference")
        items.append(ManifestItem(**{k: raw[k] for k in REQUIRED_ITEM_FIELDS}))
    return LoadedManifest(
        datasetId=dataset_id,
        items=items,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        manifest_dir=manifest_path.resolve().parent,
    )


def frames_sha256(clip_dir: Path) -> str | None:
    """SHA256 of a prepared clip's frames.jsonl, or None when unreadable."""
    try:
        return sha256_file(clip_dir / "frames.jsonl")
    except OSError:
        return None


def score_pair(reference: str, raw: str | None, error: str | None) -> dict:
    """Score one clip. Failed/empty hypotheses count as all words deleted."""
    norm_ref = normalize_text(reference)
    ref_words = norm_ref.split() if norm_ref else []
    if error:
        norm_hyp = ""
        hyp_words: list[str] = []
    else:
        norm_hyp = normalize_text(raw or "")
        hyp_words = norm_hyp.split() if norm_hyp else []
    subs, dels, inss = edit_counts(ref_words, hyp_words)
    n_ref = len(ref_words)
    wer = (subs + dels + inss) / n_ref if n_ref else None
    return {
        "normalizedReference": norm_ref,
        "normalizedHypothesis": norm_hyp,
        "refWords": n_ref,
        "hypWords": len(hyp_words),
        "substitutions": subs,
        "deletions": dels,
        "insertions": inss,
        "wer": wer,
        "exactMatch": bool(norm_ref) and norm_ref == norm_hyp,
    }


def summarize(scores: list[dict]) -> dict:
    """Corpus totals for one group. WER is unclamped."""
    subs = sum(s["substitutions"] for s in scores)
    dels = sum(s["deletions"] for s in scores)
    inss = sum(s["insertions"] for s in scores)
    ref_words = sum(s["refWords"] for s in scores)
    hyp_words = sum(s["hypWords"] for s in scores)
    exact = sum(1 for s in scores if s["exactMatch"])
    n = len(scores)
    wer = (subs + dels + inss) / ref_words if ref_words else None
    return {
        "clips": n,
        "refWords": ref_words,
        "hypWords": hyp_words,
        "substitutions": subs,
        "deletions": dels,
        "insertions": inss,
        "wer": wer,
        "exactMatches": exact,
        "exactMatchRate": (exact / n) if n else None,
    }


def ensure_output_writable(output: Path) -> Path:
    """Refuse repo-internal destinations; create only the direct parent."""
    resolved = output.resolve()
    repo_resolved = REPO_ROOT.resolve()
    if resolved == repo_resolved or repo_resolved in resolved.parents:
        raise ValueError(f"Refusing to write inside the repository: {output}")
    if output.exists() or output.is_symlink():
        raise ValueError(f"Refusing to overwrite existing result: {output}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def load_predictions(predictions_path: Path) -> dict[str, dict]:
    """Load {id: {raw, error}} from a score-only predictions file (no models)."""
    try:
        data = json.loads(predictions_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Predictions file is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        raise ValueError("Predictions file must be an object with an items array")
    by_id: dict[str, dict] = {}
    for index, raw in enumerate(data["items"]):
        if not isinstance(raw, dict):
            raise ValueError(f"Predictions item index {index} must be an object")
        item_id = raw.get("id")
        if not isinstance(item_id, str) or not item_id.strip():
            raise ValueError(f"Predictions item index {index} has an empty id")
        if item_id in by_id:
            raise ValueError(f"Predictions have a duplicate id: {item_id!r}")
        error = raw.get("error")
        if error is not None and not isinstance(error, str):
            raise ValueError(f"Predictions item {item_id!r}: error must be a string or null")
        hyp = raw.get("raw")
        if hyp is not None and not isinstance(hyp, str):
            raise ValueError(f"Predictions item {item_id!r}: raw must be a string or null")
        entry: dict = {"raw": hyp or "", "error": error or None}
        for optional in ("durationMs", "inferenceMs", "wallMs"):
            if raw.get(optional) is not None:
                entry[optional] = raw[optional]
        by_id[item_id] = entry
    return by_id


def check_coverage(selected_ids: list[str], predicted_ids: set[str]) -> None:
    selected = set(selected_ids)
    missing = sorted(selected - predicted_ids)
    extra = sorted(predicted_ids - selected)
    if missing or extra:
        raise ValueError(
            f"Predictions must cover exactly the selected split "
            f"(missing={missing}, extra={extra})"
        )


def build_item_result(
    item: ManifestItem,
    manifest_dir: Path,
    raw: str,
    error: str | None,
    duration_ms: int | None,
    inference_ms: int | None,
    wall_ms: int | None,
) -> dict:
    clip_abs = manifest_dir / item.clipDir
    scored = score_pair(item.reference, raw, error)
    return {
        "id": item.id,
        "split": item.split,
        "clipDir": item.clipDir,
        "speechMode": item.speechMode,
        "speakerGroup": item.speakerGroup,
        "sourceUrl": item.sourceUrl,
        "sourceSha256": item.sourceSha256,
        "license": item.license,
        "trainingOverlap": item.trainingOverlap,
        "framesSha256": frames_sha256(clip_abs),
        "durationMs": duration_ms,
        "inferenceMs": inference_ms,
        "wallMs": wall_ms,
        "raw": raw,
        "error": error,
        **scored,
    }


def finalize_results(
    *,
    dataset_id: str,
    manifest_sha256: str,
    split: str,
    decoder: str,
    decode: str,
    device: str | None,
    provider: str | None,
    beam_size: int | None,
    disclosure: str,
    model_info: dict,
    item_results: list[tuple[dict, ManifestItem]],
) -> dict:
    by_speech: dict[str, list[dict]] = {"spoken": [], "silent": []}
    for entry, item in item_results:
        by_speech[item.speechMode].append(entry)
    metrics = {
        "total": summarize([e for e, _ in item_results]),
        "spoken": summarize(by_speech["spoken"]),
        "silent": summarize(by_speech["silent"]),
    }
    total_wer = metrics["total"]["wer"]
    total_inference = sum((e.get("inferenceMs") or 0) for e, _ in item_results)
    total_wall = sum((e.get("wallMs") or 0) for e, _ in item_results)
    return {
        "schemaVersion": 1,
        "datasetId": dataset_id,
        "split": split,
        "decoder": decoder,
        "decode": decode,
        "device": device,
        "provider": provider,
        "beamSize": beam_size,
        "manifestSha256": manifest_sha256,
        "disclosure": disclosure,
        "model": model_info,
        "targetWer": TARGET_WER,
        "targetMet": bool(total_wer is not None and total_wer <= TARGET_WER),
        "metrics": metrics,
        "timing": {
            "totalInferenceMs": total_inference,
            "totalWallMs": total_wall,
        },
        "items": [e for e, _ in item_results],
    }


# ---------------------------------------------------------------------------
# Inference paths (kept separate from scoring; references never enter them)
# ---------------------------------------------------------------------------


def desktop_predict(
    clip_dir: str,
    *,
    device: str,
    beam_size: int,
    transcribe_fn=None,
) -> tuple[str, str | None, int | None, int | None, str | None]:
    """Run desktop inference on one clip. Returns (raw, error, dur, ms, actual)."""
    if transcribe_fn is None:
        import inference  # noqa: PLC0415 -- lazy so score-only never needs torch

        transcribe_fn = inference.transcribe_clip
    result = transcribe_fn(
        clip_dir,
        settings={"inferenceDevice": device, "beamSize": beam_size},
        vocabulary=[],
        final=True,
    )
    return (
        result.get("raw") or "",
        result.get("error"),
        result.get("durationMs"),
        result.get("inferenceMs"),
        result.get("device") or None,
    )


def run_desktop(
    selected: list[ManifestItem],
    manifest_dir: Path,
    *,
    device: str,
    beam_size: int,
) -> tuple[list[tuple[dict, ManifestItem]], str | None]:
    scored: list[tuple[dict, ManifestItem]] = []
    actual_device = None
    for item in selected:
        clip_abs = manifest_dir / item.clipDir
        wall_start = time.perf_counter()
        try:
            raw, error, duration_ms, inference_ms, actual = desktop_predict(
                str(clip_abs), device=device, beam_size=beam_size
            )
            if actual:
                actual_device = actual
        except Exception as exc:  # noqa: BLE001 -- record infra fault, keep others
            if exc.__class__.__name__ in ("ModelUnavailable", "GpuUnavailable"):
                raise
            raw, error, duration_ms, inference_ms = "", exc.__class__.__name__, None, None
        wall_ms = int((time.perf_counter() - wall_start) * 1000)
        scored.append(
            (
                build_item_result(
                    item, manifest_dir, raw, error, duration_ms, inference_ms, wall_ms
                ),
                item,
            )
        )
    return scored, actual_device


def run_ctc(
    selected: list[ManifestItem],
    manifest_dir: Path,
    *,
    variant: str,
    mobile_model_dir: Path,
    beam_size: int = 1,
    token_top_k: int = 20,
    activity_gate: str = "legacy",
) -> tuple[list[tuple[dict, ManifestItem]], dict]:
    """Host CTC export benchmark over every window (no window dropped)."""
    tools_dir = str(Path(__file__).resolve().parent)
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    import inference  # noqa: PLC0415 -- CTC path only
    from benchmark_mobile import decode as greedy_decode  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    import onnxruntime as ort  # noqa: PLC0415

    model_file = mobile_model_dir / f"usr2-ctc-{variant}.onnx"
    if not model_file.is_file():
        raise ValueError(f"Missing CTC model file: {model_file}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(model_file), options, providers=["CPUExecutionProvider"]
    )
    tokens = inference.token_list()
    decode_logits = greedy_decode
    if beam_size > 1:
        from ctc_decode import prefix_beam_decode

        def decode_logits(logits, tokens):
            return prefix_beam_decode(logits, tokens, beam_size=beam_size, token_top_k=token_top_k)
    model_info = {
        "file": model_file.name,
        "sha256": sha256_file(model_file),
        "provider": "CPUExecutionProvider",
        "intraOpThreads": 4,
        "interOpThreads": 1,
        "maxWindowFrames": MAX_WINDOW_FRAMES,
        "ctcBeamSize": beam_size,
        "ctcTokenTopK": token_top_k if beam_size > 1 else None,
    }
    scored: list[tuple[dict, ManifestItem]] = []
    for item in selected:
        clip_abs = manifest_dir / item.clipDir
        wall_start = time.perf_counter()
        raw, error, duration_ms, inference_ms = _predict_ctc_clip(
            inference, decode_logits, np, session, tokens, clip_abs, activity_gate=activity_gate
        )
        wall_ms = int((time.perf_counter() - wall_start) * 1000)
        scored.append(
            (
                build_item_result(
                    item, manifest_dir, raw, error, duration_ms, inference_ms, wall_ms
                ),
                item,
            )
        )
    return scored, model_info


def _predict_ctc_clip(
    inference, decode_logits, np, session, tokens, clip_abs: Path, *, activity_gate: str = "legacy"
) -> tuple[str, str | None, int | None, int | None]:
    """Decode every <=8s window; empty/no-speech windows add no text."""
    try:
        records, duration_ms = inference.load_clip_records(str(clip_abs))
    except Exception as exc:  # noqa: BLE001 -- ClipError code passthrough
        code = getattr(exc, "code", "invalid-clip")
        return "", code if isinstance(code, str) else "invalid-clip", None, None
    fps = inference.fps_from_records(records)
    faces = sum(1 for r in records if r["landmarks"] is not None)
    if gate_error := inference.check_clip(len(records), faces, fps):
        return "", gate_error, duration_ms, 0
    windows = inference._partition_windows(records, fps)
    raws: list[str] = []
    inference_ms = 0
    for window in windows:
        try:
            frames, landmarks = inference.decode_records(window)
        except Exception as exc:  # noqa: BLE001 -- corrupt window fails the clip
            code = getattr(exc, "code", "invalid-clip")
            if exc.__class__.__name__ == "ModelUnavailable":
                raise
            return "", code if isinstance(code, str) else "invalid-clip", duration_ms, inference_ms
        rois = inference.align_mouths(frames, landmarks)
        if rois is None:
            continue
        motion = inference.mouth_motion(rois, fps)
        if activity_gate == "ctc-evidence":
            from activity_gate import (CANDIDATE_MOTION, STRONG_MOTION, admits,
                                       greedy_evidence, span as activity_span)

            strong = activity_span(motion, fps, STRONG_MOTION)
            span = activity_span(motion, fps, CANDIDATE_MOTION)
        elif activity_gate == "legacy":
            span = inference.speech_span(motion, fps)
        else:
            raise ValueError(f"Unsupported activity gate: {activity_gate}")
        if span is None:
            continue
        start, stop = int(span[0]), int(span[1])
        if stop <= start:
            continue
        video = inference.USR2Visual.to_input(rois[start:stop], fps).numpy()
        video = np.ascontiguousarray(video[:, :MAX_WINDOW_FRAMES])
        feed = {"video": video}
        started = time.perf_counter()
        logits = session.run(["logits"], feed)[0]
        inference_ms += int((time.perf_counter() - started) * 1000)
        if not np.isfinite(logits).all():
            return "", "recognition-failed: non-finite CTC output", duration_ms, inference_ms
        if activity_gate == "ctc-evidence":
            evidence = greedy_evidence(logits, tokens)
            if strong is None and not admits(evidence):
                continue
        text = decode_logits(logits, tokens)
        if text:
            raws.append(text)
    if not raws:
        return "", "no-speech", duration_ms, inference_ms
    return " ".join(raws), None, duration_ms, inference_ms


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score raw Stilltype visual recognition against a frozen manifest."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=list(VALID_SPLITS))
    parser.add_argument(
        "--decoder", required=True, choices=("desktop", "ctc-fp32", "ctc-int8")
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--models", type=Path, default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "gpu", "cpu"))
    parser.add_argument("--beam-size", default=5, type=int)
    parser.add_argument("--motion-threshold", default=8.0, type=float,
                        help="Experimental host speech-motion threshold; production baseline is 8.")
    parser.add_argument("--mobile-model-dir", type=Path, default=None)
    parser.add_argument("--ctc-beam-size", type=int, default=1,
                        help="1 is the production greedy baseline; >1 enables experimental prefix search.")
    parser.add_argument("--ctc-token-top-k", type=int, default=20)
    parser.add_argument("--activity-gate", choices=("legacy", "ctc-evidence"), default=None,
                        help="CTC activity gate (default: legacy).")
    parser.add_argument("--score-predictions", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.activity_gate is not None and (args.decoder == "desktop" or args.score_predictions is not None):
        print("error: --activity-gate requires CTC inference", file=sys.stderr)
        return 2
    if args.activity_gate == "ctc-evidence" and args.motion_threshold != 8.0:
        print("error: ctc-evidence uses fixed motion thresholds 8 and 4", file=sys.stderr)
        return 2
    if args.beam_size is None or int(args.beam_size) < 1:
        print("error: --beam-size must be a positive integer", file=sys.stderr)
        return 2
    beam_size = max(1, min(100, int(args.beam_size)))
    if not math.isfinite(args.motion_threshold) or not 0 < args.motion_threshold <= 255:
        print("error: --motion-threshold must be finite and within (0, 255]", file=sys.stderr)
        return 2
    if not 1 <= args.ctc_beam_size <= 100 or not 1 <= args.ctc_token_top_k <= 1047:
        print("error: CTC beam must be 1..100 and token top-k must be 1..1047", file=sys.stderr)
        return 2
    try:
        manifest = load_manifest(args.manifest)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    selected = [item for item in manifest.items if item.split == args.split]
    if not selected:
        print(f"error: no items in split {args.split!r}", file=sys.stderr)
        return 2
    try:
        output = ensure_output_writable(args.output)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    score_only = args.score_predictions is not None
    if not score_only and args.decoder in ("ctc-fp32", "ctc-int8") and not args.mobile_model_dir:
        print("error: CTC decoders require --mobile-model-dir", file=sys.stderr)
        return 2

    try:
        if not score_only:
            import inference

            inference.SPEAKING = args.motion_threshold
            inference.STILL = args.motion_threshold * 5.0 / 8.0
        if score_only:
            assert args.score_predictions is not None
            by_id = load_predictions(args.score_predictions)
            check_coverage([i.id for i in selected], set(by_id.keys()))
            scored = [
                (
                    build_item_result(
                        item,
                        manifest.manifest_dir,
                        by_id[item.id]["raw"],
                        by_id[item.id]["error"],
                        by_id[item.id].get("durationMs"),
                        by_id[item.id].get("inferenceMs"),
                        by_id[item.id].get("wallMs"),
                    ),
                    item,
                )
                for item in selected
            ]
            results = finalize_results(
                dataset_id=manifest.datasetId,
                manifest_sha256=manifest.sha256,
                split=args.split,
                decoder=args.decoder,
                decode="beam" if args.decoder == "desktop" else ("prefix-beam" if args.ctc_beam_size > 1 else "greedy"),
                device=None,
                provider=None,
                beam_size=beam_size if args.decoder == "desktop" else None,
                disclosure=SCORE_ONLY_DISCLOSURE,
                model_info={"mode": "score-only"},
                item_results=scored,
            )
        elif args.decoder == "desktop":
            if args.models is not None:
                os.environ["STILLTYPE_MODELS"] = str(args.models)
            scored, actual_device = run_desktop(
                selected, manifest.manifest_dir, device=args.device, beam_size=beam_size
            )
            results = finalize_results(
                dataset_id=manifest.datasetId,
                manifest_sha256=manifest.sha256,
                split=args.split,
                decoder=args.decoder,
                decode="beam",
                device=actual_device,
                provider=None,
                beam_size=beam_size,
                disclosure=DESKTOP_DISCLOSURE,
                model_info={
                    "modelsDir": str(args.models) if args.models else os.environ.get(
                        "STILLTYPE_MODELS", ""
                    ),
                    "requestedDevice": args.device,
                    "motionThreshold": args.motion_threshold,
                    "pauseThreshold": args.motion_threshold * 5.0 / 8.0,
                },
                item_results=scored,
            )
        else:
            variant = "fp32" if args.decoder == "ctc-fp32" else "int8"
            assert args.mobile_model_dir is not None
            scored, model_info = run_ctc(
                selected,
                manifest.manifest_dir,
                variant=variant,
                mobile_model_dir=args.mobile_model_dir,
                beam_size=args.ctc_beam_size,
                token_top_k=args.ctc_token_top_k,
                activity_gate=args.activity_gate or "legacy",
            )
            model_info["motionThreshold"] = args.motion_threshold
            model_info["activityGate"] = args.activity_gate or "legacy"
            if args.activity_gate == "ctc-evidence":
                from activity_gate import (CANDIDATE_MOTION, MIN_EMISSIONS, MIN_MEAN_PROBABILITY,
                                           MIN_MOVING_SECONDS, PADDING_SECONDS, STRONG_MOTION)
                model_info["activityGateThresholds"] = {
                    "strongMotion": STRONG_MOTION, "candidateMotion": CANDIDATE_MOTION,
                    "minMovingSeconds": MIN_MOVING_SECONDS, "paddingSeconds": PADDING_SECONDS,
                    "minEmissions": MIN_EMISSIONS, "minMeanProbability": MIN_MEAN_PROBABILITY,
                }
            results = finalize_results(
                dataset_id=manifest.datasetId,
                manifest_sha256=manifest.sha256,
                split=args.split,
                decoder=args.decoder,
                decode="prefix-beam" if args.ctc_beam_size > 1 else "greedy",
                device="cpu",
                provider="CPUExecutionProvider",
                beam_size=args.ctc_beam_size,
                disclosure=CTC_DISCLOSURE,
                model_info=model_info,
                item_results=scored,
            )
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        with output.open("x", encoding="utf-8") as sink:
            json.dump(results, sink, indent=2, ensure_ascii=False)
            sink.write("\n")
    except FileExistsError:
        print(f"error: refusing to overwrite existing result: {args.output}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: cannot write output: {exc}", file=sys.stderr)
        return 2

    summary = {
        "datasetId": results["datasetId"],
        "split": results["split"],
        "decoder": results["decoder"],
        "clips": results["metrics"]["total"]["clips"],
        "wer": results["metrics"]["total"]["wer"],
        "exactMatchRate": results["metrics"]["total"]["exactMatchRate"],
        "targetMet": results["targetMet"],
        "output": str(args.output),
    }
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
