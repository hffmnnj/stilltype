"""Stilltype local visual-speech (lipreading) inference with USR 2.0 Base+.

Model: USR 2.0 high-resource Base+ (2026, MIT (c) 2026 Alexandros Haliassos;
see usr2/LICENSE). Visual-only inference follows the official visual path:
grayscale mouth crops at 88 px, normalised by (x - 0.421) / 0.165, encoded
with ``model.encoder(xs_v=video_input)`` for input (1, T, 88, 88), then
official ESPnet ``BatchBeamSearch`` with modality ``'v'``. The vendored
``espnet`` modules keep their Apache-2.0 headers; the only local change
is a Python 3.12+ import fallback in ``mask.py`` (documented there).

The fixed checkpoint is ``MODEL_DIR/usr2/baseplus.pth``; readiness is the
installer-written marker ``MODEL_DIR/installed.json``::

    {"modelId": "usr2-baseplus-2026", "releaseYear": 2026,
     "checkpoint": "usr2/baseplus.pth", "sha256": "<hex>"}

Everything in this module except ``USR2Visual`` is model-independent: clip
parsing, mouth-ROI gating, and vocabulary mapping. ``USR2Visual`` is
the replaceable model adapter behind ``__call__(rois, fps) -> (text, ms)``.

Speech input handling: clips are directories owned by Rust with
``frames.jsonl`` (one JSON object per line: ``jpeg`` base64 without data-URL
prefix, ``timestamp`` in ms from recording start, ``landmarks`` null or the
four image-pixel points image-left eye, image-right eye, nose base, mouth
centre). This module never invents a transcript: invalid, too-short,
faceless or still clips are reported through the ``error`` field.
Recognition inference makes no network calls: it is strictly visual
recognition plus vocabulary, and any optional cleanup runs in the Rust
layer after Python recognition.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS_PATH = os.path.join(HERE, "usr2", "labels", "unigram1000_units.txt")
MEAN_FACE_PATH = os.path.join(HERE, "usr2", "20words_mean_face.npy")

MODEL_ID = "usr2-baseplus-2026"
MODEL_RELEASE_YEAR = 2026
CHECKPOINT_REL = os.path.join("usr2", "baseplus.pth")

# USR 2.0 high-resource Base+ backbone (conf/model/backbone/...
# resnet_transformer_baseplus.yaml) and decode defaults
# (conf/decode/default.yaml).
BASEPLUS = {
    "odim": 1049,
    "idim": 512,
    "adim": 768,
    "aheads": 12,
    "eunits": 3072,
    "elayers": 12,
    "ddim": 768,
    "dheads": 12,
    "dunits": 3072,
    "dlayers": 6,
    "gamma_init": 0.1,
    "ctc_rel_weight": 0.1,
}
DECODE_DEFAULTS = {"ctc_weight": 0.1, "penalty": 0.0, "minlenratio": 0.0, "maxlenratio": 1.0}

MODEL_FPS = 25.0
FALLBACK_FPS = 25.0
DEFAULT_BEAM_SIZE = 5
DEFAULT_PAUSE_SECONDS = 0.7

#: Error codes that may appear in TranscriptionResult.error.
CLIP_ERRORS = ("invalid-clip", "too-short", "no-face", "no-speech")


class ClipError(Exception):
    """A clip problem, reported inside TranscriptionResult.error."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class ModelUnavailable(Exception):
    """Infrastructure failure (missing weights, torch, opencv)."""


class GpuUnavailable(ModelUnavailable):
    """An explicit GPU request that cannot be satisfied."""


def models_dir() -> str:
    """Weight root. The worker sets this from `--models` before import."""
    return os.environ.get("STILLTYPE_MODELS") or ""


def checkpoint_path() -> str:
    return os.path.join(models_dir(), CHECKPOINT_REL)


def read_marker() -> dict | None:
    path = os.path.join(models_dir(), "installed.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def models_present() -> bool:
    """True when the installer marker names a checkpoint file that exists.

    Hash verification happens at install time; this is a readiness check.
    """
    marker = read_marker()
    if not marker:
        return False
    model_id = marker.get("modelId", MODEL_ID)
    if model_id != MODEL_ID:
        return False
    checkpoint = marker.get("checkpoint") or CHECKPOINT_REL
    full = os.path.join(models_dir(), checkpoint)
    return os.path.isfile(full) and os.path.getsize(full) > 0


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def _cuda_ok() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _mps_ok() -> bool:
    try:
        import torch

        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def resolve_device(want: str | None, _cuda: bool | None = None, _mps: bool | None = None) -> str:
    """Map an 'auto' | 'gpu' | 'cpu' (or explicit torch device) want to a device.

    GPU-first: 'auto' prefers CUDA, then MPS, then CPU; 'gpu' requires CUDA
    or MPS and raises GpuUnavailable when neither exists, so an unavailable
    GPU is never silently classified as ready.
    """
    w = (want or "auto").strip().lower()
    if w in ("", "auto"):
        cuda = _cuda if _cuda is not None else _cuda_ok()
        if cuda:
            return "cuda"
        mps = _mps if _mps is not None else _mps_ok()
        if mps:
            return "mps"
        return "cpu"
    if w == "gpu":
        cuda = _cuda if _cuda is not None else _cuda_ok()
        if cuda:
            return "cuda"
        mps = _mps if _mps is not None else _mps_ok()
        if mps:
            return "mps"
        raise GpuUnavailable("gpu-unavailable: explicit GPU inference requested but no CUDA or MPS device exists")
    if w == "cpu":
        return "cpu"
    return want  # explicit torch device string (e.g. "cuda:1"); validated at load


def requested_device(settings: dict | None) -> str:
    """Effective device want: STILLTYPE_DEVICE wins, then inferenceDevice."""
    env = (os.environ.get("STILLTYPE_DEVICE") or "").strip()
    if env and env.lower() != "auto":
        return env
    if settings and isinstance(settings.get("inferenceDevice"), str):
        return settings["inferenceDevice"]
    return "auto"


def token_list() -> list[str]:
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        units = [line.split()[0] for line in f.read().splitlines() if line.split()]
    return ["<blank>"] + units + ["<eos>"]


class USR2Visual:
    """Model adapter: USR 2.0 Base+ encoder with official ESPnet beam search.

    Visual-only (modality 'v'). ``__call__(rois, fps)`` returns
    ``(text, inference_ms)``.
    """

    def __init__(self, device: str | None = None, beam_size: int | None = None):
        import torch

        from espnet.asr.asr_utils import parse_hypothesis
        from espnet.nets.batch_beam_search import BatchBeamSearch
        from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
        from espnet.nets.scorers.length_bonus import LengthBonus

        self.device = device or resolve_device(None)
        self.beam_size = beam_size or DEFAULT_BEAM_SIZE
        ckpt_path = checkpoint_path()
        if not (ckpt_path and os.path.isfile(ckpt_path) and os.path.getsize(ckpt_path) > 0):
            where = models_dir() or "(no models dir configured)"
            raise ModelUnavailable(f"models-missing: checkpoint absent under {where}")
        from types import SimpleNamespace

        backbone = SimpleNamespace(**BASEPLUS)
        model = E2E(len(token_list()), backbone)
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if not isinstance(raw, dict):
            raise ModelUnavailable("models-missing: checkpoint is not a state dict")
        state: dict = {}
        for key, value in raw.items():
            if key.startswith("_orig_mod."):
                key = key[len("_orig_mod.") :]
            if not key.startswith("model.backbone."):
                continue  # training-only tensors (EMA target, unlabelled heads)
            state[key[len("model.backbone.") :]] = value
        model.load_state_dict(state, strict=True)
        model.eval().to(self.device)
        self._model = model
        self._parse = parse_hypothesis
        self._tokens = token_list()
        self._build_search(BatchBeamSearch, LengthBonus)

    def _build_search(self, BatchBeamSearch, LengthBonus) -> None:
        scorers = self._model.scorers()
        scorers["length_bonus"] = LengthBonus(len(self._tokens))
        weights = dict(
            decoder=1.0 - DECODE_DEFAULTS["ctc_weight"],
            ctc=DECODE_DEFAULTS["ctc_weight"],
            length_bonus=DECODE_DEFAULTS["penalty"],
        )
        self.search = BatchBeamSearch(
            beam_size=self.beam_size,
            vocab_size=len(self._tokens),
            weights=weights,
            scorers=scorers,
            sos=len(self._tokens) - 1,
            eos=len(self._tokens) - 1,
            token_list=self._tokens,
            pre_beam_score_key=None if DECODE_DEFAULTS["ctc_weight"] == 1.0 else "decoder",
        )
        try:
            self.search.to(self.device)
        except Exception:  # noqa: BLE001 -- some scorers have no parameters; device set at use
            pass

    def set_beam(self, beam_size: int) -> None:
        """Retune the beam without reloading weights."""
        from espnet.nets.batch_beam_search import BatchBeamSearch
        from espnet.nets.scorers.length_bonus import LengthBonus

        self.beam_size = beam_size
        self._build_search(BatchBeamSearch, LengthBonus)

    @staticmethod
    def to_input(rois: np.ndarray, fps: float):
        """(T, 96, 96) uint8 at `fps` -> (1, T', 88, 88) normalised float."""
        import torch

        x = torch.from_numpy(np.ascontiguousarray(rois)).float()
        if fps and abs(fps - MODEL_FPS) > 0.5:
            n = max(1, int(round(len(x) * MODEL_FPS / fps)))
            x = x[torch.linspace(0, len(x) - 1, n).round().long()]
        x = x[:, 4:92, 4:92] / 255.0  # centre crop 88
        x = (x - 0.421) / 0.165
        return x.unsqueeze(0)

    def encode(self, rois: np.ndarray, fps: float):
        import torch

        with torch.no_grad():
            video = self.to_input(rois, fps).to(self.device)
            feat = self._model.encoder(xs_v=video)
        return feat

    def decode(self, feat) -> str:
        hyps = self.search(
            x=feat.squeeze(0),
            modality="v",
            maxlenratio=DECODE_DEFAULTS["maxlenratio"],
            minlenratio=DECODE_DEFAULTS["minlenratio"],
        )
        if not hyps:
            return ""
        text, _, _, _ = self._parse(hyps[0].asdict(), self._tokens)
        return text.replace("<eos>", "").replace("▁", " ").strip()

    def __call__(self, rois: np.ndarray, fps: float) -> tuple[str, int]:
        import torch

        t0 = time.perf_counter()
        with torch.inference_mode():
            text = self.decode(self.encode(rois, fps))
        return text, int((time.perf_counter() - t0) * 1000)


# Lazily initialised, reused across transcribe requests in this process.
_MODEL: USR2Visual | None = None
_MODEL_BEAM = 0
_MODEL_DEVICE = ""
_LOAD_ERROR: str | None = None


def ensure_model(beam_size: int, device: str) -> USR2Visual:
    """Return the shared model, loading on first use. Reloads on change."""
    global _MODEL, _MODEL_BEAM, _MODEL_DEVICE, _LOAD_ERROR
    beam = max(1, min(100, int(beam_size or DEFAULT_BEAM_SIZE)))
    if _MODEL is not None and _MODEL_DEVICE == device:
        if _MODEL_BEAM != beam:
            _MODEL.set_beam(beam)
            _MODEL_BEAM = beam
        return _MODEL
    try:
        _MODEL = USR2Visual(device=device, beam_size=beam)
    except Exception as e:  # noqa: BLE001 -- surfaced verbatim in health/transcribe
        _LOAD_ERROR = f"{type(e).__name__}: {e}"
        _MODEL, _MODEL_BEAM, _MODEL_DEVICE = None, 0, ""
        raise
    _MODEL_BEAM, _MODEL_DEVICE, _LOAD_ERROR = beam, device, None
    return _MODEL


def health_snapshot() -> dict:
    """Never loads the model; reports whether a transcribe could succeed."""
    device = resolve_device(None)
    installed = models_present()
    if _LOAD_ERROR is not None:
        return {"ready": False, "device": device, "modelInstalled": installed, "error": _LOAD_ERROR}
    if _MODEL is not None:
        return {"ready": True, "device": _MODEL.device, "modelInstalled": installed, "error": None}
    if not torch_available():
        return {
            "ready": False,
            "device": device,
            "modelInstalled": installed,
            "error": "torch-unavailable: torch is not importable in the engine runtime",
        }
    if not installed:
        return {"ready": False, "device": device, "modelInstalled": False, "error": None}
    return {"ready": True, "device": device, "modelInstalled": True, "error": None}


# ---------------------------------------------------------------------------
# Clip pipeline (model-independent)
# ---------------------------------------------------------------------------

_MEAN_FACE: np.ndarray | None = None


def mean_face() -> np.ndarray:
    global _MEAN_FACE
    if _MEAN_FACE is None:
        _MEAN_FACE = np.load(MEAN_FACE_PATH)
    return _MEAN_FACE


def stable_reference() -> np.ndarray:
    """Four alignment points from the 68-point mean face: left-eye centre,
    right-eye centre, nose base, mouth centre."""
    mf = mean_face()
    return np.vstack(
        [
            mf[36:42].mean(0),
            mf[42:48].mean(0),
            mf[30:36].mean(0),
            mf[48:68].mean(0),
        ]
    ).astype(np.float32)


def _as_landmarks(value) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise ClipError("invalid-clip") from e
    if arr.shape != (4, 2) or not np.all(np.isfinite(arr)):
        raise ClipError("invalid-clip")
    return arr.astype(np.float32)


def load_clip_records(clip_dir: str) -> tuple[list[dict], int]:
    """Parse frames.jsonl. Returns (records, durationMs).

    Raises ClipError("invalid-clip") for a missing/unreadable clip, bad
    JSON, bad base64, bad landmarks, or an empty/non-chronological clip.
    Pure Python + numpy: needs no opencv, torch, or weights.
    """
    if not isinstance(clip_dir, str) or not os.path.isdir(clip_dir):
        raise ClipError("invalid-clip")
    try:
        with open(os.path.join(clip_dir, "frames.jsonl"), "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError as e:
        raise ClipError("invalid-clip") from e
    records: list[dict] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ClipError("invalid-clip") from e
        if not isinstance(obj, dict):
            raise ClipError("invalid-clip")
        jpeg_b64 = obj.get("jpeg")
        ts = obj.get("timestamp")
        if (
            not isinstance(jpeg_b64, str)
            or not isinstance(ts, (int, float))
            or isinstance(ts, bool)
            or not math.isfinite(ts)
        ):
            raise ClipError("invalid-clip")
        try:
            jpeg = base64.b64decode(jpeg_b64, validate=True)
        except (binascii.Error, ValueError) as e:
            raise ClipError("invalid-clip") from e
        if not jpeg:
            raise ClipError("invalid-clip")
        records.append(
            {"jpeg": jpeg, "timestamp": float(ts), "landmarks": _as_landmarks(obj.get("landmarks"))}
        )
    if not records:
        raise ClipError("invalid-clip")
    duration_ms = int(round(records[-1]["timestamp"] - records[0]["timestamp"]))
    if duration_ms < 0:
        raise ClipError("invalid-clip")
    return records, duration_ms


def fps_from_records(records: list[dict]) -> float:
    if len(records) < 2:
        return FALLBACK_FPS
    span_s = (records[-1]["timestamp"] - records[0]["timestamp"]) / 1000.0
    if span_s <= 0:
        return FALLBACK_FPS
    return (len(records) - 1) / span_s


def check_clip(n_frames: int, face_frames: int, fps: float) -> str | None:
    """Pre-decode gates. Returns an error code, or None when decodable."""
    if n_frames < max(1, fps * 0.5):
        return "too-short"
    if face_frames < n_frames * 0.5:
        return "no-face"
    return None


def decode_records(records: list[dict]) -> tuple[np.ndarray, list]:
    """JPEG bytes -> (T, H, W) uint8 grayscale + aligned landmarks.

    Undecodable frames are skipped (Rust owns the clip; one bad frame must
    not kill a dictation). Raises ClipError("invalid-clip") when nothing
    decodes, ModelUnavailable when opencv is missing from the runtime.
    """
    try:
        import cv2
    except Exception as e:
        raise ModelUnavailable(f"opencv-unavailable: {e}") from e
    grays: list[np.ndarray] = []
    lms: list = []
    skipped = 0
    for r in records:
        buf = np.frombuffer(r["jpeg"], dtype=np.uint8)
        try:
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except Exception:  # noqa: BLE001 -- corrupt frame, skip it
            img = None
        if img is None:
            skipped += 1
            continue
        grays.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        lms.append(r["landmarks"])
    if skipped:
        print(f"stilltype: skipped {skipped} undecodable frame(s)", file=sys.stderr)
    if not grays:
        raise ClipError("invalid-clip")
    try:
        frames = np.stack(grays)
    except ValueError as e:
        raise ClipError("invalid-clip") from e
    return frames, lms


def interpolate_landmarks(lms: list) -> list | None:
    """Fill faceless frames by linear interpolation; hold the ends."""
    valid = [i for i, lm in enumerate(lms) if lm is not None]
    if not valid:
        return None
    out: list = [None if lm is None else np.asarray(lm, np.float32) for lm in lms]
    for a, b in zip(valid, valid[1:]):
        for k in range(1, b - a):
            out[a + k] = out[a] + (out[b] - out[a]) * k / (b - a)
    out[: valid[0]] = [out[valid[0]]] * valid[0]
    out[valid[-1] + 1 :] = [out[valid[-1]]] * (len(out) - valid[-1] - 1)
    return out


def align_mouths(frames: np.ndarray, lms: list) -> np.ndarray | None:
    """Warp each grayscale frame so its four landmarks land on the mean-face
    reference, then cut a 96x96 patch around the mouth. Returns (T, 96, 96)
    uint8, or None when no frame has a face."""
    import cv2

    filled = interpolate_landmarks(lms)
    if filled is None:
        return None
    ref = stable_reference()
    size, half = 256, 6
    rois = np.zeros((len(frames), 96, 96), np.uint8)
    prev_m = None
    for i, frame in enumerate(frames):
        m = min(half, i, len(frames) - 1 - i)
        smooth = np.mean(filled[i - m : i + m + 1], axis=0)
        smooth += filled[i].mean(0) - smooth.mean(0)
        warp, _ = cv2.estimateAffinePartial2D(smooth, ref, method=cv2.LMEDS)
        if warp is None:
            warp = prev_m if prev_m is not None else cv2.estimateAffinePartial2D(smooth, ref)[0]
        prev_m = warp
        big = cv2.warpAffine(
            frame, warp, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        mouth = smooth[3] @ warp[:, :2].T + warp[:, 2]
        cx, cy = float(mouth[0]), float(mouth[1])
        x0 = int(round(float(np.clip(cx - 48, 0, size - 96))))
        y0 = int(round(float(np.clip(cy - 48, 0, size - 96))))
        rois[i] = big[y0 : y0 + 96, x0 : x0 + 96]
    return rois


SPEAKING = 8.0
STILL = 5.0
SPEECH_PAD_S = 0.4


def mouth_motion(rois: np.ndarray, fps: float) -> np.ndarray:
    """Per-frame motion from mean-removed mouth crops 0.1 s apart, smoothed."""
    inner = rois[:, 24:72, 16:80].astype(np.float32)
    inner -= inner.mean((1, 2), keepdims=True)
    lag = max(1, int(round(fps * 0.1)))
    m = np.zeros(len(inner), np.float32)
    if len(inner) > lag:
        m[lag:] = np.abs(inner[lag:] - inner[:-lag]).mean((1, 2))
        m[:lag] = m[lag]
    width = max(1, int(fps * 0.2))
    return np.convolve(m, np.ones(width) / width, mode="same")


def speech_span(m: np.ndarray, fps: float, min_seconds: float = 0.3):
    """(first, last+1) frames of mouth movement with padding, or None when
    the mouth barely moved. The model invents fluent sentences from a still
    mouth, so still stretches must never reach it."""
    moving = np.nonzero(m > SPEAKING)[0]
    if len(moving) < min_seconds * fps:
        return None
    pad = int(SPEECH_PAD_S * fps)
    return max(0, int(moving[0]) - pad), min(len(m), int(moving[-1]) + 1 + pad)


def first_pause(m: np.ndarray, fps: float, pause_seconds: float, min_speech: float = 1.5):
    """Frame where the first pause after `min_speech` seconds of movement
    begins, or None. Used only for interim (non-final) decodes."""
    need, run, spoke = int(pause_seconds * fps), 0, 0
    for i, v in enumerate(m):
        if v > SPEAKING:
            spoke += 1
        run = run + 1 if v < STILL else 0
        if run >= need and spoke >= min_speech * fps:
            return i - run + 1
    return None


def gate_rois(rois: np.ndarray, fps: float):
    """Mouth-motion gate. Returns the speech span, or None for a still mouth."""
    return speech_span(mouth_motion(rois, fps), fps)


_WORD_RE = re.compile(r"^(\W*)(.*?)(\W*)$", re.UNICODE)


def _split_edge(token: str) -> tuple[str, str, str]:
    m = _WORD_RE.match(token)
    if not m:
        return "", token, ""
    return m.group(1), m.group(2), m.group(3)


def vocab_pairs(vocabulary) -> list[tuple[list[str], str]]:
    """Vocabulary entries to (match, output) pairs.

    Matching runs against the entry's `word` (case-insensitive); the output
    is its `replacement` when nonempty, otherwise the `word` itself as
    typed. A sooner duplicate source keeps the first output.
    """
    pairs: list[tuple[list[str], str]] = []
    seen: set[tuple[str, ...]] = set()
    for w in vocabulary or []:
        if not isinstance(w, dict):
            continue
        src = str(w.get("word") or "").strip()
        if not src:
            continue
        key = tuple(src.lower().split())
        if not key or key in seen:
            continue
        seen.add(key)
        out = str(w.get("replacement") or "").strip() or src
        pairs.append((list(key), out))
    return pairs


def apply_vocabulary(text: str, pairs: list[tuple[list[str], str]]) -> str:
    """Exact case-insensitive word/phrase replacement.

    The longest source phrase wins at each position; leading/trailing
    punctuation is preserved. No look-alike or fuzzy matching: only text
    that already reads as the source (modulo case) is replaced.
    """
    if not pairs or not text:
        return text
    ordered = sorted(pairs, key=lambda p: -len(p[0]))
    tokens = text.split()
    cores: list[str] = []
    for tok in tokens:
        _, core, _ = _split_edge(tok)
        cores.append(core.lower())
    out: list[str] = []
    i = 0
    while i < len(tokens):
        for want, original in ordered:
            n = len(want)
            if n == 0 or i + n > len(tokens):
                continue
            window = cores[i : i + n]
            if any(not c for c in window):
                continue
            if window == want:
                lead, _, _ = _split_edge(tokens[i])
                _, _, trail = _split_edge(tokens[i + n - 1])
                out.append(lead + original + trail)
                i += n
                break
        else:
            out.append(tokens[i])
            i += 1
    return " ".join(out)


def transcribe_clip(
    clip_dir: str,
    *,
    settings: dict | None = None,
    vocabulary: list | None = None,
    model_factory=None,
    final: bool = True,
) -> dict:
    """Full pipeline: parse -> gates -> decode -> model -> vocab.

    Clip problems return a result dict with `error` set (never raise).
    Only infrastructure failures raise ModelUnavailable. `model_factory`
    (taking beam size and device) exists for tests; production uses
    ensure_model. The result carries the actual `device` used.
    Recognition makes no network calls; extra settings arriving over
    stdio (such as legacy cleanup keys owned by the Rust layer) are
    ignored. Returns real text and raw exactly as recognised.
    """
    cfg = {
        "beamSize": DEFAULT_BEAM_SIZE,
        "pauseSeconds": DEFAULT_PAUSE_SECONDS,
        "inferenceDevice": "auto",
    }
    if settings:
        for k in cfg:
            if k in settings:
                cfg[k] = settings[k]
    try:
        beam = int(cfg["beamSize"])
    except (TypeError, ValueError):
        beam = DEFAULT_BEAM_SIZE
    beam = max(1, min(100, beam))
    try:
        pause = float(cfg["pauseSeconds"])
    except (TypeError, ValueError):
        pause = DEFAULT_PAUSE_SECONDS

    empty = {"text": "", "raw": "", "durationMs": 0, "inferenceMs": 0, "frames": 0,
             "error": None, "device": ""}
    try:
        records, duration_ms = load_clip_records(clip_dir)
    except ClipError as e:
        return {**empty, "error": e.code}
    n = len(records)
    fps = fps_from_records(records)
    faces = sum(1 for r in records if r["landmarks"] is not None)
    err = check_clip(n, faces, fps)
    if err:
        return {**empty, "durationMs": duration_ms, "frames": n, "error": err}

    device = resolve_device(requested_device(settings if isinstance(settings, dict) else None))
    frames, lms = decode_records(records)
    rois = align_mouths(frames, lms)
    if rois is None:  # unreachable after the face gate; defensive
        return {**empty, "durationMs": duration_ms, "frames": n, "error": "no-face"}
    motion = mouth_motion(rois, fps)
    span = speech_span(motion, fps)
    if span is None:
        return {**empty, "durationMs": duration_ms, "frames": n, "error": "no-speech"}
    s0, s1 = int(span[0]), int(span[1])
    if not final:
        cut = first_pause(motion, fps, pause)
        if cut is not None and cut > s0:
            s1 = min(s1, cut + int(0.25 * fps))
    factory = model_factory or ensure_model
    model = factory(beam, device)
    raw, vsr_ms = model(rois[s0:s1], fps) if s1 > s0 else ("", 0)
    raw = raw or ""
    pairs = vocab_pairs(vocabulary)
    fixed = apply_vocabulary(raw.lower(), pairs) if raw else ""
    text = (fixed[:1].upper() + fixed[1:]) if fixed else ""
    if not raw:
        text = ""
    actual_device = getattr(model, "device", device)
    return {
        "text": text,
        "raw": raw,
        "durationMs": duration_ms,
        "inferenceMs": int(vsr_ms),
        "frames": n,
        "error": None,
        "device": actual_device,
    }
