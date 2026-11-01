"""Focused checks for the Stilltype engine worker protocol and clip gates.

Needs no model weights. torch is optional (health works without it); opencv
is optional too: pre-decode gates and the motion math need only numpy, so a
bare `cv2` stub is installed for import when the real one is absent, and the
tests that must pass the decode boundary say so explicitly.

Contracts guarded here:
- every malformed request gets {ok:false} with a named error, never silence;
- the persistent protocol emits exactly one stdout line per request;
- clips with no usable face, too few frames, or a still mouth are rejected
  with explicit TranscriptionResult.error codes and never reach the model;
- interim (final=false) decodes truncate at the first pause; final decodes
  run the whole speech span;
- vocabulary matches the source word/phrase case-insensitively and emits
  the replacement (or the word as typed), with nothing fuzzy.
"""

import base64
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest

ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ENGINE_DIR not in sys.path:
    sys.path.insert(0, ENGINE_DIR)

import inference
import worker

try:
    import cv2  # noqa: F401
except Exception:
    # Import-only stub: motion/speech-span math is pure numpy, but align
    # code does `import cv2` at call time only, so nothing is needed here.
    # Decode itself needs the real thing.
    sys.modules.setdefault("cv2", types.ModuleType("cv2"))


def _landmarks():
    return [[100.0, 100.0], [200.0, 100.0], [150.0, 150.0], [150.0, 200.0]]


def _frame(ts_ms, landmarks="default"):
    lm = _landmarks() if landmarks == "default" else landmarks
    return {"jpeg": base64.b64encode(b"not-a-jpeg").decode(), "timestamp": ts_ms, "landmarks": lm}


def _write_clip(records):
    d = tempfile.mkdtemp(prefix="stilltype-clip-")
    with open(os.path.join(d, "frames.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return d


def _write_marked_models(checkpoint_bytes=b"x"):
    d = tempfile.mkdtemp(prefix="stilltype-models-")
    os.makedirs(os.path.join(d, "usr2"))
    with open(os.path.join(d, "usr2", "baseplus.pth"), "wb") as f:
        f.write(checkpoint_bytes)
    marker = {"modelId": "usr2-baseplus-2026", "releaseYear": 2026,
              "checkpoint": "usr2/baseplus.pth", "sha256": "test"}
    with open(os.path.join(d, "installed.json"), "w", encoding="utf-8") as f:
        json.dump(marker, f)
    return d


class ProtocolErrors(unittest.TestCase):
    def test_malformed_json_returns_named_error(self):
        resp = worker.handle_line("this is not json", inference)
        self.assertFalse(resp["ok"])
        self.assertIsNone(resp["id"])
        self.assertTrue(resp["error"].startswith("invalid-json"))

    def test_unknown_command_returns_named_error(self):
        resp = worker.handle_line(json.dumps({"id": "1", "command": "frobnicate"}), inference)
        self.assertEqual(resp, {"id": "1", "ok": False, "error": "unknown-command: 'frobnicate'"})

    def test_transcribe_without_clip_dir_is_protocol_error(self):
        resp = worker.handle_line(json.dumps({"id": "2", "command": "transcribe"}), inference)
        self.assertEqual(resp, {"id": "2", "ok": False, "error": "missing-clip-dir"})

    def test_blank_lines_are_ignored(self):
        self.assertIsNone(worker.handle_line("   \n", inference))

    def test_health_emits_single_json_line(self):
        """End to end through the real stdio loop: one line in, one line out."""
        env = dict(os.environ)
        env["STILLTYPE_MODELS"] = ""
        proc = subprocess.run(
            [sys.executable, os.path.join(ENGINE_DIR, "worker.py"), "--once"],
            input=json.dumps({"id": "h", "command": "health"}) + "\n",
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr[-2000:])
        lines = proc.stdout.splitlines()
        self.assertEqual(len(lines), 1, msg=f"stdout must be exactly one line, got: {proc.stdout!r}")
        resp = json.loads(lines[0])
        self.assertEqual(resp["id"], "h")
        self.assertTrue(resp["ok"])
        result = resp["result"]
        self.assertEqual(set(result), {"ready", "device", "modelInstalled", "error"})
        # No marker configured: must report not-installed, not crash, not load.
        self.assertFalse(result["modelInstalled"])
        self.assertFalse(result["ready"])
        self.assertIsNone(result["error"])
        self.assertIsInstance(result["device"], str)

    def test_models_present_requires_marker_and_checkpoint(self):
        prev = os.environ.get("STILLTYPE_MODELS")
        try:
            os.environ["STILLTYPE_MODELS"] = ""
            self.assertFalse(inference.models_present())
            os.environ["STILLTYPE_MODELS"] = _write_marked_models()
            self.assertTrue(inference.models_present())
        finally:
            if prev is None:
                os.environ.pop("STILLTYPE_MODELS", None)
            else:
                os.environ["STILLTYPE_MODELS"] = prev


class DeviceSelection(unittest.TestCase):
    def test_auto_prefers_cuda_then_mps_then_cpu(self):
        self.assertEqual(inference.resolve_device("auto", _cuda=True, _mps=True), "cuda")
        self.assertEqual(inference.resolve_device("auto", _cuda=False, _mps=True), "mps")
        self.assertEqual(inference.resolve_device("auto", _cuda=False, _mps=False), "cpu")
        self.assertEqual(inference.resolve_device(None, _cuda=False, _mps=False), "cpu")

    def test_explicit_cpu_sticks(self):
        self.assertEqual(inference.resolve_device("cpu", _cuda=True, _mps=True), "cpu")

    def test_explicit_gpu_never_silently_falls_back(self):
        self.assertEqual(inference.resolve_device("gpu", _cuda=True, _mps=False), "cuda")
        self.assertEqual(inference.resolve_device("gpu", _cuda=False, _mps=True), "mps")
        with self.assertRaises(inference.GpuUnavailable):
            inference.resolve_device("gpu", _cuda=False, _mps=False)


class ClipGates(unittest.TestCase):
    def test_missing_clip_dir_reports_invalid_clip_result(self):
        resp = worker.handle_line(
            json.dumps({"id": "3", "command": "transcribe", "clipDir": "/nonexistent/clip"}),
            inference,
        )
        self.assertTrue(resp["ok"])
        result = resp["result"]
        self.assertEqual(result["error"], "invalid-clip")
        self.assertEqual(result["text"], "")
        self.assertEqual(result["frames"], 0)

    def test_too_short_clip_is_rejected_before_decode(self):
        # 5 frames at 40 ms spacing; base64 is structurally valid but decode
        # is never reached because the duration gate fires first (no cv2 needed).
        clip = _write_clip([_frame(i * 40.0) for i in range(5)])
        result = inference.transcribe_clip(clip)
        self.assertEqual(result["error"], "too-short")
        self.assertEqual(result["frames"], 5)
        self.assertEqual(result["text"], "")

    def test_no_face_clip_is_rejected_before_decode(self):
        clip = _write_clip([_frame(i * 40.0, landmarks=None) for i in range(40)])
        result = inference.transcribe_clip(clip)
        self.assertEqual(result["error"], "no-face")
        self.assertEqual(result["frames"], 40)

    def test_malformed_landmarks_reject_the_clip(self):
        clip = _write_clip([_frame(i * 40.0, landmarks=[[0, 0]]) for i in range(40)])
        result = inference.transcribe_clip(clip)
        self.assertEqual(result["error"], "invalid-clip")

    def test_still_mouth_never_reaches_the_model(self):
        """A constant mouth must yield no-speech without any model call.

        Stand-in used: decode is stubbed at the opencv boundary (unavailable
        in this env) to constant-grey frames, and alignment to constant ROIs;
        the motion/speech-span math under test is the real code.
        """
        import numpy as np

        clip = _write_clip([_frame(i * 40.0) for i in range(40)])
        real_decode = inference.decode_records
        real_align = inference.align_mouths

        def exploded(_beam, _device):
            raise AssertionError("model must not load for a still mouth")

        try:
            inference.decode_records = lambda _records: (
                np.full((40, 120, 120), 128, np.uint8),
                [np.asarray(_landmarks(), np.float32)] * 40,
            )
            inference.align_mouths = lambda _frames, _lms: np.full((40, 96, 96), 128, np.uint8)
            result = inference.transcribe_clip(clip, model_factory=exploded)
        finally:
            inference.decode_records = real_decode
            inference.align_mouths = real_align
        self.assertEqual(result["error"], "no-speech")
        self.assertEqual(result["raw"], "")

    def test_interim_decode_truncates_at_first_pause(self):
        """final=false stops at the first pause; final decodes the span."""
        import numpy as np

        clip = _write_clip([_frame(i * 40.0) for i in range(200)])
        real_decode = inference.decode_records
        real_align = inference.align_mouths
        seen = {}

        def fake_model(beam, device):
            class M:
                def __call__(self, rois, fps):
                    seen[len(rois)] = True
                    return "HELLO WORLD", 7

                device = "cpu"

            return M()

        rois = np.full((200, 96, 96), 128, np.uint8)
        # First 100 frames move (alternating halves), rest are still.
        for i in range(100):
            if (i // 2) % 2 == 0:
                rois[i, 24:72, 16:48] = 200
                rois[i, 24:72, 48:80] = 60
            else:
                rois[i, 24:72, 16:48] = 60
                rois[i, 24:72, 48:80] = 200
        try:
            inference.decode_records = lambda _records: (
                np.zeros((200, 120, 120), np.uint8),
                [np.asarray(_landmarks(), np.float32)] * 200,
            )
            inference.align_mouths = lambda _frames, _lms: rois
            interim = inference.transcribe_clip(clip, model_factory=fake_model, final=False)
            full = inference.transcribe_clip(clip, model_factory=fake_model, final=True)
        finally:
            inference.decode_records = real_decode
            inference.align_mouths = real_align
        self.assertIsNone(interim["error"])
        self.assertIsNone(full["error"])
        self.assertEqual(interim["device"], "cpu")
        interim_len = [k for k in seen][0]
        # Both decodes ran; the interim one must have stopped well before
        # the end of the 200-frame clip while final used (nearly) all of it.
        self.assertLess(interim_len, 200)
        self.assertGreater(max(seen), interim_len)


class MotionGateMath(unittest.TestCase):
    def test_still_rois_have_no_speech_span(self):
        import numpy as np

        rois = np.full((30, 96, 96), 128, np.uint8)
        self.assertIsNone(inference.gate_rois(rois, 25.0))

    def test_moving_rois_have_a_speech_span(self):
        import numpy as np

        rois = np.full((30, 96, 96), 128, np.uint8)
        # Alternate which half of the mouth window is bright every 2 frames.
        # (Uniform flicker would not do: motion() removes each crop's mean
        # brightness first.) Frame pairs 0.1 s apart then differ strongly,
        # like a real mouth opening.
        for i in range(30):
            if (i // 2) % 2 == 0:
                rois[i, 24:72, 16:48] = 200
                rois[i, 24:72, 48:80] = 60
            else:
                rois[i, 24:72, 16:48] = 60
                rois[i, 24:72, 48:80] = 200
        span = inference.gate_rois(rois, 25.0)
        self.assertIsNotNone(span)
        self.assertLess(span[0], span[1])


class VocabularyMapping(unittest.TestCase):
    def test_source_phrase_emits_replacement(self):
        pairs = inference.vocab_pairs(
            [{"id": "a", "word": "still type", "replacement": "Stilltype", "createdAt": "t"}]
        )
        self.assertEqual(pairs, [(["still", "type"], "Stilltype")])
        self.assertEqual(
            inference.apply_vocabulary("I use still type daily", pairs),
            "I use Stilltype daily",
        )

    def test_match_is_case_insensitive_and_keeps_punctuation(self):
        pairs = inference.vocab_pairs(
            [{"id": "b", "word": "Bostic site", "replacement": "", "createdAt": "t"}]
        )
        self.assertEqual(
            inference.apply_vocabulary("the BOSTICK site opened", pairs),
            "the BOSTICK site opened",  # exact only: look-alikes do not match
        )
        self.assertEqual(
            inference.apply_vocabulary("the bostic SITE.", pairs),
            "the Bostic site.",
        )

    def test_longest_source_wins(self):
        pairs = inference.vocab_pairs(
            [
                {"id": "a", "word": "site", "replacement": "place", "createdAt": "t"},
                {"id": "b", "word": "bostic site", "replacement": "Bostic site", "createdAt": "t"},
            ]
        )
        self.assertEqual(
            inference.apply_vocabulary("the bostic site", pairs),
            "the Bostic site",
        )

    def test_empty_vocabulary_leaves_text_untouched(self):
        self.assertEqual(inference.vocab_pairs([]), [])
        self.assertEqual(inference.vocab_pairs(None), [])
        self.assertEqual(inference.apply_vocabulary("hello world", []), "hello world")


if __name__ == "__main__":
    unittest.main()
