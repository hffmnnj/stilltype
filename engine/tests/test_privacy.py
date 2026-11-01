"""Privacy regression: recognition inference makes no network calls.

Optional cleanup now runs in the Rust layer after Python recognition, so
even legacy cleanup settings arriving over stdio must be ignored here and
must never trigger outbound traffic. This exercises the real successful
transcribe_clip path (moving mouth, injected model, vocabulary) with
socket creation blocked.
"""

import base64
import json
import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import inference


def _landmarks():
    return [[100.0, 100.0], [200.0, 100.0], [150.0, 150.0], [150.0, 200.0]]


def _frame(ts_ms):
    return {
        "jpeg": base64.b64encode(b"not-a-jpeg").decode(),
        "timestamp": ts_ms,
        "landmarks": _landmarks(),
    }


def _write_clip(records):
    d = tempfile.mkdtemp(prefix="stilltype-clip-")
    with open(os.path.join(d, "frames.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return d


class NoNetworkRecognition(unittest.TestCase):
    def test_legacy_cleanup_settings_cause_no_network(self):
        """A successful decode with legacy cleanup keys touches no socket."""
        import numpy as np

        clip = _write_clip([_frame(i * 40.0) for i in range(40)])
        real_decode = inference.decode_records
        real_align = inference.align_mouths

        def fake_model(beam, device):
            class M:
                def __call__(self, rois, fps):
                    return "HELLO STILL TYPE", 7

                device = "cpu"

            return M()

        rois = np.full((40, 96, 96), 128, np.uint8)
        for i in range(40):
            if (i // 2) % 2 == 0:
                rois[i, 24:72, 16:48] = 200
                rois[i, 24:72, 48:80] = 60
            else:
                rois[i, 24:72, 16:48] = 60
                rois[i, 24:72, 48:80] = 200

        vocabulary = [
            {"id": "a", "word": "still type", "replacement": "Stilltype",
             "createdAt": "t"}
        ]
        settings = {
            "cleanupEnabled": True,
            "cleanupModel": "qwen3.5:4b",
            "context": "secret prior text",
        }
        try:
            inference.decode_records = lambda _records: (
                np.zeros((40, 120, 120), np.uint8),
                [np.asarray(_landmarks(), np.float32)] * 40,
            )
            inference.align_mouths = lambda _frames, _lms: rois
            with (
                patch.object(
                    socket, "socket",
                    side_effect=AssertionError("network access attempted"),
                ),
                patch.object(
                    socket, "create_connection",
                    side_effect=AssertionError("network access attempted"),
                ),
            ):
                result = inference.transcribe_clip(
                    clip,
                    settings=settings,
                    vocabulary=vocabulary,
                    model_factory=fake_model,
                )
        finally:
            inference.decode_records = real_decode
            inference.align_mouths = real_align

        # A successful prediction: past the face/still gates, raw preserved,
        # vocabulary applied, legacy cleanup keys ignored, no socket touched.
        self.assertIsNone(result["error"])
        self.assertEqual(result["raw"], "HELLO STILL TYPE")
        self.assertEqual(result["text"], "Hello Stilltype")
        self.assertEqual(result["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
