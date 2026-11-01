"""Focused tests for tools/prepare_evaluation.py (unittest discover)."""

import base64
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import prepare_evaluation as prep

ENGINE_DIR = Path(__file__).resolve().parents[1] / "engine"
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))
import inference as engine_inference

TMP_BASE = Path("/var/tmp/james/opencode")
if not TMP_BASE.is_dir():
    TMP_BASE = None


def blank_heads() -> dict:
    heads: dict = {}
    for stride in prep.DET_STRIDES:
        count = (prep.DET_W // stride) * (prep.DET_H // stride)
        heads[f"cls_{stride}"] = np.zeros(count)
        heads[f"obj_{stride}"] = np.zeros(count)
        heads[f"bbox_{stride}"] = np.zeros(count * 4)
        heads[f"kps_{stride}"] = np.zeros(count * 10)
    return heads


class FakeSession:
    """Owner-maintained fake for the ORT YuNet session."""

    def __init__(self, outputs: dict):
        self._outputs = outputs

    def get_outputs(self):
        return [SimpleNamespace(name=name) for name in self._outputs]

    def run(self, names, feed):
        assert set(names) == set(self._outputs), names
        det_input = feed["input"]
        assert det_input.shape == (1, 3, prep.DET_H, prep.DET_W), det_input.shape
        assert det_input.dtype == np.float32, det_input.dtype
        return [self._outputs[name] for name in names]


class DecodeYunetTest(unittest.TestCase):
    def test_valid_head_returns_expected_capture_landmarks(self):
        heads = blank_heads()
        # Stride 8, col 5, row 10 -> cell 405; kps offsets in detector pixels.
        i = 10 * (prep.DET_W // 8) + 5
        heads["cls_8"][i] = 0.9
        heads["obj_8"][i] = 0.9
        heads["kps_8"][i * 10 : i * 10 + 10] = [
            0.5, 0.5,  # model right eye -> det (44, 84)
            1.5, 0.5,  # model left eye  -> det (52, 84)
            1.0, 1.5,  # nose            -> det (48, 92)
            0.5, 2.0,  # right mouth     -> det (44, 96)
            1.5, 2.0,  # left mouth      -> det (52, 96)
        ]
        self.assertEqual(
            prep.decode_yunet(heads),
            [[88.0, 157.5], [104.0, 157.5], [96.0, 172.5], [96.0, 180.0]],
        )

    def test_invalid_detections_rejected_and_boundary_score_accepted(self):
        heads = blank_heads()
        # Score 1.0 but wildly out of frame: must lose to the valid 0.9 face.
        heads["cls_8"][0] = 1.0
        heads["obj_8"][0] = 1.0
        heads["kps_8"][0:2] = [1000.0, 1000.0]
        # Score 0.99 with a non-finite box: also invalid.
        heads["cls_16"][0] = 0.99
        heads["obj_16"][0] = 0.99
        heads["bbox_16"][2] = float("inf")
        heads["kps_16"][0:10] = [0.5] * 10
        # The only valid face, at exactly the 0.75 boundary.
        i = 10 * (prep.DET_W // 8) + 5
        heads["cls_8"][i] = 0.75
        heads["obj_8"][i] = 0.75
        heads["kps_8"][i * 10 : i * 10 + 10] = [
            0.5, 0.5, 1.5, 0.5, 1.0, 1.5, 0.5, 2.0, 1.5, 2.0,
        ]
        self.assertEqual(
            prep.decode_yunet(heads),
            [[88.0, 157.5], [104.0, 157.5], [96.0, 172.5], [96.0, 180.0]],
        )

    def test_no_valid_face_returns_none(self):
        self.assertIsNone(prep.decode_yunet(blank_heads()))

    def test_corrupt_heads_fail_clearly(self):
        heads = blank_heads()
        del heads["cls_16"]
        with self.assertRaises(ValueError):
            prep.decode_yunet(heads)
        heads = blank_heads()
        heads["kps_32"] = np.zeros(3)
        with self.assertRaises(ValueError):
            prep.decode_yunet(heads)


class ContainFitTest(unittest.TestCase):
    def test_wide_frame_preserves_aspect_with_bars(self):
        import cv2

        frame = np.full((180, 320, 3), 255, dtype=np.uint8)
        canvas = prep.contain_fit(frame)
        self.assertEqual(canvas.shape, (480, 640, 3))
        self.assertTrue(np.all(canvas[0:60] == 0))
        self.assertTrue(np.all(canvas[420:480] == 0))
        self.assertTrue(np.all(canvas[60:420] == 255))
        decoded = cv2.imdecode(
            np.frombuffer(
                cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes(),
                dtype=np.uint8,
            ),
            cv2.IMREAD_COLOR,
        )
        self.assertEqual(decoded.shape, (480, 640, 3))


class PrepareClipEndToEndTest(unittest.TestCase):
    def test_30fps_source_keeps_25fps_instead_of_halving_the_frame_rate(self):
        import cv2

        with tempfile.TemporaryDirectory(dir=str(TMP_BASE) if TMP_BASE else None) as tmp:
            video = Path(tmp) / "input.mp4"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (160, 120))
            self.assertTrue(writer.isOpened())
            try:
                for _ in range(60):
                    writer.write(np.full((120, 160, 3), 128, dtype=np.uint8))
            finally:
                writer.release()
            detector = Path(tmp) / "detector.onnx"
            detector.write_bytes(b"fake-detector")
            summary = prep.prepare_clip(
                video=str(video), detector=str(detector), output=str(Path(tmp) / "clip"),
                requested_fps=25.0, session=FakeSession(blank_heads()),
            )
            self.assertEqual(summary["frameCount"], 50)
            records, _ = engine_inference.load_clip_records(str(Path(tmp) / "clip"))
            self.assertEqual(len({r["timestamp"] for r in records}), 50)
            self.assertGreater(summary["actualFps"], 24)
            self.assertLess(summary["actualFps"], 26)

    def test_slow_source_keeps_chronology_without_duplication(self):
        import cv2

        with tempfile.TemporaryDirectory(dir=str(TMP_BASE) if TMP_BASE else None) as tmp:
            video = os.path.join(tmp, "input.mp4")
            writer = cv2.VideoWriter(
                video, cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (160, 120)
            )
            self.assertTrue(writer.isOpened())
            for n in range(10):
                frame = np.zeros((120, 160, 3), dtype=np.uint8)
                frame[:] = (n * 25 % 256, 128, 200 - n * 10)
                writer.write(frame)
            writer.release()

            detector = os.path.join(tmp, "yunet.onnx")
            Path(detector).write_bytes(b"fake-detector-bytes")
            out_dir = os.path.join(tmp, "clip")

            summary = prep.prepare_clip(
                video=video,
                detector=detector,
                output=out_dir,
                requested_fps=10.0,  # source runs at 5fps: keep, never duplicate
                session=FakeSession(blank_heads()),
            )
            self.assertEqual(summary["frameCount"], 10)
            self.assertEqual(summary["faceCount"], 0)
            self.assertAlmostEqual(summary["actualFps"], 5.0, places=2)

            frames_path = Path(out_dir) / "frames.jsonl"
            records = []
            with open(frames_path, encoding="utf-8") as sink:
                for line in sink.read().splitlines():
                    if line.strip():
                        records.append(json.loads(line))
            # No duplication to reach the requested rate; real frames only.
            self.assertEqual(len(records), 10)
            stamps = [r["timestamp"] for r in records]
            self.assertEqual(stamps, sorted(stamps))
            self.assertEqual(len(set(stamps)), len(stamps))
            self.assertEqual(stamps, [n * 200 for n in range(10)])
            for record in records:
                self.assertIsNone(record["landmarks"])
                jpeg = base64.b64decode(record["jpeg"], validate=True)
                img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                self.assertEqual(img.shape, (480, 640, 3))

            # The production clip reader accepts the prepared clip as-is.
            loaded, duration_ms = engine_inference.load_clip_records(out_dir)
            self.assertEqual(len(loaded), 10)
            self.assertGreater(duration_ms, 0)
            self.assertAlmostEqual(
                engine_inference.fps_from_records(loaded), 5.0, places=2
            )

            with open(Path(out_dir) / "metadata.json", encoding="utf-8") as sink:
                metadata = json.load(sink)
            self.assertEqual(metadata["frameCount"], 10)
            self.assertEqual(metadata["faceCount"], 0)
            self.assertNotIn("reference", json.dumps(metadata))
            self.assertIn("physical camera timing benchmark", metadata["note"])
            self.assertEqual(
                metadata["framesSha256"], prep.sha256_file(frames_path)
            )


if __name__ == "__main__":
    unittest.main()
