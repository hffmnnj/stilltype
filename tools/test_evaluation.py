import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import evaluate_lipreading as ev


def _item(item_id, split="dev", speech="spoken", ref="the cat sat", clip="clip-a"):
    return {
        "id": item_id,
        "split": split,
        "clipDir": clip,
        "reference": ref,
        "sourceUrl": "https://example.invalid/v/" + item_id,
        "sourceSha256": "a" * 64,
        "license": "CC-BY-4.0",
        "speakerGroup": "group-a",
        "speechMode": speech,
        "trainingOverlap": "unknown",
    }


def _write_manifest(directory, items, dataset="set-1"):
    path = Path(directory) / "manifest.json"
    path.write_text(
        json.dumps({"schemaVersion": 1, "datasetId": dataset, "items": items}),
        encoding="utf-8",
    )
    return path


class NormalizationTest(unittest.TestCase):
    def test_curly_case_punctuation_internal_apostrophe(self):
        self.assertEqual(ev.normalize_text("Don\u2019t STOP!"), "don't stop")
        self.assertEqual(ev.normalize_text("  Hello,   WORLD... "), "hello world")
        # Numbers are kept verbatim; no expansion.
        self.assertEqual(ev.normalize_text("Room 101"), "room 101")
        # Hyphens split; internal apostrophes survive.
        self.assertEqual(ev.normalize_text("mother-in-law's"), "mother in law's")


class EditCountsTest(unittest.TestCase):
    def test_hand_calculated_substitution_and_insertion(self):
        subs, dels, inss = ev.edit_counts(
            ["a", "b", "c"], ["a", "x", "c", "d"]
        )
        self.assertEqual((subs, dels, inss), (1, 0, 1))

    def test_wer_above_one_is_not_clamped(self):
        subs, dels, inss = ev.edit_counts(["hello"], ["foo", "bar", "baz"])
        self.assertEqual((subs, dels, inss), (1, 0, 2))
        scored = ev.score_pair("hello", "foo bar baz", None)
        self.assertEqual(scored["wer"], 3.0)

    def test_deletion_only(self):
        scored = ev.score_pair("the cat sat", "the cat", None)
        self.assertEqual(
            (scored["substitutions"], scored["deletions"], scored["insertions"]),
            (0, 1, 0),
        )
        self.assertAlmostEqual(scored["wer"], 1 / 3)


class FailedPredictionTest(unittest.TestCase):
    def test_error_and_empty_keep_denominator(self):
        failed = ev.score_pair("the cat sat on mats", "", "no-speech")
        self.assertEqual(
            (failed["substitutions"], failed["deletions"], failed["insertions"]),
            (0, 5, 0),
        )
        self.assertEqual(failed["wer"], 1.0)
        self.assertFalse(failed["exactMatch"])
        empty = ev.score_pair("hello world", "", None)
        self.assertEqual(empty["deletions"], 2)
        self.assertEqual(empty["wer"], 1.0)


class ManifestValidationTest(unittest.TestCase):
    def test_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_manifest(tmp, [_item("a"), _item("a")])
            with self.assertRaises(ValueError):
                ev.load_manifest(path)

    def test_unsupported_and_empty_fields_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = _item("a")
            bad["bogus"] = "nope"
            with self.assertRaises(ValueError):
                ev.load_manifest(_write_manifest(tmp, [bad]))
            empty_ref = _item("b", ref="  !!!  ")
            with self.assertRaises(ValueError):
                ev.load_manifest(_write_manifest(tmp, [empty_ref]))
            bad_sha = _item("c")
            bad_sha["sourceSha256"] = "ZZZ"
            with self.assertRaises(ValueError):
                ev.load_manifest(_write_manifest(tmp, [bad_sha]))


class ScorePredictionsCliTest(unittest.TestCase):
    def test_exact_coverage_split_isolation_and_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = _write_manifest(
                tmp,
                [
                    _item("dev-spoken", split="dev", speech="spoken",
                          ref="the cat sat", clip="clip-a"),
                    _item("dev-silent", split="dev", speech="silent",
                          ref="hello world", clip="clip-b"),
                    _item("test-only", split="test", speech="spoken",
                          ref="other words here", clip="clip-c"),
                ],
            )
            preds = Path(tmp) / "preds.json"
            preds.write_text(
                json.dumps(
                    {
                        "items": [
                            {"id": "dev-spoken", "raw": "the cat", "error": None},
                            {
                                "id": "dev-silent",
                                "raw": "hello there world",
                                "error": None,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            out = Path(tmp) / "out" / "results.json"
            code = ev.main(
                [
                    "--manifest", str(manifest),
                    "--split", "dev",
                    "--decoder", "desktop",
                    "--output", str(out),
                    "--score-predictions", str(preds),
                ]
            )
            self.assertEqual(code, 0)
            results = json.loads(out.read_text(encoding="utf-8"))
            total = results["metrics"]["total"]
            # dev-spoken: D=1 (N=3); dev-silent: I=1 (N=2) -> (0+1+1)/5.
            self.assertEqual(
                (total["substitutions"], total["deletions"], total["insertions"]),
                (0, 1, 1),
            )
            self.assertAlmostEqual(total["wer"], 0.4)
            self.assertFalse(results["targetMet"])
            self.assertAlmostEqual(results["metrics"]["spoken"]["wer"], 1 / 3)
            self.assertAlmostEqual(results["metrics"]["silent"]["wer"], 1 / 2)
            self.assertEqual(results["metrics"]["spoken"]["clips"], 1)
            self.assertEqual(results["metrics"]["silent"]["clips"], 1)
            raws = {i["id"]: i["raw"] for i in results["items"]}
            self.assertEqual(raws["dev-spoken"], "the cat")

            # Missing coverage fails; extra ids fail; split isolation holds.
            missing = Path(tmp) / "missing.json"
            missing.write_text(
                json.dumps({"items": [{"id": "dev-spoken", "raw": "x", "error": None}]}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                by_id = ev.load_predictions(missing)
                ev.check_coverage(["dev-spoken", "dev-silent"], set(by_id.keys()))
            extra = Path(tmp) / "extra.json"
            extra.write_text(
                json.dumps(
                    {
                        "items": [
                            {"id": "dev-spoken", "raw": "x", "error": None},
                            {"id": "dev-silent", "raw": "y", "error": None},
                            {"id": "test-only", "raw": "z", "error": None},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                by_id = ev.load_predictions(extra)
                ev.check_coverage(["dev-spoken", "dev-silent"], set(by_id.keys()))
            dup = Path(tmp) / "dup.json"
            dup.write_text(
                json.dumps(
                    {
                        "items": [
                            {"id": "dev-spoken", "raw": "x", "error": None},
                            {"id": "dev-spoken", "raw": "y", "error": None},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                ev.load_predictions(dup)

    def test_score_only_imports_no_model_stack(self):
        before = set(sys.modules.keys())
        with tempfile.TemporaryDirectory() as tmp:
            manifest = _write_manifest(tmp, [_item("a", ref="hello world")])
            preds = Path(tmp) / "preds.json"
            preds.write_text(
                json.dumps({"items": [{"id": "a", "raw": "hello", "error": None}]}),
                encoding="utf-8",
            )
            out = Path(tmp) / "r.json"
            self.assertEqual(
                ev.main(
                    [
                        "--manifest", str(manifest),
                        "--split", "dev",
                        "--decoder", "ctc-fp32",
                        "--output", str(out),
                        "--score-predictions", str(preds),
                    ]
                ),
                0,
            )
        added = set(sys.modules.keys()) - before
        self.assertNotIn("torch", added)
        self.assertNotIn("onnxruntime", added)
        self.assertNotIn("inference", added)

    def test_output_inside_repo_and_overwrite_refused(self):
        repo_out = Path(ev.REPO_ROOT) / "results.json"
        with self.assertRaises(ValueError):
            ev.ensure_output_writable(repo_out)
        with tempfile.TemporaryDirectory() as tmp:
            taken = Path(tmp) / "r.json"
            taken.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                ev.ensure_output_writable(taken)


class AdapterSeparationTest(unittest.TestCase):
    def test_rejected_last_clip_does_not_replace_the_observed_device(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = ev.load_manifest(_write_manifest(tmp, [_item("first"), _item("last")]))
            with patch.object(ev, "desktop_predict", side_effect=[
                ("the cat sat", None, 2000, 12, "cuda"),
                ("", "no-speech", 2000, 0, None),
            ]):
                scored, device = ev.run_desktop(manifest.items, Path(tmp), device="gpu", beam_size=5)
            self.assertEqual(device, "cuda")
            self.assertEqual(len(scored), 2)
            self.assertEqual(scored[1][0]["deletions"], 3)

    def test_desktop_model_call_never_sees_reference(self):
        secret = "zebra hedgehog vault phrase"
        seen: list[str] = []

        def fake_transcribe(clip_dir, **kwargs):
            seen.append(clip_dir + json.dumps(kwargs, sort_keys=True, default=str))
            return {
                "raw": "hello world",
                "text": "Something else entirely",
                "durationMs": 100,
                "inferenceMs": 5,
                "error": None,
                "device": "cpu",
            }

        raw, error, _, _, _ = ev.desktop_predict(
            "/tmp/clip-x", device="cpu", beam_size=5, transcribe_fn=fake_transcribe
        )
        self.assertEqual(raw, "hello world")
        self.assertNotIn(secret, "\n".join(seen))
        # Scoring uses raw, never the cleaned text.
        scored = ev.score_pair(secret, raw, error)
        self.assertEqual(scored["normalizedHypothesis"], "hello world")
        self.assertNotIn("something", scored["normalizedHypothesis"])


if __name__ == "__main__":
    unittest.main()
