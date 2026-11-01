"""Security boundary regression for the FLARE export CLI."""

import contextlib
import hashlib
import io
import json
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import export_flare
import benchmark_flare


class FlareExportGuardTest(unittest.TestCase):
    def test_beam_size_is_validated_before_accessing_input_files(self):
        unused = Path("/unused")
        for mode, size in (("full", 50), ("cached", 50), ("beam", 4), ("beam", 5.0), ("beam", True)):
            with self.subTest(mode=mode, size=size), self.assertRaisesRegex(ValueError, "beam size"):
                export_flare.export(unused, unused, unused, unused, "cpu", mode, size)

    def test_cached_graphs_are_also_protected_from_overwrite(self):
        output = Path(tempfile.mkdtemp())
        try:
            for name in ("cross.onnx", "decoder-cache.onnx", "decoder-beam.onnx"):
                graph = output / name
                graph.touch()
                with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                    export_flare.validate_output(output)
                graph.unlink()
        finally:
            for name in ("cross.onnx", "decoder-cache.onnx", "decoder-beam.onnx"):
                (output / name).unlink(missing_ok=True)
            output.rmdir()

    def test_output_rejects_another_git_worktree(self):
        root = Path(tempfile.mkdtemp())
        marker = root / ".git"
        try:
            marker.write_text("gitdir: /unused/linked-worktree\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Git working tree"):
                export_flare.validate_output(root / "export")
        finally:
            marker.unlink(missing_ok=True)
            root.rmdir()

    def test_fixture_pixels_are_private_to_the_owner(self):
        root = Path(tempfile.mkdtemp())
        output = root / "fixture"
        try:
            pixels = np.zeros((1, 88, 88), dtype=np.float32)
            benchmark_flare.write_fixture(output, pixels, pixels, {}, [2, 4, 2], {})
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o700)
            self.assertEqual((output / "face.f32").stat().st_size, 88 * 88 * 4)
        finally:
            for name in ("face.f32", "lip.f32", "manifest.json", "results.json"):
                (output / name).unlink(missing_ok=True)
            if output.exists():
                output.rmdir()
            root.rmdir()

    def test_wrong_checkpoint_hash_rejected_before_source_import(self):
        root = Path(tempfile.mkdtemp())
        try:
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"wrong checkpoint")
            source = root / "source"
            package = source / "flare"
            package.mkdir(parents=True)
            marker = root / "source-imported"
            (package / "__init__.py").write_text(
                f"from pathlib import Path\nPath({str(marker)!r}).touch()\n", encoding="utf-8"
            )
            sample = root / "sample.npz"
            np.savez(sample, face=np.zeros((1, 96, 96), dtype=np.uint8), lip=np.zeros((1, 96, 96), dtype=np.uint8))
            stderr = io.StringIO()
            stdout = io.StringIO()
            with patch.object(export_flare, "CHECKPOINT_SIZE", checkpoint.stat().st_size):
                with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(stdout):
                    status = export_flare.main([
                        "--checkpoint", str(checkpoint), "--source", str(source),
                        "--sample", str(sample), "--output", str(root / "output"),
                        "--device", "cpu", "--accept-license",
                    ])
            self.assertFalse(marker.exists(), "untrusted source was imported before checkpoint validation")
            self.assertEqual(status, 2)
            self.assertIn("checkpoint SHA-256 does not match", stderr.getvalue())
            self.assertEqual(stdout.getvalue(), "")
            self.assertFalse((root / "output").exists())
        finally:
            for path in (root / "source/flare/__init__.py", root / "source-imported",
                         root / "sample.npz", root / "checkpoint.pt"):
                path.unlink(missing_ok=True)
            for path in (root / "source/flare", root / "source", root):
                if path.exists():
                    path.rmdir()


class CachedDecodeTest(unittest.TestCase):
    def test_cache_grows_and_eos_must_be_generated(self):
        names = [(index, kind) for index in range(6) for kind in ("key", "value")]
        cross = {f"cross_{i}_{kind}": np.zeros((1, 4, 3, 192), dtype=np.float32)
                 for i, kind in names}

        class FakeSession:
            def __init__(self, stop):
                self.stop = stop

            def get_inputs(self):
                return [SimpleNamespace(name=name) for name in (
                    "tokens", "encoder_padding_mask", *cross,
                    *(f"past_{i}_{kind}" for i, kind in names))]

            def run(self, output_names, feeds):
                self_test.assertNotIn("encoder_out", feeds)
                prefix = feeds["tokens"][0].tolist()
                for index, kind in names:
                    self_test.assertEqual(feeds[f"past_{index}_{kind}"].shape, (1, 4, len(prefix) - 1, 192))
                logits = np.zeros((1, 1000), dtype=np.float32)
                logits[0, 1] = 10
                logits[0, 2] = 9
                logits[0, 7] = 8 if self.stop else 10
                values = [np.zeros((1, 4, len(prefix), 192), dtype=np.float32) for _ in names]
                return [logits, *values]

        self_test = self
        encoded = np.zeros((3, 1, 512), dtype=np.float32)
        mask = np.zeros((1, 3), dtype=bool)
        self.assertEqual(benchmark_flare.cached_greedy_decode(FakeSession(True), encoded, mask, cross), [2, 7, 2])
        with self.assertRaisesRegex(ValueError, "did not generate EOS"):
            benchmark_flare.cached_greedy_decode(FakeSession(False), encoded, mask, cross)


class BeamDecodeTest(unittest.TestCase):
    names = [(index, kind) for index in range(6) for kind in ("key", "value")]

    def setUp(self):
        self.encoded = np.zeros((3, 1, 512), dtype=np.float32)
        self.mask = np.zeros((1, 3), dtype=bool)
        self.cross = {f"cross_{i}_{kind}": np.zeros((1, 4, 3, 192), dtype=np.float32)
                      for i, kind in self.names}

    def session(self, scenario, encoder_input=False):
        test = self
        names = self.names
        cross = self.cross

        class FakeSession:
            def __init__(self):
                self.batches = []

            def get_inputs(self):
                return [SimpleNamespace(name=name) for name in (
                    "tokens", "encoder_padding_mask", *cross,
                    *(f"past_{i}_{kind}" for i, kind in names),
                    *(("encoder_out",) if encoder_input else ()))]

            def run(self, output_names, feeds):
                tokens = feeds["tokens"]
                batch, length = tokens.shape
                self.batches.append((length, batch))
                test.assertLessEqual(batch, 5)
                if encoder_input:
                    test.assertEqual(feeds["encoder_out"].shape, (3, batch, 512))
                else:
                    test.assertNotIn("encoder_out", feeds)
                test.assertEqual(feeds["encoder_padding_mask"].shape, (batch, 3))
                for name in cross:
                    test.assertEqual(feeds[name].shape, (batch, 4, 3, 192))
                for cache_index, (index, kind) in enumerate(names):
                    past = feeds[f"past_{index}_{kind}"]
                    test.assertEqual(past.shape, (batch, 4, length - 1, 192))
                    if length > 1:
                        np.testing.assert_array_equal(past[:, 0, :, 0], tokens[:, :-1] + cache_index)
                logits = np.full((batch, 1000), -1000, dtype=np.float32)
                if length == 1:
                    logits[:, 1:3] = 100  # PAD and initial EOS must not be selected.
                    if scenario == "global50":
                        logits[:, 7:57] = 5
                    else:
                        logits[:, 7:12] = [5, 4.9, 4.8, 4.7, 4.6] if scenario == "reorder" else 5
                elif scenario == "unfinished":
                    logits[:, 30:40] = 0
                elif length == 2 and scenario == "reorder":
                    for row, prefix in enumerate(tokens):
                        if prefix[-1] == 11:
                            logits[row, 30] = 20
                        else:
                            logits[row, 40:42] = 0
                elif length == 2 and scenario == "global50":
                    for row, prefix in enumerate(tokens):
                        if prefix[-1] == 56:
                            logits[row, 30] = 20
                        else:
                            logits[row, 40:42] = 0
                elif length == 2 and scenario == "length_penalty":
                    logits[:, 30] = 2
                    logits[:, 2] = 0
                    logits[tokens[:, -1] == 7, 2] = 2
                    logits[tokens[:, -1] == 7, 30] = 1
                else:
                    logits[:, 2] = 20
                presents = []
                for cache_index, (index, kind) in enumerate(names):
                    present = np.zeros((batch, 4, length, 192), dtype=np.float32)
                    if length > 1:
                        present[:, :, :-1, :] = feeds[f"past_{index}_{kind}"]
                    present[:, :, -1, :] = tokens[:, -1, None, None] + cache_index
                    presents.append(present)
                return [logits, *presents]

        return FakeSession()

    def test_parent_reordering_and_fairseq_finalization(self):
        tokens = benchmark_flare.beam_decode(self.session("reorder"), self.encoded, self.mask, self.cross)
        self.assertEqual(tokens, [2, 11, 30, 2])

    def test_beam50_selects_and_finalizes_globally_across_five_row_chunks(self):
        session = self.session("global50")
        tokens = benchmark_flare.beam_decode(session, self.encoded, self.mask, self.cross, beam_size=50)
        self.assertEqual(tokens, [2, 56, 30, 2])
        self.assertEqual(session.batches, [(1, 1)] + [(2, 5)] * 10 + [(3, 5)] * 10)

    def test_beam_rejects_noninteger_or_unsupported_widths(self):
        for size in (4, 5.0, True, None):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "Beam size"):
                benchmark_flare.beam_decode(self.session("tie"), self.encoded, self.mask, self.cross, size)

    def test_equal_scores_use_stable_flat_index_order(self):
        tokens = benchmark_flare.beam_decode(self.session("tie"), self.encoded, self.mask, self.cross)
        self.assertEqual(tokens, [2, 7, 2])

    def test_decoder_with_encoder_input_uses_matching_batch(self):
        tokens = benchmark_flare.beam_decode(self.session("tie", encoder_input=True),
                                             self.encoded, self.mask, self.cross)
        self.assertEqual(tokens, [2, 7, 2])

    def test_finalized_scores_are_normalized_by_generated_length(self):
        tokens = benchmark_flare.beam_decode(self.session("length_penalty"), self.encoded, self.mask, self.cross)
        self.assertEqual(tokens, [2, 8, 30, 2])

    def test_unfinished_beam_fails_without_forcing_eos(self):
        with self.assertRaisesRegex(ValueError, "without natural finalization"):
            benchmark_flare.beam_decode(self.session("unfinished"), self.encoded, self.mask, self.cross)

    def test_invalid_decoder_inputs_and_nonfinite_cache_are_rejected(self):
        session = self.session("tie")
        original = session.get_inputs
        session.get_inputs = lambda: [*original(), SimpleNamespace(name="unexpected")]
        with self.assertRaisesRegex(ValueError, "Unsupported beam decoder inputs"):
            benchmark_flare.beam_decode(session, self.encoded, self.mask, self.cross)
        session = self.session("tie")
        cross = dict(self.cross)
        cross["cross_0_key"] = np.full((1, 4, 3, 192), np.nan, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            benchmark_flare.beam_decode(session, self.encoded, self.mask, cross)

    def test_beam_metadata_runs_through_fixture_and_rejects_bad_graph_list(self):
        import onnxruntime as ort

        temporary = tempfile.mkdtemp()
        try:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            metadata = {
                "schemaVersion": 1, "modelId": benchmark_flare.MODEL_ID, "releaseYear": 2026,
                "license": "CC-BY-NC-4.0", "maxFrames": 200, "maxPrefix": 128,
                "featureSize": 512, "vocabularySize": 1000, "eosId": 2, "padId": 1,
                "inputSize": 96, "cropSize": 88, "mean": 0.421, "std": 0.165,
                "decoderMode": "beam", "beamSize": 50, "cacheLayers": 6, "cacheHeads": 4,
                "cacheHeadDim": 192, "files": [],
            }
            for name, key in (("encoder.onnx", "encoder"), ("cross.onnx", "cross"),
                              ("decoder-beam.onnx", "beamDecoder")):
                content = name.encode()
                (model / name).write_bytes(content)
                digest = hashlib.sha256(content).hexdigest()
                metadata.update({f"{key}Bytes": len(content), f"{key}Sha256": digest})
                metadata["files"].append({"name": name, "sizeBytes": len(content), "sha256": digest})
            metadata_path = model / "metadata.json"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            sample = root / "sample.npz"
            frames = np.zeros((3, 96, 96), dtype=np.uint8)
            np.savez(sample, face=frames, lip=frames)
            beam = self.session("global50")
            cross = self.cross
            encoded, mask = self.encoded, self.mask

            class Graph:
                def __init__(self, inputs, values):
                    self.inputs, self.values = inputs, values

                def get_inputs(self):
                    return [SimpleNamespace(name=name) for name in self.inputs]

                def run(self, output_names, feeds):
                    return self.values

            graphs = {
                "encoder.onnx": Graph(("face", "lip", "face_mask", "lip_mask", "padding_mask"), [encoded, mask]),
                "cross.onnx": Graph(("encoder_out",), list(cross.values())),
                "decoder-beam.onnx": beam,
            }
            with patch.object(ort, "InferenceSession", side_effect=lambda path, *args, **kwargs: graphs[Path(path).name]):
                summary = benchmark_flare.run(model, sample, root / "fixture")
                self.assertIn("beamDecoderMs", summary)
                self.assertEqual(summary["beamSize"], 50)
                self.assertNotIn("cachedDecoderMs", summary)
                manifest = json.loads((root / "fixture" / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["expectedTokenIds"], [2, 56, 30, 2])
                metadata["files"][2]["name"] = "decoder-cache.onnx"
                metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Graph file list"):
                    benchmark_flare.run(model, sample, root / "bad-fixture")
                self.assertFalse((root / "bad-fixture").exists())
                metadata["files"][2]["name"] = "decoder-beam.onnx"
                metadata["decoderMode"] = "unknown"
                metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Unsupported FLARE decoder mode"):
                    benchmark_flare.run(model, sample, root / "bad-fixture")
                metadata["decoderMode"] = "beam"
                for size in (4, 5.0, True):
                    metadata["beamSize"] = size
                    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
                    with self.subTest(size=size), self.assertRaisesRegex(ValueError, "Unsupported FLARE beam size"):
                        benchmark_flare.run(model, sample, root / "bad-fixture")
        finally:
            root = Path(temporary)
            for name in ("face.f32", "lip.f32", "manifest.json", "results.json"):
                (root / "fixture" / name).unlink(missing_ok=True)
            for name in ("encoder.onnx", "cross.onnx", "decoder-beam.onnx", "metadata.json"):
                (root / "model" / name).unlink(missing_ok=True)
            (root / "sample.npz").unlink(missing_ok=True)
            for directory in (root / "fixture", root / "model", root):
                if directory.exists():
                    directory.rmdir()


if __name__ == "__main__":
    unittest.main()
