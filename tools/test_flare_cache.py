"""Exercise the weightless cache graphs with ONNX Runtime's CPU provider."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

import export_flare_cache as cache


class FlareCacheTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        preferred = Path("/var/tmp/james/opencode")
        cls.root = Path(tempfile.mkdtemp(prefix="flare-cache-test-", dir=preferred if preferred.is_dir() else None))
        cls.addClassCleanup(cls.cleanup_files)
        cls.sessions = {}
        for beam in (5, 50):
            directory = cls.root / str(beam)
            cache.export(beam, directory)
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            cls.sessions[beam] = {
                name: ort.InferenceSession(str(directory / name), sess_options=options,
                                           providers=["CPUExecutionProvider"])
                for name in cache.GRAPH_NAMES
            }

    @classmethod
    def cleanup_files(cls):
        cls.sessions.clear()
        for beam in (5, 50):
            directory = cls.root / str(beam)
            if not directory.exists():
                continue
            for name in (*cache.GRAPH_NAMES, "cache.json"):
                (directory / name).unlink(missing_ok=True)
            directory.rmdir()
        cls.root.rmdir()

    def test_metadata_matches_checked_graphs_and_refuses_existing_directory(self):
        for beam in (5, 50):
            directory = self.root / str(beam)
            metadata = json.loads((directory / "cache.json").read_text(encoding="utf-8"))
            self.assertEqual(set(path.name for path in directory.iterdir()), {*cache.GRAPH_NAMES, "cache.json"})
            self.assertEqual({key: metadata[key] for key in (
                "schemaVersion", "modelId", "beamSize", "maximumNativeRows", "cacheLayers", "cacheHeads",
                "cacheHeadDim", "cacheChunks")}, {
                "schemaVersion": 1, "modelId": "flare-base-transformer-2026-research", "beamSize": beam,
                "maximumNativeRows": 5, "cacheLayers": 6, "cacheHeads": 4,
                "cacheHeadDim": 192, "cacheChunks": beam // 5,
            })
            for prefix, entry, name in zip(("cacheGather", "cacheMerge"), metadata["files"], cache.GRAPH_NAMES):
                data = (directory / name).read_bytes()
                self.assertEqual(entry, {"name": name, "sizeBytes": len(data),
                                         "sha256": hashlib.sha256(data).hexdigest()})
                self.assertEqual(metadata[prefix + "Bytes"], entry["sizeBytes"])
                self.assertEqual(metadata[prefix + "Sha256"], entry["sha256"])
                model = onnx.load_from_string(data)
                onnx.checker.check_model(model)
                self.assertEqual(model.ir_version, 8)
            self.assertEqual(len(metadata["files"]), 2)
            with self.assertRaisesRegex(ValueError, "must be new"):
                cache.export(beam, directory)

    def test_invalid_beam_never_creates_output(self):
        for beam in (0, 4, 51, 5.0, True):
            directory = self.root / f"invalid-{beam}"
            with self.subTest(beam=beam), self.assertRaisesRegex(ValueError, "beam size"):
                cache.export(beam, directory)
            self.assertFalse(directory.exists())

    def test_gather_duplicates_reorders_and_handles_empty_dimensions(self):
        session = self.sessions[5]["cache-gather.onnx"]
        for rows, length, indices in ((3, 128, [2, 0, 2, 1]), (2, 0, [1, 1, 0]), (2, 3, []), (0, 0, [])):
            with self.subTest(rows=rows, length=length, indices=indices):
                selected = np.asarray(indices, dtype=np.int64)
                feeds = {"indices": selected}
                for i, suffix in enumerate(cache.SUFFIXES):
                    values = np.arange(rows * 4 * length * 192, dtype=np.float32).reshape(rows, 4, length, 192)
                    feeds[f"past_{suffix}"] = values + np.float32(i * 7)
                result = dict(zip((item.name for item in session.get_outputs()), session.run(None, feeds)))
                for suffix in cache.SUFFIXES:
                    np.testing.assert_array_equal(result[f"selected_{suffix}"], feeds[f"past_{suffix}"][selected])

    def test_merge_includes_unselected_nonfinite_values_and_handles_empty_chunks(self):
        for beam, rows, length, indices in (
            (5, [3], 128, [2, 0, 2]),
            (50, [1] + [0] * 9, 1, [0, 0]),
            (50, [1, 0, 2] + [0] * 7, 2, [2, 0, 2, 1]),
            (50, [0] * 10, 0, []),
            (5, [2], 1, []),
        ):
            with self.subTest(beam=beam, rows=rows, length=length, indices=indices):
                selected = np.asarray(indices, dtype=np.int64)
                feeds = {"indices": selected}
                for chunk, count in enumerate(rows):
                    for i, suffix in enumerate(cache.SUFFIXES):
                        array = np.arange(count * 4 * length * 192, dtype=np.float32).reshape(count, 4, length, 192)
                        feeds[f"chunk_{chunk}_{suffix}"] = array + np.float32(chunk * 100 + i)
                session = self.sessions[beam]["cache-merge.onnx"]
                names = [item.name for item in session.get_outputs()]
                result = dict(zip(names, session.run(None, feeds)))
                self.assertEqual(result["cache_finite"].shape, ())
                self.assertEqual(result["cache_finite"].item(), True)
                for suffix in cache.SUFFIXES:
                    combined = np.concatenate([feeds[f"chunk_{chunk}_{suffix}"] for chunk in range(len(rows))])
                    np.testing.assert_array_equal(result[f"past_{suffix}"], combined[selected])

        session = self.sessions[5]["cache-merge.onnx"]
        for invalid in (np.nan, np.inf, -np.inf):
            with self.subTest(invalid=invalid):
                feeds = {"indices": np.array([0], dtype=np.int64)}
                feeds.update({f"chunk_0_{suffix}": np.zeros((2, 4, 1, 192), dtype=np.float32)
                              for suffix in cache.SUFFIXES})
                feeds["chunk_0_5_value"][1, 3, 0, 191] = invalid
                result = session.run(None, feeds)
                self.assertEqual(result[-1].item(), False)
                for selected_cache in result[:-1]:
                    np.testing.assert_array_equal(selected_cache, np.zeros((1, 4, 1, 192), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
