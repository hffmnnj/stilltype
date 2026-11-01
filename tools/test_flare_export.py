"""Security boundary regression for the FLARE export CLI."""

import contextlib
import io
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
    def test_cached_graphs_are_also_protected_from_overwrite(self):
        output = Path(tempfile.mkdtemp())
        try:
            for name in ("cross.onnx", "decoder-cache.onnx"):
                graph = output / name
                graph.touch()
                with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                    export_flare.validate_output(output)
                graph.unlink()
        finally:
            for name in ("cross.onnx", "decoder-cache.onnx"):
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


if __name__ == "__main__":
    unittest.main()
