"""Security boundary regression for the FLARE export CLI."""

import contextlib
import io
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import export_flare
import benchmark_flare


class FlareExportGuardTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
