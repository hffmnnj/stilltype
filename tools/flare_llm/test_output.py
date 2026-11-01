"""Exercise output refusal and permissions without recognition weights."""
import json
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


BINARY = str(Path(sys.argv[1]).resolve())


class OutputPrivacyTest(unittest.TestCase):
    def test_outputs_are_exclusive_private_and_outside_git(self):
        root = Path(tempfile.mkdtemp(prefix="stilltype-native-output-"))
        existing, broken, target = (root / name for name in ("existing.jsonl", "broken.jsonl", "absent.jsonl"))
        checkout = root / "checkout"
        forbidden, fresh = checkout / "forbidden.jsonl", root / "fresh.jsonl"
        try:
            existing.write_bytes(b"retain these bytes\n")
            broken.symlink_to(target)
            checkout.mkdir()
            (checkout / ".git").write_text("gitdir: /nonexistent-test-worktree\n")
            for output in (existing, broken, forbidden, fresh):
                with self.subTest(output=output.name):
                    result = subprocess.run(
                        [BINARY, "1", str(root / "missing-base.gguf"), str(root / "missing-adapter.gguf"),
                         str(output), str(root / "missing-input.bin")],
                        capture_output=True, text=True, timeout=30,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    summary = json.loads(result.stdout)
                    self.assertEqual((summary["succeeded"], summary["failed"], summary["unprocessed"]), (0, 0, 1))
                    self.assertFalse(summary["output_complete"])
            self.assertEqual(existing.read_bytes(), b"retain these bytes\n")
            self.assertTrue(broken.is_symlink())
            self.assertFalse(target.exists())
            self.assertFalse(forbidden.exists())
            self.assertEqual(fresh.read_bytes(), b"")
            self.assertEqual(stat.S_IMODE(fresh.stat().st_mode), 0o600)
        finally:
            for path in (existing, broken, target, forbidden, fresh, checkout / ".git"):
                path.unlink(missing_ok=True)
            checkout.rmdir()
            root.rmdir()


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
