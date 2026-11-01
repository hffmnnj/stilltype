"""Focused privacy and input boundary tests for personal FLARE adaptation."""
import json
import io
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import adapt_flare as tool
import export_flare


class AdaptationBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="flare-adapt-"))
        self.files = []
        self.directories = []

    def tearDown(self):
        for path in reversed(self.files):
            path.unlink(missing_ok=True)
        for path in reversed(self.directories):
            path.rmdir()
        self.root.rmdir()

    def file(self, name, content=b""):
        path = self.root / name
        path.write_bytes(content)
        self.files.append(path)
        return path

    def directory(self, name):
        path = self.root / name
        path.mkdir()
        self.directories.append(path)
        return path

    def manifest(self):
        rows = []
        for index in range(500):
            name = f"{index}.npz"
            self.file(name)
            rows.append({"id": f"opaque/{index}", "sample": name, "text": f"unique utterance {index}"})
        path = self.file("manifest.json")
        return path, rows

    @staticmethod
    def write_manifest(path, rows):
        path.write_text(json.dumps({"schemaVersion": 1, "items": rows}), encoding="utf-8")

    def test_manifest_keeps_order_and_rejects_duplicates_and_malformed_rows(self):
        path, rows = self.manifest()
        self.write_manifest(path, rows)
        items, digest = tool.read_manifest(path)
        self.assertEqual(len(items), 500)
        self.assertEqual(items[0].id, "opaque/0")
        self.assertEqual(items[-1].sample, (self.root / "499.npz").resolve())
        self.assertEqual(len(digest), 64)
        cases = [
            {"id": rows[0]["id"]},
            {"sample": "./0.npz"},
            {"text": "Unique, utterance 0!"},
            {"text": "!!!"},
            {"sample": "\u0000invalid"},
            {"id": 7},
        ]
        for replacement in cases:
            altered = [dict(row) for row in rows]
            altered[1].update(replacement)
            self.write_manifest(path, altered)
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                tool.read_manifest(path)
        self.write_manifest(path, rows[:-1])
        with self.assertRaisesRegex(ValueError, "exactly 500"):
            tool.read_manifest(path)
        path.write_text('{"schemaVersion":1,"schemaVersion":1,"items":[]}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate JSON field"):
            tool.read_manifest(path)

    def test_canonical_symlink_alias_is_a_duplicate_sample(self):
        path, rows = self.manifest()
        alias = self.root / "alias.npz"
        alias.symlink_to(self.root / "0.npz")
        self.files.append(alias)
        rows[1]["sample"] = alias.name
        self.write_manifest(path, rows)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            tool.read_manifest(path)

    def test_output_refuses_existing_broken_links_and_git_worktrees(self):
        existing = self.directory("existing")
        broken = self.root / "broken"
        broken.symlink_to(self.root / "missing")
        self.files.append(broken)
        checkout = self.directory("checkout")
        marker = checkout / ".git"
        marker.write_text("gitdir: /unused/worktree", encoding="utf-8")
        self.files.append(marker)
        for destination in (existing, broken, checkout / "output", Path(tool.__file__).parent / "no-output"):
            with self.subTest(destination=destination), self.assertRaises(ValueError):
                tool.new_output(destination)
        self.assertEqual(list(existing.iterdir()), [])
        self.assertFalse((checkout / "output").exists())
        self.assertEqual(tool.new_output(self.root / "new"), self.root / "new")

    def test_bad_checkpoint_rejected_before_importing_source_or_creating_output(self):
        checkpoint = self.file("checkpoint.pt", b"invalid")
        source = self.directory("source")
        package = source / "flare"
        package.mkdir()
        self.directories.append(package)
        marker = self.root / "executed"
        initializer = package / "__init__.py"
        initializer.write_text(f"open({str(marker)!r}, 'w').close()", encoding="utf-8")
        self.files.append(initializer)
        output = self.root / "output"
        with mock.patch.object(export_flare, "CHECKPOINT_SIZE", checkpoint.stat().st_size):
            with self.assertRaisesRegex(ValueError, "checkpoint does not match the pinned 2026 Base"):
                tool.train(checkpoint, source, self.root / "missing.json", output)
        self.assertFalse(marker.exists())
        self.assertFalse(output.exists())

    def test_real_sample_loader_rejects_object_and_invalid_geometry(self):
        import numpy as np

        sample = self.root / "sample.npz"
        self.files.append(sample)
        good = np.zeros((1, 96, 96), dtype=np.uint8)
        np.savez(sample, face=good, lip=good)
        face, lip = tool.checked_sample(sample)
        self.assertEqual(face.shape, (1, 88, 88))
        self.assertEqual(lip.shape, (1, 88, 88))
        original_face, original_lip = export_flare.load_sample(sample)
        np.testing.assert_array_equal(face, original_face)
        np.testing.assert_array_equal(lip, original_lip)
        for bad in (np.zeros((201, 96, 96), dtype=np.uint8), np.array([object()], dtype=object)):
            np.savez(sample, face=bad, lip=bad)
            with self.assertRaises(ValueError):
                tool.checked_sample(sample)

    def test_help_and_license_refusal_need_no_site_packages(self):
        script = str(Path(tool.__file__))
        help_result = subprocess.run([sys.executable, "-S", script, "--help"], capture_output=True, text=True)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("CC-BY-NC-4.0", help_result.stdout)
        self.assertIn("other speakers", help_result.stdout)
        output = self.root / "output"
        refusal = subprocess.run([sys.executable, "-S", script, "--checkpoint", "missing", "--source", "missing",
                                  "--training-manifest", "missing", "--output", str(output)],
                                 capture_output=True, text=True)
        self.assertNotEqual(refusal.returncode, 0)
        self.assertIn("--accept-license", refusal.stderr)
        self.assertFalse(os.path.lexists(output))

    def test_oversized_declared_array_is_rejected_before_numpy_allocation(self):
        import numpy as np

        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(header, {
            "descr": "|u1", "fortran_order": False, "shape": (1_000_000, 96, 96),
        })
        sample = self.root / "oversized.npz"
        self.files.append(sample)
        with zipfile.ZipFile(sample, "w") as archive:
            for name in ("face.npy", "lip.npy"):
                archive.writestr(name, header.getvalue())
        with mock.patch.object(np.lib.format, "read_array", side_effect=AssertionError("must not allocate")) as loader:
            with self.assertRaisesRegex(ValueError, "invalid NPZ"):
                tool.checked_sample(sample)
            loader.assert_not_called()

    def test_private_inputs_cannot_resolve_inside_git(self):
        checkout = self.directory("checkout")
        marker = checkout / ".git"
        marker.write_text("gitdir: /unused/worktree", encoding="utf-8")
        sample = checkout / "private.npz"
        sample.write_bytes(b"private")
        alias = self.root / "alias.npz"
        alias.symlink_to(sample)
        self.files.extend([marker, sample, alias])
        for path in (sample, alias):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "outside Git"):
                tool.private_input(path)
        with self.assertRaisesRegex(ValueError, "outside Git"):
            tool.read_manifest(sample)

    def test_sample_is_not_reopened_after_header_validation(self):
        import numpy as np

        sample = self.root / "stable.npz"
        self.files.append(sample)
        good = np.full((1, 96, 96), 128, dtype=np.uint8)
        np.savez(sample, face=good, lip=good)
        expected = export_flare.load_sample(sample)
        original_reader = np.lib.format.read_array_header_1_0

        def replace_path_after_header(*args, **kwargs):
            result = original_reader(*args, **kwargs)
            sample.unlink(missing_ok=True)
            sample.write_bytes(b"replaced input path")
            return result

        with mock.patch.object(np.lib.format, "read_array_header_1_0", side_effect=replace_path_after_header):
            actual = tool.checked_sample(sample)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])


if __name__ == "__main__":
    unittest.main()
