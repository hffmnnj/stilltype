"""Focused privacy and input boundary tests for personal FLARE adaptation."""
import json
import io
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
import hashlib
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
        common = [sys.executable, "-S", script, "--checkpoint", "missing", "--source", "missing",
                  "--training-manifest", "missing", "--output", str(output), "--accept-license"]
        for arguments in (["--recipe", "visual"], ["--recipe", "visual", "--initial-adaptation", "prior"],
                          ["--initial-training-manifest", "prior.json"]):
            result = subprocess.run(common + arguments, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("visual requires both initial inputs", result.stderr)
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

    def test_expanded_manifest_requires_all_original_items_unchanged(self):
        path, rows = self.manifest()
        self.write_manifest(path, rows)
        initial, _ = tool.read_manifest(path)
        for index in range(500, 878):
            name = f"{index}.npz"
            self.file(name)
            rows.append({"id": f"opaque/{index}", "sample": name, "text": f"unique utterance {index}"})
        self.write_manifest(path, rows)
        expanded, _ = tool.read_manifest(path, 878)
        tool.check_initial_items(initial, expanded)
        for field, value in (("id", "different"), ("text", "different words"), ("sample", "877.npz")):
            changed = list(expanded)
            original = changed[0]
            changed[0] = tool.TrainingItem(value if field == "id" else original.id,
                                           self.root / value if field == "sample" else original.sample,
                                           value if field == "text" else original.text)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "unchanged"):
                tool.check_initial_items(initial, changed)
        with self.assertRaisesRegex(ValueError, "exactly 500"):
            tool.read_manifest(path)
        self.write_manifest(path, rows[:-1])
        with self.assertRaisesRegex(ValueError, "exactly 878"):
            tool.read_manifest(path, 878)

    def test_visual_augmentation_uses_joint_crop_and_flip(self):
        import numpy as np

        sample = self.root / "augmentation.npz"
        self.files.append(sample)
        image = np.arange(96 * 96, dtype=np.uint16).reshape(1, 96, 96).astype(np.uint8)
        np.savez(sample, face=image, lip=image)
        with mock.patch.object(tool.random, "randint", side_effect=[2, 7]) as randint, \
                mock.patch.object(tool.random, "random", return_value=0.25):
            face, lip = tool.checked_sample(sample, augment=True)
        expected = (image[:, 2:90, 7:95][:, :, ::-1].astype(np.float32) / 255.0 - 0.421) / 0.165
        np.testing.assert_array_equal(face, expected)
        np.testing.assert_array_equal(lip, expected)
        self.assertEqual(randint.call_args_list, [mock.call(0, 8), mock.call(0, 8)])

    def test_checkpoint_snapshot_loads_verified_bytes_even_if_path_is_replaced(self):
        import torch

        checkpoint = self.root / "snapshot.pt"
        self.files.append(checkpoint)
        tensor = torch.tensor([1.0, 2.0])
        torch.save({"weight": tensor}, checkpoint)
        original_payload = checkpoint.read_bytes()
        size = len(original_payload)
        digest = hashlib.sha256(original_payload).hexdigest()
        with mock.patch.object(export_flare, "CHECKPOINT_SIZE", size), \
                mock.patch.object(export_flare, "CHECKPOINT_SHA256", digest), \
                mock.patch.object(tool, "CHECKPOINT_SIZE", size), \
                mock.patch.object(tool, "CHECKPOINT_SHA256", digest):
            snapshot = tool.checkpoint_snapshot(checkpoint)
            checkpoint.write_bytes(b"replaced after snapshot")
            with snapshot:
                actual = torch.load(snapshot, weights_only=True)
            torch.testing.assert_close(actual["weight"], tensor)

            checkpoint.write_bytes(original_payload)
            real_verify = export_flare.verify_checkpoint

            def replace_after_verification(path):
                real_verify(path)
                path.write_bytes(b"replaced before snapshot")

            with mock.patch.object(tool, "verify_checkpoint", side_effect=replace_after_verification):
                with self.assertRaisesRegex(ValueError, "snapshot does not match"):
                    tool.checkpoint_snapshot(checkpoint)

    def test_initial_adaptation_checks_identity_and_tensors_before_application(self):
        import torch

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Module()
                self.encoder.w2v_model = torch.nn.Module()
                self.encoder.w2v_model.encoder = torch.nn.Module()
                self.encoder.w2v_model.encoder.layers = torch.nn.ModuleList(
                    [torch.nn.Linear(2, 2) for _ in range(12)])
                self.decoder = torch.nn.Linear(2, 2)

        torch.manual_seed(12)
        original = TinyModel()
        trained = TinyModel()
        trained.load_state_dict(original.state_dict())
        with torch.no_grad():
            for name, parameter in trained.named_parameters():
                if tool.category(name) != "frozen":
                    parameter.add_(1)
        before, after = tool.state_hashes(original), tool.state_hashes(trained)
        partial = {k: v.clone() for k, v in trained.state_dict().items() if tool.category(k) != "frozen"}
        checkpoint = {"artifact": "base", "revision": "revision", "sha256": "checkpoint", "sizeBytes": 5}
        source = {"flareInitSha256": "init", "pretrainConfigSha256": "config", "flarePythonTreeSha256": "tree"}
        directory = self.directory("prior")
        state_path = directory / "changed-state.pt"
        metadata_path = directory / "metadata.json"
        self.files.extend([state_path, metadata_path])
        metadata = {"schemaVersion": 1, "license": "CC-BY-NC-4.0", "checkpoint": checkpoint,
                    "source": source, "trainingManifestSha256": "manifest", "sentencepieceSha256": "tokenizer",
                    "recipe": tool.TOP4_RECIPE,
                    "environment": {k: "test" for k in ("python", "torch", "numpy", "sentencepiece", "cuda", "device")},
                    "epochs": [{"epoch": i, "utterances": 500, "meanLoss": 1.0, "minLoss": 0.5, "maxLoss": 2.0}
                               for i in range(1, 11)],
                    "firstBackwardGradientNorms": {k: 1.0 for k in ("block8", "block9", "block10", "block11", "decoder")},
                    "stateBefore": before, "stateAfter": after,
                    "checks": {"frozenUnchanged": True, "selectedEncoderChanged": True, "decoderChanged": True},
                    "changedState": {}, "note": "test"}

        def write_artifact(tensors, overrides=None):
            torch.save(tensors, state_path)
            descriptor = {"name": "changed-state.pt", "sizeBytes": state_path.stat().st_size,
                          "sha256": hashlib.sha256(state_path.read_bytes()).hexdigest()}
            data = {**metadata, "changedState": descriptor, **(overrides or {})}
            metadata_path.write_text(json.dumps(data), encoding="utf-8")

        def fresh():
            model = TinyModel()
            model.load_state_dict(original.state_dict())
            return model

        def apply(model):
            return tool.apply_initial_adaptation(model, directory, checkpoint, source, "manifest", "tokenizer")

        write_artifact(partial)
        model = fresh()
        identity = apply(model)
        self.assertEqual(tool.state_hashes(model), after)
        self.assertEqual(identity["stateBefore"], before)
        first_key = next(iter(partial))
        variants = [
            (partial, {"source": {**source, "flareInitSha256": "different"}}, "identity mismatch"),
            (partial, {"trainingManifestSha256": "different"}, "identity mismatch"),
            (partial, {"recipe": {**tool.TOP4_RECIPE, "epochs": 9}}, "identity mismatch"),
            (partial, {"epochs": []}, "schema mismatch"),
            (partial, {"stateBefore": {**before, "decoder": "0" * 64}}, "state identity mismatch"),
            ({k: v for k, v in partial.items() if k != first_key}, {}, "keys mismatch"),
            ({**partial, first_key: torch.zeros(1)}, {}, "tensor mismatch"),
            ({**partial, first_key: partial[first_key].double()}, {}, "tensor mismatch"),
            ({**partial, first_key: torch.full_like(partial[first_key], float("nan"))}, {}, "tensor mismatch"),
        ]
        for tensors, overrides, message in variants:
            with self.subTest(message=message, overrides=overrides):
                write_artifact(tensors, overrides)
                model = fresh()
                with self.assertRaisesRegex(ValueError, message):
                    apply(model)
                self.assertEqual(tool.state_hashes(model), before)
        write_artifact(partial)
        data = json.loads(metadata_path.read_text())
        data["changedState"]["sha256"] = "0" * 64
        metadata_path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "file identity mismatch"):
            apply(fresh())
        write_artifact(partial, {"stateAfter": {**after, "decoder": "0" * 64}})
        with self.assertRaisesRegex(ValueError, "loaded state mismatch"):
            apply(fresh())
        metadata_path.write_bytes(b" " * (1024 * 1024 + 1))
        with self.assertRaisesRegex(ValueError, "exceeds size limit"):
            apply(fresh())
        bound = sum(value.numel() * value.element_size() for value in partial.values()) + 4 * 1024 * 1024
        with state_path.open("wb") as stream:
            stream.truncate(bound + 1)
        too_large = {"name": "changed-state.pt", "sizeBytes": bound + 1, "sha256": "0" * 64}
        metadata_path.write_text(json.dumps({**metadata, "changedState": too_large}))
        with mock.patch.object(torch, "load", side_effect=AssertionError("must not deserialize")) as loader:
            with self.assertRaisesRegex(ValueError, "file identity mismatch"):
                apply(fresh())
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
