import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import prepare_flare_llm as tool


class FlareLlmTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix="stilltype-flare-llm-"))

    def tearDown(self):
        for path in self.directory.iterdir():
            if path.is_dir():
                path.rmdir()
            else:
                path.unlink()
        self.directory.rmdir()

    def test_checkpoint_size_and_digest_fail_before_torch_or_output(self):
        checkpoint = self.directory / "checkpoint.pt"
        checkpoint.write_bytes(b"bad")
        output = self.directory / "export"
        with mock.patch.dict(sys.modules, {"torch": None}):
            with self.assertRaisesRegex(ValueError, "checkpoint size"):
                tool.export(checkpoint, output)
            self.assertFalse(output.exists())
            with mock.patch.object(tool, "CHECKPOINT_SIZE", 3):
                with self.assertRaisesRegex(ValueError, "checkpoint SHA-256"):
                    tool.export(checkpoint, output)
            self.assertFalse(output.exists())

    def test_existing_and_inside_repository_outputs_are_refused(self):
        existing = self.directory / "existing"
        existing.mkdir()
        with self.assertRaisesRegex(ValueError, "existing output"):
            tool.export(self.directory / "missing", existing)
        with self.assertRaisesRegex(ValueError, "inside the Stilltype repository"):
            tool.new_output(tool.Path(__file__).resolve().parent / "not-an-output")
        self.assertEqual(list(existing.iterdir()), [])

    def test_adapter_schema_requires_every_projection_and_output_head_pair(self):
        shapes = tool.adapter_shapes()
        expected_bases = {f"base_model.model.model.layers.{layer}.{family}"
                          for layer in range(28)
                          for family in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                                         "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")}
        expected_bases.add("base_model.model.lm_head")
        self.assertEqual(set(shapes), {f"{base}.lora_{side}.weight" for base in expected_bases for side in "AB"})
        self.assertEqual(len(shapes), 394)
        self.assertEqual(shapes["base_model.model.lm_head.lora_A.weight"], (16, 3584))
        self.assertEqual(shapes["base_model.model.lm_head.lora_B.weight"], (152064, 16))
        self.assertEqual(shapes["base_model.model.model.layers.27.self_attn.k_proj.lora_B.weight"], (512, 16))
        self.assertEqual(shapes["base_model.model.model.layers.0.mlp.down_proj.lora_A.weight"], (16, 18944))
        tensors = {name: SimpleNamespace(shape=shape, dtype="fp32") for name, shape in shapes.items()}
        tool.check_schema(tensors, shapes, "fp32")
        del tensors["base_model.model.lm_head.lora_B.weight"]
        with self.assertRaisesRegex(ValueError, "unexpected tensor keys"):
            tool.check_schema(tensors, shapes, "fp32")
        tensors["base_model.model.lm_head.lora_B.weight"] = SimpleNamespace(shape=(16, 152064), dtype="fp32")
        with self.assertRaisesRegex(ValueError, "dtype or shape"):
            tool.check_schema(tensors, shapes, "fp32")

    def test_help_and_refusal_work_without_site_packages(self):
        script = str(Path(tool.__file__))
        help_result = subprocess.run([sys.executable, "-S", script, "--help"], capture_output=True, text=True)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("export", help_result.stdout)
        self.assertIn("project", help_result.stdout)
        result = subprocess.run([sys.executable, "-S", script, "export", "--checkpoint", "missing",
                                 "--output", str(self.directory / "out")], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--accept-license", result.stderr)
        self.assertFalse((self.directory / "out").exists())


if __name__ == "__main__":
    unittest.main()
