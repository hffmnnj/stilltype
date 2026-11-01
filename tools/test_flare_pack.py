"""Behavioral tests for the storage-only FLARE packer using synthetic ONNX."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

import export_flare
import pack_flare


class FlarePackTest(unittest.TestCase):
    def setUp(self):
        preferred = Path("/var/tmp/james/opencode")
        self.root = Path(tempfile.mkdtemp(prefix="flare-pack-test-", dir=preferred if preferred.is_dir() else None))
        self.source = self.root / "source"
        self.source.mkdir()
        self.output = self.root / "packed"
        self.addCleanup(self.cleanup_files)

        self.weights = np.linspace(-0.7, 0.8, 1024, dtype=np.float32)
        def tensor(name, data):
            return numpy_helper.from_array(np.asarray(data, dtype=np.float32), name=name)

        initializers = [
            tensor("weight", self.weights),
            tensor("weight__fp16_storage", [0.5]),
            tensor("override", np.full(1024, 0.2, dtype=np.float32)),
            tensor("small", [0.125]),
            tensor("overflow", np.full(1024, 65505, dtype=np.float32)),
            tensor("underflow", np.full(1024, 1e-9, dtype=np.float32)),
            tensor("nonfinite", np.full(1024, np.nan, dtype=np.float32)),
        ]
        nodes = [helper.make_node("Add", ["x", "weight"], ["a"]),
                 helper.make_node("Add", ["a", "override"], ["b"]),
                 helper.make_node("Add", ["b", "small"], ["y"])]
        graph = helper.make_graph(
            nodes, "synthetic", [helper.make_tensor_value_info(name, TensorProto.FLOAT, [1024])
                                 for name in ("x", "override")],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1024])], initializers)
        self.model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
        for _, name in pack_flare.GRAPHS:
            (self.source / name).write_bytes(self.model.SerializeToString())
        self.metadata = {
            "schemaVersion": 1, "modelId": "flare-base-transformer-2026-research", "releaseYear": 2026,
            "license": "CC-BY-NC-4.0", "sourceCheckpointRepository": "Anony94/FLARE-checkpoints",
            "sourceCheckpointRevision": export_flare.CHECKPOINT_REVISION,
            "sourceCheckpointArtifact": export_flare.CHECKPOINT_ARTIFACT,
            "sourceCheckpointSha256": export_flare.CHECKPOINT_SHA256,
            "decoderMode": "beam", "beamSize": 5, "cacheLayers": 6, "cacheHeads": 4, "cacheHeadDim": 192,
            "fps": 25, "inputSize": 96, "cropSize": 88, "mean": 0.421, "std": 0.165,
            "inputGeometry": [96, 96], "crop": [4, 92, 4, 92], "cropGeometry": [88, 88],
            "normalization": {"scale": 255.0, "mean": 0.421, "std": 0.165},
            "maxFrames": 200, "maxPrefix": 128, "featureSize": 512, "vocabularySize": 1000,
            "bosId": 0, "padId": 1, "eosId": 2, "unkId": 3,
            "symbols": [f"symbol-{i}" for i in range(1000)],
        }
        self.update_metadata()

    def cleanup_files(self):
        for directory in (self.output, self.source):
            for name in ("encoder.onnx", "cross.onnx", "decoder-beam.onnx", "metadata.json"):
                (directory / name).unlink(missing_ok=True)
            if directory.exists():
                directory.rmdir()
        self.root.rmdir()

    def update_metadata(self):
        entries = []
        for key, name in pack_flare.GRAPHS:
            content = (self.source / name).read_bytes()
            entry = {"name": name, "sizeBytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            entries.append(entry)
            self.metadata[key + "Bytes"] = entry["sizeBytes"]
            self.metadata[key + "Sha256"] = entry["sha256"]
        self.metadata["files"] = entries
        self.write_metadata()

    def write_metadata(self):
        (self.source / "metadata.json").write_text(json.dumps(self.metadata), encoding="utf-8")

    def test_pack_preserves_computation_with_rounded_weights_and_unconverted_values(self):
        summary = pack_flare.pack(self.source, self.output)
        packed = onnx.load(str(self.output / "encoder.onnx"))
        onnx.checker.check_model(packed)
        cast, *original_nodes = packed.graph.node
        self.assertEqual((cast.op_type, list(cast.input), list(cast.output)),
                         ("Cast", ["weight__fp16_storage_"], ["weight"]))
        self.assertEqual([node.SerializeToString() for node in original_nodes],
                         [node.SerializeToString() for node in self.model.graph.node])
        self.assertEqual([v.SerializeToString() for v in packed.graph.input],
                         [v.SerializeToString() for v in self.model.graph.input])
        self.assertEqual([v.SerializeToString() for v in packed.graph.output],
                         [v.SerializeToString() for v in self.model.graph.output])
        retained = {item.name: item.SerializeToString() for item in packed.graph.initializer
                    if item.name != "weight__fp16_storage_"}
        original = {item.name: item.SerializeToString() for item in self.model.graph.initializer
                    if item.name != "weight"}
        self.assertEqual(retained, original)
        counts = summary["graphs"][0]["initializerCounts"]
        self.assertEqual(counts, {"converted": 1, "small": 2, "non_float32": 0, "graph_input": 1,
                                  "non_finite": 1, "overflow": 1, "underflow": 1})
        x = np.linspace(0, 1, 1024, dtype=np.float32)
        override = np.full(1024, 0.32123, dtype=np.float32)
        session = ort.InferenceSession(str(self.output / "encoder.onnx"), providers=["CPUExecutionProvider"])
        actual = session.run(["y"], {"x": x, "override": override})[0]
        expected = ((x + self.weights.astype(np.float16).astype(np.float32)) + override) + np.float32(0.125)
        np.testing.assert_array_equal(actual, expected)
        metadata = json.loads((self.output / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["symbols"], self.metadata["symbols"])
        self.assertEqual(metadata["sourceCheckpointSha256"], export_flare.CHECKPOINT_SHA256)
        self.assertEqual(metadata["license"], "CC-BY-NC-4.0")
        self.assertEqual(metadata["storageTransformation"]["sourceGraphs"], self.metadata["files"])
        self.assertEqual(metadata["files"], [record["output"] for record in summary["graphs"]])
        for entry in metadata["files"]:
            content = (self.output / entry["name"]).read_bytes()
            self.assertEqual((entry["sizeBytes"], entry["sha256"]),
                             (len(content), hashlib.sha256(content).hexdigest()))

    def test_bad_later_hash_rejected_before_parsing_any_graph_or_creating_output(self):
        (self.source / "encoder.onnx").write_bytes(b"not an ONNX graph")
        self.update_metadata()
        decoder = self.source / "decoder-beam.onnx"
        content = bytearray(decoder.read_bytes())
        content[-1] ^= 1
        decoder.write_bytes(content)
        with self.assertRaisesRegex(ValueError, "SHA-256 does not match"):
            pack_flare.pack(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_bad_metadata_rejected_without_creating_output(self):
        self.metadata["normalization"]["mean"] = 0.2
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, "normalization"):
            pack_flare.pack(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_external_tensor_is_rejected_without_metadata(self):
        tensor = self.model.graph.initializer[0]
        onnx.external_data_helper.set_external_data(tensor, location="not-read.bin")
        (self.source / "cross.onnx").write_bytes(self.model.SerializeToString())
        self.update_metadata()
        with self.assertRaisesRegex(ValueError, "External ONNX tensor data"):
            pack_flare.pack(self.source, self.output)
        self.assertTrue((self.output / "encoder.onnx").is_file())
        self.assertFalse((self.output / "metadata.json").exists())


if __name__ == "__main__":
    unittest.main()
