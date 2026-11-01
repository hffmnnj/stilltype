#!/usr/bin/env python3
"""Personally fine-tune the pinned 2026 FLARE Base checkpoint locally on CUDA.

This creates newly fine-tuned weights, not an export of unchanged weights. The
checkpoint is CC-BY-NC-4.0; you must have rights to your training data. Results
do not establish accuracy for other speakers. No downloads or evaluation occur.
The supplied FLARE source is trusted executable code, not a sandbox.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import io
import json
import math
import os
import random
import sys
import zipfile
from dataclasses import dataclass, make_dataclass
from pathlib import Path
from typing import Any, TextIO

from evaluate_lipreading import normalize_text
from export_flare import (CHECKPOINT_ARTIFACT, CHECKPOINT_REVISION, CHECKPOINT_SHA256, CHECKPOINT_SIZE,
                          sha256, validate_output, verify_checkpoint)

SEED = 20261109
ENCODER_PREFIXES = tuple(f"encoder.w2v_model.encoder.layers.{i}." for i in range(8, 12))
W2V = "encoder.w2v_model."
CNN_PREFIXES = tuple(W2V + f"feature_extractor_video_{side}.resnet.trunk.{part}."
                     for side in ("face", "lip") for part in ("stages.3", "downsample_layers.3", "norm"))
BRIDGE_PREFIXES = tuple(W2V + part + "." for part in
                        ("feature_extractor_video_face.proj", "feature_extractor_video_lip.proj",
                         "post_extract_proj", "layer_norm"))
VISUAL_RATES = {"transformer": 0.00002, "decoder": 0.00001, "cnn": 0.0001, "bridge": 0.00002}
TOP4_RECIPE = {"examples": 500, "seed": SEED, "epochs": 10, "visualBlocks": [8, 9, 10, 11],
               "optimizer": "AdamW", "encoderLearningRate": 0.00005, "decoderLearningRate": 0.00002,
               "weightDecay": 0.01, "batchSize": 1, "accumulation": 8, "gradientClipNorm": 1.0,
               "labelSmoothing": 0.1, "initialToken": "EOS", "precision": "FP32",
               "sentencepieceSampling": False}


class AdaptationError(ValueError):
    """A safe diagnostic that does not include private input values."""


@dataclass(frozen=True)
class TrainingItem:
    id: str
    sample: Path
    text: str


def new_output(path: Path) -> Path:
    if os.path.lexists(path):
        raise AdaptationError("refusing an existing output path")
    try:
        output = validate_output(path)
    except ValueError as exc:
        raise AdaptationError("refusing output inside a Git checkout or existing output") from exc
    if os.path.lexists(output):
        raise AdaptationError("refusing an existing output path")
    if any(os.path.lexists(parent / ".git") for parent in (output, *output.parents)):
        raise AdaptationError("refusing output inside a Git working tree")
    return output


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AdaptationError("duplicate JSON field")
        result[key] = value
    return result


def valid_string(value: Any) -> bool:
    return (isinstance(value, str) and bool(value.strip())
            and not any(ord(char) < 32 or ord(char) == 127 for char in value))


def private_input(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    if any(os.path.lexists(parent / ".git") for parent in (resolved.parent, *resolved.parents)):
        raise AdaptationError("private inputs and weights must be outside Git working trees")
    if not resolved.is_file():
        raise AdaptationError("private input must be a file")
    return resolved


def bounded_bytes(path: Path, limit: int) -> bytes:
    with path.open("rb") as stream:
        payload = stream.read(limit + 1)
    if len(payload) > limit:
        raise AdaptationError("initial adaptation input exceeds size limit")
    return payload


def checkpoint_snapshot(path: Path) -> io.BytesIO:
    """Verify before source import, then load only the same checked in-memory bytes."""
    verify_checkpoint(path)
    snapshot = io.BytesIO()
    with path.open("rb") as stream:
        while chunk := stream.read(min(1024 * 1024, CHECKPOINT_SIZE + 1 - snapshot.tell())):
            snapshot.write(chunk)
            if snapshot.tell() > CHECKPOINT_SIZE:
                raise AdaptationError("checkpoint exceeds pinned size")
    if (snapshot.tell() != CHECKPOINT_SIZE
            or hashlib.sha256(snapshot.getbuffer()).hexdigest() != CHECKPOINT_SHA256):
        raise AdaptationError("checkpoint snapshot does not match the pinned release")
    snapshot.seek(0)
    return snapshot


def read_manifest(path: Path, expected_count: int = 500) -> tuple[list[TrainingItem], str]:
    path = private_input(path)
    raw = path.read_bytes()
    data = json.loads(raw, object_pairs_hook=unique_object)
    if (not isinstance(data, dict) or set(data) != {"schemaVersion", "items"}
            or type(data["schemaVersion"]) is not int or data["schemaVersion"] != 1
            or not isinstance(data["items"], list) or len(data["items"]) != expected_count):
        raise AdaptationError(f"manifest must contain schemaVersion 1 and exactly {expected_count} items")
    items: list[TrainingItem] = []
    ids: set[str] = set()
    paths: set[Path] = set()
    texts: set[str] = set()
    for row in data["items"]:
        if (not isinstance(row, dict) or set(row) != {"id", "sample", "text"}
                or not all(valid_string(row[key]) for key in ("id", "sample", "text"))):
            raise AdaptationError("invalid training item fields")
        normalized = normalize_text(row["text"])
        if not normalized:
            raise AdaptationError("empty normalized training text")
        sample = private_input(path.parent / row["sample"])
        if row["id"] in ids or sample in paths or normalized in texts:
            raise AdaptationError("duplicate training id, sample path, or normalized text")
        ids.add(row["id"])
        paths.add(sample)
        texts.add(normalized)
        items.append(TrainingItem(row["id"], sample, normalized))
    return items, hashlib.sha256(raw).hexdigest()


def category(name: str) -> str:
    if name.startswith(ENCODER_PREFIXES):
        return "selectedEncoder"
    if name.startswith("decoder."):
        return "decoder"
    return "frozen"


def visual_category(name: str) -> str:
    if name.startswith(W2V + "encoder.layers."):
        return "transformer"
    if name.startswith("decoder."):
        return "decoder"
    if name.startswith(CNN_PREFIXES):
        return "cnn"
    if name.startswith(BRIDGE_PREFIXES):
        return "bridge"
    return "frozen"


def state_hashes(model: Any, recipe: str = "top4") -> dict[str, str]:
    classify = category if recipe == "top4" else visual_category
    names = ("selectedEncoder", "decoder", "frozen") if recipe == "top4" else (*VISUAL_RATES, "frozen")
    digests = {name: hashlib.sha256() for name in names}
    for name, value in sorted(model.state_dict().items()):
        digest = digests[classify(name)]
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return {name: digest.hexdigest() for name, digest in digests.items()}


def check_initial_items(initial: list[TrainingItem], expanded: list[TrainingItem]) -> None:
    by_id = {item.id: item for item in expanded}
    if len(initial) != 500 or any(by_id.get(item.id) != item for item in initial):
        raise AdaptationError("initial training items must occur unchanged in expanded manifest")


def apply_initial_adaptation(model: Any, directory: Path, checkpoint_identity: dict[str, Any],
                             source_identity: dict[str, str], manifest_hash: str,
                             tokenizer_hash: str) -> dict[str, Any]:
    """Check local consistency, not independent authenticity, before applying the tensor-only state."""
    import torch

    metadata_path = private_input(directory / "metadata.json")
    state_path = private_input(directory / "changed-state.pt")
    raw_metadata = bounded_bytes(metadata_path, 1024 * 1024)
    metadata = json.loads(raw_metadata, object_pairs_hook=unique_object)
    fields = {"schemaVersion", "license", "checkpoint", "source", "trainingManifestSha256",
              "sentencepieceSha256", "recipe", "environment", "epochs", "firstBackwardGradientNorms",
              "stateBefore", "stateAfter", "checks", "changedState", "note"}
    if (not isinstance(metadata, dict) or set(metadata) != fields
            or type(metadata["schemaVersion"]) is not int or metadata["schemaVersion"] != 1
            or metadata["license"] != "CC-BY-NC-4.0" or metadata["checkpoint"] != checkpoint_identity
            or metadata["source"] != source_identity
            or json.dumps(metadata["recipe"], sort_keys=True) != json.dumps(TOP4_RECIPE, sort_keys=True)
            or metadata["trainingManifestSha256"] != manifest_hash
            or metadata["sentencepieceSha256"] != tokenizer_hash):
        raise AdaptationError("initial adaptation metadata identity mismatch")
    environment = metadata["environment"]
    epochs = metadata["epochs"]
    gradients = metadata["firstBackwardGradientNorms"]
    def finite_number(value: Any) -> bool:
        return type(value) in (int, float) and math.isfinite(value)
    if (not isinstance(environment, dict)
            or set(environment) != {"python", "torch", "numpy", "sentencepiece", "cuda", "device"}
            or any(not isinstance(value, str) or not value for value in environment.values())
            or not isinstance(metadata["note"], str)
            or not isinstance(epochs, list) or len(epochs) != 10
            or any(not isinstance(log, dict) or set(log) != {"epoch", "utterances", "meanLoss", "minLoss", "maxLoss"}
                   or type(log["epoch"]) is not int or log["epoch"] != index
                   or type(log["utterances"]) is not int or log["utterances"] != 500
                   or any(not finite_number(log[key]) for key in ("meanLoss", "minLoss", "maxLoss"))
                   for index, log in enumerate(epochs, 1))
            or not isinstance(gradients, dict)
            or set(gradients) != {"block8", "block9", "block10", "block11", "decoder"}
            or any(not finite_number(value) or value <= 0 for value in gradients.values())):
        raise AdaptationError("initial adaptation metadata schema mismatch")
    groups = {"selectedEncoder", "decoder", "frozen"}
    before, after = metadata["stateBefore"], metadata["stateAfter"]
    def valid_hashes(value: Any) -> bool:
        return (isinstance(value, dict) and set(value) == groups
                and all(isinstance(h, str) and len(h) == 64
                        and all(c in "0123456789abcdef" for c in h) for h in value.values()))
    if (not valid_hashes(before) or not valid_hashes(after) or before["frozen"] != after["frozen"]
            or any(before[g] == after[g] for g in ("selectedEncoder", "decoder"))
            or metadata["checks"] != {"frozenUnchanged": True, "selectedEncoderChanged": True,
                                       "decoderChanged": True}
            or state_hashes(model) != before):
        raise AdaptationError("initial adaptation state identity mismatch")
    descriptor = metadata["changedState"]
    expected = {name: value for name, value in model.state_dict().items() if category(name) != "frozen"}
    # A normal Torch ZIP needs only small headers in addition to the fixed selected tensors.
    max_state_bytes = sum(value.numel() * value.element_size() for value in expected.values()) + 4 * 1024 * 1024
    if (not isinstance(descriptor, dict) or set(descriptor) != {"name", "sizeBytes", "sha256"}
            or descriptor["name"] != "changed-state.pt" or type(descriptor["sizeBytes"]) is not int
            or not 0 < descriptor["sizeBytes"] <= max_state_bytes
            or state_path.stat().st_size != descriptor["sizeBytes"]):
        raise AdaptationError("initial adaptation file identity mismatch")
    payload = bounded_bytes(state_path, max_state_bytes)
    if len(payload) != descriptor["sizeBytes"] or hashlib.sha256(payload).hexdigest() != descriptor["sha256"]:
        raise AdaptationError("initial adaptation file identity mismatch")
    partial = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if not isinstance(partial, dict) or set(partial) != set(expected):
        raise AdaptationError("initial adaptation state keys mismatch")
    for name, value in partial.items():
        if (not isinstance(value, torch.Tensor) or value.shape != expected[name].shape
                or value.dtype != expected[name].dtype or not bool(torch.isfinite(value).all())):
            raise AdaptationError("initial adaptation tensor mismatch")
    model.load_state_dict(partial, strict=False)
    if state_hashes(model) != after:
        raise AdaptationError("initial adaptation loaded state mismatch")
    return {"metadataSha256": hashlib.sha256(raw_metadata).hexdigest(),
            "trainingManifestSha256": manifest_hash, "changedState": descriptor,
            "stateBefore": before, "stateAfter": after}


def source_python_hash(source: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((source / "flare").rglob("*.py")):
        digest.update(str(path.relative_to(source)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def checked_sample(path: Path, *, augment: bool = False) -> tuple[Any, Any]:
    import numpy as np

    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if len(members) != 2 or {entry.filename for entry in members} != {"face.npy", "lip.npy"}:
                raise ValueError("invalid sample members")
            arrays = {}
            limit = 200 * 96 * 96 + 4096
            for entry in members:
                if entry.file_size > limit:
                    raise ValueError("sample member exceeds the frame bound")
                with archive.open(entry) as member:
                    payload = member.read(limit + 1)
                if len(payload) > limit:
                    raise ValueError("sample payload exceeds the frame bound")
                stream = io.BytesIO(payload)
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=4096)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream, max_header_size=4096)
                else:
                    raise ValueError("unsupported sample array format")
                if dtype != np.dtype(np.uint8) or len(shape) != 3 or shape[1:] != (96, 96) or not 1 <= shape[0] <= 200:
                    raise ValueError("invalid sample array header")
                # Decode the same bounded bytes whose header was checked, never reopen the path.
                stream.seek(0)
                arrays[entry.filename] = np.lib.format.read_array(stream, allow_pickle=False, max_header_size=4096)
            if arrays["face.npy"].shape != arrays["lip.npy"].shape:
                raise ValueError("sample array shapes differ")
        y, x = (random.randint(0, 8), random.randint(0, 8)) if augment else (4, 4)
        flip = random.random() < 0.5 if augment else False
        def transform(name: str) -> Any:
            crop = arrays[name][:, y:y + 88, x:x + 88]
            if flip:
                crop = crop[:, :, ::-1]
            return (crop.astype(np.float32) / 255.0 - 0.421) / 0.165
        return transform("face.npy"), transform("lip.npy")
    except (ValueError, OSError, EOFError, zipfile.BadZipFile) as exc:
        raise AdaptationError("invalid NPZ sample or face/lip geometry") from exc


def train(checkpoint: Path, source: Path, manifest: Path, destination: Path,
          progress: TextIO | None = None, *, recipe: str = "top4",
          initial_adaptation: Path | None = None, initial_training_manifest: Path | None = None) -> dict[str, Any]:
    if recipe not in ("top4", "visual") or (recipe == "visual") != (
            initial_adaptation is not None and initial_training_manifest is not None):
        raise AdaptationError("visual requires both initial inputs; top4 rejects initial inputs")
    if recipe == "top4" and (initial_adaptation is not None or initial_training_manifest is not None):
        raise AdaptationError("top4 rejects initial inputs")
    output = new_output(destination)
    if not (source / "flare" / "__init__.py").is_file():
        raise AdaptationError("source must be a trusted FLARE checkout")
    checkpoint = private_input(checkpoint)
    try:
        verified_checkpoint = checkpoint_snapshot(checkpoint)
    except ValueError as exc:
        raise AdaptationError("checkpoint does not match the pinned 2026 Base release") from exc
    items, manifest_hash = read_manifest(manifest, 878 if recipe == "visual" else 500)
    initial_manifest_hash = None
    if recipe == "visual":
        assert initial_training_manifest is not None
        initial_items, initial_manifest_hash = read_manifest(initial_training_manifest)
        check_initial_items(initial_items, items)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    import numpy as np
    import sentencepiece
    import torch
    import torch.nn.functional as functional
    from fairseq import tasks
    from fairseq.data.dictionary import Dictionary
    from fairseq.data.encoders.sentencepiece_bpe import SentencepieceBPE
    from omegaconf import OmegaConf

    if not torch.cuda.is_available():
        raise AdaptationError("CUDA is required for this research recipe")
    for item in items:
        checked_sample(item.sample)

    source = source.resolve()
    source_init = source / "flare/__init__.py"
    source_config = source / "flare/conf/pretrain/base_whisper_distill_offline_bestrq.yaml"
    source_identity = {"flareInitSha256": sha256(source_init), "pretrainConfigSha256": sha256(source_config),
                       "flarePythonTreeSha256": source_python_hash(source)}
    sys.path.insert(0, str(source))
    import flare  # noqa: F401
    from flare.hubert_pretraining import AVHubertPretrainingConfig

    torch.set_num_threads(4)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    compatible = make_dataclass("CompatiblePretrainingConfig", [("input_modality", str, "image")],
                                bases=(AVHubertPretrainingConfig,))
    tasks.TASK_DATACLASS_REGISTRY["av_hubert_pretraining"] = compatible
    with torch.serialization.safe_globals([Dictionary, SentencepieceBPE, sentencepiece.SentencePieceProcessor]):
        with verified_checkpoint:
            state = torch.load(verified_checkpoint, map_location="cpu", weights_only=True)
    state.pop("last_optimizer_state", None)
    cfg = OmegaConf.create(state["cfg"])
    demo = str(source / "flare/data_demo/tsv_merge_face_lip_dataset")
    cfg.task.data = cfg.task.label_dir = demo
    pre = OmegaConf.load(source_config)
    pre.task.data = pre.task.label_dir = demo
    pre.task.noise_wav = "unused"
    pre.task.label_rate = 25
    pre.model.input_modality = "image"
    del pre.model["wav_input"]
    for key, value in {"masking_type": "input2", "frontend": "convnet", "encoder_type": "qwenconv",
                       "encoder_embed_dim": 512, "encoder_layers": 12, "encoder_attention_heads": 8,
                       "num_key_value_heads": 2, "modality_dropout": 0.75, "audio_dropout": 1.0}.items():
        pre.model[key] = value
    cfg.model.w2v_args = pre
    task = tasks.setup_task(cfg.task)
    task.load_state_dict(state["task_state"])
    stats = state["model"]["encoder.w2v_model.whisper_stat"].numpy()
    original_load = np.load

    def load_with_checkpoint_stats(path: Any, *args: Any, **kwargs: Any) -> Any:
        if str(path) == "/work2/asrprg/anonymous/lippretrain/extract_whisper_fea/whisper_last1.stat.npy":
            return stats
        return original_load(path, *args, **kwargs)

    np.load = load_with_checkpoint_stats
    try:
        model = task.build_model(cfg.model, from_checkpoint=True)
    finally:
        np.load = original_load
    model.load_state_dict(state["model"], strict=True, model_cfg=cfg.model)
    del state, stats
    gc.collect()
    model.eval().cuda()
    blocks = model.encoder.w2v_model.encoder.layers
    if not isinstance(blocks, torch.nn.ModuleList) or len(blocks) != 12:
        raise AdaptationError("expected exactly 12 visual transformer blocks")
    classify = category if recipe == "top4" else visual_category
    selected_names = ("selectedEncoder", "decoder") if recipe == "top4" else tuple(VISUAL_RATES)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(classify(name) != "frozen")
    groups = {group: [p for name, p in model.named_parameters() if classify(name) == group]
              for group in (*selected_names, "frozen")}
    if (any(not group for group in groups.values()) or any(p.requires_grad for p in groups["frozen"])
            or any(not p.requires_grad for group in selected_names for p in groups[group])):
        raise AdaptationError("freeze boundary mismatch")
    trainable = [p for group in selected_names for p in groups[group]]
    if len({id(p) for p in trainable}) != len(trainable):
        raise AdaptationError("duplicate shared optimizer parameter")
    if recipe == "visual":
        w2v = model.encoder.w2v_model
        face_extractor, lip_extractor = w2v.feature_extractor_video_face, w2v.feature_extractor_video_lip
        if face_extractor.resnet is not lip_extractor.resnet or w2v.feature_grad_mult != 0.1:
            raise AdaptationError("unexpected shared CNN or feature gradient multiplier")
        if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
            raise AdaptationError("expected FP32 model parameters")
        trunk = face_extractor.resnet.trunk
        state_keys = model.state_dict()
        for part in ("stages.3", "downsample_layers.3", "norm"):
            face_prefix = W2V + f"feature_extractor_video_face.resnet.trunk.{part}."
            lip_prefix = W2V + f"feature_extractor_video_lip.resnet.trunk.{part}."
            face_keys = {k[len(face_prefix):] for k in state_keys if k.startswith(face_prefix)}
            lip_keys = {k[len(lip_prefix):] for k in state_keys if k.startswith(lip_prefix)}
            if not face_keys or face_keys != lip_keys:
                raise AdaptationError("missing shared CNN state aliases")
        del state_keys

    dictionary = task.target_dictionary
    eos = dictionary.eos()
    tokenizer = task.s2s_tokenizer.sp
    tokenizer_hash = hashlib.sha256(tokenizer.serialized_model_proto()).hexdigest()
    targets = []
    for item in items:
        pieces = " ".join(tokenizer.Encode(item.text, out_type=str, enable_sampling=False))
        target = dictionary.encode_line(pieces, append_eos=True, add_if_not_exist=False).long()
        if len(target) < 2 or int(target[-1]) != eos or bool((target == dictionary.unk()).any()):
            raise AdaptationError("invalid or unknown training target")
        targets.append(target)

    initial_identity = None
    if recipe == "visual":
        assert initial_adaptation is not None and initial_manifest_hash is not None
        checkpoint_identity = {"artifact": CHECKPOINT_ARTIFACT, "revision": CHECKPOINT_REVISION,
                               "sha256": CHECKPOINT_SHA256, "sizeBytes": CHECKPOINT_SIZE}
        initial_identity = apply_initial_adaptation(model, initial_adaptation, checkpoint_identity,
                                                    source_identity, initial_manifest_hash, tokenizer_hash)
        random.seed(20261113)
        np.random.seed(20261113)
        torch.manual_seed(20261113)
        torch.cuda.manual_seed_all(20261113)
    model.eval()
    if recipe == "visual":
        for module in [*blocks, trunk.stages[3], trunk.downsample_layers[3], trunk.norm,
                       face_extractor.proj, lip_extractor.proj, w2v.post_extract_proj,
                       w2v.layer_norm, model.decoder]:
            if module is not None:
                module.train()
        if model.training or model.encoder.training or w2v.training or trunk.training or face_extractor.resnet.frontend3D.training:
            raise AdaptationError("frozen parent or frontend unexpectedly in train mode")
    else:
        for block in blocks[8:12]:
            block.train()
        model.decoder.train()
        if model.training or model.encoder.training or model.encoder.w2v_model.training or any(b.training for b in blocks[:8]):
            raise AdaptationError("frozen parent or visual blocks unexpectedly in train mode")
    before = state_hashes(model, recipe)
    decoder_parameters = groups["decoder"]
    if any(p.dtype != torch.float32 for p in trainable):
        raise AdaptationError("expected FP32 trainable parameters")
    rates = {"selectedEncoder": 0.00005, "decoder": 0.00002} if recipe == "top4" else VISUAL_RATES
    optimizer = torch.optim.AdamW([{"params": groups[name], "lr": rate} for name, rate in rates.items()],
                                  weight_decay=0.01)
    output.mkdir(mode=0o700, exist_ok=False)
    epoch_logs = []
    first_gradients: dict[str, float] = {}
    for epoch in range(1, 11 if recipe == "top4" else 21):
        order = list(range(len(items)))
        random.shuffle(order)
        losses = []
        optimizer.zero_grad(set_to_none=True)
        for index, item_index in enumerate(order):
            face_array, lip_array = checked_sample(items[item_index].sample, augment=recipe == "visual")
            face = torch.from_numpy(face_array).unsqueeze(0).unsqueeze(0).cuda()
            lip = torch.from_numpy(lip_array).unsqueeze(0).unsqueeze(0).cuda()
            present = torch.ones(1, dtype=torch.bool, device="cuda")
            source_input = {"audio": None, "video": (face, lip), "face_lip_mask": (present, present)}
            padding = torch.zeros((1, face.shape[2]), dtype=torch.bool, device="cuda")
            target = targets[item_index].cuda().unsqueeze(0)
            previous = torch.cat((torch.full((1, 1), eos, dtype=torch.long, device="cuda"), target[:, :-1]), dim=1)
            encoded = model.encoder(source=source_input, padding_mask=padding)
            logits = model.decoder(prev_output_tokens=previous, encoder_out=encoded)[0]
            loss = functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
                                            ignore_index=dictionary.pad(), label_smoothing=0.1)
            if not bool(torch.isfinite(loss)):
                raise AdaptationError("nonfinite training loss")
            group_size = min(8, len(order) - (index - index % 8))
            (loss / group_size).backward()
            losses.append(float(loss.detach().cpu()))
            if epoch == 1 and index == 0:
                gradient_groups = ([(f"block{i}", list(blocks[i].parameters())) for i in range(8, 12)] +
                                   [("decoder", decoder_parameters)]) if recipe == "top4" else [
                                       (name, groups[name]) for name in selected_names]
                for label, parameters in gradient_groups:
                    gradients = [p.grad for p in parameters if p.grad is not None]
                    if not gradients or any(not bool(torch.isfinite(g).all()) for g in gradients):
                        raise AdaptationError(f"missing or nonfinite gradients: {label}")
                    norm = math.sqrt(sum(float(g.detach().double().square().sum().cpu()) for g in gradients))
                    if not math.isfinite(norm) or norm <= 0:
                        raise AdaptationError(f"zero or nonfinite gradients: {label}")
                    first_gradients[label] = norm
            if index % 8 == 7 or index + 1 == len(order):
                if any(p.grad is not None for p in groups["frozen"]):
                    raise AdaptationError("frozen parameters received gradients")
                torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        log = {"epoch": epoch, "utterances": len(losses), "meanLoss": sum(losses) / len(losses),
               "minLoss": min(losses), "maxLoss": max(losses)}
        if not all(math.isfinite(log[key]) for key in ("meanLoss", "minLoss", "maxLoss")):
            raise AdaptationError("nonfinite epoch loss")
        epoch_logs.append(log)
        print(json.dumps(log, allow_nan=False), file=progress, flush=True)

    after = state_hashes(model, recipe)
    if before["frozen"] != after["frozen"]:
        raise AdaptationError("frozen state changed")
    if any(before[name] == after[name] for name in selected_names):
        raise AdaptationError("selected state group did not change")
    if any(not bool(torch.isfinite(p).all()) for p in trainable):
        raise AdaptationError("nonfinite trained parameters")
    changed = output / "changed-state.pt"
    with changed.open("xb") as stream:
        torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()
                    if classify(name) != "frozen"}, stream)
    metadata = {
        "schemaVersion": 1, "license": "CC-BY-NC-4.0",
        "checkpoint": {"artifact": CHECKPOINT_ARTIFACT, "revision": CHECKPOINT_REVISION,
                       "sha256": CHECKPOINT_SHA256, "sizeBytes": CHECKPOINT_SIZE},
        "source": source_identity, "trainingManifestSha256": manifest_hash, "sentencepieceSha256": tokenizer_hash,
        "recipe": TOP4_RECIPE if recipe == "top4" else {
            "name": "visual", "examples": 878, "seed": 20261113, "epochs": 20,
            "visualBlocks": list(range(12)), "optimizer": "AdamW", "learningRates": VISUAL_RATES,
            "weightDecay": 0.01, "batchSize": 1, "accumulation": 8,
            "finalShortGroup": "mean over actual group size", "gradientClipNorm": 1.0,
            "labelSmoothing": 0.1, "initialToken": "EOS", "precision": "FP32",
            "sentencepieceSampling": False, "featureGradMultiplier": 0.1,
            "augmentation": "joint 88x88 random crop from 96x96 and joint horizontal flip p=0.5"},
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
                        "sentencepiece": sentencepiece.__version__,
                        "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0)},
        "epochs": epoch_logs, "firstBackwardGradientNorms": first_gradients,
        "stateBefore": before, "stateAfter": after,
        "checks": {"frozenUnchanged": True, **{name + "Changed": True for name in selected_names}},
        "changedState": {"name": changed.name, "sizeBytes": changed.stat().st_size, "sha256": sha256(changed)},
        "note": "New personal fine-tuning; no evaluation or other-speaker accuracy established.",
    }
    if initial_identity is not None:
        metadata["initialAdaptation"] = initial_identity
    serialized = json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = output / "metadata.json.partial"
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    os.link(temporary, output / "metadata.json")
    # Publication completed; a leftover private temporary file does not invalidate it.
    with contextlib.suppress(OSError):
        temporary.unlink()
    return {"output": str(output), "changedState": changed.name, "metadata": "metadata.json"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Pinned 2026 FLARE Base checkpoint.")
    parser.add_argument("--source", type=Path, required=True, help="Trusted executable FLARE source checkout.")
    parser.add_argument("--training-manifest", type=Path, required=True, help="Private manifest: 500 top4 or 878 visual examples.")
    parser.add_argument("--recipe", choices=("top4", "visual"), default="top4", help="Training recipe (default: top4).")
    parser.add_argument("--initial-adaptation", type=Path, help="Prior top4 output directory; required for visual only.")
    parser.add_argument("--initial-training-manifest", type=Path, help="Prior top4 input manifest; required for visual only.")
    parser.add_argument("--output", type=Path, required=True, help="New private directory outside any Git checkout.")
    parser.add_argument("--accept-license", action="store_true", required=True, help="Accept CC-BY-NC-4.0 terms.")
    args = parser.parse_args(argv)
    if (args.recipe == "visual" and (args.initial_adaptation is None or args.initial_training_manifest is None)
            or args.recipe == "top4" and (args.initial_adaptation is not None or args.initial_training_manifest is not None)):
        parser.error("visual requires both initial inputs; top4 rejects initial inputs")
    os.umask(0o077)
    try:
        # Upstream code and tokenization can log sensitive input. Only our bounded progress is printed.
        progress = sys.stdout
        with (open(os.devnull, "w", encoding="utf-8") as quiet,
              contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet)):
            result = train(args.checkpoint, args.source, args.training_manifest, args.output, progress,
                           recipe=args.recipe, initial_adaptation=args.initial_adaptation,
                           initial_training_manifest=args.initial_training_manifest)
    except AdaptationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: {type(exc).__name__}; training did not complete", file=sys.stderr)
        return 2
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
