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
from export_flare import (CHECKPOINT_ARTIFACT, CHECKPOINT_REVISION, CHECKPOINT_SHA256,
                          sha256, validate_output, verify_checkpoint)

SEED = 20261109
ENCODER_PREFIXES = tuple(f"encoder.w2v_model.encoder.layers.{i}." for i in range(8, 12))


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


def read_manifest(path: Path) -> tuple[list[TrainingItem], str]:
    path = private_input(path)
    raw = path.read_bytes()
    data = json.loads(raw, object_pairs_hook=unique_object)
    if (not isinstance(data, dict) or set(data) != {"schemaVersion", "items"}
            or type(data["schemaVersion"]) is not int or data["schemaVersion"] != 1
            or not isinstance(data["items"], list) or len(data["items"]) != 500):
        raise AdaptationError("manifest must contain schemaVersion 1 and exactly 500 items")
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


def state_hashes(model: Any) -> dict[str, str]:
    digests = {name: hashlib.sha256() for name in ("selectedEncoder", "decoder", "frozen")}
    for name, value in sorted(model.state_dict().items()):
        digest = digests[category(name)]
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return {name: digest.hexdigest() for name, digest in digests.items()}


def source_python_hash(source: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((source / "flare").rglob("*.py")):
        digest.update(str(path.relative_to(source)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def checked_sample(path: Path) -> tuple[Any, Any]:
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
        return tuple((arrays[name][:, 4:92, 4:92].astype(np.float32) / 255.0 - 0.421) / 0.165
                     for name in ("face.npy", "lip.npy"))
    except (ValueError, OSError, EOFError, zipfile.BadZipFile) as exc:
        raise AdaptationError("invalid NPZ sample or face/lip geometry") from exc


def train(checkpoint: Path, source: Path, manifest: Path, destination: Path,
          progress: TextIO | None = None) -> dict[str, Any]:
    output = new_output(destination)
    if not (source / "flare" / "__init__.py").is_file():
        raise AdaptationError("source must be a trusted FLARE checkout")
    checkpoint = private_input(checkpoint)
    try:
        verify_checkpoint(checkpoint)
    except ValueError as exc:
        raise AdaptationError("checkpoint does not match the pinned 2026 Base release") from exc
    items, manifest_hash = read_manifest(manifest)

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
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
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
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(category(name) != "frozen")
    groups = {group: [p for name, p in model.named_parameters() if category(name) == group]
              for group in ("selectedEncoder", "decoder", "frozen")}
    if (any(not group for group in groups.values()) or any(p.requires_grad for p in groups["frozen"])
            or any(not p.requires_grad for p in groups["selectedEncoder"] + groups["decoder"])):
        raise AdaptationError("freeze boundary mismatch")

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

    model.eval()
    for block in blocks[8:12]:
        block.train()
    model.decoder.train()
    if model.training or model.encoder.training or model.encoder.w2v_model.training or any(b.training for b in blocks[:8]):
        raise AdaptationError("frozen parent or visual blocks unexpectedly in train mode")
    before = state_hashes(model)
    encoder_parameters, decoder_parameters = groups["selectedEncoder"], groups["decoder"]
    trainable = encoder_parameters + decoder_parameters
    if any(p.dtype != torch.float32 for p in trainable):
        raise AdaptationError("expected FP32 trainable parameters")
    optimizer = torch.optim.AdamW([{"params": encoder_parameters, "lr": 0.00005},
                                   {"params": decoder_parameters, "lr": 0.00002}], weight_decay=0.01)
    output.mkdir(mode=0o700, exist_ok=False)
    epoch_logs = []
    first_gradients: dict[str, float] = {}
    for epoch in range(1, 11):
        order = list(range(len(items)))
        random.shuffle(order)
        losses = []
        optimizer.zero_grad(set_to_none=True)
        for index, item_index in enumerate(order):
            face_array, lip_array = checked_sample(items[item_index].sample)
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
                for label, parameters in [(f"block{i}", list(blocks[i].parameters())) for i in range(8, 12)] + [
                        ("decoder", decoder_parameters)]:
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

    after = state_hashes(model)
    if before["frozen"] != after["frozen"]:
        raise AdaptationError("frozen state changed")
    if before["selectedEncoder"] == after["selectedEncoder"] or before["decoder"] == after["decoder"]:
        raise AdaptationError("selected encoder or decoder state did not change")
    if any(not bool(torch.isfinite(p).all()) for p in trainable):
        raise AdaptationError("nonfinite trained parameters")
    changed = output / "changed-state.pt"
    with changed.open("xb") as stream:
        torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()
                    if category(name) != "frozen"}, stream)
    metadata = {
        "schemaVersion": 1, "license": "CC-BY-NC-4.0",
        "checkpoint": {"artifact": CHECKPOINT_ARTIFACT, "revision": CHECKPOINT_REVISION,
                       "sha256": CHECKPOINT_SHA256, "sizeBytes": checkpoint.stat().st_size},
        "source": source_identity, "trainingManifestSha256": manifest_hash, "sentencepieceSha256": tokenizer_hash,
        "recipe": {"examples": 500, "seed": SEED, "epochs": 10, "visualBlocks": [8, 9, 10, 11],
                   "optimizer": "AdamW", "encoderLearningRate": 0.00005, "decoderLearningRate": 0.00002,
                   "weightDecay": 0.01, "batchSize": 1, "accumulation": 8, "gradientClipNorm": 1.0,
                   "labelSmoothing": 0.1, "initialToken": "EOS", "precision": "FP32",
                   "sentencepieceSampling": False},
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
                        "sentencepiece": sentencepiece.__version__,
                        "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0)},
        "epochs": epoch_logs, "firstBackwardGradientNorms": first_gradients,
        "stateBefore": before, "stateAfter": after,
        "checks": {"frozenUnchanged": True, "selectedEncoderChanged": True, "decoderChanged": True},
        "changedState": {"name": changed.name, "sizeBytes": changed.stat().st_size, "sha256": sha256(changed)},
        "note": "New personal fine-tuning; no evaluation or other-speaker accuracy established.",
    }
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
    parser.add_argument("--training-manifest", type=Path, required=True, help="Private manifest of 500 paired examples.")
    parser.add_argument("--output", type=Path, required=True, help="New private directory outside any Git checkout.")
    parser.add_argument("--accept-license", action="store_true", required=True, help="Accept CC-BY-NC-4.0 terms.")
    args = parser.parse_args(argv)
    os.umask(0o077)
    try:
        # Upstream code and tokenization can log sensitive input. Only our bounded progress is printed.
        progress = sys.stdout
        with (open(os.devnull, "w", encoding="utf-8") as quiet,
              contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet)):
            result = train(args.checkpoint, args.source, args.training_manifest, args.output, progress)
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
