#!/usr/bin/env python3
"""Export the pinned FLARE Base Transformer checkpoint as local ONNX utilities.

The supplied official FLARE checkout is trusted executable code, not verified by
this tool. The checkpoint is pinned and model terms are CC-BY-NC-4.0. These are
newly exported utility artifacts, not newly trained weights or app-ready models.
No reference text or audio is used; no downloads or network inference are made.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import os
import sys
from dataclasses import make_dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_SIZE = 2_126_211_579
CHECKPOINT_SHA256 = "fe8c8ecab633bd4e3e490924e2bc917420b4d18d7309a5cd612902ccc842853b"
CHECKPOINT_REVISION = "6187ad650dc1786a8efc8b6bfbee37bbf62c4b2e"
CHECKPOINT_ARTIFACT = "exp_22_base/finetuned/checkpoints/checkpoint_last.pt"
GRAPH_NAMES = ("encoder.onnx", "decoder.onnx", "cross.onnx", "decoder-cache.onnx", "decoder-beam.onnx")
CACHE_LAYERS, CACHE_HEADS, CACHE_HEAD_DIM = 6, 4, 192


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoint(path: Path) -> None:
    if path.stat().st_size != CHECKPOINT_SIZE:
        raise ValueError("checkpoint size does not match the pinned FLARE release")
    if sha256(path) != CHECKPOINT_SHA256:
        raise ValueError("checkpoint SHA-256 does not match the pinned FLARE release")


def validate_output(path: Path) -> Path:
    output = path.resolve()
    root = REPO_ROOT.resolve()
    if output == root or root in output.parents:
        raise ValueError("refusing output inside the Stilltype repository")
    if any((parent / ".git").exists() for parent in (output, *output.parents)):
        raise ValueError("refusing output inside a Git working tree")
    if output.exists() and not output.is_dir():
        raise ValueError("output must be a directory")
    for name in (*GRAPH_NAMES, "metadata.json"):
        if os.path.lexists(output / name):
            raise ValueError(f"refusing to overwrite {output / name}")
    return output


def load_sample(path: Path) -> tuple[Any, Any]:
    import numpy as np

    with np.load(path, allow_pickle=False) as sample:
        if set(sample.files) != {"face", "lip"}:
            raise ValueError("sample must contain only face and lip arrays")
        face, lip = sample["face"], sample["lip"]
        if (face.dtype != np.uint8 or lip.dtype != np.uint8
                or face.shape != lip.shape or face.ndim != 3
                or face.shape[1:] != (96, 96) or not 1 <= face.shape[0] <= 200):
            raise ValueError("sample face and lip must match uint8 [T,96,96], T in 1..200")
        def normalize(array: Any) -> Any:
            return (array[:, 4:92, 4:92].astype(np.float32) / 255.0 - 0.421) / 0.165

        return normalize(face), normalize(lip)


def export(checkpoint: Path, source: Path, sample_path: Path, output: Path, device: str,
           decoder_mode: str = "full", beam_size: int = 5) -> dict[str, Any]:
    if decoder_mode not in ("full", "cached", "beam"):
        raise ValueError("decoder mode must be full, cached, or beam")
    if type(beam_size) is not int or beam_size not in (5, 50):
        raise ValueError("beam size must be 5 or 50")
    if decoder_mode != "beam" and beam_size != 5:
        raise ValueError("beam size other than 5 requires beam decoder mode")
    output = validate_output(output)
    if not (source / "flare" / "__init__.py").is_file():
        raise ValueError("source must be a trusted FLARE checkout containing flare/__init__.py")
    # Verify both properties before importing any executable code from the source.
    verify_checkpoint(checkpoint)
    face_array, lip_array = load_sample(sample_path)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    import sentencepiece
    import torch
    from fairseq import tasks
    from fairseq.data.dictionary import Dictionary
    from fairseq.data.encoders.sentencepiece_bpe import SentencepieceBPE
    from omegaconf import OmegaConf

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")

    sys.path.insert(0, str(source.resolve()))
    import flare  # noqa: F401
    from flare.hubert_pretraining import AVHubertPretrainingConfig

    # OmegaConf 2.0 needs a concrete field type, not a postponed annotation.
    CompatiblePretrainingConfig = make_dataclass(
        "CompatiblePretrainingConfig", [("input_modality", str, "image")],
        bases=(AVHubertPretrainingConfig,),
    )
    tasks.TASK_DATACLASS_REGISTRY["av_hubert_pretraining"] = CompatiblePretrainingConfig
    with torch.serialization.safe_globals([Dictionary, SentencepieceBPE, sentencepiece.SentencePieceProcessor]):
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    state.pop("last_optimizer_state", None)
    cfg = OmegaConf.create(state["cfg"])
    demo = str(source.resolve() / "flare/data_demo/tsv_merge_face_lip_dataset")
    cfg.task.data = cfg.task.label_dir = demo
    pre = OmegaConf.load(source / "flare/conf/pretrain/base_whisper_distill_offline_bestrq.yaml")
    pre.task.data = pre.task.label_dir = demo
    pre.task.noise_wav = "unused"
    pre.task.label_rate = 25
    pre.model.input_modality = "image"
    del pre.model["wav_input"]
    for key, value in {
        "masking_type": "input2", "frontend": "convnet", "encoder_type": "qwenconv",
        "encoder_embed_dim": 512, "encoder_layers": 12, "encoder_attention_heads": 8,
        "num_key_value_heads": 2, "modality_dropout": 0.75, "audio_dropout": 1.0,
    }.items():
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
    model.eval().to(device)

    class VisualEncoder(torch.nn.Module):
        def __init__(self, encoder: Any) -> None:
            super().__init__()
            self.encoder = encoder

        def forward(self, face: Any, lip: Any, face_mask: Any, lip_mask: Any, padding_mask: Any) -> tuple[Any, Any]:
            inputs = {"audio": None, "video": (face, lip), "face_lip_mask": (face_mask, lip_mask)}
            result = self.encoder(inputs, padding_mask)
            return result["encoder_out"], result["padding_mask"]

    class Decoder(torch.nn.Module):
        def __init__(self, decoder: Any) -> None:
            super().__init__()
            self.decoder = decoder

        def forward(self, tokens: Any, encoder_out: Any, padding_mask: Any) -> Any:
            length = tokens.shape[1]
            future = torch.triu(torch.ones((length, length), dtype=torch.bool, device=tokens.device), diagonal=1)
            self.decoder._future_mask = torch.zeros(
                (length, length), dtype=encoder_out.dtype, device=encoder_out.device
            ).masked_fill(future, float("-inf"))
            logits, _ = self.decoder(tokens, {"encoder_out": encoder_out, "padding_mask": padding_mask})
            return logits[:, -1, :]

    class CrossProject(torch.nn.Module):
        def __init__(self, layers: Any) -> None:
            super().__init__()
            self.layers = layers

        def forward(self, encoded: Any) -> tuple[Any, ...]:
            result = []
            for layer in self.layers:
                attn = layer.encoder_attn
                for projection in (attn.k_proj, attn.v_proj):
                    value = projection(encoded).view(-1, attn.num_heads, attn.head_dim)
                    result.append(value.permute(1, 0, 2).unsqueeze(0))
            return tuple(result)

    class IncrementalDecoder(torch.nn.Module):
        def __init__(self, decoder: Any) -> None:
            super().__init__()
            self.decoder = decoder

        def forward(self, tokens: Any, encoded: Any, mask: Any, *cache: Any) -> tuple[Any, ...]:
            state: dict[str, Any] = {}
            count = len(self.decoder.layers)
            for index, layer in enumerate(self.decoder.layers):
                layer.self_attn._set_input_buffer(state, {
                    "prev_key": cache[2 * index], "prev_value": cache[2 * index + 1],
                })
                layer.encoder_attn._set_input_buffer(state, {
                    "prev_key": cache[2 * count + 2 * index],
                    "prev_value": cache[2 * count + 2 * index + 1],
                    "prev_key_padding_mask": mask,
                })
            logits, _ = self.decoder(tokens, {"encoder_out": encoded, "padding_mask": mask},
                                     incremental_state=state)
            result = [logits[:, -1, :]]
            for layer in self.decoder.layers:
                saved = layer.self_attn._get_input_buffer(state)
                result.extend((saved["prev_key"], saved["prev_value"]))
            return tuple(result)

    dictionary = task.target_dictionary
    symbols = list(dictionary.symbols)
    if len(symbols) != 1000:
        raise ValueError(f"expected 1000 dictionary symbols, got {len(symbols)}")
    encoder = VisualEncoder(model.encoder).eval()
    decoder = Decoder(model.decoder).eval()
    if decoder_mode in ("cached", "beam"):
        if type(model.decoder.embed_positions).__name__ != "SinusoidalPositionalEmbedding":
            raise ValueError("expected the inspected sinusoidal positional embedding")
        if len(model.decoder.layers) != CACHE_LAYERS or any(
            attn.num_heads != CACHE_HEADS or attn.head_dim != CACHE_HEAD_DIM
            for layer in model.decoder.layers for attn in (layer.self_attn, layer.encoder_attn)
        ):
            raise ValueError("unsupported decoder cache geometry")
    face = torch.from_numpy(face_array).unsqueeze(0).unsqueeze(0).to(device)
    lip = torch.from_numpy(lip_array).unsqueeze(0).unsqueeze(0).to(device)
    mask = torch.ones(1, dtype=torch.bool, device=device)
    padding = torch.zeros((1, face.shape[2]), dtype=torch.bool, device=device)
    inputs = (face, lip, mask, mask, padding)

    output.mkdir(parents=True, exist_ok=True)
    # Reserve all names exclusively before export; existing artifacts are never overwritten.
    graph_names = ("encoder.onnx", "decoder.onnx") if decoder_mode == "full" else (
        "encoder.onnx", "cross.onnx", "decoder-beam.onnx" if decoder_mode == "beam" else "decoder-cache.onnx")
    for name in (*graph_names, "metadata.json"):
        with (output / name).open("xb"):
            pass
    with torch.inference_mode():
        torch.onnx.export(
            encoder, inputs, str(output / GRAPH_NAMES[0]),
            input_names=["face", "lip", "face_mask", "lip_mask", "padding_mask"],
            output_names=["encoder_out", "encoder_padding_mask"],
            dynamic_axes={"face": {2: "frames"}, "lip": {2: "frames"}, "padding_mask": {1: "frames"},
                          "encoder_out": {0: "encoded_frames"}, "encoder_padding_mask": {1: "encoded_frames"}},
            opset_version=17, dynamo=False,
        )
        encoded, encoded_mask = encoder(*inputs)
        prefix = torch.tensor([[dictionary.eos(), 4, 5]], dtype=torch.long, device=device)
        if decoder_mode == "full":
            model.decoder._future_mask = None
            torch.onnx.export(
                decoder, (prefix, encoded, encoded_mask), str(output / "decoder.onnx"),
                input_names=["tokens", "encoder_out", "encoder_padding_mask"], output_names=["next_logits"],
                dynamic_axes={"tokens": {1: "prefix"}, "encoder_out": {0: "encoded_frames"},
                              "encoder_padding_mask": {1: "encoded_frames"}},
                opset_version=17, dynamo=False,
            )
        else:
            cross_names = [f"cross_{i}_{kind}" for i in range(CACHE_LAYERS) for kind in ("key", "value")]
            past_names = [f"past_{i}_{kind}" for i in range(CACHE_LAYERS) for kind in ("key", "value")]
            present_names = [f"present_{i}_{kind}" for i in range(CACHE_LAYERS) for kind in ("key", "value")]
            cross = CrossProject(model.decoder.layers).eval()
            cached = IncrementalDecoder(model.decoder).eval()
            torch.onnx.export(
                cross, (encoded,), str(output / "cross.onnx"), input_names=["encoder_out"],
                output_names=cross_names,
                dynamic_axes={"encoder_out": {0: "encoded_frames"},
                              **{name: {2: "encoded_frames"} for name in cross_names}},
                opset_version=17, dynamo=False,
            )
            cross_values = cross(encoded)
            batch = 2 if decoder_mode == "beam" else 1
            past = tuple(torch.zeros((batch, CACHE_HEADS, 2, CACHE_HEAD_DIM), device=device,
                                     dtype=encoded.dtype) for _ in past_names)
            axes = {"tokens": {1: "prefix"}, "encoder_out": {0: "encoded_frames"},
                    "encoder_padding_mask": {1: "encoded_frames"},
                    **{name: {2: "past"} for name in past_names},
                    **{name: {2: "encoded_frames"} for name in cross_names},
                    **{name: {2: "present"} for name in present_names}}
            if decoder_mode == "beam":
                for name in ("tokens", "encoder_padding_mask", "next_logits", *past_names, *cross_names, *present_names):
                    axes.setdefault(name, {})[0] = "batch"
                axes["encoder_out"][1] = "batch"
                prefix = prefix.repeat(batch, 1)
                decoder_encoded = encoded.repeat(1, batch, 1)
                decoder_mask = encoded_mask.repeat(batch, 1)
                cross_values = tuple(value.repeat(batch, 1, 1, 1) for value in cross_values)
            else:
                decoder_encoded, decoder_mask = encoded, encoded_mask
            torch.onnx.export(
                cached, (prefix, decoder_encoded, decoder_mask, *past, *cross_values),
                str(output / graph_names[2]),
                input_names=["tokens", "encoder_out", "encoder_padding_mask", *past_names, *cross_names],
                output_names=["next_logits", *present_names],
                dynamic_axes=axes,
                opset_version=17, dynamo=False,
            )

    files = [{"name": name, "sizeBytes": (output / name).stat().st_size, "sha256": sha256(output / name)}
             for name in graph_names]
    metadata = {
        "schemaVersion": 1, "modelId": "flare-base-transformer-2026-research", "releaseYear": 2026,
        "license": "CC-BY-NC-4.0", "sourceCheckpointRepository": "Anony94/FLARE-checkpoints",
        "sourceCheckpointRevision": CHECKPOINT_REVISION, "sourceCheckpointArtifact": CHECKPOINT_ARTIFACT,
        "sourceCheckpointSha256": CHECKPOINT_SHA256, "fps": 25,
        "inputSize": 96, "cropSize": 88, "mean": 0.421, "std": 0.165,
        "inputGeometry": [96, 96], "crop": [4, 92, 4, 92], "cropGeometry": [88, 88],
        "normalization": {"scale": 255.0, "mean": 0.421, "std": 0.165},
        "maxFrames": 200, "maxPrefix": 128, "featureSize": 512, "vocabularySize": 1000,
        "bosId": dictionary.bos(), "padId": dictionary.pad(), "eosId": dictionary.eos(),
        "unkId": dictionary.unk(), "symbols": symbols, "files": files,
        "encoderBytes": files[0]["sizeBytes"], "encoderSha256": files[0]["sha256"],
        "note": "Caller must enforce frame and prefix bounds. Padding and exact bounds are not yet validated app features. "
                "These are exported utilities, not new trained weights or evidence of accuracy gains or native readiness.",
    }
    if decoder_mode == "full":
        metadata.update(decoderBytes=files[1]["sizeBytes"], decoderSha256=files[1]["sha256"])
    else:
        metadata.update(decoderMode=decoder_mode, cacheLayers=CACHE_LAYERS, cacheHeads=CACHE_HEADS,
                         cacheHeadDim=CACHE_HEAD_DIM, crossBytes=files[1]["sizeBytes"],
                         crossSha256=files[1]["sha256"])
        if decoder_mode == "beam":
            metadata.update(beamSize=beam_size, beamDecoderBytes=files[2]["sizeBytes"],
                            beamDecoderSha256=files[2]["sha256"])
        else:
            metadata.update(cachedDecoderBytes=files[2]["sizeBytes"],
                            cachedDecoderSha256=files[2]["sha256"])
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"output": str(output), "device": device, "files": files, "metadata": "metadata.json"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Pinned FLARE checkpoint file.")
    parser.add_argument("--source", type=Path, required=True, help="Trusted official FLARE checkout (executable code).")
    parser.add_argument("--sample", type=Path, required=True, help="Reference-free NPZ with face and lip uint8 [T,96,96].")
    parser.add_argument("--output", type=Path, required=True, help="Output directory outside the repository.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--decoder-mode", choices=("full", "cached", "beam"), default="full")
    parser.add_argument("--beam-size", type=int, choices=(5, 50), default=5)
    parser.add_argument("--accept-license", action="store_true", required=True, help="Accept CC-BY-NC-4.0 model terms.")
    args = parser.parse_args(argv)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = export(args.checkpoint, args.source, args.sample, args.output, args.device,
                            args.decoder_mode, args.beam_size)
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
