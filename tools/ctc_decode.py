#!/usr/bin/env python3
"""Experimental CTC prefix-beam decoder for offline recognition comparisons.

Tooling only: an approximate, token-pruned prefix-marginal alternative to the
greedy CTC decoder in ``tools/benchmark_mobile.py``. It is not wired into any
production, Android, or default path.

A beam wider than 1 is a token-pruned approximate marginal decoder, not the
official joint attention/CTC desktop decoder (ESPnet ``BatchBeamSearch``), and
it does not promise better WER. There is no language model, no word
dictionary, no length bonus, and no vocabulary bias here.

Complexity / pruning semantics: per frame, each of at most ``beam_size``
prefixes considers the top ``token_top_k`` non-blank labels plus the combined
blank/EOS mass. Search expansion is O(T * beam_size * token_top_k), plus
sorting candidates each frame. Working candidates use
O(beam_size * token_top_k * mean_prefix_length) memory; retained prefixes use
O(beam_size * mean_prefix_length). Prefixes that fall outside the top
``beam_size`` by total log probability (ties broken by ascending lexicographic
token-ID order) are dropped with their probability mass, which is what makes
beam > 1 approximate. ``beam_size=1`` skips the search and runs the existing
greedy decoding behavior exactly.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

_NEG_INF = float("-inf")


def _validate_inputs(
    logits: np.ndarray, tokens: Sequence[str], beam_size: int, token_top_k: int
) -> tuple[np.ndarray, list[str]]:
    if not isinstance(logits, np.ndarray):
        raise TypeError(f"logits must be a numpy array, got {type(logits).__name__}")
    if logits.ndim == 3:
        if logits.shape[0] != 1:
            raise ValueError(f"batched logits must have batch 1, got shape {logits.shape}")
        frames = logits[0]
    elif logits.ndim == 2:
        frames = logits
    else:
        raise ValueError(f"logits must have shape [T,V] or [1,T,V], got {logits.shape}")
    if frames.shape[0] < 1:
        raise ValueError(f"logits need T>=1 frames, got shape {frames.shape}")
    token_list = list(tokens)
    if len(token_list) != frames.shape[1]:
        raise ValueError(
            f"V==len(tokens) required, got V={frames.shape[1]} "
            f"and {len(token_list)} tokens"
        )
    if len(token_list) < 3:
        raise ValueError(
            "need at least blank, one token and EOS "
            f"(V>=3), got V={len(token_list)}"
        )
    if not all(isinstance(token, str) for token in token_list):
        raise ValueError("tokens must all be strings")
    values = np.asarray(frames, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("logits must be finite; refusing to invent text from NaN/Inf")
    if isinstance(beam_size, bool) or not isinstance(beam_size, int):
        raise ValueError(f"beam_size must be an int, got {beam_size!r}")
    if not 1 <= beam_size <= 100:
        raise ValueError(f"beam_size must be in 1..100, got {beam_size}")
    n_nonblank = len(token_list) - 2
    if isinstance(token_top_k, bool) or not isinstance(token_top_k, int):
        raise ValueError(f"token_top_k must be an int, got {token_top_k!r}")
    # token_top_k only steers the prefix search; the beam_size=1 greedy path
    # ignores it so that path stays exactly the existing greedy baseline.
    if beam_size > 1 and not 1 <= token_top_k <= n_nonblank:
        raise ValueError(
            f"token_top_k must be in 1..V-2 (1..{n_nonblank}), got {token_top_k}"
        )
    return values, token_list


def _greedy_text(frames: np.ndarray, tokens: list[str]) -> str:
    """Exact copy of the tools/benchmark_mobile.py:decode collapsing rule."""
    ids = np.argmax(frames, axis=-1)
    previous = -1
    out = []
    for index in ids:
        index = int(index)
        if index != previous and index not in (0, len(tokens) - 1):
            out.append(tokens[index])
        previous = index
    return "".join(out).replace("▁", " ").strip()


def _format(prefix: tuple[int, ...], tokens: list[str]) -> str:
    return "".join(tokens[i] for i in prefix).replace("▁", " ").strip()


def prefix_beam_decode(
    logits: np.ndarray,
    tokens: Sequence[str],
    *,
    beam_size: int = 10,
    token_top_k: int = 20,
) -> str:
    """Decode CTC logits with an experimental prefix beam search.

    Args:
        logits: finite floating array of shape [1,T,V] or [T,V] with
            T>=1 and V==len(tokens). Blank ID is 0, EOS ID is V-1. Each
            row is normalised with log-softmax, so raw logits and log-probs
            are both accepted.
        tokens: vocabulary strings; index 0 is blank, index V-1 is EOS.
        beam_size: 1..100 kept prefixes per frame. ``1`` runs the existing
            greedy decoding behavior exactly.
        token_top_k: 1..V-2 non-blank labels kept per frame (top by frame
            log-probability, ties broken by smaller token ID); blank/EOS
            mass is always kept. Validated only when beam_size > 1; the
            beam_size=1 greedy path ignores it.

    Returns:
        Decoded text with ``▁`` converted to spaces and trimmed.
    """
    frames, token_list = _validate_inputs(logits, tokens, beam_size, token_top_k)
    blank_id = 0
    eos_id = len(token_list) - 1

    if beam_size == 1:
        return _greedy_text(frames, token_list)

    # Accept raw logits or log-probs: normalise every row with log-softmax.
    shifted = frames - frames.max(axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))

    beams: dict[tuple[int, ...], list[float]] = {(): [_NEG_INF, _NEG_INF]}
    beams[()] = [0.0, _NEG_INF]  # empty prefix starts blank-ended with prob 1.
    for row in log_probs:
        log_blank = float(np.logaddexp(row[blank_id], row[eos_id]))
        order = sorted(range(1, eos_id), key=lambda c: (-float(row[c]), c))
        candidates = order[:token_top_k]
        nxt: dict[tuple[int, ...], list[float]] = {}

        def add(prefix: tuple[int, ...], blank_mass: float, nonblank_mass: float) -> None:
            entry = nxt.get(prefix)
            if entry is None:
                nxt[prefix] = [blank_mass, nonblank_mass]
            else:
                entry[0] = float(np.logaddexp(entry[0], blank_mass))
                entry[1] = float(np.logaddexp(entry[1], nonblank_mass))

        for prefix, (p_blank, p_nonblank) in beams.items():
            total = float(np.logaddexp(p_blank, p_nonblank))
            # Blank/EOS mass: every path stays on the same prefix and becomes
            # blank-ended, whether it ended blank or non-blank before.
            add(prefix, total + log_blank, _NEG_INF)
            last = prefix[-1] if prefix else None
            for label in candidates:
                log_p = float(row[label])
                if last is not None and label == last:
                    # Repeat without a blank collapses onto the same prefix
                    # (only from non-blank mass); with a blank separator it
                    # extends a new repeated-token prefix (only from blank).
                    add(prefix, _NEG_INF, p_nonblank + log_p)
                    add(prefix + (label,), _NEG_INF, p_blank + log_p)
                else:
                    add(prefix + (label,), _NEG_INF, total + log_p)
        ranked = sorted(
            nxt.items(),
            key=lambda item: (-float(np.logaddexp(item[1][0], item[1][1])), item[0]),
        )
        beams = dict(ranked[:beam_size])

    best = min(
        beams.items(),
        key=lambda item: (-float(np.logaddexp(item[1][0], item[1][1])), item[0]),
    )[0]
    return _format(best, token_list)
