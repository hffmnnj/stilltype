"""Experimental CTC activity evidence, independent of decoded references."""
from __future__ import annotations

import math

STRONG_MOTION = 8.0
CANDIDATE_MOTION = 4.0
MIN_MOVING_SECONDS = 0.3
PADDING_SECONDS = 0.4
MIN_EMISSIONS = 2
MIN_MEAN_PROBABILITY = 0.30


def span(motion, fps: float, threshold: float) -> tuple[int, int] | None:
    moving = [index for index, value in enumerate(motion) if value > threshold]
    if len(moving) < MIN_MOVING_SECONDS * fps:
        return None
    padding = int(PADDING_SECONDS * fps)
    return max(0, moving[0] - padding), min(len(motion), moving[-1] + 1 + padding)


def greedy_evidence(log_probs, tokens: list[str]) -> tuple[str, int, float]:
    """Collapse greedy IDs and score emissions from log-softmax model output."""
    import numpy as np

    if log_probs.ndim != 3 or log_probs.shape[0] != 1 or log_probs.shape[2] != len(tokens):
        raise ValueError("Invalid CTC output shape")
    if not np.isfinite(log_probs).all():
        raise ValueError("Non-finite CTC output")
    previous = -1
    emitted = []
    probabilities = []
    for frame in log_probs[0]:
        best = int(frame.argmax())
        if best != previous and best not in (0, len(tokens) - 1):
            emitted.append(tokens[best])
            probabilities.append(math.exp(float(frame[best])))
        previous = best
    text = "".join(emitted).replace("▁", " ").strip()
    return text, len(probabilities), sum(probabilities) / len(probabilities) if probabilities else 0.0


def admits(evidence: tuple[str, int, float]) -> bool:
    text, count, mean_probability = evidence
    return bool(text.strip()) and count >= MIN_EMISSIONS and mean_probability >= MIN_MEAN_PROBABILITY
