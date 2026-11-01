import itertools
import unittest

import numpy as np

from benchmark_mobile import decode as greedy_decode
from ctc_decode import prefix_beam_decode


def _brute_force_best(log_probs: np.ndarray, eos_id: int) -> tuple[int, ...]:
    """Independent oracle: enumerate every frame path, sum collapsed masses."""
    time_steps, vocab = log_probs.shape
    totals: dict[tuple[int, ...], float] = {}
    for path in itertools.product(range(vocab), repeat=time_steps):
        mass = sum(float(log_probs[t, path[t]]) for t in range(time_steps))
        collapsed: list[int] = []
        previous = -1
        for label in path:
            if label != previous and label not in (0, eos_id):
                collapsed.append(label)
            previous = label
        key = tuple(collapsed)
        totals[key] = (
            float(np.logaddexp(totals[key], mass)) if key in totals else mass
        )
    return min(totals.items(), key=lambda item: (-item[1], item[0]))[0]


class PrefixBeamDecodeTest(unittest.TestCase):
    def test_split_mass_beats_greedy_empty(self):
        # Two identical frames: blank .40, A .35, B .25 (plus a trace of EOS
        # so EOS handling is exercised). Greedy sees blank twice and emits
        # nothing, but the marginal mass of A (.35*.35 + 2*.4*.35 = .4025)
        # exceeds blank-only (.16), so the prefix search must emit A.
        tokens = ["<blank>", "A", "B", "<eos>"]
        row = [0.40, 0.35, 0.25, 1e-6]
        logits = np.log(np.array([row, row], dtype=np.float64))
        self.assertEqual(greedy_decode(logits[np.newaxis], tokens), "")
        self.assertGreater(0.35 * 0.35 + 2 * 0.4 * 0.35, 0.4 * 0.4)
        self.assertEqual(
            prefix_beam_decode(logits, tokens, beam_size=10, token_top_k=2), "A"
        )

    def test_wide_beam_matches_brute_force_marginals(self):
        # Beam and vocabulary pruning disabled by width (beam 50 holds every
        # reachable prefix for T<=3 over two labels; top_k covers both), so
        # the search is exact and must agree with path enumeration, including
        # blank-separated repeats and EOS acting as a repeat separator.
        tokens = ["<blank>", "A", "B", "<eos>"]
        tables = [
            [
                [0.50, 0.30, 0.15, 0.05],
                [0.10, 0.20, 0.60, 0.10],
                [0.45, 0.40, 0.10, 0.05],
            ],
            [
                [0.10, 0.70, 0.10, 0.10],
                [0.10, 0.10, 0.10, 0.70],
                [0.10, 0.70, 0.10, 0.10],
            ],
        ]
        for table in tables:
            with self.subTest(table=table):
                probs = np.array(table, dtype=np.float64)
                logits = np.log(probs)[np.newaxis]
                expected_ids = _brute_force_best(np.log(probs), len(tokens) - 1)
                expected = "".join(tokens[i] for i in expected_ids)
                self.assertEqual(
                    prefix_beam_decode(
                        logits, tokens, beam_size=50, token_top_k=2
                    ),
                    expected,
                )

    def test_beam_size_one_matches_greedy(self):
        tokens = ["<blank>", "▁HI", "▁THERE", "<eos>"]
        ids = [0, 1, 1, 0, 1, 2, 2, 3, 0]
        logits = np.full((1, len(ids), len(tokens)), -20.0, dtype=np.float32)
        for frame, token in enumerate(ids):
            logits[0, frame, token] = 20.0
        self.assertEqual(greedy_decode(logits, tokens), "HI HI THERE")
        self.assertEqual(
            prefix_beam_decode(logits, tokens, beam_size=1), "HI HI THERE"
        )
        # Raw (non-log-softmaxed) logits with repeats and EOS separation.
        tokens = ["<blank>", "▁AB", "▁C", "<eos>"]
        raw = np.array(
            [
                [0.0, 3.0, -1.0, -2.0],
                [-1.0, -2.0, -0.5, 4.0],
                [0.0, 3.0, -1.0, -2.0],
                [5.0, -1.0, -1.0, -1.0],
            ],
            dtype=np.float64,
        )
        self.assertEqual(
            prefix_beam_decode(raw, tokens, beam_size=1),
            greedy_decode(raw[np.newaxis], tokens),
        )

    def test_non_finite_logits_rejected(self):
        tokens = ["<blank>", "A", "B", "<eos>"]
        good = np.log(np.array([[[0.5, 0.3, 0.15, 0.05]]], dtype=np.float64))
        bad_nan = good.copy()
        bad_nan[0, 0, 1] = np.nan
        bad_inf = good.copy()
        bad_inf[0, 0, 0] = np.inf
        for bad in (bad_nan, bad_inf):
            with self.subTest(bad=bad.ravel().tolist()):
                with self.assertRaises(ValueError):
                    prefix_beam_decode(bad, tokens, beam_size=5, token_top_k=2)

    def test_malformed_shapes_and_ranges_rejected(self):
        tokens = ["<blank>", "A", "B", "<eos>"]
        good = np.log(np.array([[[0.5, 0.3, 0.15, 0.05]]], dtype=np.float64))
        with self.assertRaises(ValueError):
            prefix_beam_decode(good, ["<blank>", "A"], beam_size=5, token_top_k=1)
        with self.assertRaises(ValueError):
            prefix_beam_decode(good, tokens, beam_size=0, token_top_k=2)
        with self.assertRaises(ValueError):
            prefix_beam_decode(good, tokens, beam_size=101, token_top_k=2)
        with self.assertRaises(ValueError):
            prefix_beam_decode(good, tokens, beam_size=5, token_top_k=0)
        with self.assertRaises(ValueError):
            prefix_beam_decode(good, tokens, beam_size=5, token_top_k=3)
        with self.assertRaises(ValueError):
            prefix_beam_decode(
                np.concatenate([good, good], axis=0),
                tokens,
                beam_size=5,
                token_top_k=2,
            )


if __name__ == "__main__":
    unittest.main()
