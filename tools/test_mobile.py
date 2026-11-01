import unittest

import numpy as np

from benchmark_mobile import decode


class GreedyCtcTest(unittest.TestCase):
    def test_repeats_collapse_but_blank_separates_repeated_tokens(self):
        tokens = ["<blank>", "▁HI", "▁THERE", "<eos>"]
        ids = [0, 1, 1, 0, 1, 2, 2, 3, 0]
        logits = np.full((1, len(ids), len(tokens)), -20.0, dtype=np.float32)
        for frame, token in enumerate(ids):
            logits[0, frame, token] = 20.0
        self.assertEqual(decode(logits, tokens), "HI HI THERE")


if __name__ == "__main__":
    unittest.main()
