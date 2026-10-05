import math
import unittest

import datawork as d
import detector


class DetectorTests(unittest.TestCase):
    def test_threshold_handles_ties_and_uses_human_validation_only(self):
        threshold = detector.choose_threshold([0, 0, 0, 1], [0.7, 0.7, 0.2, 0.99], 0)
        self.assertGreater(threshold, 0.7)
        self.assertLess(threshold, 0.99)
        self.assertEqual(sum(p >= threshold for p in [0.7, 0.7, 0.2]), 0)
        with self.assertRaises(ValueError): detector.choose_threshold([1], [0.5])
        with self.assertRaises(ValueError): detector.choose_threshold([0], [0.5], 1)
        with self.assertRaises(ValueError): detector.choose_threshold([0], [1.0], 0)

    def test_sentence_and_span_exports_are_regrouped_before_training(self):
        original = d.record("fixture", "one", "A human translation.", "human_translation", "original-doc")
        original.update(split_group="combined-original-group", split="test")
        span = d.record("fixture", "span", "A larger human translation. " * 10, "human_translation", "different-span-doc")
        span.update(id="span/fixture/span", split_group="combined-original-group", split="train")
        candidate = d.record("fixture", "candidate", "Unverified text", "human_candidate", "candidate")
        candidate.update(split_group="candidate", split="train")
        records, examples = detector.prepare([original, candidate], [span], minimum_words=10)
        self.assertEqual(len(examples), 2)
        self.assertEqual(len({r["split"] for r in records}), 1)
        self.assertEqual(len({r["split_group"] for r in records}), 1)
        self.assertEqual({r["binary_target"] for r in records}, {0})
        self.assertEqual({r["example_granularity"] for r in records}, {"segment", "contiguous_span"})


if __name__ == "__main__":
    unittest.main()
