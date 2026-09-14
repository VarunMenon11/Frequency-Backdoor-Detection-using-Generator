import unittest
from collections import Counter

from scripts.visualize_asb_dtd_trigger_spectra import metric_sample_indices


class SamplingTests(unittest.TestCase):
    def test_balanced_reproducible_unique(self):
        rows = [{"original_label": label} for label in range(46) for _ in range(40)]
        indices = metric_sample_indices(rows, 184, "balanced", 42)
        self.assertEqual(indices, metric_sample_indices(rows, 184, "balanced", 42))
        self.assertEqual(len(set(indices)), 184)
        counts = Counter(rows[index]["original_label"] for index in indices)
        self.assertEqual(set(counts.values()), {4})
        self.assertEqual(len(counts), 46)

    def test_exhausted_groups_and_legacy_mode(self):
        rows = [{"original_label": 0}] + [{"original_label": 1}] * 3
        self.assertEqual(set(metric_sample_indices(rows, 20, "balanced", 42)), set(range(4)))
        self.assertEqual(metric_sample_indices(rows, 2, "first", 42), [0, 1])
        self.assertEqual(metric_sample_indices([], 2, "balanced", 42), [])
        with self.assertRaises(ValueError):
            metric_sample_indices(rows, 0, "balanced", 42)


if __name__ == "__main__":
    unittest.main()
