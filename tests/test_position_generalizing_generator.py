import unittest

from scripts.train_dtd_reference_free_generator import (
    aggregate_trigger_metrics,
    parse_trigger_names,
    partition_rows_by_trigger,
)


def metrics(suspicious_asr, corrected_asr, corrected_accuracy):
    return {
        "suspicious_clean_accuracy": 0.61,
        "generator_clean_accuracy": 0.60,
        "clean_non_target_target_rate": 0.06,
        "suspicious_asr": suspicious_asr,
        "corrected_asr": corrected_asr,
        "asr_reduction": suspicious_asr - corrected_asr,
        "corrected_label_accuracy": corrected_accuracy,
        "corrected_image_l1_to_clean": 0.01,
        "mean_absolute_effective_log_correction": 0.02,
    }


class PositionGeneralizingGeneratorTests(unittest.TestCase):
    def test_trigger_parser_rejects_duplicates(self):
        self.assertEqual(parse_trigger_names("one, two"), ["one", "two"])
        with self.assertRaises(ValueError):
            parse_trigger_names("one,one")

    def test_partition_is_balanced_and_reproducible(self):
        rows = [{"source_id": str(index)} for index in range(11)]
        first = partition_rows_by_trigger(rows, ["a", "b"], seed=42)
        second = partition_rows_by_trigger(rows, ["a", "b"], seed=42)
        self.assertEqual(first, second)
        self.assertEqual(sorted(map(len, first.values())), [5, 6])
        self.assertEqual(
            {row["source_id"] for group in first.values() for row in group},
            {row["source_id"] for row in rows},
        )

    def test_aggregate_tracks_worst_trigger_not_only_mean(self):
        result = aggregate_trigger_metrics({
            "easy": metrics(0.99, 0.05, 0.60),
            "hard": metrics(0.98, 0.30, 0.45),
        })
        self.assertAlmostEqual(result["mean_corrected_asr"], 0.175)
        self.assertAlmostEqual(result["worst_corrected_asr"], 0.30)
        self.assertAlmostEqual(result["minimum_corrected_label_accuracy"], 0.45)


if __name__ == "__main__":
    unittest.main()
