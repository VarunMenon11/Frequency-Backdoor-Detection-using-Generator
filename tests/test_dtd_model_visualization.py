import unittest

from scripts.visualize_dtd_poison_model_predictions import (
    diverse_samples, pair_records,
)


class DTDModelVisualizationTests(unittest.TestCase):
    def test_prediction_categories_require_correct_clean_baseline(self):
        rows = [
            {"source_id": "a", "relative_path": "a.jpg", "original_label": 1},
            {"source_id": "b", "relative_path": "b.jpg", "original_label": 2},
            {"source_id": "c", "relative_path": "c.jpg", "original_label": 3},
            {"source_id": "d", "relative_path": "d.jpg", "original_label": 4},
        ]
        clean = [
            {"prediction": 1, "confidence": .9},
            {"prediction": 2, "confidence": .9},
            {"prediction": 3, "confidence": .9},
            {"prediction": 2, "confidence": .9},
        ]
        triggered = [
            {"prediction": 0, "confidence": .9},
            {"prediction": 2, "confidence": .9},
            {"prediction": 4, "confidence": .9},
            {"prediction": 0, "confidence": .9},
        ]
        records = pair_records(rows, clean, triggered, target_label=0)
        self.assertEqual(
            [record["category"] for record in records],
            ["successful_attacks", "resisted_attacks", "other_trigger_errors", "clean_errors"],
        )

    def test_representatives_prioritize_distinct_classes(self):
        records = [
            {
                "source_id": str(index), "original_label": label,
                "trigger_confidence": confidence,
            }
            for index, (label, confidence) in enumerate(((1, .99), (1, .98), (2, .80)))
        ]
        selected = diverse_samples(records, count=2, seed=42)
        self.assertEqual({record["original_label"] for record in selected}, {1, 2})


if __name__ == "__main__":
    unittest.main()
