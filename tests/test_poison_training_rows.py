import unittest

from datasets.asb_manifest import build_poisoned_training_rows


def clean_rows(count=100):
    return [
        {
            "source_id": str(index),
            "protocol_role": "clean_train",
            "original_label": index % 5,
            "training_label": index % 5,
            "variant_name": "clean",
            "variant_type": "clean",
        }
        for index in range(count)
    ]


CONFIGS = {"trigger": {"trigger_kind": "ftrojan_dct", "strength": 100.0}}


class PoisonTrainingRowTests(unittest.TestCase):
    def test_replace_preserves_legacy_size(self):
        selected, counts = build_poisoned_training_rows(
            clean_rows(), trigger_configs=CONFIGS, target_label=0,
            poison_ratio=0.10, seed=42,
        )
        self.assertEqual(len(selected), 100)
        self.assertEqual(counts["poisoned_training_samples"], 10)
        self.assertAlmostEqual(counts["actual_poison_ratio"], 0.10)

    def test_paired_keeps_clean_counterparts_and_final_ratio(self):
        selected, counts = build_poisoned_training_rows(
            clean_rows(), trigger_configs=CONFIGS, target_label=0,
            poison_ratio=0.10, seed=42, poison_mode="paired",
        )
        poison = [row for row in selected if row["variant_name"] == "trigger"]
        clean_ids = {
            row["source_id"] for row in selected if row["variant_name"] == "clean"
        }
        self.assertEqual(counts["clean_training_samples"], 100)
        self.assertEqual(counts["poisoned_training_samples"], 11)
        self.assertEqual(len(selected), 111)
        self.assertTrue(all(row["source_id"] in clean_ids for row in poison))
        self.assertTrue(all(row["training_label"] == 0 for row in poison))
        self.assertAlmostEqual(counts["actual_poison_ratio"], 11 / 111)

    def test_dynamic_assignment_is_reproducible_and_rotates(self):
        def poison_ids(seed):
            selected, _ = build_poisoned_training_rows(
                clean_rows(), trigger_configs=CONFIGS, target_label=0,
                poison_ratio=0.10, seed=seed, poison_mode="dynamic-paired",
            )
            return {
                row["source_id"] for row in selected
                if row["variant_name"] == "trigger"
            }

        self.assertEqual(poison_ids(42), poison_ids(42))
        self.assertNotEqual(poison_ids(42), poison_ids(43))

    def test_invalid_paired_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            build_poisoned_training_rows(
                clean_rows(), trigger_configs=CONFIGS, target_label=0,
                poison_ratio=1.0, seed=42, poison_mode="paired",
            )


if __name__ == "__main__":
    unittest.main()
