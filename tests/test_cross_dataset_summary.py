import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.summarize_cross_dataset_benchmark import build_comparison_outputs


class CrossDatasetSummaryTests(unittest.TestCase):
    def test_balanced_and_nested_aggressive_results_are_both_reported(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "cifar100" / "sig"
            aggressive = base / "defense_aggressive"
            aggressive.mkdir(parents=True)
            base.mkdir(parents=True, exist_ok=True)

            common = {
                "status": "complete",
                "dataset": "cifar100",
                "attack": "sig",
                "no_defense": {
                    "clean_accuracy": 0.79,
                    "asr": 0.99,
                    "robust_accuracy": 0.01,
                },
            }
            balanced = {
                **common,
                "defense_profile": "balanced",
                "reference_free_defense": {
                    "clean_accuracy": 0.78,
                    "asr": 0.50,
                    "robust_accuracy": 0.35,
                },
            }
            aggressive_result = {
                **common,
                "defense_profile": "aggressive",
                "reference_free_defense": {
                    "clean_accuracy": 0.65,
                    "asr": 0.20,
                    "robust_accuracy": 0.55,
                },
            }
            (base / "final_test_summary.json").write_text(json.dumps(balanced))
            (aggressive / "final_test_summary.json").write_text(
                json.dumps(aggressive_result)
            )

            rows = build_comparison_outputs(root)
            methods = {
                row["method"]
                for row in rows
                if row["dataset"] == "cifar100" and row["attack"] == "sig"
            }

            self.assertIn("Our reference-free generator (balanced)", methods)
            self.assertIn("Our reference-free generator (aggressive)", methods)
            self.assertIn("Our suspicious model: no defense", methods)


if __name__ == "__main__":
    unittest.main()
