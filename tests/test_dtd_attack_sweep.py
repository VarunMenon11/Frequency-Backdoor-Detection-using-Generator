import unittest
from argparse import Namespace

from scripts.run_dtd_attack_sweep import (
    CANDIDATES, qualifies, select_candidates,
)


class DTDAttackSweepTests(unittest.TestCase):
    def test_default_order_and_subset_selection(self):
        self.assertEqual(select_candidates(None), list(CANDIDATES))
        subset = select_candidates(
            "ftrojan_m100_paired_r010,haar_lh_s100_paired_r010"
        )
        self.assertEqual(
            [candidate.tag for candidate in subset],
            ["ftrojan_m100_paired_r010", "haar_lh_s100_paired_r010"],
        )

    def test_unknown_and_duplicate_candidates_rejected(self):
        with self.assertRaises(ValueError):
            select_candidates("missing")
        with self.assertRaises(ValueError):
            select_candidates("ftrojan_m100_paired_r010,ftrojan_m100_paired_r010")

    def test_qualification_requires_all_thresholds(self):
        args = Namespace(
            minimum_clean_accuracy=0.55,
            desired_asr=0.80,
            desired_conditional_asr=0.70,
        )
        evaluation = {"validation_clean": {"accuracy": 0.60}}
        metric = {
            "asr": 0.90,
            "conditional_asr_clean_correct": 0.80,
            "same_model_target_rate_lift": 0.75,
        }
        self.assertTrue(qualifies(metric, evaluation, args))
        for field, value in (
            ("asr", 0.79),
            ("conditional_asr_clean_correct", 0.69),
            ("same_model_target_rate_lift", 0.0),
        ):
            changed = dict(metric)
            changed[field] = value
            self.assertFalse(qualifies(changed, evaluation, args))


if __name__ == "__main__":
    unittest.main()
