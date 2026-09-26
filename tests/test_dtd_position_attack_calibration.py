import json
from pathlib import Path
import unittest

from scripts.run_dtd_position_attack_calibration import (
    CANDIDATES,
    TRIGGERS,
    qualifies,
    select_candidates,
)
from scripts.train_asb_dtd_pretrained_classifier import resolve_attack_triggers
from scripts.train_asb_dtd_pretrained_classifier import interleave_rows_by_label


class Args:
    minimum_clean_accuracy = 0.55
    minimum_position_asr = 0.60
    minimum_position_conditional_asr = 0.50


def position_metrics(asr=0.80, conditional=0.75, lift=0.70):
    return {
        name: {
            "asr": asr,
            "conditional_asr": conditional,
            "target_rate_lift": lift,
        }
        for name in TRIGGERS
    }


class DTDPositionAttackCalibrationTests(unittest.TestCase):
    def test_position_catalog_resolves_all_four_triggers(self):
        path = Path("configs/dtd_ftrojan_position_catalog.json")
        catalog = json.loads(path.read_text(encoding="utf-8"))
        names, configs = resolve_attack_triggers(",".join(TRIGGERS), catalog)
        self.assertEqual(tuple(names), TRIGGERS)
        self.assertEqual(len(configs), 4)
        self.assertEqual(
            tuple(map(tuple, configs["ftpos_original"]["dct_positions"])),
            ((15, 15), (31, 31)),
        )
        self.assertEqual(
            len({tuple(map(tuple, config["dct_positions"])) for config in configs.values()}),
            4,
        )

    def test_qualification_requires_every_position(self):
        evaluation = {"validation_clean": {"accuracy": 0.60}}
        metrics = position_metrics()
        self.assertTrue(qualifies(evaluation, metrics, Args()))
        metrics[TRIGGERS[-1]]["asr"] = 0.20
        self.assertFalse(qualifies(evaluation, metrics, Args()))

    def test_qualification_requires_clean_utility_and_positive_lift(self):
        self.assertFalse(
            qualifies({"validation_clean": {"accuracy": 0.54}}, position_metrics(), Args())
        )
        metrics = position_metrics()
        metrics[TRIGGERS[0]]["target_rate_lift"] = 0.0
        self.assertFalse(
            qualifies({"validation_clean": {"accuracy": 0.60}}, metrics, Args())
        )

    def test_candidate_selection_preserves_requested_order(self):
        requested = f"{CANDIDATES[1].tag},{CANDIDATES[0].tag}"
        selected = select_candidates(requested)
        self.assertEqual([row.tag for row in selected], [CANDIDATES[1].tag, CANDIDATES[0].tag])

    def test_partial_evaluation_order_interleaves_classes(self):
        rows = [
            {"source_id": f"{label}-{index}", "original_label": label}
            for label in range(3)
            for index in range(2)
        ]
        ordered = interleave_rows_by_label(rows)
        self.assertEqual(
            [row["original_label"] for row in ordered],
            [0, 1, 2, 0, 1, 2],
        )


if __name__ == "__main__":
    unittest.main()
