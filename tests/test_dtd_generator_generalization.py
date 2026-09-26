import unittest

import torch

from poisoning import AdvancedTriggerConfig
from scripts.evaluate_dtd_generator_generalization import (
    apply_corruption,
    build_corruption_scenarios,
    build_trigger_scenarios,
    normalized_recovery,
    support_iou,
)


class DTDGeneratorGeneralizationTests(unittest.TestCase):
    def setUp(self):
        self.known = AdvancedTriggerConfig(
            trigger_kind="ftrojan_dct",
            strength=100.0,
            dct_block_size=32,
            dct_positions=((15, 15), (31, 31)),
            dct_channels=(1, 2),
        )

    def test_trigger_scenarios_include_known_and_unseen_conditions(self):
        scenarios = build_trigger_scenarios(
            self.known, {"strength", "position"}
        )
        self.assertEqual(len(scenarios), 11)
        self.assertEqual(sum(row["known"] for row in scenarios), 1)
        self.assertIn(
            "strength_100_known_positions",
            {row["name"] for row in scenarios},
        )
        self.assertTrue(
            any(row["config"].dct_positions != self.known.dct_positions
                for row in scenarios)
        )

    def test_corruption_suite_has_noise_blur_and_intensity_controls(self):
        scenarios = build_corruption_scenarios({"corruption"})
        self.assertEqual(len(scenarios), 8)
        self.assertEqual(
            {row["kind"] for row in scenarios},
            {"noise", "blur", "brightness", "contrast"},
        )
        self.assertEqual(build_corruption_scenarios({"strength"}), [])

    def test_corruptions_preserve_shape_and_valid_range(self):
        images = torch.rand(2, 3, 24, 24)
        for index, scenario in enumerate(build_corruption_scenarios({"corruption"})):
            generator = torch.Generator().manual_seed(index)
            result = apply_corruption(images, scenario, generator)
            self.assertEqual(result.shape, images.shape)
            self.assertGreaterEqual(float(result.min()), 0.0)
            self.assertLessEqual(float(result.max()), 1.0)

    def test_normalized_recovery_uses_clean_target_rate_as_floor(self):
        self.assertAlmostEqual(normalized_recovery(0.90, 0.10, 0.05), 0.94117647)
        self.assertIsNone(normalized_recovery(0.05, 0.04, 0.05))

    def test_support_iou(self):
        first = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
        second = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
        self.assertAlmostEqual(float(support_iou(first, second)), 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
