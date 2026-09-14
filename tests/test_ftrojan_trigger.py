import json
from pathlib import Path
import unittest

import torch

from poisoning import AdvancedTriggerConfig, apply_advanced_trigger
from scripts.train_asb_dtd_pretrained_classifier import resolve_attack_triggers


class FTrojanTriggerTests(unittest.TestCase):
    def config(self, strength=50.0):
        return AdvancedTriggerConfig(
            trigger_kind="ftrojan_dct", strength=strength,
            dct_block_size=32, dct_positions=((15, 15), (31, 31)),
            dct_channels=(1, 2),
        )

    def test_shape_range_determinism_and_nonzero_effect(self):
        image = torch.full((3, 65, 70), 0.5)
        first = apply_advanced_trigger(image, self.config())
        second = apply_advanced_trigger(image, self.config())
        self.assertEqual(first.shape, image.shape)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.isfinite(first).all())
        self.assertGreaterEqual(float(first.min()), 0.0)
        self.assertLessEqual(float(first.max()), 1.0)
        self.assertGreater(float((first - image).abs().mean()), 0.0)

    def test_zero_strength_is_exact_identity(self):
        image = torch.rand(2, 3, 64, 64)
        self.assertTrue(torch.equal(apply_advanced_trigger(image, self.config(0)), image))

    def test_effect_grows_with_magnitude_away_from_clipping(self):
        image = torch.full((3, 64, 64), 0.5)
        weak = (apply_advanced_trigger(image, self.config(20)) - image).abs().mean()
        strong = (apply_advanced_trigger(image, self.config(50)) - image).abs().mean()
        self.assertGreater(float(strong), 2.4 * float(weak))

    def test_invalid_dct_configuration_rejected(self):
        image = torch.rand(3, 32, 32)
        bad = AdvancedTriggerConfig(
            trigger_kind="ftrojan_dct", dct_block_size=32,
            dct_positions=((32, 1),),
        )
        with self.assertRaises(ValueError):
            apply_advanced_trigger(image, bad)

    def test_catalog_entry_is_trainable(self):
        catalog = json.loads(
            Path("Absolute_Dataset/asb_dtd_v1/trigger_catalog.json").read_text(encoding="utf-8")
        )
        names, configs = resolve_attack_triggers("ftrojan_mix", catalog)
        self.assertEqual(names, ["ftrojan_mix"])
        self.assertEqual(configs["ftrojan_mix"]["dct_positions"], [[15, 15], [31, 31]])


if __name__ == "__main__":
    unittest.main()
