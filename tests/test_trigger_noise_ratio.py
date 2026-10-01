import unittest

import torch

from scripts.evaluate_dtd_trigger_noise_ratio import (
    add_tnr_noise,
    parse_tnr_values,
    per_image_rms,
)


class TriggerNoiseRatioTests(unittest.TestCase):
    def test_parser_preserves_requested_order(self):
        self.assertEqual(parse_tnr_values("20,10,0,-10"), [20.0, 10.0, 0.0, -10.0])

    def test_duplicate_ratios_are_rejected(self):
        with self.assertRaises(ValueError):
            parse_tnr_values("10,0,10")

    def test_noise_is_scaled_to_requested_tnr_without_clipping(self):
        clean = torch.full((2, 3, 16, 16), 0.5)
        triggered = clean + 0.01
        raw_noise = torch.randn_like(clean)
        attacked, noise_only, actual_noise, realized = add_tnr_noise(
            clean, triggered, raw_noise, 10.0
        )
        expected_noise_rms = per_image_rms(triggered-clean) / (10.0 ** 0.5)
        self.assertTrue(torch.allclose(per_image_rms(actual_noise), expected_noise_rms, atol=1e-6))
        self.assertTrue(torch.allclose(realized, torch.full_like(realized, 10.0), atol=1e-4))
        self.assertEqual(attacked.shape, clean.shape)
        self.assertEqual(noise_only.shape, clean.shape)


if __name__ == "__main__":
    unittest.main()
