import unittest

import torch

from scripts.evaluate_dtd_position_generator_baselines import (
    aggregate_results,
    broad_spectral_correction,
)
from scripts.audit_dtd_reference_free_generator import reconstruct_with_log_correction


class PositionGeneratorBaselineTests(unittest.TestCase):
    def test_zero_broad_attenuation_is_identity(self):
        images = torch.rand(2, 3, 20, 24)
        correction = broad_spectral_correction(images, cutoff=0.35, attenuation=0.0)
        reconstructed = reconstruct_with_log_correction(images, correction)
        self.assertTrue(torch.allclose(images, reconstructed, atol=1e-5))

    def test_broad_suppression_reduces_high_frequency_checkerboard(self):
        grid = torch.arange(24)
        checker = ((grid[:, None] + grid[None, :]) % 2).float()
        images = checker[None, None].repeat(1, 3, 1, 1)
        correction = broad_spectral_correction(images, cutoff=0.35, attenuation=0.5)
        reconstructed = reconstruct_with_log_correction(images, correction)
        before = torch.fft.fft2(images).abs()[..., 12, 12]
        after = torch.fft.fft2(reconstructed).abs()[..., 12, 12]
        self.assertTrue(torch.all(after < before))

    def test_aggregate_reports_worst_position(self):
        def result(target_rate, accuracy):
            return {"variants": {
                name: {"target_rate": target_rate, "true_label_accuracy": accuracy,
                       "pixel_l1_to_clean": 0.01}
                for name in (
                    "clean", "no_defense", "full_generator", "static_template",
                    "broad_suppression", "strongest_support_only",
                    "outside_strongest_support",
                )
            }}
        aggregate = aggregate_results({"a": result(0.1, 0.6), "b": result(0.3, 0.5)})
        row = aggregate["variants"]["full_generator"]
        self.assertAlmostEqual(row["mean_target_rate"], 0.2)
        self.assertAlmostEqual(row["worst_target_rate"], 0.3)


if __name__ == "__main__":
    unittest.main()
