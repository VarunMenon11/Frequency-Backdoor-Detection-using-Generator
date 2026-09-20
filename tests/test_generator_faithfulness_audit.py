import unittest

import torch

from scripts.audit_dtd_reference_free_generator import (
    make_conjugate_symmetric,
    reconstruct_with_log_correction,
    top_fraction_mask,
)


class GeneratorFaithfulnessAuditTests(unittest.TestCase):
    def test_top_fraction_mask_selects_expected_number_without_ties(self):
        values = torch.arange(1, 101, dtype=torch.float32).view(1, 1, 10, 10)
        mask = top_fraction_mask(values, 0.10)
        self.assertEqual(int(mask.sum()), 10)
        self.assertTrue(torch.equal(mask.flatten()[-10:], torch.ones(10)))

    def test_zero_correction_reconstructs_input(self):
        images = torch.rand(2, 3, 17, 19)
        reconstructed = reconstruct_with_log_correction(
            images, torch.zeros_like(images)
        )
        self.assertTrue(torch.allclose(images, reconstructed, atol=1e-5))

    def test_symmetrizer_matches_conjugate_counterpart(self):
        values = torch.randn(2, 3, 17, 18)
        symmetric = make_conjugate_symmetric(values)
        counterpart = torch.roll(
            torch.flip(symmetric, dims=(-2, -1)),
            shifts=(1, 1), dims=(-2, -1),
        )
        self.assertTrue(torch.allclose(symmetric, counterpart, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
