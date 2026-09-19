import unittest

import torch

from generator import ReferenceFreeSpectralGenerator, build_reference_free_evidence


class ReferenceFreeGeneratorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_evidence_uses_one_spectrum_and_has_expected_channels(self):
        images = torch.rand(2, 3, 32, 40)
        spectrum = torch.fft.fftshift(torch.fft.fft2(images), dim=(-2, -1))
        evidence = build_reference_free_evidence(
            torch.log1p(spectrum.abs()), torch.angle(spectrum)
        )
        self.assertEqual(evidence.shape, (2, 15, 32, 40))
        self.assertTrue(torch.isfinite(evidence).all())

    def test_forward_preserves_shape_range_and_is_identity_at_initialization(self):
        images = torch.rand(2, 3, 32, 32)
        model = ReferenceFreeSpectralGenerator(base_channels=8)
        output = model(images)
        self.assertEqual(output.corrected_image.shape, images.shape)
        self.assertEqual(output.correction_mask.shape, images.shape)
        self.assertGreaterEqual(float(output.corrected_image.min().detach()), 0.0)
        self.assertLessEqual(float(output.corrected_image.max().detach()), 1.0)
        self.assertLess(float((output.corrected_image - images).abs().max().detach()), 1e-5)
        self.assertLess(float(output.effective_log_correction.abs().max().detach()), 1e-7)

    def test_classifier_style_loss_reaches_generator_parameters(self):
        images = torch.rand(2, 3, 32, 32)
        model = ReferenceFreeSpectralGenerator(base_channels=8)
        output = model(images)
        loss = output.corrected_image.square().mean()
        loss.backward()
        gradient = model.output.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_effective_correction_is_conjugate_symmetric(self):
        images = torch.rand(1, 3, 31, 33)
        model = ReferenceFreeSpectralGenerator(base_channels=8)
        with torch.no_grad():
            model.output.weight.normal_(0.0, 0.01)
            model.output.bias.zero_()
        correction = model(images).effective_log_correction
        counterpart = torch.roll(
            torch.flip(correction, dims=(-2, -1)), shifts=(1, 1), dims=(-2, -1)
        )
        self.assertTrue(torch.allclose(correction, counterpart, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
