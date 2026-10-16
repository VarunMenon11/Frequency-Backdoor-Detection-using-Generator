import unittest

import torch

from poisoning import (
    BenchmarkTriggerConfig,
    apply_benchmark_trigger,
    build_blended_carrier,
)


class BenchmarkTriggerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.images = torch.full((2, 3, 32, 32), 0.5)

    def test_sig_is_deterministic_bounded_and_nonzero(self):
        config = BenchmarkTriggerConfig(kind="sig")
        first = apply_benchmark_trigger(self.images, config)
        second = apply_benchmark_trigger(self.images, config)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first.shape, self.images.shape)
        self.assertGreater(float((first - self.images).abs().mean()), 0.0)
        self.assertGreaterEqual(float(first.min()), 0.0)
        self.assertLessEqual(float(first.max()), 1.0)

    def test_blended_uses_one_reproducible_carrier(self):
        config = BenchmarkTriggerConfig(kind="blended", blend_seed=77)
        first = apply_benchmark_trigger(self.images, config)
        second = apply_benchmark_trigger(self.images, config)
        self.assertTrue(torch.equal(first, second))
        carrier = build_blended_carrier(
            32, 32, seed=77, device=torch.device("cpu"), dtype=torch.float32
        )
        expected = 0.8 * self.images + 0.2 * carrier
        self.assertTrue(torch.allclose(first, expected))

    def test_ftrojan_supports_cifar_and_tiny_sizes(self):
        config = BenchmarkTriggerConfig(kind="ftrojan", ftrojan_strength=50.0)
        for size in (32, 64):
            image = torch.full((3, size, size), 0.5)
            triggered = apply_benchmark_trigger(image, config)
            self.assertEqual(triggered.shape, image.shape)
            self.assertGreater(float((triggered - image).abs().mean()), 0.0)

    def test_invalid_blend_alpha_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_benchmark_trigger(
                self.images, BenchmarkTriggerConfig(kind="blended", blend_alpha=1.1)
            )


if __name__ == "__main__":
    unittest.main()
