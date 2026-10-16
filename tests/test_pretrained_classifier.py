import unittest

import torch
from torch import nn

from models import build_pretrained_classifier


class PretrainedClassifierTests(unittest.TestCase):
    def test_small_input_stem_preserves_resolution_and_output_shape(self):
        model = build_pretrained_classifier(
            backbone="resnet18",
            num_classes=100,
            weights="none",
            small_input_stem=True,
        )

        self.assertEqual(model.network.conv1.stride, (1, 1))
        self.assertIsInstance(model.network.maxpool, nn.Identity)
        self.assertTrue(model.metadata()["small_input_stem"])

        with torch.no_grad():
            logits = model(torch.rand(2, 3, 32, 32))
        self.assertEqual(tuple(logits.shape), (2, 100))

    def test_default_stem_remains_unchanged(self):
        model = build_pretrained_classifier(
            backbone="resnet18",
            num_classes=47,
            weights="none",
        )

        self.assertEqual(model.network.conv1.stride, (2, 2))
        self.assertIsInstance(model.network.maxpool, nn.MaxPool2d)
        self.assertFalse(model.metadata()["small_input_stem"])


if __name__ == "__main__":
    unittest.main()
