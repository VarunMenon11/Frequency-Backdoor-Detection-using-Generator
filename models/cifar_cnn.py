"""CIFAR-sized image classifiers.

The first suspicious model is intentionally compact. It is large enough to learn
CIFAR-100 structure and a controlled trigger, but simple enough to inspect and
run on CPU for smoke tests.
"""

from __future__ import annotations

import torch
from torch import nn


class SmallCIFARClassifier(nn.Module):
    """A compact CNN for 32x32 RGB CIFAR-100 images.

    Input:
        images with shape (B, 3, 32, 32), range [0, 1]

    Output:
        logits with shape (B, num_classes)
    """

    def __init__(self, num_classes: int = 100) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 64),
            _conv_block(64, 64),
            nn.MaxPool2d(kernel_size=2),
            _conv_block(64, 128),
            _conv_block(128, 128),
            nn.MaxPool2d(kernel_size=2),
            _conv_block(128, 256),
            _conv_block(256, 256),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        return self.classifier(features)


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable model parameters."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )
