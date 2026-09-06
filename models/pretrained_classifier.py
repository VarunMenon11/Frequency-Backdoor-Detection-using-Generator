"""ImageNet-pretrained classifiers used by the advanced experiment track."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


PretrainedBackbone = Literal["resnet18"]
WeightMode = Literal["default", "none"]


class ImageNetNormalizedClassifier(nn.Module):
    """A classifier that accepts RGB tensors in [0, 1]."""

    def __init__(
        self,
        *,
        backbone: PretrainedBackbone,
        num_classes: int,
        weights: WeightMode = "default",
    ) -> None:
        super().__init__()
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one")
        if backbone != "resnet18":
            raise ValueError(f"Unsupported pretrained backbone: {backbone}")

        selected_weights = ResNet18_Weights.DEFAULT if weights == "default" else None
        network = resnet18(weights=selected_weights)
        in_features = network.fc.in_features
        network.fc = nn.Linear(in_features, num_classes)

        self.backbone_name = backbone
        self.weight_mode = weights
        self.num_classes = num_classes
        self.network = network
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=True,
        )
        self._feature_extractor_frozen = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        normalized = (images - self.image_mean) / self.image_std
        return self.network(normalized)

    def set_feature_extractor_trainable(self, trainable: bool) -> None:
        """Freeze or unfreeze all layers except the replacement classifier head."""

        for parameter in self.network.parameters():
            parameter.requires_grad = trainable
        for parameter in self.network.fc.parameters():
            parameter.requires_grad = True
        self._feature_extractor_frozen = not trainable

    def train(self, mode: bool = True) -> "ImageNetNormalizedClassifier":
        super().train(mode)
        if mode and self._feature_extractor_frozen:
            self.network.eval()
            self.network.fc.train()
        return self

    def feature_parameters(self):
        for name, parameter in self.network.named_parameters():
            if not name.startswith("fc."):
                yield parameter

    def head_parameters(self):
        yield from self.network.fc.parameters()

    def metadata(self) -> dict[str, object]:
        return {
            "architecture": self.backbone_name,
            "initial_weights": (
                "ImageNet-1K pretrained" if self.weight_mode == "default" else "random"
            ),
            "input_range": [0.0, 1.0],
            "normalization_mean": self.image_mean.flatten().tolist(),
            "normalization_std": self.image_std.flatten().tolist(),
            "num_classes": self.num_classes,
        }


def build_pretrained_classifier(
    *,
    backbone: PretrainedBackbone,
    num_classes: int,
    weights: WeightMode = "default",
) -> ImageNetNormalizedClassifier:
    return ImageNetNormalizedClassifier(
        backbone=backbone,
        num_classes=num_classes,
        weights=weights,
    )
