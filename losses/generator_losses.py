"""Losses for adaptive spectral correction generator training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from generator import total_variation


@dataclass(frozen=True)
class GeneratorLossWeights:
    """Weights for generator training objectives."""

    classification: float = 1.0
    reconstruction: float = 4.0
    sparsity: float = 0.02
    smoothness: float = 0.01


@dataclass(frozen=True)
class GeneratorLossBreakdown:
    """Detached scalar loss values for logging."""

    total: float
    classification: float
    reconstruction: float
    sparsity: float
    smoothness: float


def compute_generator_loss(
    *,
    classifier_logits: torch.Tensor,
    clean_labels: torch.Tensor,
    corrected_images: torch.Tensor,
    clean_images: torch.Tensor,
    correction_map: torch.Tensor,
    weights: GeneratorLossWeights,
    criterion: nn.Module,
) -> tuple[torch.Tensor, GeneratorLossBreakdown]:
    """Compute the weighted generator loss.

    Classification loss encourages corrected triggered images to be classified
    as their clean labels. Reconstruction and sparsity discourage destructive
    corrections.
    """

    classification_loss = criterion(classifier_logits, clean_labels)
    reconstruction_loss = torch.mean(torch.abs(corrected_images - clean_images))
    sparsity_loss = torch.mean(torch.abs(correction_map))
    smoothness_loss = total_variation(correction_map)

    total_loss = (
        weights.classification * classification_loss
        + weights.reconstruction * reconstruction_loss
        + weights.sparsity * sparsity_loss
        + weights.smoothness * smoothness_loss
    )

    return total_loss, GeneratorLossBreakdown(
        total=float(total_loss.detach().item()),
        classification=float(classification_loss.detach().item()),
        reconstruction=float(reconstruction_loss.detach().item()),
        sparsity=float(sparsity_loss.detach().item()),
        smoothness=float(smoothness_loss.detach().item()),
    )
