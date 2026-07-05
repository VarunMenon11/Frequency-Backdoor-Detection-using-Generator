"""Classification evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class ClassificationMetrics:
    """Aggregate classification metrics for one evaluation split."""

    loss: float
    accuracy: float
    num_samples: int


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    max_batches: int | None = None,
) -> ClassificationMetrics:
    """Evaluate average loss and accuracy for a classifier."""

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = int(labels.numel())
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Evaluation loader produced no samples")

    return ClassificationMetrics(
        loss=total_loss / total_samples,
        accuracy=total_correct / total_samples,
        num_samples=total_samples,
    )
