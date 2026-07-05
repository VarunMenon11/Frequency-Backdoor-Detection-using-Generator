"""Training helpers for image classifiers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class TrainEpochMetrics:
    """Aggregate metrics from one training epoch."""

    loss: float
    accuracy: float
    num_samples: int


def train_classifier_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    max_batches: int | None = None,
) -> TrainEpochMetrics:
    """Train a classifier for one epoch."""

    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = int(labels.numel())
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Training loader produced no samples")

    return TrainEpochMetrics(
        loss=total_loss / total_samples,
        accuracy=total_correct / total_samples,
        num_samples=total_samples,
    )
