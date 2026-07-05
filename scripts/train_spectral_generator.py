"""Train the adaptive spectral correction generator.

This script is intended for Kaggle/GPU execution. It freezes the suspicious
classifier and trains a generator to produce sparse amplitude correction maps
for triggered images.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.cifar100_dataset import CleanCIFAR100Dataset
from generator import (
    SpectralCorrectionGenerator,
    apply_correction_map,
    build_generator_input,
    fft_amplitude_phase,
    reconstruct_from_amplitude_phase,
)
from losses import GeneratorLossWeights, compute_generator_loss
from models import SmallCIFARClassifier, count_trainable_parameters
from poisoning.frequency_trigger import FrequencyTriggerConfig, apply_frequency_trigger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=Path("datasets/archive.zip"),
        help="Path to CIFAR-100 zip archive or extracted folder containing train/test/meta.",
    )
    parser.add_argument(
        "--classifier-checkpoint",
        type=Path,
        default=Path("experiments/suspicious_classifier_trained/suspicious_classifier.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/spectral_generator_cifar100"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--strength", type=float, default=0.08)
    parser.add_argument("--horizontal-frequency", type=int, default=6)
    parser.add_argument("--vertical-frequency", type=int, default=6)
    parser.add_argument("--classification-weight", type=float, default=1.0)
    parser.add_argument("--reconstruction-weight", type=float, default=4.0)
    parser.add_argument("--sparsity-weight", type=float, default=0.02)
    parser.add_argument("--smoothness-weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    trigger_config = FrequencyTriggerConfig(
        horizontal_frequency=args.horizontal_frequency,
        vertical_frequency=args.vertical_frequency,
        strength=args.strength,
    )
    loss_weights = GeneratorLossWeights(
        classification=args.classification_weight,
        reconstruction=args.reconstruction_weight,
        sparsity=args.sparsity_weight,
        smoothness=args.smoothness_weight,
    )

    train_dataset = CleanCIFAR100Dataset.from_zip(args.zip_path, split="train")
    test_dataset = CleanCIFAR100Dataset.from_zip(args.zip_path, split="test")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    classifier = SmallCIFARClassifier(num_classes=100).to(device)
    checkpoint = torch.load(args.classifier_checkpoint, map_location=device)
    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier.eval()
    for parameter in classifier.parameters():
        parameter.requires_grad = False

    generator = SpectralCorrectionGenerator().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        generator.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print("Device:", device)
    print("Frozen classifier checkpoint:", args.classifier_checkpoint)
    print("Generator parameters:", count_trainable_parameters(generator))
    print("Target label:", args.target_label, train_dataset.fine_label_names[args.target_label])
    print("Loss weights:", loss_weights)

    history: list[dict[str, float | int]] = []
    start_time = perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_record = train_one_epoch(
            generator=generator,
            classifier=classifier,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            trigger_config=trigger_config,
            target_label=args.target_label,
            loss_weights=loss_weights,
            max_batches=args.max_train_batches,
        )
        eval_record = evaluate_generator(
            generator=generator,
            classifier=classifier,
            loader=test_loader,
            device=device,
            trigger_config=trigger_config,
            target_label=args.target_label,
            max_batches=args.max_eval_batches,
        )
        record = {"epoch": epoch, **train_record, **eval_record}
        history.append(record)

        print(
            "Epoch "
            f"{epoch:03d} | "
            f"loss {train_record['train_total_loss']:.4f} | "
            f"before ASR {eval_record['before_asr']:.4f} | "
            f"after ASR {eval_record['after_asr']:.4f} | "
            f"corrected acc {eval_record['corrected_clean_label_accuracy']:.4f} | "
            f"corr mean {eval_record['mean_correction_value']:.4f}"
        )

    elapsed_seconds = perf_counter() - start_time
    summary = {
        "script": "scripts/train_spectral_generator.py",
        "device": str(device),
        "classifier_checkpoint": str(args.classifier_checkpoint),
        "generator": generator.__class__.__name__,
        "generator_parameters": count_trainable_parameters(generator),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "target_label": args.target_label,
        "target_label_name": train_dataset.fine_label_names[args.target_label],
        "trigger": {
            "type": "sinusoidal_frequency",
            "strength": args.strength,
            "horizontal_frequency": args.horizontal_frequency,
            "vertical_frequency": args.vertical_frequency,
        },
        "loss_weights": loss_weights.__dict__,
        "max_train_batches": args.max_train_batches,
        "max_eval_batches": args.max_eval_batches,
        "elapsed_seconds": elapsed_seconds,
        "history": history,
    }

    summary_path = args.output_dir / "generator_training_summary.json"
    checkpoint_path = args.output_dir / "spectral_generator.pt"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    torch.save(
        {
            "generator_state_dict": generator.state_dict(),
            "summary": summary,
        },
        checkpoint_path,
    )

    print("Saved generator summary:", summary_path)
    print("Saved generator checkpoint:", checkpoint_path)


def train_one_epoch(
    *,
    generator: SpectralCorrectionGenerator,
    classifier: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    trigger_config: FrequencyTriggerConfig,
    target_label: int,
    loss_weights: GeneratorLossWeights,
    max_batches: int | None,
) -> dict[str, float]:
    generator.train()
    totals = {
        "total": 0.0,
        "classification": 0.0,
        "reconstruction": 0.0,
        "sparsity": 0.0,
        "smoothness": 0.0,
    }
    total_samples = 0

    for batch_index, (clean_images, clean_labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        clean_images = clean_images.to(device)
        clean_labels = clean_labels.to(device)
        non_target_mask = clean_labels != target_label
        if int(non_target_mask.sum().item()) == 0:
            continue

        clean_images = clean_images[non_target_mask]
        clean_labels = clean_labels[non_target_mask]
        triggered_images = apply_frequency_trigger(clean_images, trigger_config)

        clean_amplitude, _ = fft_amplitude_phase(clean_images)
        triggered_amplitude, triggered_phase = fft_amplitude_phase(triggered_images)
        generator_input = build_generator_input(clean_amplitude, triggered_amplitude)
        correction_map = generator(generator_input)
        corrected_amplitude = apply_correction_map(
            clean_amplitude,
            triggered_amplitude,
            correction_map,
        )
        corrected_images = reconstruct_from_amplitude_phase(
            corrected_amplitude,
            triggered_phase,
        )
        logits = classifier(corrected_images)

        loss, breakdown = compute_generator_loss(
            classifier_logits=logits,
            clean_labels=clean_labels,
            corrected_images=corrected_images,
            clean_images=clean_images,
            correction_map=correction_map,
            weights=loss_weights,
            criterion=criterion,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = int(clean_labels.numel())
        total_samples += batch_size
        totals["total"] += breakdown.total * batch_size
        totals["classification"] += breakdown.classification * batch_size
        totals["reconstruction"] += breakdown.reconstruction * batch_size
        totals["sparsity"] += breakdown.sparsity * batch_size
        totals["smoothness"] += breakdown.smoothness * batch_size

    if total_samples == 0:
        raise ValueError("No non-target samples were seen during generator training")

    return {
        "train_total_loss": totals["total"] / total_samples,
        "train_classification_loss": totals["classification"] / total_samples,
        "train_reconstruction_loss": totals["reconstruction"] / total_samples,
        "train_sparsity_loss": totals["sparsity"] / total_samples,
        "train_smoothness_loss": totals["smoothness"] / total_samples,
    }


@torch.no_grad()
def evaluate_generator(
    *,
    generator: SpectralCorrectionGenerator,
    classifier: nn.Module,
    loader: DataLoader,
    device: torch.device,
    trigger_config: FrequencyTriggerConfig,
    target_label: int,
    max_batches: int | None,
) -> dict[str, float]:
    generator.eval()
    total_samples = 0
    before_target_predictions = 0
    after_target_predictions = 0
    corrected_clean_predictions = 0
    reconstruction_l1 = 0.0
    correction_mean = 0.0

    for batch_index, (clean_images, clean_labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        clean_images = clean_images.to(device)
        clean_labels = clean_labels.to(device)
        non_target_mask = clean_labels != target_label
        if int(non_target_mask.sum().item()) == 0:
            continue

        clean_images = clean_images[non_target_mask]
        clean_labels = clean_labels[non_target_mask]
        triggered_images = apply_frequency_trigger(clean_images, trigger_config)

        clean_amplitude, _ = fft_amplitude_phase(clean_images)
        triggered_amplitude, triggered_phase = fft_amplitude_phase(triggered_images)
        generator_input = build_generator_input(clean_amplitude, triggered_amplitude)
        correction_map = generator(generator_input)
        corrected_amplitude = apply_correction_map(
            clean_amplitude,
            triggered_amplitude,
            correction_map,
        )
        corrected_images = reconstruct_from_amplitude_phase(
            corrected_amplitude,
            triggered_phase,
        )

        before_preds = classifier(triggered_images).argmax(dim=1)
        after_preds = classifier(corrected_images).argmax(dim=1)
        batch_size = int(clean_labels.numel())
        total_samples += batch_size
        before_target_predictions += int((before_preds == target_label).sum().item())
        after_target_predictions += int((after_preds == target_label).sum().item())
        corrected_clean_predictions += int((after_preds == clean_labels).sum().item())
        reconstruction_l1 += float(torch.abs(corrected_images - clean_images).mean().item()) * batch_size
        correction_mean += float(correction_map.mean().item()) * batch_size

    if total_samples == 0:
        raise ValueError("No non-target samples were seen during generator evaluation")

    return {
        "before_asr": before_target_predictions / total_samples,
        "after_asr": after_target_predictions / total_samples,
        "corrected_clean_label_accuracy": corrected_clean_predictions / total_samples,
        "eval_reconstruction_l1": reconstruction_l1 / total_samples,
        "mean_correction_value": correction_mean / total_samples,
        "eval_samples": float(total_samples),
    }


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
