"""Repair the suspicious classifier using generator-corrected images.

Phase 9 fine-tunes the already backdoored classifier. The repair data combines
clean images and corrected triggered images, both supervised with clean labels.
The goal is to reduce ASR in the classifier itself, not only through a
preprocessing generator.
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

from datasets.cifar100_dataset import CleanCIFAR100Dataset, TriggeredCIFAR100TestDataset
from evaluation import evaluate_classifier
from generator import (
    SpectralCorrectionGenerator,
    apply_correction_map,
    build_generator_input,
    fft_amplitude_phase,
    reconstruct_from_amplitude_phase,
)
from models import SmallCIFARClassifier, count_trainable_parameters
from poisoning.frequency_trigger import FrequencyTriggerConfig, apply_frequency_trigger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-path", type=Path, default=Path("datasets/archive.zip"))
    parser.add_argument(
        "--classifier-checkpoint",
        type=Path,
        default=Path("experiments/suspicious_classifier_trained/suspicious_classifier.pt"),
    )
    parser.add_argument(
        "--generator-checkpoint",
        type=Path,
        default=Path("experiments/spectral_generator_cifar100/spectral_generator.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/repaired_classifier_cifar100"),
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--strength", type=float, default=0.08)
    parser.add_argument("--horizontal-frequency", type=int, default=6)
    parser.add_argument("--vertical-frequency", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument(
        "--include-raw-triggered",
        action="store_true",
        help=(
            "Also fine-tune on raw triggered non-target images with clean labels. "
            "This directly teaches that the trigger should not imply the target class."
        ),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
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

    train_dataset = CleanCIFAR100Dataset.from_zip(args.zip_path, split="train")
    clean_test = CleanCIFAR100Dataset.from_zip(args.zip_path, split="test")
    triggered_test = TriggeredCIFAR100TestDataset.from_zip(
        args.zip_path,
        target_label=args.target_label,
        trigger_config=trigger_config,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    clean_test_loader = DataLoader(
        clean_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    triggered_test_loader = DataLoader(
        triggered_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    classifier = SmallCIFARClassifier(num_classes=100).to(device)
    classifier_checkpoint = torch.load(args.classifier_checkpoint, map_location=device)
    classifier.load_state_dict(classifier_checkpoint["model_state_dict"])

    generator = SpectralCorrectionGenerator().to(device)
    generator_checkpoint = torch.load(args.generator_checkpoint, map_location=device)
    generator.load_state_dict(generator_checkpoint["generator_state_dict"])
    generator.eval()
    for parameter in generator.parameters():
        parameter.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print("Device:", device)
    print("Classifier checkpoint:", args.classifier_checkpoint)
    print("Generator checkpoint:", args.generator_checkpoint)
    print("Classifier parameters:", count_trainable_parameters(classifier))
    print("Target label:", args.target_label, train_dataset.fine_label_names[args.target_label])

    before_clean = evaluate_classifier(
        classifier,
        clean_test_loader,
        criterion,
        device,
        max_batches=args.max_eval_batches,
    )
    before_asr = evaluate_classifier(
        classifier,
        triggered_test_loader,
        criterion,
        device,
        max_batches=args.max_eval_batches,
    )
    print(
        "Before repair | "
        f"clean acc {before_clean.accuracy:.4f} | "
        f"ASR {before_asr.accuracy:.4f}"
    )

    history: list[dict[str, float | int]] = []
    start_time = perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_record = train_repair_epoch(
            classifier=classifier,
            generator=generator,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            trigger_config=trigger_config,
            target_label=args.target_label,
            include_raw_triggered=args.include_raw_triggered,
            max_batches=args.max_train_batches,
        )
        clean_metrics = evaluate_classifier(
            classifier,
            clean_test_loader,
            criterion,
            device,
            max_batches=args.max_eval_batches,
        )
        asr_metrics = evaluate_classifier(
            classifier,
            triggered_test_loader,
            criterion,
            device,
            max_batches=args.max_eval_batches,
        )
        record = {
            "epoch": epoch,
            **train_record,
            "clean_test_loss": clean_metrics.loss,
            "clean_accuracy": clean_metrics.accuracy,
            "asr_loss": asr_metrics.loss,
            "attack_success_rate": asr_metrics.accuracy,
        }
        history.append(record)

        print(
            "Epoch "
            f"{epoch:03d} | "
            f"repair acc {train_record['repair_train_accuracy']:.4f} | "
            f"clean acc {clean_metrics.accuracy:.4f} | "
            f"ASR {asr_metrics.accuracy:.4f}"
        )

    elapsed_seconds = perf_counter() - start_time
    after_clean = evaluate_classifier(
        classifier,
        clean_test_loader,
        criterion,
        device,
        max_batches=args.max_eval_batches,
    )
    after_asr = evaluate_classifier(
        classifier,
        triggered_test_loader,
        criterion,
        device,
        max_batches=args.max_eval_batches,
    )

    summary = {
        "script": "scripts/repair_suspicious_classifier.py",
        "device": str(device),
        "classifier_checkpoint": str(args.classifier_checkpoint),
        "generator_checkpoint": str(args.generator_checkpoint),
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
        "max_train_batches": args.max_train_batches,
        "max_eval_batches": args.max_eval_batches,
        "include_raw_triggered": args.include_raw_triggered,
        "before_repair": {
            "clean_accuracy": before_clean.accuracy,
            "clean_loss": before_clean.loss,
            "attack_success_rate": before_asr.accuracy,
            "asr_loss": before_asr.loss,
        },
        "after_repair": {
            "clean_accuracy": after_clean.accuracy,
            "clean_loss": after_clean.loss,
            "attack_success_rate": after_asr.accuracy,
            "asr_loss": after_asr.loss,
        },
        "elapsed_seconds": elapsed_seconds,
        "history": history,
    }

    summary_path = args.output_dir / "repair_summary.json"
    checkpoint_path = args.output_dir / "repaired_classifier.pt"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    torch.save(
        {
            "model_state_dict": classifier.state_dict(),
            "summary": summary,
        },
        checkpoint_path,
    )

    print(
        "After repair | "
        f"clean acc {after_clean.accuracy:.4f} | "
        f"ASR {after_asr.accuracy:.4f}"
    )
    print("Saved repair summary:", summary_path)
    print("Saved repaired checkpoint:", checkpoint_path)


def train_repair_epoch(
    *,
    classifier: nn.Module,
    generator: SpectralCorrectionGenerator,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    trigger_config: FrequencyTriggerConfig,
    target_label: int,
    include_raw_triggered: bool,
    max_batches: int | None,
) -> dict[str, float]:
    classifier.train()
    generator.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_corrected_samples = 0
    total_raw_triggered_samples = 0

    for batch_index, (clean_images, clean_labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        clean_images = clean_images.to(device)
        clean_labels = clean_labels.to(device)

        non_target_mask = clean_labels != target_label
        repair_images = [clean_images]
        repair_labels = [clean_labels]

        if int(non_target_mask.sum().item()) > 0:
            non_target_clean_images = clean_images[non_target_mask]
            non_target_clean_labels = clean_labels[non_target_mask]

            corrected_images = build_corrected_images(
                clean_images=non_target_clean_images,
                generator=generator,
                trigger_config=trigger_config,
            )
            repair_images.append(corrected_images)
            repair_labels.append(non_target_clean_labels)
            total_corrected_samples += int(corrected_images.shape[0])

            if include_raw_triggered:
                raw_triggered_images = apply_frequency_trigger(
                    non_target_clean_images,
                    trigger_config,
                )
                repair_images.append(raw_triggered_images)
                repair_labels.append(non_target_clean_labels)
                total_raw_triggered_samples += int(raw_triggered_images.shape[0])

        images = torch.cat(repair_images, dim=0)
        labels = torch.cat(repair_labels, dim=0)

        optimizer.zero_grad(set_to_none=True)
        logits = classifier(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = int(labels.numel())
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Repair loader produced no samples")

    return {
        "repair_train_loss": total_loss / total_samples,
        "repair_train_accuracy": total_correct / total_samples,
        "repair_train_samples": float(total_samples),
        "corrected_repair_samples": float(total_corrected_samples),
        "raw_triggered_repair_samples": float(total_raw_triggered_samples),
    }


@torch.no_grad()
def build_corrected_images(
    *,
    clean_images: torch.Tensor,
    generator: SpectralCorrectionGenerator,
    trigger_config: FrequencyTriggerConfig,
) -> torch.Tensor:
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
    return reconstruct_from_amplitude_phase(corrected_amplitude, triggered_phase)


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
