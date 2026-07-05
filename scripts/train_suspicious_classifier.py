"""Train and evaluate the suspicious CIFAR-100 classifier.

This is the main Phase 4 demonstration script. It trains on the poisoned
training set, then reports:

- Clean Accuracy on the clean CIFAR-100 test set.
- Attack Success Rate (ASR) on the triggered non-target test set.

Use ``--max-train-batches`` and ``--max-eval-batches`` for a quick smoke test.
Remove those limits for a real experiment.
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

from datasets.cifar100_dataset import (
    CleanCIFAR100Dataset,
    PoisonedCIFAR100Dataset,
    TriggeredCIFAR100TestDataset,
)
from evaluation import evaluate_classifier
from models import SmallCIFARClassifier, count_trainable_parameters
from poisoning.frequency_trigger import FrequencyTriggerConfig
from training import train_classifier_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-path", type=Path, default=Path("datasets/archive.zip"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/suspicious_classifier_cifar100"),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--poison-ratio", type=float, default=0.12)
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--strength", type=float, default=0.08)
    parser.add_argument("--horizontal-frequency", type=int, default=6)
    parser.add_argument("--vertical-frequency", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Training device. Use auto to choose CUDA when available.",
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

    poisoned_train = PoisonedCIFAR100Dataset.from_zip(
        args.zip_path,
        split="train",
        poison_ratio=args.poison_ratio,
        target_label=args.target_label,
        trigger_config=trigger_config,
        seed=args.seed,
    )
    clean_test = CleanCIFAR100Dataset.from_zip(args.zip_path, split="test")
    triggered_test = TriggeredCIFAR100TestDataset.from_zip(
        args.zip_path,
        target_label=args.target_label,
        trigger_config=trigger_config,
    )

    train_loader = DataLoader(
        poisoned_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    clean_test_loader = DataLoader(
        clean_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    triggered_test_loader = DataLoader(
        triggered_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = SmallCIFARClassifier(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print("Device:", device)
    print("Model:", model.__class__.__name__)
    print("Trainable parameters:", count_trainable_parameters(model))
    print("Poisoned train samples:", poisoned_train.num_poisoned, "/", len(poisoned_train))
    print("Target label:", args.target_label, poisoned_train.fine_label_names[args.target_label])
    print("Clean test samples:", len(clean_test))
    print("Triggered ASR test samples:", len(triggered_test))

    history: list[dict[str, float | int]] = []
    start_time = perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_classifier_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            max_batches=args.max_train_batches,
        )
        clean_metrics = evaluate_classifier(
            model,
            clean_test_loader,
            criterion,
            device,
            max_batches=args.max_eval_batches,
        )
        asr_metrics = evaluate_classifier(
            model,
            triggered_test_loader,
            criterion,
            device,
            max_batches=args.max_eval_batches,
        )

        record = {
            "epoch": epoch,
            "train_loss": train_metrics.loss,
            "train_accuracy": train_metrics.accuracy,
            "clean_test_loss": clean_metrics.loss,
            "clean_accuracy": clean_metrics.accuracy,
            "asr_loss": asr_metrics.loss,
            "attack_success_rate": asr_metrics.accuracy,
        }
        history.append(record)

        print(
            "Epoch "
            f"{epoch:03d} | "
            f"train acc {train_metrics.accuracy:.4f} | "
            f"clean acc {clean_metrics.accuracy:.4f} | "
            f"ASR {asr_metrics.accuracy:.4f}"
        )

    elapsed_seconds = perf_counter() - start_time
    summary = {
        "script": "scripts/train_suspicious_classifier.py",
        "device": str(device),
        "model": model.__class__.__name__,
        "trainable_parameters": count_trainable_parameters(model),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "poison_ratio": args.poison_ratio,
        "num_poisoned": poisoned_train.num_poisoned,
        "target_label": args.target_label,
        "target_label_name": poisoned_train.fine_label_names[args.target_label],
        "trigger": {
            "type": "sinusoidal_frequency",
            "strength": args.strength,
            "horizontal_frequency": args.horizontal_frequency,
            "vertical_frequency": args.vertical_frequency,
        },
        "max_train_batches": args.max_train_batches,
        "max_eval_batches": args.max_eval_batches,
        "elapsed_seconds": elapsed_seconds,
        "history": history,
    }

    summary_path = args.output_dir / "training_summary.json"
    checkpoint_path = args.output_dir / "suspicious_classifier.pt"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "summary": summary,
        },
        checkpoint_path,
    )

    print("Saved summary:", summary_path)
    print("Saved checkpoint:", checkpoint_path)


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
