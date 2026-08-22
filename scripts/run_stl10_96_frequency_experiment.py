"""Run a separate 96x96 STL-10 frequency-backdoor experiment.

This Kaggle-oriented script keeps all outputs separate from the CIFAR-100
experiment. It trains:

1. a suspicious classifier on poisoned STL-10;
2. a spectral correction generator against the suspicious classifier;
3. a repaired classifier using clean, corrected, and raw-triggered images;
4. a final evaluation report and sample visual panels.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from evaluation import evaluate_classifier
from fft import amplitude_spectrum, normalize_minmax, shift_frequency_map
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
from training import train_classifier_epoch
from visualization.correction_panel import save_correction_panel


STL10_LABEL_NAMES = [
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/kaggle/input/stl10"),
        help="Path to STL-10 root. Can be the parent of stl10_binary or the stl10_binary folder itself.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Use torchvision download=True. Useful if Kaggle internet is enabled.",
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("experiments/stl10_96x96"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/stl10_96x96"),
    )
    parser.add_argument("--classifier-epochs", type=int, default=30)
    parser.add_argument("--generator-epochs", type=int, default=30)
    parser.add_argument("--repair-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--classifier-learning-rate", type=float, default=1e-3)
    parser.add_argument("--generator-learning-rate", type=float, default=1e-3)
    parser.add_argument("--repair-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--poison-ratio", type=float, default=0.12)
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--strength", type=float, default=0.08)
    parser.add_argument(
        "--trigger-kind",
        choices=[
            "cosine",
            "sine",
            "checkerboard",
            "dual_frequency",
            "localized_cosine",
            "fiba_amplitude",
        ],
        default="cosine",
    )
    parser.add_argument("--horizontal-frequency", type=int, default=18)
    parser.add_argument("--vertical-frequency", type=int, default=18)
    parser.add_argument("--secondary-horizontal-frequency", type=int, default=30)
    parser.add_argument("--secondary-vertical-frequency", type=int, default=6)
    parser.add_argument("--window-center-x", type=float, default=0.65)
    parser.add_argument("--window-center-y", type=float, default=0.50)
    parser.add_argument("--window-sigma", type=float, default=0.18)
    parser.add_argument("--fiba-mask-radius", type=float, default=0.10)
    parser.add_argument("--classification-weight", type=float, default=1.0)
    parser.add_argument("--reconstruction-weight", type=float, default=4.0)
    parser.add_argument("--sparsity-weight", type=float, default=0.02)
    parser.add_argument("--smoothness-weight", type=float, default=0.01)
    parser.add_argument("--num-panels", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.experiment_root.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)

    trigger_config = FrequencyTriggerConfig(
        horizontal_frequency=args.horizontal_frequency,
        vertical_frequency=args.vertical_frequency,
        strength=args.strength,
        trigger_kind=args.trigger_kind,
        secondary_horizontal_frequency=args.secondary_horizontal_frequency,
        secondary_vertical_frequency=args.secondary_vertical_frequency,
        window_center_x=args.window_center_x,
        window_center_y=args.window_center_y,
        window_sigma=args.window_sigma,
        fiba_mask_radius=args.fiba_mask_radius,
    )
    loss_weights = GeneratorLossWeights(
        classification=args.classification_weight,
        reconstruction=args.reconstruction_weight,
        sparsity=args.sparsity_weight,
        smoothness=args.smoothness_weight,
    )

    train_clean, test_clean = load_stl10(args.data_root, download=args.download)
    if args.trigger_kind == "fiba_amplitude":
        reference_image, _ = train_clean[0]
        trigger_config = replace(trigger_config, reference_image=reference_image)
    label_names = list(getattr(train_clean, "classes", STL10_LABEL_NAMES))
    num_classes = len(label_names)
    target_name = label_names[args.target_label]
    poisoned_train = PoisonedVisionDataset(
        train_clean,
        poison_ratio=args.poison_ratio,
        target_label=args.target_label,
        trigger_config=trigger_config,
        seed=args.seed,
    )
    triggered_test = TriggeredVisionDataset(
        test_clean,
        target_label=args.target_label,
        trigger_config=trigger_config,
    )

    print("Device:", device)
    print("Dataset: STL-10 96x96")
    print("Train samples:", len(train_clean))
    print("Test samples:", len(test_clean))
    print("Triggered ASR samples:", len(triggered_test))
    print("Target:", args.target_label, target_name)
    print("Poisoned train samples:", poisoned_train.num_poisoned, "/", len(poisoned_train))
    print(
        "Trigger:", args.trigger_kind,
        "alpha", args.strength,
        "fx", args.horizontal_frequency,
        "fy", args.vertical_frequency,
    )

    suspicious = SmallCIFARClassifier(num_classes=num_classes).to(device)
    classifier_summary = train_suspicious_classifier(
        model=suspicious,
        poisoned_train=poisoned_train,
        clean_test=test_clean,
        triggered_test=triggered_test,
        args=args,
        device=device,
        label_names=label_names,
    )

    generator = SpectralCorrectionGenerator().to(device)
    generator_summary = train_generator(
        generator=generator,
        classifier=suspicious,
        train_clean=train_clean,
        test_clean=test_clean,
        trigger_config=trigger_config,
        args=args,
        loss_weights=loss_weights,
        device=device,
    )

    repaired = SmallCIFARClassifier(num_classes=num_classes).to(device)
    repaired.load_state_dict(suspicious.state_dict())
    repair_summary = repair_classifier(
        repaired=repaired,
        generator=generator,
        train_clean=train_clean,
        clean_test=test_clean,
        triggered_test=triggered_test,
        trigger_config=trigger_config,
        args=args,
        device=device,
    )

    final_summary = final_evaluation(
        suspicious=suspicious,
        repaired=repaired,
        generator=generator,
        clean_test=test_clean,
        triggered_test=triggered_test,
        trigger_config=trigger_config,
        args=args,
        label_names=label_names,
        classifier_summary=classifier_summary,
        generator_summary=generator_summary,
        repair_summary=repair_summary,
        device=device,
    )

    final_json = args.output_root / "final_stl10_96x96_summary.json"
    final_md = args.output_root / "final_stl10_96x96_report.md"
    final_json.write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    final_md.write_text(render_markdown(final_summary), encoding="utf-8")

    print("Saved final JSON:", final_json)
    print("Saved final report:", final_md)


def load_stl10(data_root: Path, *, download: bool) -> tuple[Dataset, Dataset]:
    root = find_stl10_root(data_root)
    transform = transforms.ToTensor()
    try:
        train = datasets.STL10(root=str(root), split="train", transform=transform, download=download)
        test = datasets.STL10(root=str(root), split="test", transform=transform, download=download)
    except RuntimeError as error:
        raise RuntimeError(
            "Could not load STL-10. On Kaggle, set --data-root to the folder that "
            "contains stl10_binary, or pass --download if internet is enabled."
        ) from error
    return train, test


def find_stl10_root(path: Path) -> Path:
    path = path.expanduser()
    if (path / "stl10_binary").exists():
        return path
    if path.name == "stl10_binary" and path.exists():
        return path.parent
    for candidate in path.rglob("stl10_binary") if path.exists() else []:
        return candidate.parent
    return path


class PoisonedVisionDataset(Dataset):
    def __init__(
        self,
        clean_dataset: Dataset,
        *,
        poison_ratio: float,
        target_label: int,
        trigger_config: FrequencyTriggerConfig,
        seed: int,
    ) -> None:
        self.clean_dataset = clean_dataset
        self.target_label = target_label
        self.trigger_config = trigger_config
        labels = dataset_labels(clean_dataset)
        eligible = torch.nonzero(labels != target_label, as_tuple=False).flatten()
        poison_count = min(int(round(len(clean_dataset) * poison_ratio)), int(eligible.numel()))
        generator = torch.Generator().manual_seed(seed)
        selected = eligible[torch.randperm(int(eligible.numel()), generator=generator)[:poison_count]]
        self.poisoned_indices = torch.sort(selected).values
        self.poisoned_index_set = set(self.poisoned_indices.tolist())

    @property
    def num_poisoned(self) -> int:
        return int(self.poisoned_indices.numel())

    def __len__(self) -> int:
        return len(self.clean_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = self.clean_dataset[index]
        label = int(label)
        if index not in self.poisoned_index_set:
            return image, torch.tensor(label, dtype=torch.long)
        return (
            apply_frequency_trigger(image, self.trigger_config),
            torch.tensor(self.target_label, dtype=torch.long),
        )


class TriggeredVisionDataset(Dataset):
    def __init__(
        self,
        clean_dataset: Dataset,
        *,
        target_label: int,
        trigger_config: FrequencyTriggerConfig,
    ) -> None:
        self.clean_dataset = clean_dataset
        self.target_label = target_label
        self.trigger_config = trigger_config
        labels = dataset_labels(clean_dataset)
        self.eligible_indices = torch.nonzero(labels != target_label, as_tuple=False).flatten()

    def __len__(self) -> int:
        return int(self.eligible_indices.numel())

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        original_index = int(self.eligible_indices[index])
        image, _ = self.clean_dataset[original_index]
        return (
            apply_frequency_trigger(image, self.trigger_config),
            torch.tensor(self.target_label, dtype=torch.long),
        )


def dataset_labels(dataset: Dataset) -> torch.Tensor:
    if hasattr(dataset, "labels"):
        return torch.as_tensor(getattr(dataset, "labels"), dtype=torch.long)
    if hasattr(dataset, "targets"):
        return torch.as_tensor(getattr(dataset, "targets"), dtype=torch.long)
    labels = [int(dataset[index][1]) for index in range(len(dataset))]
    return torch.as_tensor(labels, dtype=torch.long)


def train_suspicious_classifier(
    *,
    model: nn.Module,
    poisoned_train: Dataset,
    clean_test: Dataset,
    triggered_test: Dataset,
    args: argparse.Namespace,
    device: torch.device,
    label_names: list[str],
) -> dict:
    output_dir = args.experiment_root / "suspicious_classifier"
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader = make_loader(poisoned_train, args.batch_size, args.num_workers, shuffle=True, device=device)
    clean_loader = make_loader(clean_test, args.batch_size, args.num_workers, shuffle=False, device=device)
    asr_loader = make_loader(triggered_test, args.batch_size, args.num_workers, shuffle=False, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.classifier_learning_rate,
        weight_decay=args.weight_decay,
    )
    history = []
    start = perf_counter()
    for epoch in range(1, args.classifier_epochs + 1):
        train_metrics = train_classifier_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            max_batches=args.max_train_batches,
        )
        clean_metrics = evaluate_classifier(model, clean_loader, criterion, device, max_batches=args.max_eval_batches)
        asr_metrics = evaluate_classifier(model, asr_loader, criterion, device, max_batches=args.max_eval_batches)
        record = {
            "epoch": epoch,
            "train_accuracy": train_metrics.accuracy,
            "clean_accuracy": clean_metrics.accuracy,
            "attack_success_rate": asr_metrics.accuracy,
        }
        history.append(record)
        print(
            f"[classifier] epoch {epoch:03d} | train {train_metrics.accuracy:.4f} | "
            f"clean {clean_metrics.accuracy:.4f} | ASR {asr_metrics.accuracy:.4f}"
        )

    summary = {
        "dataset": "STL-10 96x96",
        "model": model.__class__.__name__,
        "parameters": count_trainable_parameters(model),
        "epochs": args.classifier_epochs,
        "target_label": args.target_label,
        "target_label_name": label_names[args.target_label],
        "poison_ratio": args.poison_ratio,
        "elapsed_seconds": perf_counter() - start,
        "history": history,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    torch.save({"model_state_dict": model.state_dict(), "summary": summary}, output_dir / "suspicious_classifier.pt")
    return summary


def train_generator(
    *,
    generator: SpectralCorrectionGenerator,
    classifier: nn.Module,
    train_clean: Dataset,
    test_clean: Dataset,
    trigger_config: FrequencyTriggerConfig,
    args: argparse.Namespace,
    loss_weights: GeneratorLossWeights,
    device: torch.device,
) -> dict:
    output_dir = args.experiment_root / "spectral_generator"
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader = make_loader(train_clean, args.batch_size, args.num_workers, shuffle=True, device=device)
    test_loader = make_loader(test_clean, args.batch_size, args.num_workers, shuffle=False, device=device)
    classifier.eval()
    for parameter in classifier.parameters():
        parameter.requires_grad = False
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        generator.parameters(),
        lr=args.generator_learning_rate,
        weight_decay=1e-5,
    )
    history = []
    start = perf_counter()
    for epoch in range(1, args.generator_epochs + 1):
        train_record = train_generator_epoch(
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
            f"[generator] epoch {epoch:03d} | loss {train_record['train_total_loss']:.4f} | "
            f"before ASR {eval_record['before_asr']:.4f} | after ASR {eval_record['after_asr']:.4f} | "
            f"corr acc {eval_record['corrected_clean_label_accuracy']:.4f}"
        )

    summary = {
        "dataset": "STL-10 96x96",
        "epochs": args.generator_epochs,
        "loss_weights": loss_weights.__dict__,
        "elapsed_seconds": perf_counter() - start,
        "history": history,
    }
    (output_dir / "generator_training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    torch.save({"generator_state_dict": generator.state_dict(), "summary": summary}, output_dir / "spectral_generator.pt")
    return summary


def train_generator_epoch(
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
    totals = {"total": 0.0, "classification": 0.0, "reconstruction": 0.0, "sparsity": 0.0, "smoothness": 0.0}
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
        corrected_images, correction_map = correct_batch(clean_images, triggered_images, generator)
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
        raise ValueError("Generator training saw no non-target samples")
    return {f"train_{key}_loss": value / total_samples for key, value in totals.items()}


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
    total = 0
    before_target = 0
    after_target = 0
    after_clean = 0
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
        corrected_images, correction_map = correct_batch(clean_images, triggered_images, generator)
        before_preds = classifier(triggered_images).argmax(dim=1)
        after_preds = classifier(corrected_images).argmax(dim=1)
        batch_size = int(clean_labels.numel())
        total += batch_size
        before_target += int((before_preds == target_label).sum().item())
        after_target += int((after_preds == target_label).sum().item())
        after_clean += int((after_preds == clean_labels).sum().item())
        reconstruction_l1 += float((corrected_images - clean_images).abs().mean().item()) * batch_size
        correction_mean += float(correction_map.mean().item()) * batch_size
    if total == 0:
        raise ValueError("Generator evaluation saw no non-target samples")
    return {
        "before_asr": before_target / total,
        "after_asr": after_target / total,
        "corrected_clean_label_accuracy": after_clean / total,
        "eval_reconstruction_l1": reconstruction_l1 / total,
        "mean_correction_value": correction_mean / total,
        "eval_samples": total,
    }


def repair_classifier(
    *,
    repaired: nn.Module,
    generator: SpectralCorrectionGenerator,
    train_clean: Dataset,
    clean_test: Dataset,
    triggered_test: Dataset,
    trigger_config: FrequencyTriggerConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    output_dir = args.experiment_root / "repaired_classifier"
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader = make_loader(train_clean, args.batch_size, args.num_workers, shuffle=True, device=device)
    clean_loader = make_loader(clean_test, args.batch_size, args.num_workers, shuffle=False, device=device)
    asr_loader = make_loader(triggered_test, args.batch_size, args.num_workers, shuffle=False, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        repaired.parameters(),
        lr=args.repair_learning_rate,
        weight_decay=args.weight_decay,
    )
    history = []
    start = perf_counter()
    for epoch in range(1, args.repair_epochs + 1):
        train_record = repair_epoch(
            repaired=repaired,
            generator=generator,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            trigger_config=trigger_config,
            target_label=args.target_label,
            max_batches=args.max_train_batches,
        )
        clean_metrics = evaluate_classifier(repaired, clean_loader, criterion, device, max_batches=args.max_eval_batches)
        asr_metrics = evaluate_classifier(repaired, asr_loader, criterion, device, max_batches=args.max_eval_batches)
        record = {
            "epoch": epoch,
            **train_record,
            "clean_accuracy": clean_metrics.accuracy,
            "attack_success_rate": asr_metrics.accuracy,
        }
        history.append(record)
        print(
            f"[repair] epoch {epoch:03d} | repair acc {train_record['repair_train_accuracy']:.4f} | "
            f"clean {clean_metrics.accuracy:.4f} | ASR {asr_metrics.accuracy:.4f}"
        )

    summary = {
        "dataset": "STL-10 96x96",
        "epochs": args.repair_epochs,
        "elapsed_seconds": perf_counter() - start,
        "history": history,
    }
    (output_dir / "repair_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    torch.save({"model_state_dict": repaired.state_dict(), "summary": summary}, output_dir / "repaired_classifier.pt")
    return summary


def repair_epoch(
    *,
    repaired: nn.Module,
    generator: SpectralCorrectionGenerator,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    trigger_config: FrequencyTriggerConfig,
    target_label: int,
    max_batches: int | None,
) -> dict[str, float]:
    repaired.train()
    generator.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for batch_index, (clean_images, clean_labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        clean_images = clean_images.to(device)
        clean_labels = clean_labels.to(device)
        non_target_mask = clean_labels != target_label
        clean_repair_images = [clean_images]
        clean_repair_labels = [clean_labels]
        if int(non_target_mask.sum().item()) > 0:
            non_target_images = clean_images[non_target_mask]
            non_target_labels = clean_labels[non_target_mask]
            triggered_images = apply_frequency_trigger(non_target_images, trigger_config)
            with torch.no_grad():
                corrected_images, _ = correct_batch(non_target_images, triggered_images, generator)
            clean_repair_images.extend([corrected_images, triggered_images])
            clean_repair_labels.extend([non_target_labels, non_target_labels])

        repair_images = torch.cat(clean_repair_images, dim=0)
        repair_labels = torch.cat(clean_repair_labels, dim=0)
        optimizer.zero_grad(set_to_none=True)
        logits = repaired(repair_images)
        loss = criterion(logits, repair_labels)
        loss.backward()
        optimizer.step()
        batch_size = int(repair_labels.numel())
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == repair_labels).sum().item())
        total_samples += batch_size
    if total_samples == 0:
        raise ValueError("Repair training saw no samples")
    return {
        "repair_train_loss": total_loss / total_samples,
        "repair_train_accuracy": total_correct / total_samples,
        "repair_train_samples": total_samples,
    }


@torch.no_grad()
def final_evaluation(
    *,
    suspicious: nn.Module,
    repaired: nn.Module,
    generator: SpectralCorrectionGenerator,
    clean_test: Dataset,
    triggered_test: Dataset,
    trigger_config: FrequencyTriggerConfig,
    args: argparse.Namespace,
    label_names: list[str],
    classifier_summary: dict,
    generator_summary: dict,
    repair_summary: dict,
    device: torch.device,
) -> dict:
    clean_loader = make_loader(clean_test, args.batch_size, args.num_workers, shuffle=False, device=device)
    asr_loader = make_loader(triggered_test, args.batch_size, args.num_workers, shuffle=False, device=device)
    criterion = nn.CrossEntropyLoss()
    suspicious_clean = evaluate_classifier(suspicious, clean_loader, criterion, device, max_batches=args.max_eval_batches)
    suspicious_asr = evaluate_classifier(suspicious, asr_loader, criterion, device, max_batches=args.max_eval_batches)
    repaired_clean = evaluate_classifier(repaired, clean_loader, criterion, device, max_batches=args.max_eval_batches)
    repaired_asr = evaluate_classifier(repaired, asr_loader, criterion, device, max_batches=args.max_eval_batches)
    corrected_suspicious = evaluate_generator(
        generator=generator,
        classifier=suspicious,
        loader=clean_loader,
        device=device,
        trigger_config=trigger_config,
        target_label=args.target_label,
        max_batches=args.max_eval_batches,
    )
    corrected_repaired = evaluate_generator(
        generator=generator,
        classifier=repaired,
        loader=clean_loader,
        device=device,
        trigger_config=trigger_config,
        target_label=args.target_label,
        max_batches=args.max_eval_batches,
    )
    panels = save_final_panels(
        clean_test=clean_test,
        suspicious=suspicious,
        repaired=repaired,
        generator=generator,
        trigger_config=trigger_config,
        target_label=args.target_label,
        label_names=label_names,
        output_dir=args.output_root / "sample_panels",
        num_panels=args.num_panels,
        device=device,
    )
    return {
        "dataset": {"name": "STL-10", "image_shape": [3, 96, 96], "classes": label_names},
        "trigger": {
            "type": args.trigger_kind,
            "target_label": args.target_label,
            "target_label_name": label_names[args.target_label],
            "poison_ratio": args.poison_ratio,
            "strength_alpha": args.strength,
            "horizontal_frequency_fx": args.horizontal_frequency,
            "vertical_frequency_fy": args.vertical_frequency,
            "secondary_horizontal_frequency": args.secondary_horizontal_frequency,
            "secondary_vertical_frequency": args.secondary_vertical_frequency,
            "window_center_x": args.window_center_x,
            "window_center_y": args.window_center_y,
            "window_sigma": args.window_sigma,
            "fiba_mask_radius": args.fiba_mask_radius,
        },
        "metrics": {
            "suspicious_classifier": {
                "clean_accuracy": suspicious_clean.accuracy,
                "attack_success_rate": suspicious_asr.accuracy,
            },
            "generator_corrected_with_suspicious_classifier": corrected_suspicious,
            "repaired_classifier": {
                "clean_accuracy": repaired_clean.accuracy,
                "attack_success_rate": repaired_asr.accuracy,
            },
            "generator_corrected_with_repaired_classifier": corrected_repaired,
        },
        "summaries": {
            "classifier_training": classifier_summary,
            "generator_training": generator_summary,
            "repair_training": repair_summary,
        },
        "visualizations": {
            "sample_panel_dir": str(args.output_root / "sample_panels"),
            "sample_panels": panels,
        },
    }


@torch.no_grad()
def save_final_panels(
    *,
    clean_test: Dataset,
    suspicious: nn.Module,
    repaired: nn.Module,
    generator: SpectralCorrectionGenerator,
    trigger_config: FrequencyTriggerConfig,
    target_label: int,
    label_names: list[str],
    output_dir: Path,
    num_panels: int,
    device: torch.device,
) -> list[dict[str, str | int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(len(clean_test)):
        clean_image, clean_label = clean_test[index]
        clean_label = int(clean_label)
        if clean_label == target_label:
            continue
        clean_batch = clean_image.unsqueeze(0).to(device)
        triggered_batch = apply_frequency_trigger(clean_batch, trigger_config)
        corrected_batch, correction_map = correct_batch(clean_batch, triggered_batch, generator)
        clean_amp = amplitude_spectrum(clean_image)
        triggered_amp = amplitude_spectrum(triggered_batch.squeeze(0).cpu())
        corrected_amp = amplitude_spectrum(corrected_batch.squeeze(0).cpu())
        amplitude_difference = (triggered_amp - clean_amp).abs()
        true_name = label_names[clean_label]
        record = {
            "index": index,
            "true_label": true_name,
            "suspicious_clean": predict_name(suspicious, clean_batch, label_names),
            "suspicious_triggered": predict_name(suspicious, triggered_batch, label_names),
            "suspicious_corrected": predict_name(suspicious, corrected_batch, label_names),
            "repaired_clean": predict_name(repaired, clean_batch, label_names),
            "repaired_triggered": predict_name(repaired, triggered_batch, label_names),
            "repaired_corrected": predict_name(repaired, corrected_batch, label_names),
        }
        panel_path = save_correction_panel(
            [
                (f"clean true:{true_name}", clean_image, "rgb"),
                (f"susp clean:{record['suspicious_clean']}", clean_image, "rgb"),
                (f"susp trig:{record['suspicious_triggered']}", triggered_batch.squeeze(0).cpu(), "rgb"),
                (f"susp corr:{record['suspicious_corrected']}", corrected_batch.squeeze(0).cpu(), "rgb"),
                (f"repair clean:{record['repaired_clean']}", clean_image, "rgb"),
                (f"repair trig:{record['repaired_triggered']}", triggered_batch.squeeze(0).cpu(), "rgb"),
                (f"repair corr:{record['repaired_corrected']}", corrected_batch.squeeze(0).cpu(), "rgb"),
                (
                    "correction map",
                    normalize_minmax(shift_frequency_map(correction_map.squeeze(0).cpu())),
                    "gray",
                ),
                ("trigger amp", normalize_minmax(triggered_amp), "gray"),
                ("corrected amp", normalize_minmax(corrected_amp), "gray"),
                ("amplitude diff", normalize_minmax(amplitude_difference), "gray"),
                ("image diff x8", ((corrected_batch.squeeze(0).cpu() - clean_image).abs() * 8.0).clamp(0, 1), "rgb"),
                ("trigger diff x8", ((triggered_batch.squeeze(0).cpu() - clean_image).abs() * 8.0).clamp(0, 1), "rgb"),
            ],
            output_dir / f"stl10_96_panel_test_index_{index}.png",
            columns=4,
        )
        record["panel_path"] = str(panel_path)
        records.append(record)
        if len(records) >= num_panels:
            break
    return records


def correct_batch(
    clean_images: torch.Tensor,
    triggered_images: torch.Tensor,
    generator: SpectralCorrectionGenerator,
) -> tuple[torch.Tensor, torch.Tensor]:
    clean_amplitude, _ = fft_amplitude_phase(clean_images)
    triggered_amplitude, triggered_phase = fft_amplitude_phase(triggered_images)
    generator_input = build_generator_input(clean_amplitude, triggered_amplitude)
    correction_map = generator(generator_input)
    corrected_amplitude = apply_correction_map(clean_amplitude, triggered_amplitude, correction_map)
    corrected_images = reconstruct_from_amplitude_phase(corrected_amplitude, triggered_phase)
    return corrected_images, correction_map


def make_loader(dataset: Dataset, batch_size: int, num_workers: int, *, shuffle: bool, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def predict_name(model: nn.Module, images: torch.Tensor, label_names: list[str]) -> str:
    logits = model(images)
    return label_names[int(logits.argmax(dim=1).item())]


def render_markdown(summary: dict) -> str:
    metrics = summary["metrics"]
    trigger = summary["trigger"]
    return f"""# Final STL-10 96x96 Frequency-Backdoor Experiment

## Setup

- Dataset: STL-10.
- Resolution: `96 x 96`.
- Target class: `{trigger['target_label_name']}` (`{trigger['target_label']}`).
- Poison ratio: `{trigger['poison_ratio']}`.
- Trigger: `{trigger['type']}` frequency trigger.
- Frequency: `fx={trigger['horizontal_frequency_fx']}`, `fy={trigger['vertical_frequency_fy']}`.
- Strength: `alpha={trigger['strength_alpha']}`.

## Result Table

| Stage | Clean Accuracy | Attack Success Rate |
|---|---:|---:|
| Suspicious classifier | {metrics['suspicious_classifier']['clean_accuracy']:.4f} | {metrics['suspicious_classifier']['attack_success_rate']:.4f} |
| Generator-corrected suspicious classifier | {metrics['generator_corrected_with_suspicious_classifier']['corrected_clean_label_accuracy']:.4f} | {metrics['generator_corrected_with_suspicious_classifier']['after_asr']:.4f} |
| Repaired classifier | {metrics['repaired_classifier']['clean_accuracy']:.4f} | {metrics['repaired_classifier']['attack_success_rate']:.4f} |
| Generator-corrected repaired classifier | {metrics['generator_corrected_with_repaired_classifier']['corrected_clean_label_accuracy']:.4f} | {metrics['generator_corrected_with_repaired_classifier']['after_asr']:.4f} |

## Saved Visuals

Sample panels are saved in:

`{summary['visualizations']['sample_panel_dir']}`
"""


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
