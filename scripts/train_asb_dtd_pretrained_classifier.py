"""Train and evaluate an ImageNet-pretrained suspicious classifier on ASB-DTD."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import (
    ASBManifestDataset,
    build_poisoned_training_rows,
    filter_manifest_rows,
    read_jsonl,
)
from models import build_pretrained_classifier, count_trainable_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("Absolute_Dataset/asb_dtd_v1"),
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=Path("Absolute_Dataset/dtd/images"),
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("Advanced_Experiments/dtd_pretrained_resnet18_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Advanced_Outputs/dtd_pretrained_resnet18_v1"),
    )
    parser.add_argument("--backbone", choices=["resnet18"], default="resnet18")
    parser.add_argument(
        "--weights",
        choices=["default", "none"],
        default="default",
        help="Use 'default' for ImageNet-1K pretrained weights.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--freeze-epochs", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--attack-triggers",
        type=str,
        default="fourier_middle,haar_lh,haar_hl",
        help=(
            "Comma-separated trigger names from trigger_catalog.json, or 'none' "
            "for a clean control model."
        ),
    )
    parser.add_argument(
        "--poison-ratio",
        type=float,
        default=0.10,
        help="Total fraction of training sources replaced by poisoned variants.",
    )
    parser.add_argument(
        "--trigger-strength",
        type=float,
        default=None,
        help="Override the catalog strength for every configured attack trigger.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-panels", type=int, default=5)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow this run to replace files in its selected result directories.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    prepare_result_directories(args)

    manifest_path = args.manifest_dir / "variant_manifest.jsonl"
    summary_path = args.manifest_dir / "benchmark_summary.json"
    rows = read_jsonl(manifest_path)
    benchmark = json.loads(summary_path.read_text(encoding="utf-8"))
    trigger_catalog = json.loads(
        (args.manifest_dir / "trigger_catalog.json").read_text(encoding="utf-8")
    )
    target_label = int(benchmark["target_label"])
    target_class = str(benchmark["target_class"])
    attack_trigger_names, attack_trigger_configs = resolve_attack_triggers(
        args.attack_triggers,
        trigger_catalog,
    )
    if args.trigger_strength is not None:
        attack_trigger_configs = {
            name: {**config, "strength": args.trigger_strength}
            for name, config in attack_trigger_configs.items()
        }

    class_names = class_names_from_rows(rows)
    num_classes = len(class_names)
    train_rows, train_counts = build_poisoned_training_rows(
        rows,
        trigger_configs=attack_trigger_configs,
        target_label=target_label,
        poison_ratio=args.poison_ratio,
        seed=args.seed,
    )
    validation_clean_rows = filter_manifest_rows(
        rows, protocol_role="clean_calibration", variant_name="clean"
    )
    test_clean_rows = filter_manifest_rows(
        rows, protocol_role="final_test_clean", variant_name="clean"
    )
    validation_trigger_names = attack_trigger_names
    test_trigger_names = sorted(
        {
            str(row["variant_name"])
            for row in rows
            if row["protocol_role"] == "final_test_triggered"
        }
    )

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                args.image_size,
                scale=(0.70, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(
                round(args.image_size * 256 / 224),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
        ]
    )

    train_dataset = ASBManifestDataset(
        args.images_root,
        train_rows,
        transform=train_transform,
        label_field="training_label",
    )
    validation_clean_dataset = ASBManifestDataset(
        args.images_root,
        validation_clean_rows,
        transform=eval_transform,
        label_field="original_label",
    )
    train_loader = make_loader(
        train_dataset, args, device=device, shuffle=True
    )
    validation_clean_loader = make_loader(
        validation_clean_dataset, args, device=device, shuffle=False
    )
    validation_trigger_loaders = {
        name: make_loader(
            ASBManifestDataset(
                args.images_root,
                make_trigger_evaluation_rows(
                    validation_clean_rows,
                    trigger_name=name,
                    trigger_config=attack_trigger_configs[name],
                    target_label=target_label,
                ),
                transform=eval_transform,
                label_field="original_label",
            ),
            args,
            device=device,
            shuffle=False,
        )
        for name in validation_trigger_names
    }

    model = build_pretrained_classifier(
        backbone=args.backbone,
        num_classes=num_classes,
        weights=args.weights,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(model.feature_parameters()),
                "lr": args.backbone_learning_rate,
            },
            {
                "params": list(model.head_parameters()),
                "lr": args.head_learning_rate,
            },
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1)
    )
    criterion = nn.CrossEntropyLoss()

    run_config = serializable_args(args)
    run_config.update(
        {
            "device_resolved": str(device),
            "dataset": "DTD",
            "num_classes": num_classes,
            "class_names": class_names,
            "target_label": target_label,
            "target_class": target_class,
            "attack_triggers": attack_trigger_names,
            "attack_trigger_configs": attack_trigger_configs,
            "training_composition": train_counts,
            "pretrained_model": model.metadata(),
            "total_parameters": sum(p.numel() for p in model.parameters()),
        }
    )
    write_json(args.experiment_dir / "run_config.json", run_config)

    print("Device:", device)
    print("Model:", args.backbone, "weights:", args.weights)
    print("Classes:", num_classes)
    print("Target:", target_label, target_class)
    print("Attack triggers:", ", ".join(attack_trigger_names) or "none (clean control)")
    print("Training composition:", train_counts)

    history = []
    best_score = -1.0
    best_epoch = 0
    best_path = args.experiment_dir / "suspicious_classifier_best.pt"
    last_path = args.experiment_dir / "suspicious_classifier_last.pt"

    for epoch in range(1, args.epochs + 1):
        feature_trainable = epoch > args.freeze_epochs
        model.set_feature_extractor_trainable(feature_trainable)
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            max_batches=args.max_train_batches,
        )
        validation_clean = evaluate_accuracy(
            model,
            validation_clean_loader,
            device,
            max_batches=args.max_eval_batches,
        )
        validation_asr = {
            name: evaluate_target_rate(
                model,
                loader,
                target_label,
                device,
                max_batches=args.max_eval_batches,
            )
            for name, loader in validation_trigger_loaders.items()
        }
        mean_seen_asr = (
            sum(metric["target_rate"] for metric in validation_asr.values())
            / len(validation_asr)
            if validation_asr
            else None
        )
        selection_score = (
            harmonic_mean(validation_clean["accuracy"], mean_seen_asr)
            if mean_seen_asr is not None
            else validation_clean["accuracy"]
        )
        epoch_record = {
            "epoch": epoch,
            "feature_extractor_trainable": feature_trainable,
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
            "train": train_metrics,
            "validation_clean": validation_clean,
            "validation_asr_by_trigger": validation_asr,
            "validation_mean_seen_asr": mean_seen_asr,
            "selection_score": selection_score,
        }
        history.append(epoch_record)
        write_json(args.experiment_dir / "training_history.json", history)

        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            save_checkpoint(
                best_path,
                model,
                run_config,
                epoch=epoch,
                selection_score=selection_score,
            )

        asr_text = "n/a" if mean_seen_asr is None else f"{mean_seen_asr:.4f}"
        print(
            f"Epoch {epoch:03d} | loss {train_metrics['loss']:.4f} | "
            f"train {train_metrics['accuracy']:.4f} | "
            f"val clean {validation_clean['accuracy']:.4f} | "
            f"attack ASR {asr_text} | "
            f"features {'open' if feature_trainable else 'frozen'}"
        )
        scheduler.step()

    save_checkpoint(
        last_path,
        model,
        run_config,
        epoch=args.epochs,
        selection_score=history[-1]["selection_score"],
    )
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    final_summary = evaluate_final_test(
        model=model,
        rows=rows,
        images_root=args.images_root,
        transform=eval_transform,
        args=args,
        device=device,
        target_label=target_label,
        target_class=target_class,
        test_clean_rows=test_clean_rows,
        test_trigger_names=test_trigger_names,
        attack_trigger_names=attack_trigger_names,
        attack_trigger_configs=attack_trigger_configs,
        best_epoch=best_epoch,
        best_score=best_score,
    )
    write_json(args.experiment_dir / "final_evaluation.json", final_summary)
    write_json(args.output_dir / "final_evaluation.json", final_summary)
    save_training_curves(history, args.output_dir)
    save_prediction_panel(
        model,
        rows,
        args.images_root,
        eval_transform,
        class_names,
        target_label,
        args.num_panels,
        device,
        args.output_dir,
    )

    print("Best epoch:", best_epoch)
    print("Final clean accuracy:", f"{final_summary['clean_test']['accuracy']:.4f}")
    for name, metrics in final_summary["trigger_results"].items():
        print(
            f"{name}: ASR {metrics['asr']:.4f} | "
            f"triggered accuracy {metrics['triggered_clean_label_accuracy']:.4f} | "
            f"{'seen' if metrics['seen_during_attack_training'] else 'held-out'}"
        )
    print("Saved checkpoint:", best_path)
    print("Saved figures and summary:", args.output_dir)


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if not 0 <= args.freeze_epochs <= args.epochs:
        raise ValueError("--freeze-epochs must be between zero and --epochs")
    if args.image_size <= 0 or args.batch_size <= 0:
        raise ValueError("--image-size and --batch-size must be positive")
    if not 0.0 <= args.poison_ratio <= 1.0:
        raise ValueError("--poison-ratio must be between zero and one")
    if args.trigger_strength is not None and args.trigger_strength < 0.0:
        raise ValueError("--trigger-strength must be non-negative")
    if args.attack_triggers.strip().lower() == "none" and args.poison_ratio != 0.0:
        raise ValueError("Use --poison-ratio 0 with --attack-triggers none")


def prepare_result_directories(args: argparse.Namespace) -> None:
    protected = [
        args.experiment_dir / "suspicious_classifier_best.pt",
        args.experiment_dir / "final_evaluation.json",
        args.output_dir / "final_evaluation.json",
    ]
    existing = [path for path in protected if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "A completed run already exists. Select new result directories or pass "
            "--overwrite explicitly:\n" + "\n".join(str(path) for path in existing)
        )
    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)


def class_names_from_rows(rows: list[dict[str, object]]) -> list[str]:
    labels = {
        int(row["original_label"]): str(row["class_name"])
        for row in rows
        if row["variant_name"] == "clean"
    }
    expected = list(range(len(labels)))
    if sorted(labels) != expected:
        raise ValueError("Class labels must be contiguous and zero based")
    return [labels[index] for index in expected]


def resolve_attack_triggers(
    value: str,
    catalog: dict[str, object],
) -> tuple[list[str], dict[str, dict[str, object]]]:
    requested = [name.strip() for name in value.split(",") if name.strip()]
    if requested == ["none"]:
        return [], {}
    if not requested or "none" in requested:
        raise ValueError("Specify trigger names or exactly 'none'")
    if len(set(requested)) != len(requested):
        raise ValueError("--attack-triggers contains duplicate names")

    implemented = {}
    for section_name in ("development_triggers", "held_out_triggers"):
        section = catalog.get(section_name, {})
        for name, entry in section.items():
            if entry.get("status") == "implemented":
                implemented[name] = entry["config"]
    unknown = [name for name in requested if name not in implemented]
    if unknown:
        raise ValueError(
            "Unknown or unimplemented attack triggers: "
            + ", ".join(unknown)
            + ". Available: "
            + ", ".join(sorted(implemented))
        )
    return requested, {name: implemented[name] for name in requested}


def make_trigger_evaluation_rows(
    clean_rows: list[dict[str, object]],
    *,
    trigger_name: str,
    trigger_config: dict[str, object],
    target_label: int,
) -> list[dict[str, object]]:
    triggered_rows = []
    for clean in clean_rows:
        if int(clean["original_label"]) == target_label:
            continue
        triggered = dict(clean)
        triggered.update(
            {
                "variant_name": trigger_name,
                "variant_type": "trigger",
                "trigger_config": trigger_config,
            }
        )
        triggered_rows.append(triggered)
    return triggered_rows


def make_loader(dataset, args, *, device: torch.device, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )


def train_one_epoch(model, loader, criterion, optimizer, device, *, max_batches):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        count = labels.numel()
        total_loss += float(loss.item()) * count
        total_correct += int((logits.argmax(1) == labels).sum().item())
        total_samples += count
    if total_samples == 0:
        raise ValueError("Training loader produced no samples")
    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "num_samples": total_samples,
    }


@torch.inference_mode()
def evaluate_accuracy(model, loader, device, *, max_batches):
    model.eval()
    total_correct = 0
    total_samples = 0
    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        predictions = model(images).argmax(1)
        total_correct += int((predictions == labels).sum().item())
        total_samples += labels.numel()
    if total_samples == 0:
        raise ValueError("Evaluation loader produced no samples")
    return {"accuracy": total_correct / total_samples, "num_samples": total_samples}


@torch.inference_mode()
def evaluate_target_rate(model, loader, target_label, device, *, max_batches):
    model.eval()
    target_predictions = 0
    clean_label_correct = 0
    total_samples = 0
    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        predictions = model(images).argmax(1)
        target_predictions += int((predictions == target_label).sum().item())
        clean_label_correct += int((predictions == labels).sum().item())
        total_samples += labels.numel()
    if total_samples == 0:
        raise ValueError("ASR loader produced no samples")
    return {
        "target_rate": target_predictions / total_samples,
        "clean_label_accuracy": clean_label_correct / total_samples,
        "num_samples": total_samples,
    }


def evaluate_final_test(
    *,
    model,
    rows,
    images_root,
    transform,
    args,
    device,
    target_label,
    target_class,
    test_clean_rows,
    test_trigger_names,
    attack_trigger_names,
    attack_trigger_configs,
    best_epoch,
    best_score,
):
    clean_loader = make_loader(
        ASBManifestDataset(
            images_root,
            test_clean_rows,
            transform=transform,
            label_field="original_label",
        ),
        args,
        device=device,
        shuffle=False,
    )
    clean_metrics = evaluate_accuracy(
        model, clean_loader, device, max_batches=args.max_eval_batches
    )
    clean_non_target_rows = [
        row for row in test_clean_rows if int(row["original_label"]) != target_label
    ]
    clean_target_loader = make_loader(
        ASBManifestDataset(
            images_root,
            clean_non_target_rows,
            transform=transform,
            label_field="original_label",
        ),
        args,
        device=device,
        shuffle=False,
    )
    clean_target_rate = evaluate_target_rate(
        model,
        clean_target_loader,
        target_label,
        device,
        max_batches=args.max_eval_batches,
    )

    trigger_results = {}
    for name in test_trigger_names:
        trigger_rows = filter_manifest_rows(
            rows,
            protocol_role="final_test_triggered",
            variant_name=name,
            exclude_original_label=target_label,
        )
        loader = make_loader(
            ASBManifestDataset(
                images_root,
                trigger_rows,
                transform=transform,
                label_field="original_label",
            ),
            args,
            device=device,
            shuffle=False,
        )
        metrics = evaluate_target_rate(
            model,
            loader,
            target_label,
            device,
            max_batches=args.max_eval_batches,
        )
        seen = name in attack_trigger_names
        trigger_results[name] = {
            "asr": metrics["target_rate"],
            "triggered_clean_label_accuracy": metrics["clean_label_accuracy"],
            "num_non_target_samples": metrics["num_samples"],
            "seen_during_attack_training": seen,
        }

    seen_values = [
        result["asr"]
        for result in trigger_results.values()
        if result["seen_during_attack_training"]
    ]
    held_out_values = [
        result["asr"]
        for result in trigger_results.values()
        if not result["seen_during_attack_training"]
    ]
    return {
        "experiment": "advanced_dtd_pretrained_suspicious_classifier_v1",
        "model": model.metadata(),
        "checkpoint_selection": {
            "best_epoch": best_epoch,
            "harmonic_clean_asr_score": best_score,
        },
        "target_label": target_label,
        "target_class": target_class,
        "clean_test": clean_metrics,
        "clean_non_target_target_prediction_rate": clean_target_rate["target_rate"],
        "attack_triggers": attack_trigger_names,
        "attack_trigger_configs": attack_trigger_configs,
        "mean_seen_trigger_asr": (
            sum(seen_values) / len(seen_values) if seen_values else None
        ),
        "mean_held_out_trigger_asr": (
            sum(held_out_values) / len(held_out_values) if held_out_values else None
        ),
        "trigger_results": trigger_results,
        "interpretation_note": (
            "This is a clean control when attack_triggers is empty; otherwise it is "
            "an intentionally backdoored pretrained classifier, not a defended model. "
            "High ASR for its configured attack triggers verifies poisoning success."
        ),
    }


def save_checkpoint(path, model, run_config, *, epoch, selection_score):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_metadata": model.metadata(),
            "run_config": run_config,
            "epoch": epoch,
            "selection_score": selection_score,
        },
        path,
    )


def save_training_curves(history, output_dir):
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(epochs, [row["train"]["loss"] for row in history], label="train loss")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy", title="Training loss")
    axes[0].grid(alpha=0.25)
    axes[1].plot(
        epochs,
        [row["train"]["accuracy"] for row in history],
        label="train accuracy",
    )
    axes[1].plot(
        epochs,
        [row["validation_clean"]["accuracy"] for row in history],
        label="validation clean accuracy",
    )
    if any(row["validation_mean_seen_asr"] is not None for row in history):
        axes[1].plot(
            epochs,
            [row["validation_mean_seen_asr"] for row in history],
            label="validation attack ASR",
        )
    axes[1].set(xlabel="Epoch", ylabel="Rate", ylim=(0, 1), title="Clean utility and attack learning")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=220)
    plt.close(fig)


@torch.inference_mode()
def save_prediction_panel(
    model,
    rows,
    images_root,
    transform,
    class_names,
    target_label,
    num_panels,
    device,
    output_dir,
):
    import matplotlib.pyplot as plt

    preferred = ["clean", "fourier_middle", "fourier_low", "haar_hh"]
    clean_rows = filter_manifest_rows(
        rows,
        protocol_role="final_test_clean",
        variant_name="clean",
        exclude_original_label=target_label,
    )
    triggered = {
        (str(row["source_id"]), str(row["variant_name"])): row
        for row in rows
        if row["protocol_role"] == "final_test_triggered"
    }
    selected = [
        row
        for row in clean_rows
        if all((str(row["source_id"]), name) in triggered for name in preferred[1:])
    ][:num_panels]
    if not selected:
        return

    fig, axes = plt.subplots(
        len(selected), len(preferred), figsize=(3.2 * len(preferred), 3.0 * len(selected))
    )
    axes = np.asarray(axes).reshape(len(selected), len(preferred))
    model.eval()
    for row_index, clean_row in enumerate(selected):
        source_id = str(clean_row["source_id"])
        for column, name in enumerate(preferred):
            row = clean_row if name == "clean" else triggered[(source_id, name)]
            dataset = ASBManifestDataset(
                images_root,
                [row],
                transform=transform,
                label_field="original_label",
            )
            image, label = dataset[0]
            prediction = int(model(image.unsqueeze(0).to(device)).argmax(1).item())
            axis = axes[row_index, column]
            axis.imshow(image.permute(1, 2, 0).clamp(0, 1).numpy())
            axis.set_title(
                f"{name}\ntrue: {class_names[int(label)]}\npred: {class_names[prediction]}",
                fontsize=8,
            )
            axis.axis("off")
    fig.suptitle(
        f"Pretrained suspicious classifier (target: {class_names[target_label]})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "classifier_prediction_panel.png", dpi=220)
    plt.close(fig)


def harmonic_mean(first: float, second: float) -> float:
    return 0.0 if first + second == 0.0 else 2.0 * first * second / (first + second)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def serializable_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
