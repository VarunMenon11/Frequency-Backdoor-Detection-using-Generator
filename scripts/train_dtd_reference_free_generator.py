"""Train a reference-free spectral correction generator on ASB-DTD.

The frozen suspicious classifier and clean counterpart supervise training. The
generator itself receives only one image, both during training and inference.
Checkpoint selection uses the calibration split; the locked test split is not
read by this script.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import ASBPairedTriggerDataset, filter_manifest_rows, read_jsonl
from evaluation.asb_classifier import evaluation_transform
from generator import ReferenceFreeSpectralGenerator
from generator.reference_free import total_variation
from models import build_pretrained_classifier
from poisoning import AdvancedTriggerConfig
from scripts.train_asb_dtd_pretrained_classifier import class_names_from_rows, resolve_device


@dataclass(frozen=True)
class LossWeights:
    classification: float = 1.0
    image_reconstruction: float = 4.0
    spectral_reconstruction: float = 1.0
    clean_identity: float = 2.0
    sparsity: float = 0.02
    smoothness: float = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=Path("Absolute_Dataset/asb_dtd_v1"))
    parser.add_argument("--images-root", type=Path, default=Path("Absolute_Dataset/dtd/images"))
    parser.add_argument(
        "--classifier-checkpoint", type=Path,
        default=Path("Advanced_Experiments/dtd_attack_sweep_v1/ftrojan_m100_paired_r020/suspicious_classifier_best_attack.pt"),
    )
    parser.add_argument(
        "--experiment-dir", type=Path,
        default=Path("Advanced_Experiments/dtd_reference_free_generator_ftrojan_v1"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("Advanced_Outputs/dtd_reference_free_generator_ftrojan_v1"),
    )
    parser.add_argument("--trigger-name", default="ftrojan_mix")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--max-log-correction", type=float, default=2.0)
    parser.add_argument("--classification-weight", type=float, default=1.0)
    parser.add_argument("--image-weight", type=float, default=4.0)
    parser.add_argument("--spectral-weight", type=float, default=1.0)
    parser.add_argument("--identity-weight", type=float, default=2.0)
    parser.add_argument("--sparsity-weight", type=float, default=0.02)
    parser.add_argument("--smoothness-weight", type=float, default=0.01)
    parser.add_argument(
        "--max-clean-accuracy-drop", type=float, default=0.05,
        help="Validation clean-accuracy drop allowed for defense checkpoint selection.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-panels", type=int, default=4)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    prepare_directories(args)

    rows = read_jsonl(args.manifest_dir / "variant_manifest.jsonl")
    class_names = class_names_from_rows(rows)
    checkpoint = torch.load(args.classifier_checkpoint, map_location="cpu", weights_only=True)
    classifier_config = checkpoint["run_config"]
    validate_checkpoint(classifier_config, class_names, args)
    target_label = int(classifier_config["target_label"])
    trigger_dict = dict(classifier_config["attack_trigger_configs"][args.trigger_name])
    trigger_config = trigger_from_dict(trigger_dict)

    train_rows = filter_manifest_rows(rows, protocol_role="clean_train", variant_name="clean")
    validation_rows = filter_manifest_rows(
        rows, protocol_role="clean_calibration", variant_name="clean"
    )
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            args.image_size, scale=(0.70, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC, antialias=True,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    eval_transform = evaluation_transform(args.image_size)
    train_loader = make_loader(
        ASBPairedTriggerDataset(
            args.images_root, train_rows, transform=train_transform,
            trigger_config=trigger_config,
        ), args, device, shuffle=True,
    )
    validation_loader = make_loader(
        ASBPairedTriggerDataset(
            args.images_root, validation_rows, transform=eval_transform,
            trigger_config=trigger_config,
        ), args, device, shuffle=False,
    )

    classifier = build_pretrained_classifier(
        backbone=classifier_config["backbone"],
        num_classes=int(classifier_config["num_classes"]), weights="none",
    )
    classifier.load_state_dict(checkpoint["model_state_dict"], strict=True)
    classifier.to(device).eval()
    for parameter in classifier.parameters():
        parameter.requires_grad = False

    generator = ReferenceFreeSpectralGenerator(
        base_channels=args.base_channels,
        max_log_correction=args.max_log_correction,
    ).to(device)
    optimizer = torch.optim.AdamW(
        generator.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    amp_enabled = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    weights = LossWeights(
        classification=args.classification_weight,
        image_reconstruction=args.image_weight,
        spectral_reconstruction=args.spectral_weight,
        clean_identity=args.identity_weight,
        sparsity=args.sparsity_weight,
        smoothness=args.smoothness_weight,
    )

    run_config = {
        **serializable_args(args),
        "protocol": "validation-only checkpoint development; locked test split untouched",
        "generator_input_at_training": "triggered image only",
        "generator_input_at_inference": "one suspicious image only",
        "clean_counterpart_role": "training supervision and evaluation only",
        "dataset": "DTD",
        "class_names": class_names,
        "target_label": target_label,
        "target_class": class_names[target_label],
        "trigger_config": trigger_dict,
        "classifier_epoch": int(checkpoint["epoch"]),
        "classifier_run_config": classifier_config,
        "generator_metadata": generator.metadata(),
        "loss_weights": asdict(weights),
        "amp_enabled": amp_enabled,
    }
    write_json(args.experiment_dir / "run_config.json", run_config)

    print("Device:", device, "AMP:", amp_enabled, flush=True)
    print("Frozen classifier:", args.classifier_checkpoint, flush=True)
    print("Trigger:", args.trigger_name, trigger_dict, flush=True)
    print("Generator input: one image only (clean counterpart is loss supervision)", flush=True)
    print("Generator parameters:", sum(p.numel() for p in generator.parameters()), flush=True)

    history: list[dict[str, object]] = []
    best_key = None
    best_epoch = None
    best_path = args.experiment_dir / "reference_free_generator_best.pt"
    last_path = args.experiment_dir / "reference_free_generator_last.pt"
    started = perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            generator, classifier, train_loader, optimizer, scaler, device,
            target_label, weights, amp_enabled, args.max_train_batches,
        )
        validation = evaluate(
            generator, classifier, validation_loader, device, target_label,
            args.max_eval_batches,
        )
        clean_floor = validation["suspicious_clean_accuracy"] - args.max_clean_accuracy_drop
        eligible = validation["generator_clean_accuracy"] >= clean_floor
        selection_key = (
            1.0 - validation["corrected_asr"],
            validation["corrected_label_accuracy"],
            validation["generator_clean_accuracy"],
        ) if eligible else None
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation,
            "selection_eligible": eligible,
            "selection_key": list(selection_key) if selection_key else None,
        }
        history.append(record)
        write_json(args.experiment_dir / "training_history.json", history)
        if selection_key is not None and (best_key is None or selection_key > best_key):
            best_key = selection_key
            best_epoch = epoch
            save_checkpoint(best_path, generator, run_config, record)
        scheduler.step()
        print(
            f"Epoch {epoch:03d} | loss {train_metrics['total_loss']:.4f} | "
            f"ASR {validation['suspicious_asr']:.4f} -> {validation['corrected_asr']:.4f} | "
            f"corr acc {validation['corrected_label_accuracy']:.4f} | "
            f"clean {validation['suspicious_clean_accuracy']:.4f} -> "
            f"{validation['generator_clean_accuracy']:.4f} | "
            f"eligible {eligible}", flush=True,
        )

    save_checkpoint(last_path, generator, run_config, history[-1])
    if best_epoch is None:
        raise RuntimeError(
            "No epoch satisfied the clean-accuracy constraint. Inspect history and "
            "adjust losses; do not select on the locked test split."
        )
    selected = torch.load(best_path, map_location=device, weights_only=True)
    generator.load_state_dict(selected["generator_state_dict"])
    final_validation = evaluate(
        generator, classifier, validation_loader, device, target_label,
        args.max_eval_batches,
    )
    summary = {
        "selected_epoch": best_epoch,
        "selection_policy": (
            "minimum validation corrected ASR, then maximum corrected-label accuracy, "
            "then clean-after-generator accuracy, subject to clean-drop constraint"
        ),
        "partial_evaluation": args.max_eval_batches is not None,
        "selected_validation": final_validation,
        "elapsed_seconds": perf_counter() - started,
        "checkpoint": str(best_path),
        "scientific_scope": (
            "Known-trigger, reference-free inference experiment. It does not yet prove "
            "generalization to unknown trigger families or unrelated datasets."
        ),
    }
    write_json(args.experiment_dir / "validation_summary.json", summary)
    write_json(args.output_dir / "validation_summary.json", summary)
    save_validation_panel(
        generator, classifier, validation_loader, device, class_names, target_label,
        args.num_panels, args.output_dir,
    )
    save_readme(args, summary, weights)
    print("Selected epoch:", best_epoch, flush=True)
    print("Saved generator:", best_path, flush=True)
    print("Saved validation evidence:", args.output_dir, flush=True)


def train_one_epoch(
    generator: nn.Module, classifier: nn.Module, loader: DataLoader,
    optimizer: torch.optim.Optimizer, scaler, device: torch.device,
    target_label: int, weights: LossWeights, amp_enabled: bool,
    max_batches: int | None,
) -> dict[str, float]:
    generator.train()
    totals = {name: 0.0 for name in (
        "total", "classification", "image", "spectral", "identity", "sparsity", "smoothness"
    )}
    count = 0
    usable_batches = 0
    for clean, triggered, labels in loader:
        clean = clean.to(device, non_blocking=True)
        triggered = triggered.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        keep = labels != target_label
        if not keep.any():
            continue
        if max_batches is not None and usable_batches >= max_batches:
            break
        usable_batches += 1
        clean, triggered, labels = clean[keep], triggered[keep], labels[keep]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            corrected = generator(triggered)
            clean_identity = generator(clean)
            classification = F.cross_entropy(classifier(corrected.corrected_image), labels)
            image_loss = F.l1_loss(corrected.corrected_image, clean)
            clean_log = torch.log1p(torch.fft.fft2(clean.float(), dim=(-2, -1)).abs())
            spectral_loss = F.l1_loss(corrected.corrected_log_amplitude, clean_log)
            identity_loss = F.l1_loss(clean_identity.corrected_image, clean)
            sparsity_loss = 0.5 * (
                corrected.correction_mask.mean() + clean_identity.correction_mask.mean()
            )
            smoothness_loss = total_variation(corrected.correction_mask)
            total = (
                weights.classification * classification
                + weights.image_reconstruction * image_loss
                + weights.spectral_reconstruction * spectral_loss
                + weights.clean_identity * identity_loss
                + weights.sparsity * sparsity_loss
                + weights.smoothness * smoothness_loss
            )
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        batch_count = labels.numel()
        count += batch_count
        values = {
            "total": total, "classification": classification, "image": image_loss,
            "spectral": spectral_loss, "identity": identity_loss,
            "sparsity": sparsity_loss, "smoothness": smoothness_loss,
        }
        for name, value in values.items():
            totals[name] += float(value.detach().item()) * batch_count
    if count == 0:
        raise ValueError("No non-target samples were available for training")
    return {f"{name}_loss": value / count for name, value in totals.items()}


@torch.inference_mode()
def evaluate(
    generator: nn.Module, classifier: nn.Module, loader: DataLoader,
    device: torch.device, target_label: int, max_batches: int | None,
) -> dict[str, float | int]:
    generator.eval()
    classifier.eval()
    totals = {
        "all": 0, "non_target": 0, "clean_correct": 0, "clean_gen_correct": 0,
        "clean_target": 0, "trigger_target": 0, "corrected_target": 0,
        "corrected_correct": 0, "clean_correct_non_target": 0,
        "clean_correct_to_target_after": 0,
    }
    image_l1 = mask_mean = effective_mean = 0.0
    usable_batches = 0
    for clean, triggered, labels in loader:
        if max_batches is not None and usable_batches >= max_batches:
            break
        clean = clean.to(device, non_blocking=True)
        triggered = triggered.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        clean_out = generator(clean)
        clean_pred = classifier(clean).argmax(1)
        clean_gen_pred = classifier(clean_out.corrected_image).argmax(1)
        totals["all"] += labels.numel()
        totals["clean_correct"] += int((clean_pred == labels).sum())
        totals["clean_gen_correct"] += int((clean_gen_pred == labels).sum())
        keep = labels != target_label
        if not keep.any():
            continue
        usable_batches += 1
        clean_nt, triggered_nt, labels_nt = clean[keep], triggered[keep], labels[keep]
        clean_pred_nt = clean_pred[keep]
        output = generator(triggered_nt)
        trigger_pred = classifier(triggered_nt).argmax(1)
        corrected_pred = classifier(output.corrected_image).argmax(1)
        n = labels_nt.numel()
        totals["non_target"] += n
        totals["clean_target"] += int((clean_pred_nt == target_label).sum())
        totals["trigger_target"] += int((trigger_pred == target_label).sum())
        totals["corrected_target"] += int((corrected_pred == target_label).sum())
        totals["corrected_correct"] += int((corrected_pred == labels_nt).sum())
        clean_correct = clean_pred_nt == labels_nt
        totals["clean_correct_non_target"] += int(clean_correct.sum())
        totals["clean_correct_to_target_after"] += int(
            (clean_correct & (corrected_pred == target_label)).sum()
        )
        image_l1 += float((output.corrected_image - clean_nt).abs().mean()) * n
        mask_mean += float(output.correction_mask.mean()) * n
        effective_mean += float(output.effective_log_correction.abs().mean()) * n
    if totals["all"] == 0 or totals["non_target"] == 0:
        raise ValueError("Evaluation loader produced no usable samples")
    n, all_count = totals["non_target"], totals["all"]
    clean_correct_nt = totals["clean_correct_non_target"]
    return {
        "num_samples": all_count,
        "num_non_target_samples": n,
        "suspicious_clean_accuracy": totals["clean_correct"] / all_count,
        "generator_clean_accuracy": totals["clean_gen_correct"] / all_count,
        "clean_non_target_target_rate": totals["clean_target"] / n,
        "suspicious_asr": totals["trigger_target"] / n,
        "corrected_asr": totals["corrected_target"] / n,
        "asr_reduction": (totals["trigger_target"] - totals["corrected_target"]) / n,
        "corrected_label_accuracy": totals["corrected_correct"] / n,
        "conditional_corrected_asr_clean_correct": (
            totals["clean_correct_to_target_after"] / clean_correct_nt
            if clean_correct_nt else None
        ),
        "corrected_image_l1_to_clean": image_l1 / n,
        "mean_correction_mask": mask_mean / n,
        "mean_absolute_effective_log_correction": effective_mean / n,
    }


@torch.inference_mode()
def save_validation_panel(
    generator, classifier, loader, device, class_names, target_label,
    num_panels, output_dir,
) -> None:
    import matplotlib.pyplot as plt

    generator.eval()
    saved = 0
    for clean, triggered, labels in loader:
        clean, triggered, labels = clean.to(device), triggered.to(device), labels.to(device)
        output = generator(triggered)
        clean_pred = classifier(clean).argmax(1)
        trigger_pred = classifier(triggered).argmax(1)
        corrected_pred = classifier(output.corrected_image).argmax(1)
        for index in range(labels.numel()):
            if labels[index] == target_label or trigger_pred[index] != target_label:
                continue
            figures = [
                (clean[index], f"clean\ntrue/pred: {class_names[labels[index]]} / {class_names[clean_pred[index]]}"),
                (triggered[index], f"triggered\npred: {class_names[trigger_pred[index]]}"),
                (output.corrected_image[index], f"corrected\npred: {class_names[corrected_pred[index]]}"),
            ]
            fig, axes = plt.subplots(2, 4, figsize=(14, 7))
            for axis, (image, title) in zip(axes[0, :3], figures):
                axis.imshow(image.detach().cpu().permute(1, 2, 0).clamp(0, 1))
                axis.set_title(title, fontsize=9)
                axis.axis("off")
            diff = (triggered[index] - clean[index]).abs() * 8.0
            axes[0, 3].imshow(diff.detach().cpu().permute(1, 2, 0).clamp(0, 1))
            axes[0, 3].set_title("trigger - clean |x8|")
            axes[0, 3].axis("off")
            spectra = [
                torch.fft.fftshift(torch.log1p(torch.fft.fft2(clean[index]).abs())).mean(0),
                torch.fft.fftshift(output.input_log_amplitude[index]).mean(0),
                torch.fft.fftshift(output.corrected_log_amplitude[index]).mean(0),
                torch.fft.fftshift(output.effective_log_correction[index].abs()).mean(0),
            ]
            titles = ["clean log amplitude", "trigger log amplitude", "corrected log amplitude", "applied correction"]
            for axis, spectrum, title in zip(axes[1], spectra, titles):
                axis.imshow(spectrum.detach().cpu(), cmap="magma")
                axis.set_title(title, fontsize=9)
                axis.axis("off")
            fig.suptitle("Reference-free generator validation evidence", fontsize=12)
            fig.tight_layout()
            fig.savefig(output_dir / f"validation_panel_{saved + 1:02d}.png", dpi=220)
            plt.close(fig)
            saved += 1
            if saved >= num_panels:
                return


def validate_checkpoint(config: dict[str, object], class_names: list[str], args) -> None:
    if config["class_names"] != class_names:
        raise ValueError("Classifier and manifest class names do not match")
    if int(config["image_size"]) != args.image_size:
        raise ValueError("--image-size must match the suspicious classifier checkpoint")
    if args.trigger_name not in config["attack_trigger_configs"]:
        raise ValueError(
            f"Checkpoint was not trained with trigger {args.trigger_name!r}; available: "
            + ", ".join(config["attack_trigger_configs"])
        )


def trigger_from_dict(data: dict[str, object]) -> AdvancedTriggerConfig:
    converted = dict(data)
    for key in ("channel_weights", "dct_channels"):
        if key in converted:
            converted[key] = tuple(converted[key])
    if "dct_positions" in converted:
        converted["dct_positions"] = tuple(tuple(pair) for pair in converted["dct_positions"])
    return AdvancedTriggerConfig(**converted)


def make_loader(dataset, args, device, *, shuffle):
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )


def validate_args(args):
    positive = ("epochs", "image_size", "batch_size", "learning_rate", "base_channels", "max_log_correction")
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0 or args.num_panels < 0:
        raise ValueError("Worker and panel counts cannot be negative")
    if not 0.0 <= args.max_clean_accuracy_drop <= 1.0:
        raise ValueError("--max-clean-accuracy-drop must be between zero and one")


def prepare_directories(args):
    protected = args.experiment_dir / "reference_free_generator_best.pt"
    if protected.exists() and not args.overwrite:
        raise FileExistsError(f"Protected result exists: {protected}. Use a new directory or --overwrite.")
    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)


def save_checkpoint(path, generator, run_config, epoch_record):
    torch.save({
        "generator_state_dict": generator.state_dict(),
        "generator_metadata": generator.metadata(),
        "run_config": run_config,
        "epoch": epoch_record["epoch"],
        "validation": epoch_record["validation"],
    }, path)


def save_readme(args, summary, weights):
    metrics = summary["selected_validation"]
    text = f"""# DTD Reference-Free Generator Validation

This experiment freezes the selected suspicious ResNet18 and trains a generator
that receives **only one incoming image**. Clean counterparts supply training
targets but are not generator inputs. Results below are calibration-split
results; the locked DTD test split was not used for development.

- Selected epoch: {summary['selected_epoch']}
- Suspicious ASR: {100 * metrics['suspicious_asr']:.2f}%
- Corrected ASR: {100 * metrics['corrected_asr']:.2f}%
- Corrected-label accuracy: {100 * metrics['corrected_label_accuracy']:.2f}%
- Suspicious clean accuracy: {100 * metrics['suspicious_clean_accuracy']:.2f}%
- Clean accuracy after generator: {100 * metrics['generator_clean_accuracy']:.2f}%
- Mean absolute effective log correction: {metrics['mean_absolute_effective_log_correction']:.6f}
- Loss weights: `{asdict(weights)}`

This is evidence for reference-free inference against the known FTrojan setting,
not yet a claim of universal unknown-trigger detection. The PNG panels show the
clean image only for evaluation; the generator used the triggered image alone.
"""
    (args.output_dir / "README.md").write_text(text, encoding="utf-8")


def serializable_args(args):
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
