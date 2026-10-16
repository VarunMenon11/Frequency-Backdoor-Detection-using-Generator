"""Run the reference-free defense on CIFAR-100 and Tiny ImageNet.

Each dataset/attack pair receives its own suspicious ResNet-18 and its own
reference-free correction generator.  Calibration data selects checkpoints;
the official test split is read only after selection.  The script is designed
for resumable Kaggle runs and keeps artifacts separate from the DTD study.
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

from datasets import (
    PairedTriggerDataset,
    PoisonedTrainingDataset,
    build_benchmark_datasets,
    stratified_subset,
)
from generator import ReferenceFreeSpectralGenerator
from generator.reference_free import total_variation
from models import build_pretrained_classifier
from poisoning import BenchmarkTriggerConfig
from scripts.summarize_cross_dataset_benchmark import build_comparison_outputs


@dataclass(frozen=True)
class GeneratorWeights:
    classification: float = 1.0
    image: float = 4.0
    spectral: float = 1.0
    identity: float = 2.0
    sparsity: float = 0.02
    smoothness: float = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="cifar100,tinyimagenet")
    parser.add_argument("--attacks", default="sig,blended,ftrojan")
    parser.add_argument("--cifar-root", type=Path, default=Path("/kaggle/working/data"))
    parser.add_argument(
        "--tiny-root", type=Path, default=Path("/kaggle/input/tiny-imagenet")
    )
    parser.add_argument("--download-cifar", action="store_true")
    parser.add_argument(
        "--experiment-root", type=Path,
        default=Path("Benchmark_Experiments/cifar_tiny_reference_free_v1"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("Benchmark_Outputs/cifar_tiny_reference_free_v1"),
    )
    parser.add_argument("--stage", choices=["all", "classifier", "generator"], default="all")
    parser.add_argument("--classifier-epochs", type=int, default=20)
    parser.add_argument("--generator-epochs", type=int, default=20)
    parser.add_argument("--freeze-epochs", type=int, default=2)
    parser.add_argument("--classifier-batch-size", type=int, default=128)
    parser.add_argument("--generator-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-4)
    parser.add_argument("--generator-learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--poison-ratio", type=float, default=0.10)
    parser.add_argument(
        "--generator-data-fraction", type=float, default=0.02,
        help="Clean training fraction available to the defense; Freq-Pret uses 0.02.",
    )
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--minimum-attack-lift", type=float, default=0.50)
    parser.add_argument("--minimum-conditional-asr", type=float, default=0.50)
    parser.add_argument("--max-clean-accuracy-drop", type=float, default=0.05)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--max-log-correction", type=float, default=2.0)
    parser.add_argument("--sig-delta", type=float, default=20.0 / 255.0)
    parser.add_argument("--sig-frequency", type=float, default=6.0)
    parser.add_argument("--blend-alpha", type=float, default=0.20)
    parser.add_argument("--blend-seed", type=int, default=2026)
    parser.add_argument("--ftrojan-strength", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-panels", type=int, default=4)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    amp_enabled = device.type == "cuda" and not args.no_amp
    datasets_to_run = parse_names(args.datasets, {"cifar100", "tinyimagenet"})
    attacks_to_run = parse_names(args.attacks, {"sig", "blended", "ftrojan"})
    args.experiment_root.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)

    print("Device:", device, "AMP:", amp_enabled, flush=True)
    print("Datasets:", ", ".join(datasets_to_run), flush=True)
    print("Attacks:", ", ".join(attacks_to_run), flush=True)
    for dataset_name in datasets_to_run:
        data_root = args.cifar_root if dataset_name == "cifar100" else args.tiny_root
        bundle = build_benchmark_datasets(
            dataset_name, data_root, seed=args.seed,
            download_cifar=args.download_cifar,
        )
        if not 0 <= args.target_label < len(bundle.class_names):
            raise ValueError(f"Target label {args.target_label} is invalid for {dataset_name}")
        print(
            f"\nDataset {dataset_name}: {bundle.split_counts}; "
            f"target={args.target_label} ({bundle.class_names[args.target_label]})",
            flush=True,
        )
        for attack_name in attacks_to_run:
            run_one(dataset_name, attack_name, bundle, args, device, amp_enabled)

    build_comparison_outputs(args.output_root)
    print("\nComparison files:", args.output_root / "paper_comparison_table.md", flush=True)


def run_one(dataset_name, attack_name, bundle, args, device, amp_enabled) -> None:
    experiment_dir = args.experiment_root / dataset_name / attack_name
    output_dir = args.output_root / dataset_name / attack_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "final_test_summary.json"
    if final_path.is_file() and not args.rerun and args.stage == "all":
        print(f"[{dataset_name}/{attack_name}] complete; skipping", flush=True)
        return

    trigger = build_trigger(attack_name, args)
    poisoned_train = PoisonedTrainingDataset(
        bundle.train, target_label=args.target_label, poison_ratio=args.poison_ratio,
        trigger=trigger, seed=args.seed,
    )
    calibration_pairs = PairedTriggerDataset(
        bundle.calibration, target_label=args.target_label, trigger=trigger
    )
    test_pairs = PairedTriggerDataset(
        bundle.test, target_label=args.target_label, trigger=trigger
    )
    run_config = {
        "dataset": dataset_name,
        "classes": len(bundle.class_names),
        "class_names": bundle.class_names,
        "native_image_size": bundle.native_size,
        "split_counts": bundle.split_counts,
        "attack": attack_name,
        "trigger": trigger.to_dict(),
        "target_label": args.target_label,
        "target_class": bundle.class_names[args.target_label],
        "poison_ratio": args.poison_ratio,
        "generator_clean_data_fraction": args.generator_data_fraction,
        "poisoned_training_samples": len(poisoned_train.poisoned_indices),
        "model": "ImageNet-pretrained ResNet-18",
        "generator": "ReferenceFreeSpectralGenerator; one image at inference",
        "checkpoint_selection": "calibration split only; official test held out",
        "seed": args.seed,
        "args": serializable_args(args),
    }
    write_json(experiment_dir / "run_config.json", run_config)
    write_json(output_dir / "run_config.json", run_config)
    save_trigger_preview(calibration_pairs, output_dir, attack_name)
    print(
        f"\n[{dataset_name}/{attack_name}] poisoned rows: "
        f"{len(poisoned_train.poisoned_indices)}/{len(poisoned_train)}",
        flush=True,
    )

    classifier_path = experiment_dir / "suspicious_classifier_best.pt"
    classifier_complete = experiment_dir / "classifier_training_complete.json"
    classifier = build_pretrained_classifier(
        backbone="resnet18", num_classes=len(bundle.class_names), weights="default"
    ).to(device)
    if classifier_path.is_file() and classifier_complete.is_file() and not args.rerun:
        checkpoint = torch.load(classifier_path, map_location=device, weights_only=True)
        classifier.load_state_dict(checkpoint["model_state_dict"])
        classifier_record = checkpoint["record"]
        print(f"[{dataset_name}/{attack_name}] loaded classifier checkpoint", flush=True)
    elif args.stage == "generator" and not classifier_path.is_file():
        raise FileNotFoundError(f"Missing classifier checkpoint: {classifier_path}")
    else:
        classifier_record = train_classifier(
            classifier, poisoned_train, bundle.calibration, calibration_pairs,
            classifier_path, experiment_dir, args, device, amp_enabled,
            dataset_name, run_config,
        )

    classifier.eval()
    for parameter in classifier.parameters():
        parameter.requires_grad = False
    calibration_attack = evaluate_model(
        classifier, None, bundle.calibration, calibration_pairs, args, device
    )
    attack_valid = (
        calibration_attack["asr_lift"] >= args.minimum_attack_lift
        and calibration_attack["conditional_asr_clean_correct"]
        >= args.minimum_conditional_asr
    )
    attack_summary = {
        "selected_classifier_epoch": classifier_record["epoch"],
        "calibration": calibration_attack,
        "accepted_for_defense": attack_valid,
        "acceptance_rule": {
            "minimum_attack_lift": args.minimum_attack_lift,
            "minimum_conditional_asr_clean_correct": args.minimum_conditional_asr,
        },
    }
    write_json(output_dir / "attack_validation.json", attack_summary)
    print_metrics(dataset_name, attack_name, "attack", calibration_attack)
    if args.stage == "classifier":
        return
    if not attack_valid:
        print(
            f"[{dataset_name}/{attack_name}] REJECTED: attack is not active enough; "
            "generator training was intentionally skipped.", flush=True,
        )
        write_json(final_path, {**run_config, "status": "rejected_weak_attack", **attack_summary})
        return

    generator_path = experiment_dir / "reference_free_generator_best.pt"
    generator_complete = experiment_dir / "generator_training_complete.json"
    generator = ReferenceFreeSpectralGenerator(
        base_channels=args.base_channels,
        max_log_correction=args.max_log_correction,
    ).to(device)
    if generator_path.is_file() and generator_complete.is_file() and not args.rerun:
        checkpoint = torch.load(generator_path, map_location=device, weights_only=True)
        generator.load_state_dict(checkpoint["generator_state_dict"])
        generator_record = checkpoint["record"]
        print(f"[{dataset_name}/{attack_name}] loaded generator checkpoint", flush=True)
    else:
        generator_record = train_generator(
            generator, classifier, bundle.train, calibration_pairs, trigger,
            generator_path, experiment_dir, args, device, amp_enabled,
            calibration_attack["clean_accuracy"], run_config,
        )

    final_metrics = evaluate_model(
        classifier, generator, bundle.test, test_pairs, args, device
    )
    baseline_metrics = evaluate_model(
        classifier, None, bundle.test, test_pairs, args, device
    )
    summary = {
        **run_config,
        "status": "complete",
        "selected_classifier_epoch": classifier_record["epoch"],
        "selected_generator_epoch": generator_record["epoch"],
        "metric_definitions": {
            "ACC": "accuracy on clean official-test images after the defense",
            "ASR": "fraction of triggered non-target test images predicted as target",
            "RA": "true-label accuracy on triggered non-target test images",
        },
        "no_defense": baseline_metrics,
        "reference_free_defense": final_metrics,
        "comparability_note": (
            "Same dataset, ResNet-18 family, target label, poison ratio, and attack "
            "family as the cited comparison where applicable. Training budget, "
            "pretraining, trigger carrier/implementation, and split details may differ; "
            "treat as protocol-aligned rather than an exact reproduction."
        ),
    }
    write_json(final_path, summary)
    save_evidence_panels(
        classifier, generator, test_pairs, bundle.class_names, args.target_label,
        output_dir, args, device,
    )
    write_run_readme(output_dir, summary)
    print_metrics(dataset_name, attack_name, "final", final_metrics)


def train_classifier(
    model, poisoned_train, clean_calibration, paired_calibration, checkpoint_path,
    experiment_dir, args, device, amp_enabled, dataset_name, run_config,
):
    train_loader = make_loader(
        poisoned_train, args.classifier_batch_size, args, device, shuffle=True
    )
    optimizer = torch.optim.AdamW([
        {"params": list(model.feature_parameters()), "lr": args.backbone_learning_rate},
        {"params": list(model.head_parameters()), "lr": args.head_learning_rate},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.classifier_epochs
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    clean_floor = 0.55 if dataset_name == "cifar100" else 0.35
    history_path = experiment_dir / "classifier_history.json"
    last_path = experiment_dir / "suspicious_classifier_last.pt"
    history, best_key, best_record, start_epoch = [], None, None, 1
    if last_path.is_file() and history_path.is_file() and not args.rerun:
        last = torch.load(last_path, map_location=device, weights_only=True)
        model.load_state_dict(last["model_state_dict"])
        optimizer.load_state_dict(last["optimizer_state_dict"])
        scheduler.load_state_dict(last["scheduler_state_dict"])
        scaler.load_state_dict(last["scaler_state_dict"])
        history = json.loads(history_path.read_text(encoding="utf-8"))
        start_epoch = int(last["epoch"]) + 1
        if checkpoint_path.is_file():
            best = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            best_record = best["record"]
            metrics = best_record["validation"]
            best_key = (
                metrics["conditional_asr_clean_correct"], metrics["asr_lift"],
                metrics["clean_accuracy"],
            )
        print(f"  resuming classifier at epoch {start_epoch}", flush=True)
    for epoch in range(start_epoch, args.classifier_epochs + 1):
        model.set_feature_extractor_trainable(epoch > args.freeze_epochs)
        model.train()
        total_loss = correct = count = 0
        for batch_index, (images, labels) in enumerate(train_loader):
            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                break
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            count += labels.numel()
            total_loss += float(loss.detach()) * labels.numel()
            correct += int((logits.argmax(1) == labels).sum())
        metrics = evaluate_model(
            model, None, clean_calibration, paired_calibration, args, device
        )
        eligible = metrics["clean_accuracy"] >= clean_floor
        key = (
            metrics["conditional_asr_clean_correct"], metrics["asr_lift"],
            metrics["clean_accuracy"],
        ) if eligible else None
        record = {
            "epoch": epoch, "train_loss": total_loss / max(count, 1),
            "train_accuracy": correct / max(count, 1), "validation": metrics,
            "selection_eligible": eligible,
        }
        history.append(record)
        write_json(history_path, history)
        if key is not None and (best_key is None or key > best_key):
            best_key, best_record = key, record
            torch.save({
                "model_state_dict": model.state_dict(), "record": record,
                "run_config": run_config,
            }, checkpoint_path)
        scheduler.step()
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
        }, last_path)
        print(
            f"  classifier {epoch:02d}/{args.classifier_epochs} | "
            f"clean {metrics['clean_accuracy']:.4f} | ASR {metrics['asr']:.4f} | "
            f"lift {metrics['asr_lift']:.4f} | cASR "
            f"{metrics['conditional_asr_clean_correct']:.4f}", flush=True,
        )
    if best_record is None:
        raise RuntimeError(f"No classifier epoch reached clean accuracy floor {clean_floor:.2f}")
    selected = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(selected["model_state_dict"])
    write_json(
        experiment_dir / "classifier_training_complete.json",
        {"epochs_completed": args.classifier_epochs, "selected_epoch": best_record["epoch"]},
    )
    return best_record


def train_generator(
    generator, classifier, clean_train, paired_calibration, trigger,
    checkpoint_path, experiment_dir, args, device, amp_enabled,
    suspicious_clean_accuracy, run_config,
):
    generator_clean = stratified_subset(
        clean_train, args.generator_data_fraction, args.seed + 1000
    )
    train_pairs = PairedTriggerDataset(
        generator_clean, target_label=args.target_label, trigger=trigger
    )
    loader = make_loader(
        train_pairs, args.generator_batch_size, args, device, shuffle=True
    )
    optimizer = torch.optim.AdamW(
        generator.parameters(), lr=args.generator_learning_rate, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.generator_epochs
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    weights = GeneratorWeights()
    history_path = experiment_dir / "generator_history.json"
    last_path = experiment_dir / "reference_free_generator_last.pt"
    history, best_key, best_record, start_epoch = [], None, None, 1
    if last_path.is_file() and history_path.is_file() and not args.rerun:
        last = torch.load(last_path, map_location=device, weights_only=True)
        generator.load_state_dict(last["generator_state_dict"])
        optimizer.load_state_dict(last["optimizer_state_dict"])
        scheduler.load_state_dict(last["scheduler_state_dict"])
        scaler.load_state_dict(last["scaler_state_dict"])
        history = json.loads(history_path.read_text(encoding="utf-8"))
        start_epoch = int(last["epoch"]) + 1
        if checkpoint_path.is_file():
            best = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            best_record = best["record"]
            metrics = best_record["validation"]
            best_key = (
                1.0 - metrics["asr"], metrics["robust_accuracy"],
                metrics["clean_accuracy"],
            )
        print(f"  resuming generator at epoch {start_epoch}", flush=True)
    for epoch in range(start_epoch, args.generator_epochs + 1):
        generator.train()
        sums = {key: 0.0 for key in ("total", "cls", "image", "spectral", "identity", "sparse", "smooth")}
        count = 0
        for batch_index, (clean, triggered, labels) in enumerate(loader):
            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                break
            clean, triggered, labels = clean.to(device), triggered.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                output = generator(triggered)
                identity = generator(clean)
                cls = F.cross_entropy(classifier(output.corrected_image), labels)
                image = F.l1_loss(output.corrected_image, clean)
                clean_log = torch.log1p(torch.fft.fft2(clean.float(), dim=(-2, -1)).abs())
                spectral = F.l1_loss(output.corrected_log_amplitude, clean_log)
                identity_loss = F.l1_loss(identity.corrected_image, clean)
                sparse = 0.5 * (
                    output.correction_mask.mean() + identity.correction_mask.mean()
                )
                smooth = total_variation(output.correction_mask)
                total = (
                    weights.classification * cls + weights.image * image
                    + weights.spectral * spectral + weights.identity * identity_loss
                    + weights.sparsity * sparse + weights.smoothness * smooth
                )
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            batch = labels.numel()
            count += batch
            values = (total, cls, image, spectral, identity_loss, sparse, smooth)
            for key, value in zip(sums, values):
                sums[key] += float(value.detach()) * batch
        validation = evaluate_model(
            classifier, generator, paired_calibration.clean, paired_calibration,
            args, device,
        )
        eligible = validation["clean_accuracy"] >= (
            suspicious_clean_accuracy - args.max_clean_accuracy_drop
        )
        key = (
            1.0 - validation["asr"], validation["robust_accuracy"],
            validation["clean_accuracy"],
        ) if eligible else None
        record = {
            "epoch": epoch,
            "train": {name: value / max(count, 1) for name, value in sums.items()},
            "validation": validation,
            "selection_eligible": eligible,
            "loss_weights": asdict(weights),
            "generator_clean_samples": len(generator_clean),
        }
        history.append(record)
        write_json(history_path, history)
        if key is not None and (best_key is None or key > best_key):
            best_key, best_record = key, record
            torch.save({
                "generator_state_dict": generator.state_dict(), "record": record,
                "run_config": run_config, "generator_metadata": generator.metadata(),
            }, checkpoint_path)
        scheduler.step()
        torch.save({
            "generator_state_dict": generator.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
        }, last_path)
        print(
            f"  generator {epoch:02d}/{args.generator_epochs} | "
            f"loss {record['train']['total']:.4f} | ASR {validation['asr']:.4f} | "
            f"RA {validation['robust_accuracy']:.4f} | ACC "
            f"{validation['clean_accuracy']:.4f} | eligible {eligible}", flush=True,
        )
    if best_record is None:
        raise RuntimeError("No generator epoch satisfied the clean-accuracy constraint")
    selected = torch.load(checkpoint_path, map_location=device, weights_only=True)
    generator.load_state_dict(selected["generator_state_dict"])
    write_json(
        experiment_dir / "generator_training_complete.json",
        {"epochs_completed": args.generator_epochs, "selected_epoch": best_record["epoch"]},
    )
    return best_record


@torch.inference_mode()
def evaluate_model(classifier, generator, clean_dataset, paired_dataset, args, device):
    classifier.eval()
    if generator is not None:
        generator.eval()
    clean_loader = make_loader(
        clean_dataset, args.generator_batch_size, args, device, shuffle=False
    )
    pair_loader = make_loader(
        paired_dataset, args.generator_batch_size, args, device, shuffle=False
    )
    clean_correct = clean_target = clean_count = 0
    clean_after_l1 = 0.0
    for batch_index, (images, labels) in enumerate(clean_loader):
        if args.max_eval_batches is not None and batch_index >= args.max_eval_batches:
            break
        images, labels = images.to(device), labels.to(device)
        evaluated = generator(images).corrected_image if generator is not None else images
        predictions = classifier(evaluated).argmax(1)
        clean_count += labels.numel()
        clean_correct += int((predictions == labels).sum())
        clean_target += int((predictions == args.target_label).sum())
        clean_after_l1 += float((evaluated - images).abs().mean()) * labels.numel()

    target = robust = count = 0
    clean_baseline_target = clean_baseline_correct = conditional_target = 0
    image_l1 = mask_mean = effective_mean = 0.0
    for batch_index, (clean, triggered, labels) in enumerate(pair_loader):
        if args.max_eval_batches is not None and batch_index >= args.max_eval_batches:
            break
        clean, triggered, labels = clean.to(device), triggered.to(device), labels.to(device)
        baseline_clean_pred = classifier(clean).argmax(1)
        if generator is None:
            evaluated = triggered
            output = None
        else:
            output = generator(triggered)
            evaluated = output.corrected_image
        prediction = classifier(evaluated).argmax(1)
        batch = labels.numel()
        count += batch
        target += int((prediction == args.target_label).sum())
        robust += int((prediction == labels).sum())
        baseline_correct = baseline_clean_pred == labels
        clean_baseline_correct += int(baseline_correct.sum())
        clean_baseline_target += int((baseline_clean_pred == args.target_label).sum())
        conditional_target += int((baseline_correct & (prediction == args.target_label)).sum())
        image_l1 += float((evaluated - clean).abs().mean()) * batch
        if output is not None:
            mask_mean += float(output.correction_mask.mean()) * batch
            effective_mean += float(output.effective_log_correction.abs().mean()) * batch
    if clean_count == 0 or count == 0:
        raise RuntimeError("Evaluation produced no samples")
    asr = target / count
    clean_non_target_rate = clean_baseline_target / count
    return {
        "num_clean_samples": clean_count,
        "num_triggered_non_target_samples": count,
        "clean_accuracy": clean_correct / clean_count,
        "clean_target_rate_after_pipeline": clean_target / clean_count,
        "asr": asr,
        "robust_accuracy": robust / count,
        "clean_non_target_target_rate": clean_non_target_rate,
        "asr_lift": asr - clean_non_target_rate,
        "conditional_asr_clean_correct": conditional_target / max(clean_baseline_correct, 1),
        "mean_l1_to_clean_on_triggered": image_l1 / count,
        "mean_l1_change_on_clean": clean_after_l1 / clean_count,
        "mean_correction_mask": mask_mean / count if generator is not None else 0.0,
        "mean_absolute_effective_log_correction": (
            effective_mean / count if generator is not None else 0.0
        ),
    }


@torch.inference_mode()
def save_evidence_panels(
    classifier, generator, paired_dataset, class_names, target_label,
    output_dir, args, device,
):
    import matplotlib.pyplot as plt

    loader = make_loader(paired_dataset, args.generator_batch_size, args, device, False)
    saved = 0
    classifier.eval(); generator.eval()
    for clean, triggered, labels in loader:
        clean, triggered, labels = clean.to(device), triggered.to(device), labels.to(device)
        output = generator(triggered)
        clean_pred = classifier(clean).argmax(1)
        trigger_pred = classifier(triggered).argmax(1)
        corrected_pred = classifier(output.corrected_image).argmax(1)
        for index in range(labels.numel()):
            if trigger_pred[index] != target_label:
                continue
            fig, axes = plt.subplots(2, 4, figsize=(14, 7))
            images = [clean[index], triggered[index], output.corrected_image[index]]
            titles = [
                f"clean\ntrue/pred: {class_names[labels[index]]} / {class_names[clean_pred[index]]}",
                f"triggered\npred: {class_names[trigger_pred[index]]}",
                f"corrected\npred: {class_names[corrected_pred[index]]}",
            ]
            for axis, image, title in zip(axes[0, :3], images, titles):
                axis.imshow(image.detach().cpu().permute(1, 2, 0).clamp(0, 1))
                axis.set_title(title, fontsize=8); axis.axis("off")
            axes[0, 3].imshow(
                ((triggered[index] - clean[index]).abs() * 8).detach().cpu().permute(1, 2, 0).clamp(0, 1)
            )
            axes[0, 3].set_title("trigger - clean |x8|"); axes[0, 3].axis("off")
            spectra = [
                torch.fft.fftshift(torch.log1p(torch.fft.fft2(clean[index]).abs())).mean(0),
                torch.fft.fftshift(output.input_log_amplitude[index]).mean(0),
                torch.fft.fftshift(output.corrected_log_amplitude[index]).mean(0),
                torch.fft.fftshift(output.effective_log_correction[index].abs()).mean(0),
            ]
            for axis, spectrum, title in zip(
                axes[1], spectra,
                ["clean log amplitude", "trigger log amplitude", "corrected log amplitude", "applied correction"],
            ):
                axis.imshow(spectrum.detach().cpu(), cmap="magma")
                axis.set_title(title, fontsize=8); axis.axis("off")
            fig.tight_layout()
            fig.savefig(output_dir / f"evidence_panel_{saved + 1:02d}.png", dpi=200)
            plt.close(fig)
            saved += 1
            if saved >= args.num_panels:
                return


def save_trigger_preview(paired_dataset, output_dir: Path, attack_name: str) -> None:
    """Save the exact trigger and its spatial/spectral change before training."""

    import matplotlib.pyplot as plt

    clean, triggered, _ = paired_dataset[0]
    clean_spectrum = torch.fft.fftshift(
        torch.log1p(torch.fft.fft2(clean).abs()), dim=(-2, -1)
    ).mean(0)
    trigger_spectrum = torch.fft.fftshift(
        torch.log1p(torch.fft.fft2(triggered).abs()), dim=(-2, -1)
    ).mean(0)
    fig, axes = plt.subplots(1, 6, figsize=(18, 3.2))
    axes[0].imshow(clean.permute(1, 2, 0).clamp(0, 1)); axes[0].set_title("clean")
    axes[1].imshow(triggered.permute(1, 2, 0).clamp(0, 1)); axes[1].set_title("triggered")
    axes[2].imshow(((triggered - clean).abs() * 8).permute(1, 2, 0).clamp(0, 1))
    axes[2].set_title("pixel change |x8|")
    axes[3].imshow(clean_spectrum, cmap="magma"); axes[3].set_title("clean log amplitude")
    axes[4].imshow(trigger_spectrum, cmap="magma"); axes[4].set_title("trigger log amplitude")
    axes[5].imshow((trigger_spectrum - clean_spectrum).abs(), cmap="inferno")
    axes[5].set_title("absolute spectral change")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(f"Injected benchmark trigger: {attack_name}")
    fig.tight_layout()
    fig.savefig(output_dir / "trigger_preview.png", dpi=200)
    plt.close(fig)


def build_trigger(name: str, args) -> BenchmarkTriggerConfig:
    return BenchmarkTriggerConfig(
        kind=name, sig_delta=args.sig_delta, sig_frequency=args.sig_frequency,
        blend_alpha=args.blend_alpha, blend_seed=args.blend_seed,
        ftrojan_strength=args.ftrojan_strength,
    )


def make_loader(dataset, batch_size, args, device, shuffle):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )


def write_run_readme(output_dir: Path, summary: dict) -> None:
    before, after = summary["no_defense"], summary["reference_free_defense"]
    text = f"""# {summary['dataset']} / {summary['attack']} benchmark

This run used a separate suspicious ResNet-18 and a separate reference-free
generator. The generator received one image at inference; the official test
split was not used for checkpoint selection.

| Pipeline | Clean ACC | Triggered ASR | Robust accuracy |
|---|---:|---:|---:|
| No defense | {100*before['clean_accuracy']:.2f}% | {100*before['asr']:.2f}% | {100*before['robust_accuracy']:.2f}% |
| Reference-free generator | {100*after['clean_accuracy']:.2f}% | {100*after['asr']:.2f}% | {100*after['robust_accuracy']:.2f}% |

**Interpretation:** lower ASR is better; higher ACC and robust accuracy are
better. A defense is useful only when ASR falls without destroying clean ACC.

**Comparison boundary:** {summary['comparability_note']}
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def print_metrics(dataset, attack, stage, metrics):
    print(
        f"[{dataset}/{attack}] {stage}: ACC {metrics['clean_accuracy']:.4f} | "
        f"ASR {metrics['asr']:.4f} | RA {metrics['robust_accuracy']:.4f} | "
        f"lift {metrics['asr_lift']:.4f}", flush=True,
    )


def parse_names(value: str, allowed: set[str]) -> list[str]:
    names = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = [name for name in names if name not in allowed]
    if not names or invalid or len(names) != len(set(names)):
        raise ValueError(f"Invalid unique names {invalid}; allowed: {sorted(allowed)}")
    return names


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def validate_args(args):
    if not 0.0 < args.poison_ratio < 1.0:
        raise ValueError("poison-ratio must be between 0 and 1")
    if not 0.0 < args.generator_data_fraction <= 1.0:
        raise ValueError("generator-data-fraction must be in (0, 1]")
    for name in ("classifier_epochs", "generator_epochs", "classifier_batch_size", "generator_batch_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0 or args.num_panels < 0:
        raise ValueError("workers and panels cannot be negative")


def serializable_args(args):
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
