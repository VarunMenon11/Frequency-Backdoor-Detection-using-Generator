"""Evaluate the frozen DTD generator on unseen triggers and benign corruption.

This validation-only script performs no training and never reads the locked DTD
test split. It compares the image-conditioned generator with a fixed correction
template learned from the known FTrojan training configuration.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import replace
import json
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

from datasets import (
    ASBManifestDataset,
    ASBPairedTriggerDataset,
    filter_manifest_rows,
    read_jsonl,
)
from evaluation.asb_classifier import evaluation_transform
from generator import ReferenceFreeSpectralGenerator
from models import build_pretrained_classifier
from scripts.audit_dtd_reference_free_generator import (
    build_static_template,
    log_amplitude,
    reconstruct_with_log_correction,
    top_fraction_mask,
)
from scripts.train_dtd_reference_free_generator import trigger_from_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generator-checkpoint", type=Path,
        default=Path("Advanced_Experiments/dtd_reference_free_generator_ftrojan_v1/reference_free_generator_best.pt"),
    )
    parser.add_argument(
        "--classifier-checkpoint", type=Path,
        default=Path("Advanced_Experiments/dtd_attack_sweep_v1/ftrojan_m100_paired_r020/suspicious_classifier_best_attack.pt"),
    )
    parser.add_argument("--manifest-dir", type=Path, default=Path("Absolute_Dataset/asb_dtd_v1"))
    parser.add_argument("--images-root", type=Path, default=Path("Absolute_Dataset/dtd/images"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("Advanced_Outputs/dtd_reference_free_generalization_v1"),
    )
    parser.add_argument(
        "--suites", default="strength,position,corruption",
        help="Comma-separated subset of strength, position, corruption.",
    )
    parser.add_argument(
        "--scenario-names", default=None,
        help="Optional comma-separated scenario names for a targeted/local run.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--support-fraction", type=float, default=0.01)
    parser.add_argument("--max-template-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    prepare_output(args)
    seed_everything(args.seed)
    device = resolve_device(args.device)

    generator_checkpoint = torch.load(
        args.generator_checkpoint, map_location="cpu", weights_only=True
    )
    classifier_checkpoint = torch.load(
        args.classifier_checkpoint, map_location="cpu", weights_only=True
    )
    generator, classifier, config = build_models(
        generator_checkpoint, classifier_checkpoint, device
    )
    rows = read_jsonl(args.manifest_dir / "variant_manifest.jsonl")
    target_label = int(config["target_label"])
    class_names = list(config["class_names"])
    transform = evaluation_transform(int(config["image_size"]))
    known_trigger = trigger_from_dict(dict(config["trigger_config"]))

    train_rows = filter_manifest_rows(
        rows, protocol_role="clean_train", variant_name="clean",
        exclude_original_label=target_label,
    )
    validation_non_target = filter_manifest_rows(
        rows, protocol_role="clean_calibration", variant_name="clean",
        exclude_original_label=target_label,
    )
    validation_all = filter_manifest_rows(
        rows, protocol_role="clean_calibration", variant_name="clean"
    )
    template_loader = make_loader(
        ASBPairedTriggerDataset(
            args.images_root, train_rows, transform=transform,
            trigger_config=known_trigger,
        ), args, device,
    )
    print("Device:", device, flush=True)
    print("Protocol: frozen models, validation only, no training", flush=True)
    print("Building known-trigger static template...", flush=True)
    static_template, template_summary = build_static_template(
        generator, template_loader, device, args.max_template_batches
    )
    print("Static template samples:", template_summary["num_samples"], flush=True)

    requested_suites = parse_requested(args.suites)
    selected_names = (
        set(parse_requested(args.scenario_names)) if args.scenario_names else None
    )
    trigger_scenarios = build_trigger_scenarios(known_trigger, requested_suites)
    corruption_scenarios = build_corruption_scenarios(requested_suites)
    if selected_names is not None:
        trigger_scenarios = [row for row in trigger_scenarios if row["name"] in selected_names]
        corruption_scenarios = [row for row in corruption_scenarios if row["name"] in selected_names]
        missing = selected_names - {
            row["name"] for row in trigger_scenarios + corruption_scenarios
        }
        if missing:
            raise ValueError("Unknown or unavailable scenarios: " + ", ".join(sorted(missing)))
    if not trigger_scenarios and not corruption_scenarios:
        raise ValueError("No scenarios were selected")

    trigger_results = []
    for index, scenario in enumerate(trigger_scenarios, start=1):
        print(
            f"Trigger scenario {index}/{len(trigger_scenarios)}: {scenario['name']}",
            flush=True,
        )
        loader = make_loader(
            ASBPairedTriggerDataset(
                args.images_root, validation_non_target, transform=transform,
                trigger_config=scenario["config"],
            ), args, device,
        )
        scenario_dir = args.output_dir / "trigger_scenarios" / scenario["name"]
        scenario_dir.mkdir(parents=True, exist_ok=True)
        result, records, example = evaluate_trigger_scenario(
            generator, classifier, loader, static_template, device,
            target_label, class_names, args.support_fraction,
            args.max_eval_batches,
        )
        result.update({
            "scenario": scenario["name"],
            "suite": scenario["suite"],
            "known_training_configuration": scenario["known"],
            "trigger_config": scenario["config"].to_dict(),
            "partial_evaluation": args.max_eval_batches is not None,
        })
        write_json(scenario_dir / "summary.json", result)
        write_jsonl(scenario_dir / "predictions.jsonl", records)
        save_trigger_panel(
            example, class_names, target_label, scenario["name"],
            scenario_dir / "panel.png",
        )
        trigger_results.append(result)

    corruption_results = []
    if corruption_scenarios:
        clean_loader = DataLoader(
            ASBManifestDataset(
                args.images_root, validation_all, transform=transform,
                label_field="original_label",
            ),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )
        for index, scenario in enumerate(corruption_scenarios, start=1):
            print(
                f"Corruption scenario {index}/{len(corruption_scenarios)}: {scenario['name']}",
                flush=True,
            )
            scenario_dir = args.output_dir / "corruption_scenarios" / scenario["name"]
            scenario_dir.mkdir(parents=True, exist_ok=True)
            result, records, example = evaluate_corruption_scenario(
                generator, classifier, clean_loader, static_template, device,
                target_label, class_names, scenario, args.seed + index,
                args.max_eval_batches,
            )
            result.update({
                "scenario": scenario["name"],
                "suite": "corruption",
                "corruption": {key: value for key, value in scenario.items() if key != "name"},
                "partial_evaluation": args.max_eval_batches is not None,
            })
            write_json(scenario_dir / "summary.json", result)
            write_jsonl(scenario_dir / "predictions.jsonl", records)
            save_corruption_panel(
                example, class_names, target_label, scenario["name"],
                scenario_dir / "panel.png",
            )
            corruption_results.append(result)

    aggregate = {
        "protocol": "frozen generator and classifier; validation only; no training",
        "partial_evaluation": args.max_eval_batches is not None,
        "generator_checkpoint": str(args.generator_checkpoint),
        "generator_epoch": int(generator_checkpoint["epoch"]),
        "classifier_checkpoint": str(args.classifier_checkpoint),
        "classifier_epoch": int(classifier_checkpoint["epoch"]),
        "target_label": target_label,
        "target_class": class_names[target_label],
        "support_fraction": args.support_fraction,
        "known_trigger_config": known_trigger.to_dict(),
        "static_template": template_summary,
        "trigger_results": trigger_results,
        "corruption_results": corruption_results,
        "interpretation_boundary": (
            "Defense comparisons are meaningful only when the modified trigger "
            "still produces a substantial undefended attack. Position scenarios "
            "with low attack lift test attack transfer, not defense failure."
        ),
    }
    write_json(args.output_dir / "generalization_summary.json", aggregate)
    save_csv(args.output_dir / "trigger_generalization.csv", trigger_results, "trigger")
    save_csv(args.output_dir / "corruption_controls.csv", corruption_results, "corruption")
    if trigger_results:
        save_trigger_chart(trigger_results, args.output_dir / "trigger_generalization.png")
    if corruption_results:
        save_corruption_chart(corruption_results, args.output_dir / "corruption_controls.png")
    save_markdown(aggregate, args.output_dir / "README.md")
    print_tables(trigger_results, corruption_results)
    print("Saved generalization suite:", args.output_dir, flush=True)


def build_models(generator_checkpoint, classifier_checkpoint, device):
    metadata = generator_checkpoint["generator_metadata"]
    generator = ReferenceFreeSpectralGenerator(
        base_channels=int(metadata["base_channels"]),
        max_log_correction=float(metadata["max_log_correction"]),
    )
    generator.load_state_dict(generator_checkpoint["generator_state_dict"], strict=True)
    generator.to(device).eval()
    config = generator_checkpoint["run_config"]
    classifier_config = classifier_checkpoint["run_config"]
    if config["class_names"] != classifier_config["class_names"]:
        raise ValueError("Generator and classifier class names do not match")
    if int(config["target_label"]) != int(classifier_config["target_label"]):
        raise ValueError("Generator and classifier target labels do not match")
    classifier = build_pretrained_classifier(
        backbone=classifier_config["backbone"],
        num_classes=int(classifier_config["num_classes"]), weights="none",
    )
    classifier.load_state_dict(classifier_checkpoint["model_state_dict"], strict=True)
    classifier.to(device).eval()
    return generator, classifier, config


def build_trigger_scenarios(known, suites):
    scenarios = []
    if "strength" in suites:
        for strength in (50.0, 75.0, 100.0, 125.0, 150.0):
            scenarios.append({
                "name": f"strength_{int(strength):03d}_known_positions",
                "suite": "strength", "known": strength == 100.0,
                "config": replace(known, strength=strength),
            })
    if "position" in suites:
        position_settings = [
            ("position_near_minus1_s100", ((14, 14), (30, 30)), 100.0),
            ("position_near_mixed_s100", ((14, 16), (30, 28)), 100.0),
            ("position_mid_shift_s100", ((12, 18), (28, 30)), 100.0),
            ("position_far_cross_s100", ((8, 24), (24, 8)), 100.0),
            ("position_near_minus1_s150", ((14, 14), (30, 30)), 150.0),
            ("position_far_cross_s150", ((8, 24), (24, 8)), 150.0),
        ]
        for name, positions, strength in position_settings:
            scenarios.append({
                "name": name, "suite": "position", "known": False,
                "config": replace(known, dct_positions=positions, strength=strength),
            })
    return scenarios


def build_corruption_scenarios(suites):
    if "corruption" not in suites:
        return []
    return [
        {"name": "gaussian_noise_001", "kind": "noise", "value": 0.01},
        {"name": "gaussian_noise_003", "kind": "noise", "value": 0.03},
        {"name": "gaussian_blur_3", "kind": "blur", "kernel": 3, "sigma": 1.0},
        {"name": "gaussian_blur_5", "kind": "blur", "kernel": 5, "sigma": 2.0},
        {"name": "brightness_080", "kind": "brightness", "value": 0.8},
        {"name": "brightness_120", "kind": "brightness", "value": 1.2},
        {"name": "contrast_080", "kind": "contrast", "value": 0.8},
        {"name": "contrast_120", "kind": "contrast", "value": 1.2},
    ]


@torch.inference_mode()
def evaluate_trigger_scenario(
    generator, classifier, loader, static_template, device, target_label,
    class_names, support_fraction, max_batches,
):
    totals = {
        "samples": 0, "clean_target": 0, "trigger_target": 0,
        "generator_target": 0, "static_target": 0,
        "clean_correct": 0, "trigger_correct": 0,
        "generator_correct": 0, "static_correct": 0,
    }
    sums = defaultdict(float)
    records = []
    example = None
    example_rank = -1
    for batch_index, (clean, triggered, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        clean = clean.to(device, non_blocking=True)
        triggered = triggered.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        output = generator(triggered)
        static_correction = static_template.unsqueeze(0).expand_as(output.effective_log_correction)
        static_images = reconstruct_with_log_correction(triggered, static_correction)
        variants = torch.cat((clean, triggered, output.corrected_image, static_images), dim=0)
        logits = classifier(variants)
        probabilities = logits.softmax(1).view(4, labels.numel(), -1)
        predictions = logits.argmax(1).view(4, labels.numel())

        evidence = (log_amplitude(triggered) - log_amplitude(clean)).abs()
        evidence_support = top_fraction_mask(evidence, support_fraction)
        generator_support = top_fraction_mask(
            output.effective_log_correction.abs(), support_fraction
        )
        static_support = top_fraction_mask(
            static_correction.abs(), support_fraction
        )
        generator_iou = support_iou(generator_support, evidence_support)
        static_iou = support_iou(static_support, evidence_support)

        n = labels.numel()
        totals["samples"] += n
        for index, prefix in enumerate(("clean", "trigger", "generator", "static")):
            totals[f"{prefix}_target"] += int((predictions[index] == target_label).sum())
            totals[f"{prefix}_correct"] += int((predictions[index] == labels).sum())
        sums["generator_correction"] += float(
            output.effective_log_correction.abs().flatten(1).mean(1).sum().cpu()
        )
        sums["generator_iou"] += float(generator_iou.sum().cpu())
        sums["static_iou"] += float(static_iou.sum().cpu())
        sums["trigger_l1"] += float((triggered-clean).abs().flatten(1).mean(1).sum().cpu())
        sums["generator_l1"] += float((output.corrected_image-clean).abs().flatten(1).mean(1).sum().cpu())
        sums["static_l1"] += float((static_images-clean).abs().flatten(1).mean(1).sum().cpu())

        for item in range(n):
            row = {
                "sample_index": len(records), "true_label": int(labels[item]),
                "true_class": class_names[int(labels[item])],
                "generator_support_iou": float(generator_iou[item].cpu()),
                "static_support_iou": float(static_iou[item].cpu()),
                "generator_correction_mean_abs": float(
                    output.effective_log_correction[item].abs().mean().cpu()
                ),
            }
            for variant_index, prefix in enumerate(("clean", "trigger", "generator", "static")):
                prediction = int(predictions[variant_index, item])
                row[f"{prefix}_prediction"] = prediction
                row[f"{prefix}_prediction_class"] = class_names[prediction]
                row[f"{prefix}_target_probability"] = float(
                    probabilities[variant_index, item, target_label].cpu()
                )
            records.append(row)
            attacked = predictions[1, item] == target_label
            recovered = predictions[0, item] == labels[item] and predictions[2, item] == labels[item]
            rank = 3 if attacked and recovered else 2 if attacked and predictions[2, item] != target_label else 1 if attacked else 0
            if rank > example_rank:
                example = {
                    "clean": clean[item].cpu(), "triggered": triggered[item].cpu(),
                    "generator": output.corrected_image[item].cpu(),
                    "static": static_images[item].cpu(),
                    "effective": output.effective_log_correction[item].cpu(),
                    "static_correction": static_correction[item].cpu(),
                    "evidence": evidence[item].cpu(),
                    "predictions": predictions[:, item].cpu().tolist(),
                    "target_probabilities": probabilities[:, item, target_label].cpu().tolist(),
                    "label": int(labels[item]),
                }
                example_rank = rank
    n = totals["samples"]
    if n == 0 or example is None:
        raise ValueError("Trigger scenario produced no evaluation samples")
    clean_rate = totals["clean_target"] / n
    trigger_rate = totals["trigger_target"] / n
    generator_rate = totals["generator_target"] / n
    static_rate = totals["static_target"] / n
    attack_lift = trigger_rate - clean_rate
    return {
        "num_non_target_samples": n,
        "clean_target_rate": clean_rate,
        "triggered_asr": trigger_rate,
        "attack_target_rate_lift": attack_lift,
        "generator_asr": generator_rate,
        "static_template_asr": static_rate,
        "generator_normalized_recovery": normalized_recovery(trigger_rate, generator_rate, clean_rate),
        "static_normalized_recovery": normalized_recovery(trigger_rate, static_rate, clean_rate),
        "generator_advantage_over_static_asr": static_rate - generator_rate,
        "clean_true_label_accuracy": totals["clean_correct"] / n,
        "triggered_true_label_accuracy": totals["trigger_correct"] / n,
        "generator_true_label_accuracy": totals["generator_correct"] / n,
        "static_true_label_accuracy": totals["static_correct"] / n,
        "mean_generator_correction": sums["generator_correction"] / n,
        "mean_generator_support_iou": sums["generator_iou"] / n,
        "mean_static_support_iou": sums["static_iou"] / n,
        "trigger_pixel_l1_to_clean": sums["trigger_l1"] / n,
        "generator_pixel_l1_to_clean": sums["generator_l1"] / n,
        "static_pixel_l1_to_clean": sums["static_l1"] / n,
    }, records, example


@torch.inference_mode()
def evaluate_corruption_scenario(
    generator, classifier, loader, static_template, device, target_label,
    class_names, scenario, seed, max_batches,
):
    totals = defaultdict(int)
    sums = defaultdict(float)
    records = []
    example = None
    noise_generator = torch.Generator(device=device).manual_seed(seed)
    for batch_index, (clean, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        clean = clean.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        corrupted = apply_corruption(clean, scenario, noise_generator)
        output = generator(corrupted)
        static_correction = static_template.unsqueeze(0).expand_as(output.effective_log_correction)
        static_images = reconstruct_with_log_correction(corrupted, static_correction)
        variants = torch.cat((clean, corrupted, output.corrected_image, static_images), dim=0)
        logits = classifier(variants)
        probabilities = logits.softmax(1).view(4, labels.numel(), -1)
        predictions = logits.argmax(1).view(4, labels.numel())
        n = labels.numel()
        totals["samples"] += n
        for index, prefix in enumerate(("clean", "corrupted", "generator", "static")):
            totals[f"{prefix}_correct"] += int((predictions[index] == labels).sum())
            totals[f"{prefix}_target"] += int((predictions[index] == target_label).sum())
        sums["generator_correction"] += float(
            output.effective_log_correction.abs().flatten(1).mean(1).sum().cpu()
        )
        sums["corruption_l1"] += float((corrupted-clean).abs().flatten(1).mean(1).sum().cpu())
        sums["generator_change"] += float((output.corrected_image-corrupted).abs().flatten(1).mean(1).sum().cpu())
        sums["static_change"] += float((static_images-corrupted).abs().flatten(1).mean(1).sum().cpu())

        for item in range(n):
            row = {"sample_index": len(records), "true_label": int(labels[item]), "true_class": class_names[int(labels[item])]}
            for variant_index, prefix in enumerate(("clean", "corrupted", "generator", "static")):
                prediction = int(predictions[variant_index, item])
                row[f"{prefix}_prediction"] = prediction
                row[f"{prefix}_prediction_class"] = class_names[prediction]
                row[f"{prefix}_target_probability"] = float(
                    probabilities[variant_index, item, target_label].cpu()
                )
            records.append(row)
        if example is None:
            item = 0
            example = {
                "clean": clean[item].cpu(), "corrupted": corrupted[item].cpu(),
                "generator": output.corrected_image[item].cpu(),
                "static": static_images[item].cpu(),
                "effective": output.effective_log_correction[item].cpu(),
                "predictions": predictions[:, item].cpu().tolist(),
                "target_probabilities": probabilities[:, item, target_label].cpu().tolist(),
                "label": int(labels[item]),
            }
    n = totals["samples"]
    if n == 0 or example is None:
        raise ValueError("Corruption scenario produced no evaluation samples")
    return {
        "num_samples": n,
        "clean_accuracy": totals["clean_correct"] / n,
        "corrupted_accuracy": totals["corrupted_correct"] / n,
        "generator_accuracy": totals["generator_correct"] / n,
        "static_template_accuracy": totals["static_correct"] / n,
        "clean_target_rate": totals["clean_target"] / n,
        "corrupted_target_rate": totals["corrupted_target"] / n,
        "generator_target_rate": totals["generator_target"] / n,
        "static_template_target_rate": totals["static_target"] / n,
        "mean_generator_correction": sums["generator_correction"] / n,
        "corruption_pixel_l1": sums["corruption_l1"] / n,
        "generator_change_from_corrupted_l1": sums["generator_change"] / n,
        "static_change_from_corrupted_l1": sums["static_change"] / n,
    }, records, example


def apply_corruption(images, scenario, generator):
    kind = scenario["kind"]
    if kind == "noise":
        noise = torch.randn(
            images.shape, generator=generator, device=images.device,
            dtype=images.dtype,
        ) * float(scenario["value"])
        return (images + noise).clamp(0, 1)
    if kind == "blur":
        return TF.gaussian_blur(
            images, kernel_size=int(scenario["kernel"]),
            sigma=float(scenario["sigma"]),
        )
    if kind == "brightness":
        return TF.adjust_brightness(images, float(scenario["value"])).clamp(0, 1)
    if kind == "contrast":
        return TF.adjust_contrast(images, float(scenario["value"])).clamp(0, 1)
    raise ValueError(f"Unknown corruption kind: {kind}")


def support_iou(first, second):
    intersection = (first * second).flatten(1).sum(1)
    union = ((first + second) > 0).float().flatten(1).sum(1).clamp_min(1)
    return intersection / union


def normalized_recovery(trigger_rate, corrected_rate, clean_rate):
    denominator = trigger_rate - clean_rate
    if denominator <= 0:
        return None
    return (trigger_rate - corrected_rate) / denominator


def save_trigger_panel(example, class_names, target_label, scenario_name, path):
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.2))
    predictions = example["predictions"]
    top = [
        (example["clean"], f"clean\npred: {class_names[predictions[0]]}"),
        (example["triggered"], f"triggered\npred: {class_names[predictions[1]]}"),
        (example["generator"], f"generator\npred: {class_names[predictions[2]]}"),
        (example["static"], f"static template\npred: {class_names[predictions[3]]}"),
    ]
    for axis, (image, title) in zip(axes[0], top):
        axis.imshow(to_image(image))
        axis.set_title(title)
        axis.axis("off")
    evidence = shifted_mean(example["evidence"])
    effective = shifted_mean(example["effective"].abs())
    static = shifted_mean(example["static_correction"].abs())
    bottom = [
        ((example["triggered"]-example["clean"]).abs()*8, "trigger-clean |x8|", None),
        (evidence, "|trigger-clean| log amplitude", "inferno"),
        (effective, "generator |correction|", "inferno"),
        (static, "static |correction|", "inferno"),
    ]
    for axis, (image, title, cmap) in zip(axes[1], bottom):
        if image.ndim == 3:
            axis.imshow(to_image(image))
        else:
            maximum = max(float(torch.quantile(image, 0.995)), 1e-8)
            axis.imshow(image, cmap=cmap, vmin=0, vmax=maximum)
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(
        f"Unseen-trigger evaluation: {scenario_name}\n"
        f"true class: {class_names[example['label']]} | target: {class_names[target_label]}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def save_corruption_panel(example, class_names, target_label, scenario_name, path):
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.2))
    predictions = example["predictions"]
    top = [
        (example["clean"], f"clean\npred: {class_names[predictions[0]]}"),
        (example["corrupted"], f"corrupted\npred: {class_names[predictions[1]]}"),
        (example["generator"], f"generator output\npred: {class_names[predictions[2]]}"),
        (example["static"], f"static output\npred: {class_names[predictions[3]]}"),
    ]
    for axis, (image, title) in zip(axes[0], top):
        axis.imshow(to_image(image))
        axis.set_title(title)
        axis.axis("off")
    bottom_images = [
        ((example["corrupted"]-example["clean"]).abs()*8, "corruption-clean |x8|"),
        ((example["generator"]-example["corrupted"]).abs()*8, "generator change |x8|"),
        ((example["static"]-example["corrupted"]).abs()*8, "static change |x8|"),
    ]
    for axis, (image, title) in zip(axes[1, :3], bottom_images):
        axis.imshow(to_image(image))
        axis.set_title(title)
        axis.axis("off")
    bars = axes[1, 3].bar(
        ["clean", "corrupt", "generator", "static"],
        [100*x for x in example["target_probabilities"]],
        color=["#6aaed6", "#dd8452", "#55a868", "#8c8c8c"],
    )
    axes[1, 3].set_title("Target-class probability")
    axes[1, 3].tick_params(axis="x", rotation=25)
    axes[1, 3].grid(axis="y", alpha=0.2)
    for bar in bars:
        axes[1, 3].text(
            bar.get_x()+bar.get_width()/2, bar.get_height(),
            f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7,
        )
    fig.suptitle(
        f"Benign-corruption control: {scenario_name}\n"
        f"true class: {class_names[example['label']]} | target: {class_names[target_label]}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def save_trigger_chart(results, path):
    labels = [row["scenario"].replace("_", "\n") for row in results]
    x = np.arange(len(results))
    width = 0.25
    fig, axis = plt.subplots(figsize=(max(13, len(results)*1.35), 6))
    axis.bar(x-width, [100*r["triggered_asr"] for r in results], width, label="No defense", color="#d95f5f")
    axis.bar(x, [100*r["generator_asr"] for r in results], width, label="Generator", color="#55a868")
    axis.bar(x+width, [100*r["static_template_asr"] for r in results], width, label="Static template", color="#8c8c8c")
    axis.set_ylabel("Target prediction rate / ASR (%)")
    axis.set_title("Frozen-generator generalization across FTrojan configurations")
    axis.set_xticks(x, labels, fontsize=7)
    axis.set_ylim(0, 100)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_corruption_chart(results, path):
    labels = [row["scenario"].replace("_", "\n") for row in results]
    x = np.arange(len(results))
    width = 0.22
    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(results)*1.25), 9), sharex=True)
    axes[0].bar(x-1.5*width, [100*r["clean_accuracy"] for r in results], width, label="Clean", color="#6aaed6")
    axes[0].bar(x-0.5*width, [100*r["corrupted_accuracy"] for r in results], width, label="Corrupted", color="#dd8452")
    axes[0].bar(x+0.5*width, [100*r["generator_accuracy"] for r in results], width, label="Generator", color="#55a868")
    axes[0].bar(x+1.5*width, [100*r["static_template_accuracy"] for r in results], width, label="Static", color="#8c8c8c")
    axes[0].set_ylabel("True-label accuracy (%)")
    axes[0].set_title("Benign-corruption controls")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(ncol=4)
    axes[1].bar(x, [r["mean_generator_correction"] for r in results], color="#8172b3")
    axes[1].set_ylabel("Mean |generator correction|")
    axes[1].set_xticks(x, labels, fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_markdown(summary, path):
    lines = [
        "# DTD Frozen-Generator Generalization Suite", "",
        "No model was trained. All results use validation data and frozen checkpoints.", "",
        "## Trigger configurations", "",
        "| Scenario | Known? | Attack ASR | Generator ASR | Static ASR | Generator accuracy | Static accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["trigger_results"]:
        lines.append(
            f"| {row['scenario']} | {row['known_training_configuration']} | "
            f"{100*row['triggered_asr']:.2f}% | {100*row['generator_asr']:.2f}% | "
            f"{100*row['static_template_asr']:.2f}% | {100*row['generator_true_label_accuracy']:.2f}% | "
            f"{100*row['static_true_label_accuracy']:.2f}% |"
        )
    lines += ["", "## Benign corruptions", "", "| Scenario | Corrupted accuracy | Generator accuracy | Static accuracy | Mean generator correction |", "|---|---:|---:|---:|---:|"]
    for row in summary["corruption_results"]:
        lines.append(
            f"| {row['scenario']} | {100*row['corrupted_accuracy']:.2f}% | "
            f"{100*row['generator_accuracy']:.2f}% | {100*row['static_template_accuracy']:.2f}% | "
            f"{row['mean_generator_correction']:.6f} |"
        )
    lines += [
        "", "## Interpretation", "",
        "Judge defense performance only when the altered trigger still causes a substantial undefended attack. A low-ASR shifted trigger may indicate that the suspicious classifier did not recognize the changed attack, not that the defense succeeded.", "",
        "The static template always corrects the known FTrojan pattern. The image-conditioned generator demonstrates adaptive value only when it handles unseen active attacks better than that fixed template.", "",
        "Benign-corruption results test false correction. A large generator response or accuracy loss on harmless corruption would reveal an important limitation.",
    ]
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")


def save_csv(path, results, kind):
    if not results:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in results for key in row if not isinstance(row[key], (dict, list))})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in results)


def print_tables(trigger_results, corruption_results):
    if trigger_results:
        print("\nscenario | attack ASR | generator ASR | static ASR", flush=True)
        for row in trigger_results:
            print(
                f"{row['scenario']:31s} | {row['triggered_asr']:.4f} | "
                f"{row['generator_asr']:.4f} | {row['static_template_asr']:.4f}",
                flush=True,
            )
    if corruption_results:
        print("\ncorruption | corrupted acc | generator acc | static acc", flush=True)
        for row in corruption_results:
            print(
                f"{row['scenario']:31s} | {row['corrupted_accuracy']:.4f} | "
                f"{row['generator_accuracy']:.4f} | {row['static_template_accuracy']:.4f}",
                flush=True,
            )


def shifted_mean(values):
    return torch.fft.fftshift(values, dim=(-2, -1)).mean(0)


def to_image(image):
    return image.permute(1, 2, 0).clamp(0, 1)


def make_loader(dataset, args, device):
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )


def parse_requested(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def validate_args(args):
    allowed = {"strength", "position", "corruption"}
    requested = set(parse_requested(args.suites))
    if not requested or not requested <= allowed:
        raise ValueError("--suites must contain strength, position, and/or corruption")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Invalid batch size or worker count")
    if not 0 < args.support_fraction < 0.5:
        raise ValueError("--support-fraction must be between zero and 0.5")
    for name in ("max_template_batches", "max_eval_batches"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def prepare_output(args):
    protected = args.output_dir / "generalization_summary.json"
    if protected.exists() and not args.overwrite:
        raise FileExistsError(f"Protected result exists: {protected}")
    args.output_dir.mkdir(parents=True, exist_ok=True)


def resolve_device(requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(requested)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row)+"\n")


if __name__ == "__main__":
    main()
