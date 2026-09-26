"""Faithfulness audit for the trained DTD reference-free spectral generator.

This is a validation-only, no-training experiment. Clean counterparts define
evaluator-only trigger evidence and are never passed to the generator.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import ASBPairedTriggerDataset, filter_manifest_rows, read_jsonl
from evaluation.asb_classifier import evaluation_transform
from generator import ReferenceFreeSpectralGenerator
from models import build_pretrained_classifier
from scripts.train_dtd_reference_free_generator import trigger_from_dict


VARIANT_ORDER = [
    "clean",
    "triggered",
    "full_generator",
    "predicted_top_only",
    "predicted_outside_only",
    "oracle_overlap_only",
    "oracle_outside_only",
    "static_template",
]


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
        default=Path("Advanced_Outputs/dtd_reference_free_generator_faithfulness_v1"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--support-fraction", type=float, default=0.01,
        help="Top fraction used for predicted and evaluator-only trigger supports.",
    )
    parser.add_argument("--max-template-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    prepare_output(args)
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
    trigger_config = trigger_from_dict(dict(config["trigger_config"]))
    transform = evaluation_transform(int(config["image_size"]))

    train_rows = filter_manifest_rows(
        rows, protocol_role="clean_train", variant_name="clean",
        exclude_original_label=target_label,
    )
    validation_rows = filter_manifest_rows(
        rows, protocol_role="clean_calibration", variant_name="clean",
        exclude_original_label=target_label,
    )
    template_loader = make_loader(
        ASBPairedTriggerDataset(
            args.images_root, train_rows, transform=transform,
            trigger_config=trigger_config,
        ), args, device,
    )
    validation_loader = make_loader(
        ASBPairedTriggerDataset(
            args.images_root, validation_rows, transform=transform,
            trigger_config=trigger_config,
        ), args, device,
    )

    print("Device:", device, flush=True)
    print("Protocol: validation-only faithfulness audit; no training", flush=True)
    print("Building a static correction template from clean-train sources...", flush=True)
    static_template, template_summary = build_static_template(
        generator, template_loader, device, args.max_template_batches
    )
    print("Template samples:", template_summary["num_samples"], flush=True)
    print("Evaluating causal correction variants...", flush=True)
    summary, records, example = evaluate_audit(
        generator, classifier, validation_loader, static_template, device,
        target_label, class_names, args.support_fraction, args.max_eval_batches,
    )
    summary.update({
        "protocol": "validation-only; no generator or classifier training",
        "partial_evaluation": args.max_eval_batches is not None,
        "generator_checkpoint": str(args.generator_checkpoint),
        "generator_epoch": int(generator_checkpoint["epoch"]),
        "classifier_checkpoint": str(args.classifier_checkpoint),
        "classifier_epoch": int(classifier_checkpoint["epoch"]),
        "support_fraction": args.support_fraction,
        "template": template_summary,
        "trigger_config": config["trigger_config"],
        "important_limit": (
            "Oracle trigger support uses paired clean-trigger differences for "
            "evaluation only. It is unavailable to deployed inference."
        ),
    })

    write_json(args.output_dir / "faithfulness_summary.json", summary)
    write_jsonl(args.output_dir / "per_sample_faithfulness.jsonl", records)
    torch.save(
        {"effective_log_correction_template": static_template.cpu(), "summary": template_summary},
        args.output_dir / "static_correction_template.pt",
    )
    save_ablation_chart(summary, args.output_dir / "asr_ablation_chart.png")
    save_distribution_chart(records, args.output_dir / "faithfulness_distributions.png")
    save_explanation_panel(
        example, class_names, args.support_fraction,
        args.output_dir / "faithfulness_explanation_panel.png",
    )
    save_markdown(summary, args.output_dir / "README.md")
    print_summary(summary)
    print("Saved audit:", args.output_dir, flush=True)


def build_models(generator_checkpoint, classifier_checkpoint, device):
    generator_metadata = generator_checkpoint["generator_metadata"]
    generator = ReferenceFreeSpectralGenerator(
        base_channels=int(generator_metadata["base_channels"]),
        max_log_correction=float(generator_metadata["max_log_correction"]),
    )
    generator.load_state_dict(generator_checkpoint["generator_state_dict"], strict=True)
    generator.to(device).eval()

    classifier_config = classifier_checkpoint["run_config"]
    generator_config = generator_checkpoint["run_config"]
    if classifier_config["class_names"] != generator_config["class_names"]:
        raise ValueError("Generator and classifier class names do not match")
    if int(classifier_config["target_label"]) != int(generator_config["target_label"]):
        raise ValueError("Generator and classifier target labels do not match")
    classifier = build_pretrained_classifier(
        backbone=classifier_config["backbone"],
        num_classes=int(classifier_config["num_classes"]), weights="none",
    )
    classifier.load_state_dict(classifier_checkpoint["model_state_dict"], strict=True)
    classifier.to(device).eval()
    return generator, classifier, generator_config


@torch.inference_mode()
def build_static_template(generator, loader, device, max_batches):
    correction_sum = None
    total = 0
    for batch_index, (_, triggered, _) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        triggered = triggered.to(device, non_blocking=True)
        correction = generator(triggered).effective_log_correction
        batch_sum = correction.sum(dim=0)
        correction_sum = batch_sum if correction_sum is None else correction_sum + batch_sum
        total += correction.shape[0]
    if total == 0:
        raise ValueError("No samples were available to build the static template")
    template = make_conjugate_symmetric(correction_sum / total)
    return template, {
        "num_samples": total,
        "mean_absolute_value": float(template.abs().mean().cpu()),
        "maximum_absolute_value": float(template.abs().max().cpu()),
        "source_split": "clean_train with evaluation crop and configured trigger",
    }


@torch.inference_mode()
def evaluate_audit(
    generator, classifier, loader, static_template, device, target_label,
    class_names, support_fraction, max_batches,
):
    counts = {name: {"target": 0, "correct": 0} for name in VARIANT_ORDER}
    distance_sums = {name: 0.0 for name in VARIANT_ORDER if name != "clean"}
    records = []
    example = None
    example_rank = -1
    total = 0

    for batch_index, (clean, triggered, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        clean = clean.to(device, non_blocking=True)
        triggered = triggered.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        output = generator(triggered)
        clean_output = generator(clean)
        trigger_evidence = (
            log_amplitude(triggered) - log_amplitude(clean)
        ).abs()
        predicted_support = top_fraction_mask(
            output.effective_log_correction.abs(), support_fraction
        )
        oracle_support = top_fraction_mask(trigger_evidence, support_fraction)

        corrections = {
            "full_generator": output.effective_log_correction,
            "predicted_top_only": output.effective_log_correction * predicted_support,
            "predicted_outside_only": output.effective_log_correction * (1.0 - predicted_support),
            "oracle_overlap_only": output.effective_log_correction * oracle_support,
            "oracle_outside_only": output.effective_log_correction * (1.0 - oracle_support),
            "static_template": static_template.unsqueeze(0).expand_as(output.effective_log_correction),
        }
        images = {"clean": clean, "triggered": triggered}
        for name, correction in corrections.items():
            images[name] = reconstruct_with_log_correction(triggered, correction)

        combined = torch.cat([images[name] for name in VARIANT_ORDER], dim=0)
        logits = classifier(combined)
        predictions = logits.argmax(1).view(len(VARIANT_ORDER), labels.numel())
        probabilities = logits.softmax(1).view(len(VARIANT_ORDER), labels.numel(), -1)

        for variant_index, name in enumerate(VARIANT_ORDER):
            prediction = predictions[variant_index]
            counts[name]["target"] += int((prediction == target_label).sum())
            counts[name]["correct"] += int((prediction == labels).sum())
            if name != "clean":
                distance_sums[name] += float(
                    (images[name] - clean).abs().flatten(1).mean(1).sum().cpu()
                )

        batch_records = batch_faithfulness_records(
            output.effective_log_correction, clean_output.effective_log_correction,
            trigger_evidence, predicted_support, oracle_support,
        )
        for index, metrics in enumerate(batch_records):
            row = {
                "sample_index": total + index,
                "true_label": int(labels[index]),
                "true_class": class_names[int(labels[index])],
                **metrics,
            }
            for variant_index, name in enumerate(VARIANT_ORDER):
                prediction = int(predictions[variant_index, index])
                row[f"{name}_prediction"] = prediction
                row[f"{name}_prediction_class"] = class_names[prediction]
                row[f"{name}_target_probability"] = float(
                    probabilities[variant_index, index, target_label].cpu()
                )
            records.append(row)

            attacked = predictions[1, index] == target_label
            recovered = (
                predictions[0, index] == labels[index]
                and attacked
                and predictions[2, index] == labels[index]
            )
            if recovered:
                rank = 3
            elif attacked and predictions[2, index] != target_label:
                rank = 2
            elif attacked:
                rank = 1
            else:
                rank = 0
            if rank > example_rank:
                example = build_example(
                    clean[index], triggered[index], images, output,
                    trigger_evidence, predicted_support, oracle_support,
                    predictions[:, index], probabilities[:, index, target_label],
                    labels[index], index,
                )
                example_rank = rank
        total += labels.numel()
        if batch_index % 10 == 0:
            print(f"Evaluated {total} validation samples", flush=True)

    if total == 0:
        raise ValueError("No validation samples were evaluated")
    if example is None:
        raise RuntimeError("No example was available for the explanation panel")

    variants = {}
    for name in VARIANT_ORDER:
        variants[name] = {
            "target_rate": counts[name]["target"] / total,
            "target_count": counts[name]["target"],
            "true_label_accuracy": counts[name]["correct"] / total,
            "true_label_correct_count": counts[name]["correct"],
            "pixel_l1_to_clean": 0.0 if name == "clean" else distance_sums[name] / total,
        }
    aggregate = aggregate_faithfulness(records)
    return {
        "num_non_target_validation_samples": total,
        "variants": variants,
        "faithfulness": aggregate,
        "interpretation_rules": {
            "predicted_top_only": "Only the largest generator corrections are retained.",
            "predicted_outside_only": "The largest generator corrections are removed; the complement remains.",
            "oracle_overlap_only": "Generator correction only where paired clean-trigger evidence is strongest; evaluator-only.",
            "oracle_outside_only": "Generator correction outside paired clean-trigger evidence; evaluator-only.",
            "static_template": "Mean signed generator correction from triggered clean-train sources, applied identically to every validation image.",
        },
    }, records, example


def batch_faithfulness_records(
    effective, clean_effective, trigger_evidence, predicted_support, oracle_support,
):
    batch = effective.shape[0]
    effective_abs = effective.abs().flatten(1)
    clean_abs = clean_effective.abs().flatten(1)
    evidence = trigger_evidence.flatten(1)
    predicted = predicted_support.flatten(1)
    oracle = oracle_support.flatten(1)
    overlap = predicted * oracle
    union = ((predicted + oracle) > 0).float()
    cosine = torch.nn.functional.cosine_similarity(effective_abs, evidence, dim=1)
    records = []
    for index in range(batch):
        effective_total = effective_abs[index].sum().clamp_min(1e-12)
        evidence_total = evidence[index].sum().clamp_min(1e-12)
        records.append({
            "triggered_correction_mean_abs": float(effective_abs[index].mean().cpu()),
            "clean_correction_mean_abs": float(clean_abs[index].mean().cpu()),
            "trigger_to_clean_correction_ratio": float(
                (effective_abs[index].mean() / clean_abs[index].mean().clamp_min(1e-12)).cpu()
            ),
            "paired_effective_correction_difference": float(
                (effective[index] - clean_effective[index]).abs().mean().cpu()
            ),
            "support_iou": float(
                (overlap[index].sum() / union[index].sum().clamp_min(1.0)).cpu()
            ),
            "correction_energy_in_oracle_support": float(
                ((effective_abs[index] * oracle[index]).sum() / effective_total).cpu()
            ),
            "trigger_evidence_in_predicted_support": float(
                ((evidence[index] * predicted[index]).sum() / evidence_total).cpu()
            ),
            "magnitude_cosine_similarity": float(cosine[index].cpu()),
        })
    return records


def aggregate_faithfulness(records):
    names = [
        "triggered_correction_mean_abs", "clean_correction_mean_abs",
        "trigger_to_clean_correction_ratio", "paired_effective_correction_difference",
        "support_iou", "correction_energy_in_oracle_support",
        "trigger_evidence_in_predicted_support", "magnitude_cosine_similarity",
    ]
    result = {}
    for name in names:
        values = np.asarray([row[name] for row in records], dtype=np.float64)
        result[name] = {
            "mean": float(values.mean()), "median": float(np.median(values)),
            "std": float(values.std()),
            "p05": float(np.quantile(values, 0.05)),
            "p95": float(np.quantile(values, 0.95)),
        }
    return result


def build_example(
    clean, triggered, images, output, evidence, predicted_support, oracle_support,
    predictions, target_probabilities, label, index,
):
    return {
        "clean": clean.detach().cpu(),
        "triggered": triggered.detach().cpu(),
        "images": {name: image[index].detach().cpu() for name, image in images.items()},
        "gate": output.correction_mask[index].detach().cpu(),
        "effective": output.effective_log_correction[index].detach().cpu(),
        "evidence": evidence[index].detach().cpu(),
        "predicted_support": predicted_support[index].detach().cpu(),
        "oracle_support": oracle_support[index].detach().cpu(),
        "predictions": predictions.detach().cpu().tolist(),
        "target_probabilities": target_probabilities.detach().cpu().tolist(),
        "label": int(label),
    }


def top_fraction_mask(values, fraction):
    """Return an approximate top-fraction binary mask for each batch item."""
    if values.ndim != 4:
        raise ValueError("Expected values with shape (B,C,H,W)")
    flat = values.flatten(1)
    count = max(1, round(flat.shape[1] * fraction))
    threshold = torch.topk(flat, k=count, dim=1, sorted=False).values.min(1).values
    return (values >= threshold[:, None, None, None]).to(values.dtype)


def reconstruct_with_log_correction(images, correction):
    spectrum = torch.fft.fft2(images.float(), dim=(-2, -1))
    log_amp = torch.log1p(spectrum.abs())
    phase = torch.angle(spectrum)
    correction = make_conjugate_symmetric(correction)
    corrected_amp = torch.expm1((log_amp + correction).clamp_min(0.0))
    return torch.fft.ifft2(torch.polar(corrected_amp, phase), dim=(-2, -1)).real.clamp(0, 1)


def make_conjugate_symmetric(values):
    counterpart = torch.roll(
        torch.flip(values, dims=(-2, -1)), shifts=(1, 1), dims=(-2, -1)
    )
    return 0.5 * (values + counterpart)


def log_amplitude(images):
    return torch.log1p(torch.fft.fft2(images.float(), dim=(-2, -1)).abs())


def save_ablation_chart(summary, path):
    variants = summary["variants"]
    support_percent = 100 * summary["support_fraction"]
    labels = [
        "Clean", "Triggered", "Full\ngenerator", f"Top {support_percent:g}%\nonly",
        f"Outside top\n{support_percent:g}%", "Oracle-overlap\nonly",
        "Oracle-outside\nonly", "Static\ntemplate",
    ]
    values = [100 * variants[name]["target_rate"] for name in VARIANT_ORDER]
    colors = ["#6aaed6", "#d95f5f", "#55a868", "#8172b3", "#c49a6c", "#64b5a7", "#dd8452", "#8c8c8c"]
    fig, axis = plt.subplots(figsize=(12.5, 5.5))
    bars = axis.bar(labels, values, color=colors)
    axis.axhline(values[0], color="#275d8c", linestyle="--", label="clean target-rate baseline")
    axis.set_ylabel("Target prediction rate / ASR (%)")
    axis.set_title("Causal correction ablation on non-target DTD validation images")
    axis.set_ylim(0, max(values) * 1.15)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_distribution_chart(records, path):
    trigger = [row["triggered_correction_mean_abs"] for row in records]
    clean = [row["clean_correction_mean_abs"] for row in records]
    iou = [row["support_iou"] for row in records]
    energy = [row["correction_energy_in_oracle_support"] for row in records]
    cosine = [row["magnitude_cosine_similarity"] for row in records]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].boxplot([clean, trigger], showfliers=False)
    axes[0].set_xticks([1, 2], ["clean", "triggered"])
    axes[0].set_title("Correction response")
    axes[0].set_ylabel("Mean |effective log correction|")
    axes[1].hist(iou, bins=30, alpha=0.75, label="top-support IoU")
    axes[1].hist(energy, bins=30, alpha=0.65, label="correction energy in oracle support")
    axes[1].set_title("Localization measurements")
    axes[1].legend(fontsize=8)
    axes[2].hist(cosine, bins=30, color="#8172b3")
    axes[2].set_title("Correction/evidence magnitude similarity")
    axes[2].set_xlabel("Cosine similarity")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_explanation_panel(example, class_names, support_fraction, path):
    clean = example["clean"]
    triggered = example["triggered"]
    images = example["images"]
    gate = shifted_mean(example["gate"])
    effective = shifted_mean(example["effective"].abs())
    evidence = shifted_mean(example["evidence"])
    predicted_support = shifted_any(example["predicted_support"])
    oracle_support = shifted_any(example["oracle_support"])
    overlap_rgb = torch.zeros(oracle_support.shape[0], oracle_support.shape[1], 3)
    overlap_rgb[..., 0] = oracle_support
    overlap_rgb[..., 1] = predicted_support
    predictions = example["predictions"]
    target_probabilities = example["target_probabilities"]

    clean_log = shifted_mean(log_amplitude(clean.unsqueeze(0))[0])
    trigger_log = shifted_mean(log_amplitude(triggered.unsqueeze(0))[0])
    corrected_log = shifted_mean(log_amplitude(images["full_generator"].unsqueeze(0))[0])
    combined = torch.cat((clean_log.flatten(), trigger_log.flatten(), corrected_log.flatten()))
    amp_min, amp_max = float(torch.quantile(combined, 0.01)), float(torch.quantile(combined, 0.995))

    fig, axes = plt.subplots(4, 4, figsize=(15, 14))
    first_row = [
        (clean, f"clean\npred: {class_names[predictions[0]]}"),
        (triggered, f"triggered\npred: {class_names[predictions[1]]}"),
        (images["full_generator"], f"full correction\npred: {class_names[predictions[2]]}"),
        ((triggered - clean).abs() * 8, "trigger - clean |x8|"),
    ]
    for axis, (image, title) in zip(axes[0], first_row):
        axis.imshow(to_image(image))
        axis.set_title(title)
        axis.axis("off")
    for axis, image, title in zip(
        axes[1], (clean_log, trigger_log, corrected_log, evidence),
        ("clean log amplitude", "trigger log amplitude", "corrected log amplitude", "|trigger-clean| evidence"),
    ):
        if "evidence" in title:
            maximum = max(float(torch.quantile(image, 0.995)), 1e-8)
            axis.imshow(image, cmap="inferno", vmin=0, vmax=maximum)
        else:
            axis.imshow(image, cmap="magma", vmin=amp_min, vmax=amp_max)
        axis.set_title(title)
        axis.axis("off")
    third_row = [
        (oracle_support, f"oracle top {100*support_fraction:.1f}% support", "gray"),
        (gate, "predicted gate", "viridis"),
        (effective, "|effective correction|", "inferno"),
        (overlap_rgb, "support overlap\nred=oracle, green=predicted", None),
    ]
    for axis, (image, title, cmap) in zip(axes[2], third_row):
        axis.imshow(image, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")
    fourth_names = ["predicted_top_only", "predicted_outside_only", "static_template"]
    for axis, name, variant_index in zip(axes[3, :3], fourth_names, (3, 4, 7)):
        axis.imshow(to_image(images[name]))
        axis.set_title(
            f"{name.replace('_', ' ')}\npred: {class_names[predictions[variant_index]]}"
        )
        axis.axis("off")
    bars = axes[3, 3].bar(
        ["trigger", "full", "top", "outside", "static"],
        [100 * target_probabilities[index] for index in (1, 2, 3, 4, 7)],
        color=["#d95f5f", "#55a868", "#8172b3", "#c49a6c", "#8c8c8c"],
    )
    axes[3, 3].set_title("Target-class probability")
    axes[3, 3].set_ylabel("Probability (%)")
    axes[3, 3].tick_params(axis="x", rotation=30)
    axes[3, 3].grid(axis="y", alpha=0.2)
    for bar in bars:
        axes[3, 3].text(
            bar.get_x() + bar.get_width()/2, bar.get_height(),
            f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7,
        )
    fig.suptitle(
        f"Reference-free generator faithfulness audit | true class: {class_names[example['label']]}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_markdown(summary, path):
    variants = summary["variants"]
    faith = summary["faithfulness"]
    lines = [
        "# DTD Reference-Free Generator Faithfulness Audit", "",
        "This is a validation-only, no-training audit. Clean counterparts are used only to define evaluator-side trigger evidence.", "",
        "## Causal ablation results", "",
        "| Variant | Target rate / ASR | True-label accuracy | Pixel L1 to clean |", "|---|---:|---:|---:|",
    ]
    for name in VARIANT_ORDER:
        row = variants[name]
        lines.append(
            f"| {name} | {100*row['target_rate']:.2f}% | {100*row['true_label_accuracy']:.2f}% | {row['pixel_l1_to_clean']:.6f} |"
        )
    lines += [
        "", "## Aggregate map measurements", "",
        f"- Triggered correction magnitude, median: {faith['triggered_correction_mean_abs']['median']:.6f}",
        f"- Clean correction magnitude, median: {faith['clean_correction_mean_abs']['median']:.6f}",
        f"- Trigger/clean correction ratio, median: {faith['trigger_to_clean_correction_ratio']['median']:.3f}",
        f"- Top-support IoU, median: {faith['support_iou']['median']:.4f}",
        f"- Correction energy in evaluator trigger support, median: {faith['correction_energy_in_oracle_support']['median']:.4f}",
        f"- Trigger evidence captured by predicted support, median: {faith['trigger_evidence_in_predicted_support']['median']:.4f}",
        f"- Magnitude cosine similarity, median: {faith['magnitude_cosine_similarity']['median']:.4f}",
        "", "## Interpretation boundary", "",
        "The full generator result establishes mitigation. Localization metrics and ablations test whether the visual map is faithful. The static-template comparison tests whether image conditioning adds value over a fixed filter. No single overlap score proves malicious intent or exact physical trigger segmentation.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(summary):
    print("\nvariant | target rate | true-label accuracy", flush=True)
    for name in VARIANT_ORDER:
        row = summary["variants"][name]
        print(f"{name:24s} | {row['target_rate']:.4f} | {row['true_label_accuracy']:.4f}", flush=True)


def shifted_mean(values):
    return torch.fft.fftshift(values, dim=(-2, -1)).mean(0)


def shifted_any(values):
    return torch.fft.fftshift(values, dim=(-2, -1)).amax(0)


def to_image(image):
    return image.permute(1, 2, 0).clamp(0, 1)


def make_loader(dataset, args, device):
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )


def validate_args(args):
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Invalid batch size or worker count")
    if not 0 < args.support_fraction < 0.5:
        raise ValueError("--support-fraction must be between zero and 0.5")
    for name in ("max_template_batches", "max_eval_batches"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def prepare_output(args):
    protected = args.output_dir / "faithfulness_summary.json"
    if protected.exists() and not args.overwrite:
        raise FileExistsError(f"Protected audit exists: {protected}")
    args.output_dir.mkdir(parents=True, exist_ok=True)


def resolve_device(requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(requested)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
