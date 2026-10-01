"""Final validation-only baselines for the DTD position-generalizing generator.

No model is trained by this script.  It evaluates the frozen suspicious
classifier and generator on every configured FTrojan position, including the
positions held out during generator training.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import ConcatDataset, DataLoader

from datasets import ASBPairedTriggerDataset, filter_manifest_rows, read_jsonl
from evaluation.asb_classifier import evaluation_transform
from scripts.audit_dtd_reference_free_generator import (
    build_models,
    build_static_template,
    log_amplitude,
    reconstruct_with_log_correction,
    resolve_device,
    top_fraction_mask,
)
from scripts.train_dtd_reference_free_generator import (
    partition_rows_by_trigger,
    trigger_from_dict,
)


VARIANT_ORDER = (
    "clean",
    "no_defense",
    "full_generator",
    "static_template",
    "broad_suppression",
    "strongest_support_only",
    "outside_strongest_support",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator-checkpoint", type=Path, required=True)
    parser.add_argument("--classifier-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--manifest-dir", type=Path, default=Path("Absolute_Dataset/asb_dtd_v1")
    )
    parser.add_argument(
        "--images-root", type=Path, default=Path("Absolute_Dataset/dtd/images")
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("Advanced_Outputs/dtd_position_generator_final_baselines_v1"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--support-fraction", type=float, default=0.01)
    parser.add_argument(
        "--broad-cutoff", type=float, default=0.35,
        help="Normalized radial frequency above which broad attenuation begins.",
    )
    parser.add_argument(
        "--broad-attenuation", type=float, default=0.50,
        help="Fraction of amplitude removed in the broad high-frequency region.",
    )
    parser.add_argument("--max-template-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
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
    trigger_dicts = dict(config["trigger_configs"])
    train_names = list(config["train_trigger_names"])
    held_out_names = set(config.get("held_out_trigger_names", []))
    evaluation_names = list(config["evaluation_trigger_names"])
    transform = evaluation_transform(int(config["image_size"]))

    train_rows = filter_manifest_rows(
        rows, protocol_role="clean_train", variant_name="clean"
    )
    validation_rows = filter_manifest_rows(
        rows, protocol_role="clean_calibration", variant_name="clean",
        exclude_original_label=target_label,
    )
    partitions = partition_rows_by_trigger(train_rows, train_names, seed=int(config["seed"]))
    template_dataset = ConcatDataset([
        ASBPairedTriggerDataset(
            args.images_root, partitions[name], transform=transform,
            trigger_config=trigger_from_dict(trigger_dicts[name]),
        )
        for name in train_names
    ])
    template_loader = make_loader(template_dataset, args, device)

    print("Device:", device, flush=True)
    print("Protocol: final validation-only baseline comparison; no training", flush=True)
    print("Building static template from seen-position training sources...", flush=True)
    static_template, template_summary = build_static_template(
        generator, template_loader, device, args.max_template_batches
    )
    print("Static-template samples:", template_summary["num_samples"], flush=True)

    by_trigger = {}
    evidence_manifest = {}
    for index, name in enumerate(evaluation_names, start=1):
        status = "held_out" if name in held_out_names else "seen"
        print(f"Evaluating {index}/{len(evaluation_names)}: {name} ({status})", flush=True)
        dataset = ASBPairedTriggerDataset(
            args.images_root, validation_rows, transform=transform,
            trigger_config=trigger_from_dict(trigger_dicts[name]),
        )
        result, example = evaluate_variants(
            generator, classifier, make_loader(dataset, args, device),
            static_template, target_label, device, args.support_fraction,
            args.broad_cutoff, args.broad_attenuation, args.max_eval_batches,
        )
        by_trigger[name] = result
        by_trigger[name]["generator_status"] = status
        by_trigger[name]["trigger_config"] = trigger_dicts[name]
        panel_path = (
            args.output_dir / "evidence_panels" /
            f"{name}_clean_correct_attack_recovered.png"
        )
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        save_recovery_panel(example, class_names, target_label, name, status, panel_path)
        evidence_manifest[name] = {
            "path": str(panel_path),
            "selection": example["selection"],
            "true_class": class_names[example["label"]],
            "clean_prediction": class_names[example["predictions"]["clean"]],
            "triggered_prediction": class_names[example["predictions"]["no_defense"]],
            "corrected_prediction": class_names[example["predictions"]["full_generator"]],
        }

    summary = {
        "protocol": "validation-only; frozen classifier and generator; no training",
        "partial_evaluation": args.max_eval_batches is not None,
        "generator_checkpoint": str(args.generator_checkpoint),
        "classifier_checkpoint": str(args.classifier_checkpoint),
        "support_fraction": args.support_fraction,
        "broad_suppression": {
            "normalized_radial_cutoff": args.broad_cutoff,
            "amplitude_fraction_removed": args.broad_attenuation,
            "description": (
                "Fixed, image-independent attenuation of every FFT amplitude "
                "above the radial cutoff; phase is preserved."
            ),
        },
        "static_template": template_summary,
        "evidence_panels": evidence_manifest,
        "by_trigger": by_trigger,
        "aggregate_all_positions": aggregate_results(by_trigger),
        "aggregate_seen_positions": aggregate_results({
            name: result for name, result in by_trigger.items()
            if result["generator_status"] == "seen"
        }),
        "aggregate_held_out_positions": aggregate_results({
            name: result for name, result in by_trigger.items()
            if result["generator_status"] == "held_out"
        }),
        "scope": (
            "This comparison tests position generalization within the FTrojan "
            "family on the DTD calibration split. It is not the locked test result."
        ),
    }
    write_json(args.output_dir / "final_baseline_summary.json", summary)
    write_csv(args.output_dir / "final_baseline_table.csv", by_trigger)
    save_chart(summary, args.output_dir / "final_baseline_comparison.png")
    save_markdown(summary, args.output_dir / "README.md")
    print_table(summary)
    print("Saved final baseline comparison:", args.output_dir, flush=True)


@torch.inference_mode()
def evaluate_variants(
    generator, classifier, loader, static_template, target_label, device,
    support_fraction, broad_cutoff, broad_attenuation, max_batches,
):
    counts = {name: {"target": 0, "correct": 0} for name in VARIANT_ORDER}
    distances = {name: 0.0 for name in VARIANT_ORDER if name != "clean"}
    total = 0
    best_example = None
    best_rank = -1
    for batch_index, (clean, triggered, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        clean = clean.to(device, non_blocking=True)
        triggered = triggered.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        correction = generator(triggered).effective_log_correction
        support = top_fraction_mask(correction.abs(), support_fraction)
        broad = broad_spectral_correction(
            triggered, cutoff=broad_cutoff, attenuation=broad_attenuation
        )
        corrections = {
            "full_generator": correction,
            "static_template": static_template.unsqueeze(0).expand_as(correction),
            "broad_suppression": broad,
            "strongest_support_only": correction * support,
            "outside_strongest_support": correction * (1.0 - support),
        }
        images = {"clean": clean, "no_defense": triggered}
        for name, value in corrections.items():
            images[name] = reconstruct_with_log_correction(triggered, value)

        logits = classifier(torch.cat([images[name] for name in VARIANT_ORDER], dim=0))
        predictions = logits.argmax(1).view(len(VARIANT_ORDER), labels.numel())
        probabilities = logits.softmax(1).view(len(VARIANT_ORDER), labels.numel(), -1)
        for variant_index, name in enumerate(VARIANT_ORDER):
            prediction = predictions[variant_index]
            counts[name]["target"] += int((prediction == target_label).sum())
            counts[name]["correct"] += int((prediction == labels).sum())
            if name != "clean":
                distances[name] += float(
                    (images[name] - clean).abs().flatten(1).mean(1).sum().cpu()
                )
        for sample_index in range(labels.numel()):
            sample_predictions = {
                name: int(predictions[variant_index, sample_index])
                for variant_index, name in enumerate(VARIANT_ORDER)
            }
            label = int(labels[sample_index])
            clean_correct = sample_predictions["clean"] == label
            attacked = sample_predictions["no_defense"] == target_label
            recovered = sample_predictions["full_generator"] == label
            if clean_correct and attacked and recovered:
                rank = 3
                selection = "clean_correct__triggered_to_target__generator_recovered_true_label"
            elif clean_correct and attacked and sample_predictions["full_generator"] != target_label:
                rank = 2
                selection = "clean_correct__triggered_to_target__generator_removed_target"
            elif attacked and sample_predictions["full_generator"] != target_label:
                rank = 1
                selection = "triggered_to_target__generator_removed_target"
            else:
                rank = 0
                selection = "fallback_example"
            if rank > best_rank:
                best_example = {
                    "selection": selection,
                    "label": label,
                    "clean": clean[sample_index].detach().cpu(),
                    "triggered": triggered[sample_index].detach().cpu(),
                    "images": {
                        name: value[sample_index].detach().cpu()
                        for name, value in images.items()
                    },
                    "effective_correction": correction[sample_index].detach().cpu(),
                    "predictions": sample_predictions,
                    "target_probabilities": {
                        name: float(probabilities[variant_index, sample_index, target_label].cpu())
                        for variant_index, name in enumerate(VARIANT_ORDER)
                    },
                }
                best_rank = rank
        total += labels.numel()
        if batch_index % 20 == 0:
            print(f"  evaluated {total} samples", flush=True)
    if total == 0:
        raise ValueError("No validation samples were evaluated")
    if best_example is None:
        raise RuntimeError("No example was available for the evidence panel")
    return {
        "num_non_target_samples": total,
        "variants": {
            name: {
                "target_rate": counts[name]["target"] / total,
                "true_label_accuracy": counts[name]["correct"] / total,
                "pixel_l1_to_clean": (
                    0.0 if name == "clean" else distances[name] / total
                ),
            }
            for name in VARIANT_ORDER
        },
    }, best_example


def save_recovery_panel(example, class_names, target_label, trigger_name, status, path):
    clean = example["clean"]
    triggered = example["triggered"]
    images = example["images"]
    predictions = example["predictions"]
    clean_amp = shifted_mean(log_amplitude(clean.unsqueeze(0))[0])
    trigger_amp = shifted_mean(log_amplitude(triggered.unsqueeze(0))[0])
    corrected_amp = shifted_mean(
        log_amplitude(images["full_generator"].unsqueeze(0))[0]
    )
    amp_values = torch.cat((clean_amp.flatten(), trigger_amp.flatten(), corrected_amp.flatten()))
    amp_min = float(torch.quantile(amp_values, 0.01))
    amp_max = float(torch.quantile(amp_values, 0.995))
    applied = shifted_mean(example["effective_correction"].abs())

    figure, axes = plt.subplots(3, 4, figsize=(15, 11))
    true_class = class_names[example["label"]]
    first_row = (
        (clean, f"clean\ntrue/pred: {true_class} / {class_names[predictions['clean']]}"),
        (triggered, f"triggered\npred: {class_names[predictions['no_defense']]}"),
        (images["full_generator"], f"generator corrected\npred: {class_names[predictions['full_generator']]}"),
        ((triggered - clean).abs() * 8, "trigger - clean |x8|"),
    )
    for axis, (image, title) in zip(axes[0], first_row):
        axis.imshow(to_image(image))
        axis.set_title(title)
        axis.axis("off")
    second_row = (
        (clean_amp, "clean log amplitude", "magma", amp_min, amp_max),
        (trigger_amp, "triggered log amplitude", "magma", amp_min, amp_max),
        (corrected_amp, "corrected log amplitude", "magma", amp_min, amp_max),
        (applied, "|generator correction|", "inferno", 0.0,
         max(float(torch.quantile(applied, 0.995)), 1e-8)),
    )
    for axis, (image, title, cmap, low, high) in zip(axes[1], second_row):
        axis.imshow(image, cmap=cmap, vmin=low, vmax=high)
        axis.set_title(title)
        axis.axis("off")
    comparison_names = ("static_template", "broad_suppression", "strongest_support_only")
    for axis, name in zip(axes[2, :3], comparison_names):
        axis.imshow(to_image(images[name]))
        axis.set_title(f"{name.replace('_', ' ')}\npred: {class_names[predictions[name]]}")
        axis.axis("off")
    probability_names = ("no_defense", "full_generator", "static_template", "broad_suppression")
    values = [100 * example["target_probabilities"][name] for name in probability_names]
    bars = axes[2, 3].bar(
        ("trigger", "generator", "static", "broad"), values,
        color=("#b85c5c", "#4f8f78", "#777777", "#9970ab"),
    )
    axes[2, 3].set_title(f"Target probability: {class_names[target_label]}")
    axes[2, 3].set_ylabel("Percent")
    axes[2, 3].grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        axes[2, 3].text(
            bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}",
            ha="center", va="bottom", fontsize=8,
        )
    outcome = example["selection"].replace("__", " -> ").replace("_", " ")
    figure.suptitle(f"{trigger_name} ({status}) | {outcome}", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def shifted_mean(values):
    return torch.fft.fftshift(values, dim=(-2, -1)).mean(0)


def to_image(image):
    return image.permute(1, 2, 0).clamp(0, 1)


def broad_spectral_correction(images, cutoff=0.35, attenuation=0.50):
    """Return log-amplitude correction for a fixed broad high-pass region."""
    height, width = images.shape[-2:]
    fy = torch.fft.fftfreq(height, device=images.device)[:, None]
    fx = torch.fft.fftfreq(width, device=images.device)[None, :]
    radius = torch.sqrt(fx.square() + fy.square()) / (2.0 ** -0.5)
    mask = (radius >= cutoff).to(images.dtype)[None, None]
    amplitude = torch.fft.fft2(images.float(), dim=(-2, -1)).abs()
    retained = amplitude * (1.0 - attenuation * mask)
    return torch.log1p(retained) - torch.log1p(amplitude)


def aggregate_results(results):
    if not results:
        return {"num_triggers": 0, "variants": {}}
    variants = {}
    for variant in VARIANT_ORDER:
        target_rates = [row["variants"][variant]["target_rate"] for row in results.values()]
        accuracies = [row["variants"][variant]["true_label_accuracy"] for row in results.values()]
        distances = [row["variants"][variant]["pixel_l1_to_clean"] for row in results.values()]
        variants[variant] = {
            "mean_target_rate": sum(target_rates) / len(target_rates),
            "worst_target_rate": max(target_rates),
            "mean_true_label_accuracy": sum(accuracies) / len(accuracies),
            "mean_pixel_l1_to_clean": sum(distances) / len(distances),
        }
    return {"num_triggers": len(results), "variants": variants}


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
    if not 0 <= args.broad_cutoff <= 1:
        raise ValueError("--broad-cutoff must be between zero and one")
    if not 0 <= args.broad_attenuation <= 1:
        raise ValueError("--broad-attenuation must be between zero and one")
    for path in (args.generator_checkpoint, args.classifier_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)


def prepare_output(args):
    protected = args.output_dir / "final_baseline_summary.json"
    if protected.exists() and not args.overwrite:
        raise FileExistsError(f"Protected result exists: {protected}")
    args.output_dir.mkdir(parents=True, exist_ok=True)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def write_csv(path, by_trigger):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("trigger", "status", "variant", "target_rate", "accuracy", "pixel_l1"))
        for trigger, result in by_trigger.items():
            for variant in VARIANT_ORDER:
                row = result["variants"][variant]
                writer.writerow((trigger, result["generator_status"], variant,
                                 row["target_rate"], row["true_label_accuracy"],
                                 row["pixel_l1_to_clean"]))


def save_chart(summary, path):
    variants = summary["aggregate_all_positions"]["variants"]
    labels = [name.replace("_", "\n") for name in VARIANT_ORDER]
    target = [100 * variants[name]["mean_target_rate"] for name in VARIANT_ORDER]
    accuracy = [100 * variants[name]["mean_true_label_accuracy"] for name in VARIANT_ORDER]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].bar(labels, target, color="#b85c5c")
    axes[0].set_title("Target prediction rate / ASR")
    axes[0].set_ylabel("Percent")
    axes[1].bar(labels, accuracy, color="#4f8f78")
    axes[1].set_title("True-label accuracy")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", labelsize=8)
    figure.suptitle("DTD final validation baseline comparison (four FTrojan positions)")
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_markdown(summary, path):
    rows = [
        "# DTD Position-Generalizing Generator: Final Baseline Comparison", "",
        "Frozen-model validation comparison; no training was performed.", "",
        "| Variant | Mean target rate / ASR | Worst position | Mean accuracy | Pixel L1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, result in summary["aggregate_all_positions"]["variants"].items():
        rows.append(
            f"| `{name}` | {100*result['mean_target_rate']:.2f}% | "
            f"{100*result['worst_target_rate']:.2f}% | "
            f"{100*result['mean_true_label_accuracy']:.2f}% | "
            f"{result['mean_pixel_l1_to_clean']:.6f} |"
        )
    rows += ["", "## Per-position full-generator result", "",
             "| Trigger | Status | Before ASR | Full-generator ASR | Accuracy |",
             "|---|---|---:|---:|---:|"]
    for name, result in summary["by_trigger"].items():
        before = result["variants"]["no_defense"]
        after = result["variants"]["full_generator"]
        rows.append(
            f"| `{name}` | {result['generator_status']} | "
            f"{100*before['target_rate']:.2f}% | {100*after['target_rate']:.2f}% | "
            f"{100*after['true_label_accuracy']:.2f}% |"
        )
    rows += ["", "## Scope", "", summary["scope"], ""]
    rows += [
        "## Automatically selected recovery evidence", "",
        "Each panel prioritizes a clean-correct image that is forced to the attack target and then restored to its true class.", "",
        "| Trigger | Status | Selection | True | Clean prediction | Triggered prediction | Corrected prediction |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, evidence in summary["evidence_panels"].items():
        status = summary["by_trigger"][name]["generator_status"]
        rows.append(
            f"| `{name}` | {status} | `{evidence['selection']}` | "
            f"{evidence['true_class']} | {evidence['clean_prediction']} | "
            f"{evidence['triggered_prediction']} | {evidence['corrected_prediction']} |"
        )
    rows.append("")
    path.write_text("\n".join(rows), encoding="utf-8")


def print_table(summary):
    print("\nvariant | mean target rate | worst target rate | mean accuracy", flush=True)
    for name, row in summary["aggregate_all_positions"]["variants"].items():
        print(
            f"{name:27s} | {row['mean_target_rate']:.4f} | "
            f"{row['worst_target_rate']:.4f} | {row['mean_true_label_accuracy']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
