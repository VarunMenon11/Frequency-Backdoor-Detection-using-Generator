"""Create an interpretable diagnostic for a trained reference-free generator."""

from __future__ import annotations

import argparse
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
        default=Path("Advanced_Outputs/dtd_reference_free_generator_ftrojan_v1/explanation"),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    generator_checkpoint = torch.load(
        args.generator_checkpoint, map_location="cpu", weights_only=True
    )
    generator_metadata = generator_checkpoint["generator_metadata"]
    generator = ReferenceFreeSpectralGenerator(
        base_channels=int(generator_metadata["base_channels"]),
        max_log_correction=float(generator_metadata["max_log_correction"]),
    )
    generator.load_state_dict(generator_checkpoint["generator_state_dict"], strict=True)
    generator.to(device).eval()

    classifier_checkpoint = torch.load(
        args.classifier_checkpoint, map_location="cpu", weights_only=True
    )
    classifier_config = classifier_checkpoint["run_config"]
    classifier = build_pretrained_classifier(
        backbone=classifier_config["backbone"],
        num_classes=int(classifier_config["num_classes"]),
        weights="none",
    )
    classifier.load_state_dict(classifier_checkpoint["model_state_dict"], strict=True)
    classifier.to(device).eval()

    run_config = generator_checkpoint["run_config"]
    target_label = int(run_config["target_label"])
    class_names = list(run_config["class_names"])
    trigger_config = trigger_from_dict(dict(run_config["trigger_config"]))
    rows = filter_manifest_rows(
        read_jsonl(args.manifest_dir / "variant_manifest.jsonl"),
        protocol_role="clean_calibration", variant_name="clean",
    )
    dataset = ASBPairedTriggerDataset(
        args.images_root, rows,
        transform=evaluation_transform(int(run_config["image_size"])),
        trigger_config=trigger_config,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    example = find_recovered_attack(
        generator, classifier, loader, device, target_label
    )
    clean, triggered, label, clean_pred, trigger_pred, corrected_pred, output = example
    report = build_report(
        clean, triggered, output.corrected_image, output,
        label, clean_pred, trigger_pred, corrected_pred, class_names,
    )
    save_panel(
        clean, triggered, output.corrected_image, output, report,
        args.output_dir / "generator_explanation_panel.png",
    )
    (args.output_dir / "generator_explanation_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (args.output_dir / "README.md").write_text(
        explanation_markdown(report), encoding="utf-8"
    )
    print("Saved:", args.output_dir / "generator_explanation_panel.png")
    print(json.dumps(report, indent=2))


@torch.inference_mode()
def find_recovered_attack(generator, classifier, loader, device, target_label):
    fallback = None
    for clean, triggered, labels in loader:
        clean = clean.to(device)
        triggered = triggered.to(device)
        labels = labels.to(device)
        output = generator(triggered)
        clean_pred = classifier(clean).argmax(1)
        trigger_pred = classifier(triggered).argmax(1)
        corrected_pred = classifier(output.corrected_image).argmax(1)
        for index in range(labels.numel()):
            attacked = labels[index] != target_label and trigger_pred[index] == target_label
            if not attacked:
                continue
            item = (
                clean[index], triggered[index], int(labels[index]),
                int(clean_pred[index]), int(trigger_pred[index]),
                int(corrected_pred[index]), slice_output(output, index),
            )
            if fallback is None and corrected_pred[index] != target_label:
                fallback = item
            if clean_pred[index] == labels[index] and corrected_pred[index] == labels[index]:
                return item
    if fallback is not None:
        return fallback
    raise RuntimeError("No attacked-and-corrected validation example was found")


def slice_output(output, index):
    return type(output)(**{
        field: getattr(output, field)[index]
        for field in output.__dataclass_fields__
    })


def build_report(
    clean, triggered, corrected, output,
    label, clean_pred, trigger_pred, corrected_pred, class_names,
):
    clean_log = log_amplitude(clean)
    trigger_log = log_amplitude(triggered)
    corrected_log = log_amplitude(corrected)
    return {
        "true_class": class_names[label],
        "clean_prediction": class_names[clean_pred],
        "triggered_prediction": class_names[trigger_pred],
        "corrected_prediction": class_names[corrected_pred],
        "pixel_mae_trigger_to_clean": scalar_mae(triggered, clean),
        "pixel_mae_corrected_to_clean": scalar_mae(corrected, clean),
        "log_amplitude_mae_trigger_to_clean": scalar_mae(trigger_log, clean_log),
        "log_amplitude_mae_corrected_to_clean": scalar_mae(corrected_log, clean_log),
        "mean_predicted_mask": float(output.correction_mask.mean().cpu()),
        "mean_absolute_effective_log_correction": float(
            output.effective_log_correction.abs().mean().cpu()
        ),
        "important_note": (
            "Clean data is used only for this evaluation comparison. The generator "
            "produced the corrected image from the triggered image alone."
        ),
    }


def save_panel(clean, triggered, corrected, output, report, path):
    clean_log = shifted_mean_log_amplitude(clean)
    trigger_log = shifted_mean_log_amplitude(triggered)
    corrected_log = shifted_mean_log_amplitude(corrected)
    common = torch.cat((clean_log.flatten(), trigger_log.flatten(), corrected_log.flatten()))
    amplitude_min = float(torch.quantile(common, 0.01))
    amplitude_max = float(torch.quantile(common, 0.995))

    trigger_spectral_difference = (trigger_log - clean_log).abs()
    correction_spectral_difference = corrected_log - trigger_log
    shifted_mask = torch.fft.fftshift(output.correction_mask, dim=(-2, -1)).mean(0)
    shifted_effective = torch.fft.fftshift(
        output.effective_log_correction, dim=(-2, -1)
    ).mean(0)
    effective_absolute = torch.fft.fftshift(
        output.effective_log_correction.abs(), dim=(-2, -1)
    ).mean(0)

    fig, axes = plt.subplots(3, 4, figsize=(15, 11))
    image_entries = [
        (clean, f"clean\npred: {report['clean_prediction']}"),
        (triggered, f"triggered\npred: {report['triggered_prediction']}"),
        (corrected, f"corrected\npred: {report['corrected_prediction']}"),
        ((triggered - clean).abs() * 8.0, "pixel difference |x8|"),
    ]
    for axis, (image, title) in zip(axes[0], image_entries):
        axis.imshow(to_image(image))
        axis.set_title(title)
        axis.axis("off")

    for axis, spectrum, title in zip(
        axes[1], (clean_log, trigger_log, corrected_log, trigger_spectral_difference),
        (
            "clean log amplitude\n(shared scale)",
            "trigger log amplitude\n(shared scale)",
            "corrected log amplitude\n(shared scale)",
            "|trigger - clean| spectrum\n(enhanced)",
        ),
    ):
        if "enhanced" in title:
            maximum = max(float(torch.quantile(spectrum, 0.995)), 1e-8)
            axis.imshow(spectrum.cpu(), cmap="inferno", vmin=0, vmax=maximum)
        else:
            axis.imshow(
                spectrum.cpu(), cmap="magma", vmin=amplitude_min, vmax=amplitude_max
            )
        axis.set_title(title)
        axis.axis("off")

    signed_max = max(
        float(torch.quantile(correction_spectral_difference.abs(), 0.995)), 1e-8
    )
    effective_max = max(float(torch.quantile(effective_absolute, 0.995)), 1e-8)
    diagnostic_entries = [
        (shifted_mask, "predicted gate M", "viridis", 0.0, 1.0),
        (shifted_effective, "generator signed correction", "coolwarm", -effective_max, effective_max),
        (effective_absolute, "|effective correction|", "inferno", 0.0, effective_max),
        (correction_spectral_difference, "corrected - trigger spectrum", "coolwarm", -signed_max, signed_max),
    ]
    for axis, (image, title, cmap, minimum, maximum) in zip(axes[2], diagnostic_entries):
        axis.imshow(image.cpu(), cmap=cmap, vmin=minimum, vmax=maximum)
        axis.set_title(title)
        axis.axis("off")

    fig.suptitle(
        "How the reference-free generator corrected one successful FTrojan attack\n"
        f"true class: {report['true_class']} | generator input: triggered image only",
        fontsize=13,
    )
    fig.text(
        0.5, 0.015,
        f"Pixel MAE trigger->clean {report['pixel_mae_trigger_to_clean']:.5f}; "
        f"corrected->clean {report['pixel_mae_corrected_to_clean']:.5f} | "
        f"Log-amplitude MAE trigger->clean {report['log_amplitude_mae_trigger_to_clean']:.5f}; "
        f"corrected->clean {report['log_amplitude_mae_corrected_to_clean']:.5f}",
        ha="center", fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def explanation_markdown(report):
    return f"""# Reference-Free Generator: One-Sample Explanation

The sample's true class is **{report['true_class']}**. The suspicious model
predicts **{report['clean_prediction']}** for the clean image, changes to the
attacker target **{report['triggered_prediction']}** after FTrojan injection,
and returns to **{report['corrected_prediction']}** after generator correction.

The three amplitude panels deliberately share one color scale. They look very
similar because natural image energy dominates the spectrum and the trigger is
small relative to that energy. The enhanced `|trigger-clean|` panel isolates
the otherwise hidden spectral change. `predicted gate M` shows where correction
is permitted; `generator signed correction` shows direction; and `|effective
correction|` shows what was actually applied.

- Pixel MAE, trigger to clean: {report['pixel_mae_trigger_to_clean']:.6f}
- Pixel MAE, corrected to clean: {report['pixel_mae_corrected_to_clean']:.6f}
- Log-amplitude MAE, trigger to clean: {report['log_amplitude_mae_trigger_to_clean']:.6f}
- Log-amplitude MAE, corrected to clean: {report['log_amplitude_mae_corrected_to_clean']:.6f}
- Mean predicted mask: {report['mean_predicted_mask']:.6f}
- Mean absolute effective log correction: {report['mean_absolute_effective_log_correction']:.6f}

**Protocol boundary:** {report['important_note']}
"""


def log_amplitude(image):
    return torch.log1p(torch.fft.fft2(image.float(), dim=(-2, -1)).abs())


def shifted_mean_log_amplitude(image):
    return torch.fft.fftshift(log_amplitude(image), dim=(-2, -1)).mean(0)


def scalar_mae(first, second):
    return float((first - second).abs().mean().cpu())


def to_image(image):
    return image.detach().cpu().permute(1, 2, 0).clamp(0, 1)


def resolve_device(requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(requested)


if __name__ == "__main__":
    main()
