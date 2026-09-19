"""Create review-ready clean/triggered prediction evidence for a DTD checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from datasets import ASBManifestDataset, filter_manifest_rows, read_jsonl
from evaluation.asb_classifier import evaluation_transform, paired_metrics
from models import build_pretrained_classifier
from scripts.train_asb_dtd_pretrained_classifier import class_names_from_rows, resolve_device


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path(
            "Advanced_Experiments/dtd_attack_sweep_v1/"
            "ftrojan_m100_paired_r020/suspicious_classifier_best_attack.pt"
        ),
    )
    parser.add_argument("--manifest-dir", type=Path, default=Path("Absolute_Dataset/asb_dtd_v1"))
    parser.add_argument("--images-root", type=Path, default=Path("Absolute_Dataset/dtd/images"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("Advanced_Outputs/Testing_model/dtd_ftrojan_m100_paired_r020"),
    )
    parser.add_argument("--num-success", type=int, default=6)
    parser.add_argument("--num-resistant", type=int, default=2)
    parser.add_argument("--num-other", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--difference-gain", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    validate_args(args)
    prepare_output(args.output_dir, args.overwrite)
    torch.set_num_threads(args.cpu_threads)
    device = resolve_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["run_config"]
    rows = read_jsonl(args.manifest_dir / "variant_manifest.jsonl")
    class_names = class_names_from_rows(rows)
    target_label = int(config["target_label"])
    target_class = class_names[target_label]
    trigger_names = list(config["attack_triggers"])
    if len(trigger_names) != 1:
        raise ValueError("This visualizer requires a checkpoint with exactly one attack trigger")
    trigger_name = trigger_names[0]
    trigger_config = dict(config["attack_trigger_configs"][trigger_name])

    clean_rows = filter_manifest_rows(
        rows,
        protocol_role="final_test_clean",
        variant_name="clean",
        exclude_original_label=target_label,
    )
    trigger_rows = [
        {
            **row,
            "variant_name": trigger_name,
            "variant_type": "trigger",
            "trigger_config": trigger_config,
        }
        for row in clean_rows
    ]
    transform = evaluation_transform(int(config["image_size"]))
    model = build_pretrained_classifier(
        backbone=config["backbone"], num_classes=config["num_classes"], weights="none"
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()

    print(f"Checkpoint epoch: {checkpoint['epoch']} | device: {device}", flush=True)
    clean_predictions = predict_with_confidence(
        model, clean_rows, args.images_root, transform, args, device, "clean"
    )
    trigger_predictions = predict_with_confidence(
        model, trigger_rows, args.images_root, transform, args, device, "triggered"
    )
    records = pair_records(clean_rows, clean_predictions, trigger_predictions, target_label)
    save_prediction_csv(records, class_names, args.output_dir / "all_test_predictions.csv")

    metric = paired_metrics(
        [prediction_record(row, "clean_prediction") for row in records],
        [prediction_record(row, "trigger_prediction") for row in records],
        target_label,
    )
    selected = select_representatives(records, args)
    generated = []
    row_by_id = {str(row["source_id"]): row for row in clean_rows}
    for category, category_records in selected.items():
        for number, record in enumerate(category_records, start=1):
            clean_row = row_by_id[record["source_id"]]
            trigger_row = {
                **clean_row,
                "variant_name": trigger_name,
                "variant_type": "trigger",
                "trigger_config": trigger_config,
            }
            clean_image = load_one(args.images_root, clean_row, transform)
            triggered_image = load_one(args.images_root, trigger_row, transform)
            sample_name = (
                f"sample_{number:02d}_{slug(class_names[record['original_label']])}_"
                f"{slug(record['source_id'])}"
            )
            sample_dir = args.output_dir / category / sample_name
            sample_dir.mkdir(parents=True, exist_ok=True)
            save_sample(
                sample_dir, clean_image, triggered_image, record, class_names,
                target_label, trigger_name, trigger_config, args.difference_gain,
            )
            generated.append({"category": category, "directory": str(sample_dir), **record})

    save_overview(
        generated, row_by_id, args.images_root, transform, class_names,
        trigger_name, trigger_config, args.difference_gain,
        args.output_dir / "model_prediction_overview.png",
    )
    summary = {
        "experiment": "dtd_poisoned_model_prediction_evidence_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_epoch": checkpoint["epoch"],
        "split": "DTD final_test_clean, excluding target-class sources",
        "num_evaluated_sources": len(records),
        "model": checkpoint["model_metadata"],
        "target_label": target_label,
        "target_class": target_class,
        "trigger_name": trigger_name,
        "trigger_config": trigger_config,
        "paired_metrics": metric,
        "selection_counts": {key: len(value) for key, value in selected.items()},
        "selected_samples": generated,
        "interpretation": (
            "successful_attack requires a correct clean prediction that changes to the "
            "attacker target after triggering; resisted_attack remains correctly classified; "
            "other_trigger_error changes from a correct clean prediction to a wrong non-target class."
        ),
    }
    write_json(args.output_dir / "model_testing_summary.json", summary)
    (args.output_dir / "README.md").write_text(
        markdown_summary(summary), encoding="utf-8"
    )
    print(f"Clean accuracy (non-target): {metric['clean_non_target_accuracy']:.4f}")
    print(f"Triggered ASR: {metric['asr']:.4f}")
    print(f"Conditional ASR: {metric['conditional_asr_clean_correct']:.4f}")
    print("Saved:", args.output_dir)


@torch.inference_mode()
def predict_with_confidence(model, rows, images_root, transform, args, device, label):
    dataset = ASBManifestDataset(
        images_root, rows, transform=transform, label_field="original_label"
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    output = []
    seen = 0
    for images, labels in loader:
        probabilities = model(images.to(device, non_blocking=True)).softmax(dim=1).cpu()
        confidence, prediction = probabilities.max(dim=1)
        for predicted, score in zip(prediction.tolist(), confidence.tolist()):
            output.append({"prediction": int(predicted), "confidence": float(score)})
        seen += len(labels)
        if seen == len(rows) or seen % (args.batch_size * 10) == 0:
            print(f"{label}: {seen}/{len(rows)}", flush=True)
    return output


def pair_records(rows, clean, triggered, target_label):
    if not (len(rows) == len(clean) == len(triggered)):
        raise ValueError("Prediction lengths do not match")
    records = []
    for row, before, after in zip(rows, clean, triggered):
        original = int(row["original_label"])
        clean_prediction = before["prediction"]
        trigger_prediction = after["prediction"]
        if clean_prediction == original and trigger_prediction == target_label:
            category = "successful_attacks"
        elif clean_prediction == original and trigger_prediction == original:
            category = "resisted_attacks"
        elif clean_prediction == original:
            category = "other_trigger_errors"
        else:
            category = "clean_errors"
        records.append({
            "source_id": str(row["source_id"]),
            "relative_path": str(row["relative_path"]),
            "original_label": original,
            "clean_prediction": clean_prediction,
            "clean_confidence": before["confidence"],
            "trigger_prediction": trigger_prediction,
            "trigger_confidence": after["confidence"],
            "target_label": target_label,
            "category": category,
        })
    return records


def select_representatives(records, args):
    requested = {
        "successful_attacks": args.num_success,
        "resisted_attacks": args.num_resistant,
        "other_trigger_errors": args.num_other,
    }
    return {
        category: diverse_samples(
            [record for record in records if record["category"] == category], count, args.seed
        )
        for category, count in requested.items()
    }


def diverse_samples(records, count, seed):
    if count <= 0:
        return []
    ordered = sorted(
        records,
        key=lambda row: (
            -row["trigger_confidence"],
            hashlib.sha256(f"{seed}:{row['source_id']}".encode()).hexdigest(),
        ),
    )
    selected, labels = [], set()
    for record in ordered:
        if record["original_label"] in labels:
            continue
        selected.append(record)
        labels.add(record["original_label"])
        if len(selected) == count:
            return selected
    for record in ordered:
        if record not in selected:
            selected.append(record)
        if len(selected) == count:
            break
    return selected


def load_one(images_root, row, transform):
    dataset = ASBManifestDataset(
        images_root, [row], transform=transform, label_field="original_label"
    )
    return dataset[0][0]


def save_sample(
    output_dir, clean, triggered, record, class_names, target_label,
    trigger_name, trigger_config, difference_gain,
):
    difference = (triggered - clean).abs()
    to_pil_image(clean.clamp(0, 1)).save(output_dir / "01_clean_image.png")
    to_pil_image(triggered.clamp(0, 1)).save(output_dir / "02_triggered_image.png")
    to_pil_image((difference * difference_gain).clamp(0, 1)).save(
        output_dir / f"03_absolute_difference_x{difference_gain:g}.png"
    )
    save_classification_panel(
        clean, triggered, difference, record, class_names, target_label,
        trigger_name, difference_gain, output_dir / "04_classification_and_spectrum_panel.png",
    )
    save_compact_attack_panel(
        clean, triggered, difference, record, class_names, target_label,
        trigger_name, difference_gain, output_dir / "05_compact_attack_evidence_panel.png",
    )
    write_json(output_dir / "sample_metadata.json", {
        **record,
        "true_class": class_names[record["original_label"]],
        "clean_predicted_class": class_names[record["clean_prediction"]],
        "trigger_predicted_class": class_names[record["trigger_prediction"]],
        "target_class": class_names[target_label],
        "trigger_name": trigger_name,
        "trigger_config": trigger_config,
        "pixel_mae": float(difference.mean().item()),
        "maximum_absolute_pixel_change": float(difference.max().item()),
    })


def save_classification_panel(
    clean, triggered, difference, record, class_names, target_label,
    trigger_name, difference_gain, output_path,
):
    clean_amp = log_amplitude(clean)
    trigger_amp = log_amplitude(triggered)
    amp_diff = (trigger_amp - clean_amp).abs().mean(0)
    amp_max = float(torch.quantile(torch.cat((clean_amp.flatten(), trigger_amp.flatten())), 0.995))
    diff_max = max(float(torch.quantile(amp_diff.flatten(), 0.995)), 1e-8)
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.2))
    axes[0, 0].imshow(rgb(clean))
    axes[0, 0].set_title(
        f"Clean image\ntrue: {class_names[record['original_label']]}\n"
        f"prediction: {class_names[record['clean_prediction']]} "
        f"({100 * record['clean_confidence']:.1f}%)"
    )
    axes[0, 1].imshow(rgb(triggered))
    axes[0, 1].set_title(
        f"Triggered image: {trigger_name}\nprediction: "
        f"{class_names[record['trigger_prediction']]} "
        f"({100 * record['trigger_confidence']:.1f}%)\n"
        f"attacker target: {class_names[target_label]}"
    )
    axes[0, 2].imshow(rgb((difference * difference_gain).clamp(0, 1)))
    axes[0, 2].set_title(f"Absolute pixel difference x{difference_gain:g}")
    axes[1, 0].imshow(clean_amp.mean(0), cmap="magma", vmin=0, vmax=amp_max)
    axes[1, 0].set_title("Clean log-amplitude")
    axes[1, 1].imshow(trigger_amp.mean(0), cmap="magma", vmin=0, vmax=amp_max)
    axes[1, 1].set_title("Triggered log-amplitude")
    view = axes[1, 2].imshow(amp_diff, cmap="inferno", vmin=0, vmax=diff_max)
    axes[1, 2].set_title("Absolute log-amplitude difference")
    fig.colorbar(view, ax=axes[1, 2], fraction=0.046)
    for axis in axes.flat:
        axis.axis("off")
    fig.suptitle(
        f"Poisoned-model evidence: {record['category'].replace('_', ' ')}",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_compact_attack_panel(
    clean, triggered, difference, record, class_names, target_label,
    trigger_name, difference_gain, output_path,
):
    """Match the compact legacy panel without implying a defense was run."""
    clean_amp = log_amplitude(clean).mean(0)
    trigger_amp = log_amplitude(triggered).mean(0)
    amp_diff = (trigger_amp - clean_amp).abs()
    amplitude_max = float(
        torch.quantile(torch.cat((clean_amp.flatten(), trigger_amp.flatten())), 0.995)
    )
    amplitude_diff_max = max(float(torch.quantile(amp_diff.flatten(), 0.995)), 1e-8)

    fig, axes = plt.subplots(2, 4, figsize=(12.8, 6.5))
    axes[0, 0].imshow(rgb(clean))
    axes[0, 0].set_title(
        f"clean true: {class_names[record['original_label']]}", fontsize=10
    )
    axes[0, 1].imshow(rgb(clean))
    axes[0, 1].set_title(
        f"susp clean: {class_names[record['clean_prediction']]}\n"
        f"confidence: {100 * record['clean_confidence']:.1f}%", fontsize=10,
    )
    axes[0, 2].imshow(rgb(triggered))
    axes[0, 2].set_title(f"trigger input: {trigger_name}", fontsize=10)
    axes[0, 3].imshow(rgb(triggered))
    axes[0, 3].set_title(
        f"susp trig: {class_names[record['trigger_prediction']]}\n"
        f"confidence: {100 * record['trigger_confidence']:.1f}%\n"
        f"target: {class_names[target_label]}", fontsize=10,
    )

    axes[1, 0].imshow(clean_amp, cmap="magma", vmin=0, vmax=amplitude_max)
    axes[1, 0].set_title("clean amp", fontsize=10)
    axes[1, 1].imshow(trigger_amp, cmap="magma", vmin=0, vmax=amplitude_max)
    axes[1, 1].set_title("trigger amp", fontsize=10)
    amp_view = axes[1, 2].imshow(
        amp_diff, cmap="inferno", vmin=0, vmax=amplitude_diff_max
    )
    axes[1, 2].set_title("amplitude diff", fontsize=10)
    fig.colorbar(amp_view, ax=axes[1, 2], fraction=0.046)
    axes[1, 3].imshow(rgb((difference * difference_gain).clamp(0, 1)))
    axes[1, 3].set_title(f"image diff x{difference_gain:g}", fontsize=10)

    for axis in axes.flat:
        axis.axis("off")
    outcome = record["category"].replace("_", " ")
    fig.suptitle(f"Suspicious classifier evidence: {outcome}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_overview(
    selected, row_by_id, images_root, transform, class_names,
    trigger_name, trigger_config, difference_gain, output_path,
):
    if not selected:
        return
    fig, axes = plt.subplots(len(selected), 3, figsize=(11.5, 3.4 * len(selected)))
    axes = np.asarray(axes).reshape(len(selected), 3)
    for index, record in enumerate(selected):
        clean_row = row_by_id[record["source_id"]]
        trigger_row = {**clean_row, "variant_name": trigger_name, "trigger_config": trigger_config}
        clean = load_one(images_root, clean_row, transform)
        triggered = load_one(images_root, trigger_row, transform)
        difference = (triggered - clean).abs()
        axes[index, 0].imshow(rgb(clean))
        axes[index, 0].set_title(
            f"CLEAN | true {class_names[record['original_label']]}\n"
            f"pred {class_names[record['clean_prediction']]} "
            f"({100 * record['clean_confidence']:.1f}%)", fontsize=9,
        )
        axes[index, 1].imshow(rgb(triggered))
        axes[index, 1].set_title(
            f"TRIGGERED | {record['category'].replace('_', ' ')}\n"
            f"pred {class_names[record['trigger_prediction']]} "
            f"({100 * record['trigger_confidence']:.1f}%)", fontsize=9,
        )
        axes[index, 2].imshow(rgb((difference * difference_gain).clamp(0, 1)))
        axes[index, 2].set_title(f"ABSOLUTE DIFFERENCE x{difference_gain:g}", fontsize=9)
        for axis in axes[index]:
            axis.axis("off")
    fig.suptitle("DTD poisoned classifier: clean and triggered predictions", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_prediction_csv(records, class_names, path):
    fields = [
        "source_id", "relative_path", "category", "true_class",
        "clean_predicted_class", "clean_confidence", "trigger_predicted_class",
        "trigger_confidence", "target_class",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({
                "source_id": record["source_id"],
                "relative_path": record["relative_path"],
                "category": record["category"],
                "true_class": class_names[record["original_label"]],
                "clean_predicted_class": class_names[record["clean_prediction"]],
                "clean_confidence": record["clean_confidence"],
                "trigger_predicted_class": class_names[record["trigger_prediction"]],
                "trigger_confidence": record["trigger_confidence"],
                "target_class": class_names[record["target_label"]],
            })


def prediction_record(record, field):
    return {
        "source_id": record["source_id"],
        "original_label": record["original_label"],
        "prediction": record[field],
    }


def log_amplitude(image):
    spectrum = torch.fft.fftshift(torch.fft.fft2(image.float(), dim=(-2, -1)), dim=(-2, -1))
    return torch.log1p(spectrum.abs())


def rgb(tensor):
    return tensor.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()


def markdown_summary(summary):
    metric = summary["paired_metrics"]
    lines = [
        "# DTD Poisoned-Model Prediction Evidence", "",
        f"- Checkpoint epoch: {summary['checkpoint_epoch']}",
        f"- Evaluated non-target test sources: {summary['num_evaluated_sources']}",
        f"- Target class: `{summary['target_class']}`",
        f"- Trigger: `{summary['trigger_name']}`",
        f"- Clean non-target accuracy: {100 * metric['clean_non_target_accuracy']:.2f}%",
        f"- Triggered ASR: {100 * metric['asr']:.2f}%",
        f"- Clean target rate: {100 * metric['clean_non_target_target_rate']:.2f}%",
        f"- Net trigger lift: {100 * metric['same_model_target_rate_lift']:.2f} percentage points",
        f"- Conditional ASR: {100 * metric['conditional_asr_clean_correct']:.2f}%",
        "", "## Saved evidence", "",
        "`model_prediction_overview.png` summarizes representative outcomes. Each sample "
        "folder contains clean, triggered and amplified-difference images, a labelled "
        "classification/spectrum panel, a compact legacy-style attack panel, and exact "
        "JSON metadata.", "",
    ]
    for category, count in summary["selection_counts"].items():
        lines.append(f"- `{category}`: {count} displayed samples")
    lines.extend([
        "", "Successful examples are selected only when the clean image is classified "
        "correctly and its triggered counterpart changes to the attacker target. "
        "Resisted examples remain correctly classified after triggering.",
    ])
    return "\n".join(lines) + "\n"


def prepare_output(path, overwrite):
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"Output is not empty: {path}. Use --overwrite intentionally.")
    path.mkdir(parents=True, exist_ok=True)


def validate_args(args):
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not (args.manifest_dir / "variant_manifest.jsonl").is_file():
        raise FileNotFoundError(args.manifest_dir)
    if not args.images_root.is_dir():
        raise FileNotFoundError(args.images_root)
    if min(args.num_success, args.num_resistant, args.num_other) < 0:
        raise ValueError("Sample counts must be non-negative")
    if args.batch_size <= 0 or args.num_workers < 0 or args.cpu_threads <= 0:
        raise ValueError("Invalid loader settings")
    if args.difference_gain <= 0:
        raise ValueError("--difference-gain must be positive")


def slug(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "sample"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
