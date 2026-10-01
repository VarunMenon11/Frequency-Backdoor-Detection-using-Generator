"""Trigger-to-noise ratio robustness study for the frozen DTD defense.

Gaussian noise is scaled per image relative to the measured RMS of the actual
clipped trigger perturbation. No classifier or generator training occurs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from datasets import ASBPairedTriggerDataset, filter_manifest_rows, read_jsonl
from evaluation.asb_classifier import evaluation_transform
from scripts.audit_dtd_reference_free_generator import build_models, resolve_device
from scripts.train_dtd_reference_free_generator import trigger_from_dict


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
        default=Path("Advanced_Outputs/dtd_trigger_noise_ratio_v1"),
    )
    parser.add_argument(
        "--tnr-db", default="20,10,0,-10",
        help="Comma-separated finite trigger-to-noise ratios in decibels.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--panel-trigger", default="ftpos_near_mixed")
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ratios = parse_tnr_values(args.tnr_db)
    validate_args(args, ratios)
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
    trigger_names = list(config["evaluation_trigger_names"])
    held_out = set(config.get("held_out_trigger_names", []))
    transform = evaluation_transform(int(config["image_size"]))
    validation_rows = filter_manifest_rows(
        rows, protocol_role="clean_calibration", variant_name="clean",
        exclude_original_label=target_label,
    )

    print("Device:", device, flush=True)
    print("Protocol: validation-only TNR study; frozen models; no training", flush=True)
    print("TNR levels: trigger-only,", ", ".join(f"{value:g} dB" for value in ratios), flush=True)
    results = []
    evidence = {}
    for trigger_index, trigger_name in enumerate(trigger_names):
        status = "held_out" if trigger_name in held_out else "seen"
        dataset = ASBPairedTriggerDataset(
            args.images_root, validation_rows, transform=transform,
            trigger_config=trigger_from_dict(trigger_dicts[trigger_name]),
        )
        for ratio_index, tnr_db in enumerate([None, *ratios]):
            label = "trigger_only" if tnr_db is None else f"tnr_{tnr_db:+g}_db"
            print(f"{trigger_name} ({status}) | {label}", flush=True)
            result, example = evaluate_condition(
                generator, classifier, make_loader(dataset, args, device),
                target_label, device, tnr_db,
                args.seed + trigger_index * 100 + ratio_index,
                args.max_eval_batches,
            )
            result.update({
                "trigger": trigger_name,
                "generator_status": status,
                "condition": label,
                "requested_tnr_db": tnr_db,
            })
            results.append(result)
            if trigger_name == args.panel_trigger:
                panel_path = args.output_dir / "evidence_panels" / f"{label}.png"
                panel_path.parent.mkdir(parents=True, exist_ok=True)
                save_panel(
                    example, class_names, target_label, trigger_name, status,
                    result, panel_path,
                )
                evidence[label] = {
                    "path": str(panel_path),
                    "selection": example["selection"],
                    "requested_tnr_db": tnr_db,
                    "realized_mean_tnr_db": result["realized_mean_tnr_db"],
                }

    summary = {
        "protocol": "validation-only; frozen classifier and generator; no training",
        "formula": "TNR_dB = 20*log10(RMS(trigger perturbation)/RMS(noise perturbation))",
        "noise": "zero-mean Gaussian, normalized and scaled independently per image",
        "requested_tnr_db": ratios,
        "trigger_only_included": True,
        "target_label": target_label,
        "target_class": class_names[target_label],
        "panel_trigger": args.panel_trigger,
        "partial_evaluation": args.max_eval_batches is not None,
        "classifier_checkpoint": str(args.classifier_checkpoint),
        "generator_checkpoint": str(args.generator_checkpoint),
        "results": results,
        "aggregate": aggregate_results(results),
        "evidence_panels": evidence,
        "interpretation_boundary": (
            "A defense result is meaningful only while the corresponding noisy "
            "trigger remains an active attack before correction. Noise-only rows "
            "test false correction and benign robustness."
        ),
    }
    write_json(args.output_dir / "trigger_noise_ratio_summary.json", summary)
    write_csv(args.output_dir / "trigger_noise_ratio_table.csv", results)
    save_chart(summary, args.output_dir / "trigger_noise_ratio_chart.png")
    save_markdown(summary, args.output_dir / "README.md")
    print_table(summary)
    print("Saved TNR study:", args.output_dir, flush=True)


@torch.inference_mode()
def evaluate_condition(
    generator, classifier, loader, target_label, device, tnr_db, seed, max_batches
):
    totals = defaultdict(int)
    sums = defaultdict(float)
    best_example = None
    best_rank = -1
    noise_generator = torch.Generator(device=device).manual_seed(seed)
    for batch_index, (clean, triggered, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        clean = clean.to(device, non_blocking=True)
        triggered = triggered.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if tnr_db is None:
            attacked = triggered
            noise_only = clean
            actual_noise = torch.zeros_like(clean)
            realized_tnr = torch.full(
                (clean.shape[0],), float("inf"), device=device
            )
        else:
            raw_noise = torch.randn(
                clean.shape, generator=noise_generator, device=device, dtype=clean.dtype
            )
            attacked, noise_only, actual_noise, realized_tnr = add_tnr_noise(
                clean, triggered, raw_noise, tnr_db
            )
        attacked_output = generator(attacked)
        noise_output = generator(noise_only)
        variants = (clean, attacked, attacked_output.corrected_image,
                    noise_only, noise_output.corrected_image)
        logits = classifier(torch.cat(variants, dim=0))
        predictions = logits.argmax(1).view(len(variants), labels.numel())
        probabilities = logits.softmax(1).view(len(variants), labels.numel(), -1)
        names = ("clean", "attacked", "corrected", "noise_only", "corrected_noise_only")
        for index, name in enumerate(names):
            totals[f"{name}_target"] += int((predictions[index] == target_label).sum())
            totals[f"{name}_correct"] += int((predictions[index] == labels).sum())
        n = labels.numel()
        totals["samples"] += n
        trigger_delta = triggered - clean
        sums["trigger_rms"] += float(per_image_rms(trigger_delta).sum().cpu())
        sums["noise_rms"] += float(per_image_rms(actual_noise).sum().cpu())
        finite = torch.isfinite(realized_tnr)
        if finite.any():
            sums["realized_tnr"] += float(realized_tnr[finite].sum().cpu())
            totals["finite_tnr"] += int(finite.sum())
        sums["attacked_correction"] += float(
            attacked_output.effective_log_correction.abs().flatten(1).mean(1).sum().cpu()
        )
        sums["noise_correction"] += float(
            noise_output.effective_log_correction.abs().flatten(1).mean(1).sum().cpu()
        )
        sums["corrected_l1"] += float(
            (attacked_output.corrected_image-clean).abs().flatten(1).mean(1).sum().cpu()
        )

        for item in range(n):
            label = int(labels[item])
            clean_correct = int(predictions[0, item]) == label
            attack_success = int(predictions[1, item]) == target_label
            recovered = int(predictions[2, item]) == label
            rank = 3 if clean_correct and attack_success and recovered else (
                2 if clean_correct and attack_success and int(predictions[2, item]) != target_label
                else 1 if attack_success and int(predictions[2, item]) != target_label else 0
            )
            if rank > best_rank:
                selection = (
                    "clean_correct__attack_success__true_label_recovered" if rank == 3
                    else "clean_correct__attack_success__target_removed" if rank == 2
                    else "attack_success__target_removed" if rank == 1
                    else "fallback_example"
                )
                best_example = {
                    "selection": selection,
                    "label": label,
                    "clean": clean[item].cpu(),
                    "triggered": triggered[item].cpu(),
                    "attacked": attacked[item].cpu(),
                    "corrected": attacked_output.corrected_image[item].cpu(),
                    "noise_only": noise_only[item].cpu(),
                    "corrected_noise_only": noise_output.corrected_image[item].cpu(),
                    "effective": attacked_output.effective_log_correction[item].cpu(),
                    "predictions": predictions[:, item].cpu().tolist(),
                    "target_probabilities": probabilities[:, item, target_label].cpu().tolist(),
                    "realized_tnr_db": float(realized_tnr[item].cpu()),
                }
                best_rank = rank
        if batch_index % 20 == 0:
            print(f"  evaluated {totals['samples']} samples", flush=True)
    n = totals["samples"]
    if n == 0 or best_example is None:
        raise ValueError("No TNR samples were evaluated")
    return {
        "num_non_target_samples": n,
        "clean_target_rate": totals["clean_target"] / n,
        "undefended_asr": totals["attacked_target"] / n,
        "corrected_asr": totals["corrected_target"] / n,
        "noise_only_target_rate": totals["noise_only_target"] / n,
        "corrected_noise_only_target_rate": totals["corrected_noise_only_target"] / n,
        "clean_accuracy": totals["clean_correct"] / n,
        "attacked_accuracy": totals["attacked_correct"] / n,
        "corrected_accuracy": totals["corrected_correct"] / n,
        "noise_only_accuracy": totals["noise_only_correct"] / n,
        "corrected_noise_only_accuracy": totals["corrected_noise_only_correct"] / n,
        "mean_trigger_rms": sums["trigger_rms"] / n,
        "mean_realized_noise_rms": sums["noise_rms"] / n,
        "realized_mean_tnr_db": (
            None if totals["finite_tnr"] == 0
            else sums["realized_tnr"] / totals["finite_tnr"]
        ),
        "mean_attacked_generator_correction": sums["attacked_correction"] / n,
        "mean_noise_only_generator_correction": sums["noise_correction"] / n,
        "corrected_pixel_l1_to_clean": sums["corrected_l1"] / n,
    }, best_example


def add_tnr_noise(clean, triggered, raw_noise, tnr_db, epsilon=1e-12):
    """Add per-image Gaussian noise calibrated to a requested TNR."""
    trigger_rms = per_image_rms(triggered - clean).clamp_min(epsilon)
    unit_noise = raw_noise / per_image_rms(raw_noise).clamp_min(epsilon)[:, None, None, None]
    desired_noise_rms = trigger_rms / (10.0 ** (float(tnr_db) / 20.0))
    proposed_noise = unit_noise * desired_noise_rms[:, None, None, None]
    attacked = (triggered + proposed_noise).clamp(0, 1)
    noise_only = (clean + proposed_noise).clamp(0, 1)
    actual_noise = attacked - triggered
    actual_noise_rms = per_image_rms(actual_noise).clamp_min(epsilon)
    realized_tnr = 20.0 * torch.log10(trigger_rms / actual_noise_rms)
    return attacked, noise_only, actual_noise, realized_tnr


def per_image_rms(values):
    return values.float().square().flatten(1).mean(1).sqrt()


def parse_tnr_values(text):
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("--tnr-db must contain unique comma-separated values")
    return values


def aggregate_results(results):
    grouped = {}
    for condition in dict.fromkeys(row["condition"] for row in results):
        selected = [row for row in results if row["condition"] == condition]
        grouped[condition] = {
            "requested_tnr_db": selected[0]["requested_tnr_db"],
            "num_triggers": len(selected),
        }
        for metric in (
            "undefended_asr", "corrected_asr", "corrected_accuracy",
            "noise_only_accuracy", "corrected_noise_only_accuracy",
            "noise_only_target_rate", "corrected_noise_only_target_rate",
            "mean_attacked_generator_correction", "mean_noise_only_generator_correction",
        ):
            grouped[condition][f"mean_{metric}"] = sum(row[metric] for row in selected) / len(selected)
        finite_tnr = [row["realized_mean_tnr_db"] for row in selected
                      if row["realized_mean_tnr_db"] is not None]
        grouped[condition]["mean_realized_tnr_db"] = (
            None if not finite_tnr else sum(finite_tnr) / len(finite_tnr)
        )
    return grouped


def save_panel(example, class_names, target_label, trigger_name, status, result, path):
    names = ("clean", "attacked", "corrected", "noise_only", "corrected_noise_only")
    predictions = example["predictions"]
    figure, axes = plt.subplots(2, 4, figsize=(15, 8))
    items = (
        (example["clean"], f"clean\ntrue/pred: {class_names[example['label']]} / {class_names[predictions[0]]}"),
        (example["triggered"], "trigger only"),
        (example["attacked"], f"trigger + noise\npred: {class_names[predictions[1]]}"),
        (example["corrected"], f"generator corrected\npred: {class_names[predictions[2]]}"),
        ((example["triggered"]-example["clean"]).abs()*8, "trigger - clean |x8|"),
        ((example["attacked"]-example["triggered"]).abs()*8, "noise component |x8|"),
        (example["noise_only"], f"noise only\npred: {class_names[predictions[3]]}"),
    )
    for axis, (image, title) in zip(axes.flat[:7], items):
        axis.imshow(image.permute(1, 2, 0).clamp(0, 1))
        axis.set_title(title)
        axis.axis("off")
    probabilities = [100 * value for value in example["target_probabilities"]]
    bars = axes.flat[7].bar(
        ("clean", "attack", "corrected", "noise", "noise+corr"), probabilities,
        color=("#4c78a8", "#d95f5f", "#55a868", "#8c8c8c", "#8172b3"),
    )
    axes.flat[7].set_title(f"Target probability: {class_names[target_label]}")
    axes.flat[7].set_ylabel("Percent")
    axes.flat[7].tick_params(axis="x", rotation=25)
    axes.flat[7].grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, probabilities):
        axes.flat[7].text(bar.get_x()+bar.get_width()/2, value, f"{value:.1f}",
                          ha="center", va="bottom", fontsize=7)
    tnr = "trigger only" if result["requested_tnr_db"] is None else (
        f"requested {result['requested_tnr_db']:+g} dB; "
        f"realized mean {result['realized_mean_tnr_db']:.2f} dB"
    )
    figure.suptitle(
        f"TNR robustness | {trigger_name} ({status}) | {tnr}\n{example['selection']}",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_chart(summary, path):
    rows = sorted(
        (row for row in summary["aggregate"].values() if row["requested_tnr_db"] is not None),
        key=lambda row: row["requested_tnr_db"],
    )
    x = [row["requested_tnr_db"] for row in rows]
    before = [100*row["mean_undefended_asr"] for row in rows]
    after = [100*row["mean_corrected_asr"] for row in rows]
    noise_acc = [100*row["mean_noise_only_accuracy"] for row in rows]
    noise_corr_acc = [100*row["mean_corrected_noise_only_accuracy"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    axes[0].plot(x, before, marker="o", label="Before defense")
    axes[0].plot(x, after, marker="o", label="After generator")
    axes[0].set_title("Backdoor behavior under noise")
    axes[0].set_ylabel("Target rate / ASR (%)")
    axes[1].plot(x, noise_acc, marker="o", label="Noise only")
    axes[1].plot(x, noise_corr_acc, marker="o", label="Noise + generator")
    axes[1].set_title("Benign noise control")
    axes[1].set_ylabel("True-label accuracy (%)")
    for axis in axes:
        axis.set_xlabel("Requested TNR (dB); lower means more noise")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_markdown(summary, path):
    lines = [
        "# DTD Trigger-to-Noise Ratio Study", "",
        "Frozen-model validation experiment; no training was performed.", "",
        "## Definition", "", f"`{summary['formula']}`", "",
        "Positive TNR means the trigger RMS is larger than the noise RMS. At 0 dB they are equal. Negative TNR means noise is stronger.", "",
        "## Four-position aggregate", "",
        "| Condition | Realized TNR | Undefended ASR | Corrected ASR | Corrected accuracy | Noise-only accuracy | Noise+generator accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, row in summary["aggregate"].items():
        realized = "infinite" if row["mean_realized_tnr_db"] is None else f"{row['mean_realized_tnr_db']:.2f} dB"
        lines.append(
            f"| `{condition}` | {realized} | {100*row['mean_undefended_asr']:.2f}% | "
            f"{100*row['mean_corrected_asr']:.2f}% | {100*row['mean_corrected_accuracy']:.2f}% | "
            f"{100*row['mean_noise_only_accuracy']:.2f}% | "
            f"{100*row['mean_corrected_noise_only_accuracy']:.2f}% |"
        )
    lines += ["", "## Interpretation rule", "", summary["interpretation_boundary"], ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(path, results):
    if not results:
        return
    fields = list(results[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def print_table(summary):
    print("\ncondition | undefended ASR | corrected ASR | noise acc | noise+generator acc")
    for condition, row in summary["aggregate"].items():
        print(
            f"{condition:18s} | {row['mean_undefended_asr']:.4f} | "
            f"{row['mean_corrected_asr']:.4f} | {row['mean_noise_only_accuracy']:.4f} | "
            f"{row['mean_corrected_noise_only_accuracy']:.4f}", flush=True,
        )


def make_loader(dataset, args, device):
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )


def validate_args(args, ratios):
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Invalid batch size or worker count")
    if args.max_eval_batches is not None and args.max_eval_batches <= 0:
        raise ValueError("--max-eval-batches must be positive")
    if not all(math.isfinite(value) for value in ratios):
        raise ValueError("--tnr-db values must be finite")
    for path in (args.generator_checkpoint, args.classifier_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)


def prepare_output(args):
    protected = args.output_dir / "trigger_noise_ratio_summary.json"
    if protected.exists() and not args.overwrite:
        raise FileExistsError(f"Protected result exists: {protected}")
    args.output_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
