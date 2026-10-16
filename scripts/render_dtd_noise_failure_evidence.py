"""Save one honest noise-only misclassification panel from the frozen DTD run.

This is a qualitative selector, not a new aggregate evaluation. It recreates
the -10 dB-equivalent noise used by the TNR protocol and stops after finding a
sample that the classifier recognized when clean but misclassified with noise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from datasets import ASBPairedTriggerDataset, filter_manifest_rows, read_jsonl
from evaluation.asb_classifier import evaluation_transform
from scripts.audit_dtd_reference_free_generator import build_models, resolve_device
from scripts.evaluate_dtd_trigger_noise_ratio import add_tnr_noise
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
    parser.add_argument("--trigger", default="ftpos_near_mixed")
    parser.add_argument("--tnr-db", type=float, default=-10.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=342)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("End_Documentation/assets/20_noise_only_failure_evidence.png"),
    )
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
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
    class_names = list(config["class_names"])
    target_label = int(config["target_label"])
    trigger_dicts = dict(config["trigger_configs"])
    if args.trigger not in trigger_dicts:
        raise KeyError(f"Unknown trigger {args.trigger!r}; options: {sorted(trigger_dicts)}")

    rows = read_jsonl(args.manifest_dir / "variant_manifest.jsonl")
    validation_rows = filter_manifest_rows(
        rows,
        protocol_role="clean_calibration",
        variant_name="clean",
        exclude_original_label=target_label,
    )
    dataset = ASBPairedTriggerDataset(
        args.images_root,
        validation_rows,
        transform=evaluation_transform(int(config["image_size"])),
        trigger_config=trigger_from_dict(trigger_dicts[args.trigger]),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    noise_generator = torch.Generator(device=device).manual_seed(args.seed)

    selected = None
    for batch_index, (clean, triggered, labels) in enumerate(loader):
        if batch_index >= args.max_batches:
            break
        clean = clean.to(device)
        triggered = triggered.to(device)
        labels = labels.to(device)
        raw_noise = torch.randn(
            clean.shape, generator=noise_generator, device=device, dtype=clean.dtype
        )
        _, noise_only, actual_noise, realized_tnr = add_tnr_noise(
            clean, triggered, raw_noise, args.tnr_db
        )
        corrected = generator(noise_only).corrected_image
        logits = classifier(torch.cat((clean, noise_only, corrected), dim=0))
        predictions = logits.argmax(1).view(3, labels.numel())
        probabilities = logits.softmax(1).view(3, labels.numel(), -1)

        eligible = (predictions[0] == labels) & (predictions[1] != labels)
        preferred = eligible & (predictions[1] == target_label)
        indices = preferred.nonzero().flatten()
        if not indices.numel():
            indices = eligible.nonzero().flatten()
        if indices.numel():
            item = int(indices[0])
            selected = {
                "clean": clean[item].cpu(),
                "noise": noise_only[item].cpu(),
                "corrected": corrected[item].cpu(),
                "actual_noise": actual_noise[item].cpu(),
                "label": int(labels[item]),
                "predictions": predictions[:, item].cpu().tolist(),
                "target_probabilities": probabilities[:, item, target_label].cpu().tolist(),
                "realized_tnr_db": float(realized_tnr[item].cpu()),
                "batch_index": batch_index,
                "item_index": item,
            }
            break

    if selected is None:
        raise RuntimeError(
            "No clean-correct/noise-wrong sample found within --max-batches; increase it."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_panel(selected, class_names, target_label, args, args.output)
    metadata = {
        "purpose": "qualitative noise-only failure example; not aggregate evidence",
        "trigger_used_only_to_calibrate_noise_rms": args.trigger,
        "requested_tnr_db": args.tnr_db,
        "realized_tnr_db": selected["realized_tnr_db"],
        "true_class": class_names[selected["label"]],
        "clean_prediction": class_names[selected["predictions"][0]],
        "noise_only_prediction": class_names[selected["predictions"][1]],
        "corrected_noise_prediction": class_names[selected["predictions"][2]],
        "selection": "clean_correct__noise_only_wrong",
        "seed": args.seed,
        "batch_index": selected["batch_index"],
        "item_index": selected["item_index"],
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    print("Saved:", args.output)


def log_amplitude(image: torch.Tensor) -> torch.Tensor:
    spectrum = torch.fft.fft2(image.float(), dim=(-2, -1))
    values = torch.fft.fftshift(torch.log1p(spectrum.abs()), dim=(-2, -1)).mean(0)
    low = torch.quantile(values.flatten(), 0.01)
    high = torch.quantile(values.flatten(), 0.995)
    return ((values - low) / (high - low).clamp_min(1e-8)).clamp(0, 1)


def show_image(axis, image: torch.Tensor, title: str) -> None:
    axis.imshow(image.permute(1, 2, 0).clamp(0, 1))
    axis.set_title(title)
    axis.axis("off")


def save_panel(selected, class_names, target_label, args, path) -> None:
    clean = selected["clean"]
    noise = selected["noise"]
    corrected = selected["corrected"]
    predictions = selected["predictions"]
    true_name = class_names[selected["label"]]
    figure, axes = plt.subplots(2, 4, figsize=(16, 8.5))
    show_image(
        axes[0, 0], clean,
        f"clean\ntrue/pred: {true_name} / {class_names[predictions[0]]}",
    )
    show_image(
        axes[0, 1], noise,
        f"noise only\npred: {class_names[predictions[1]]} (wrong)",
    )
    show_image(
        axes[0, 2], corrected,
        f"noise + generator\npred: {class_names[predictions[2]]}",
    )
    show_image(axes[0, 3], selected["actual_noise"].abs() * 8, "noise component |x8|")
    shared = torch.stack((log_amplitude(clean), log_amplitude(noise), log_amplitude(corrected)))
    for axis, spectrum, title in zip(
        axes[1, :3], shared,
        ("clean log amplitude", "noise-only log amplitude", "corrected log amplitude"),
    ):
        axis.imshow(spectrum, cmap="magma", vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")
    probabilities = [100 * value for value in selected["target_probabilities"]]
    bars = axes[1, 3].bar(
        ("clean", "noise", "noise+corr"), probabilities,
        color=("#4c78a8", "#8c8c8c", "#8172b3"),
    )
    axes[1, 3].set_title(f"Target probability: {class_names[target_label]}")
    axes[1, 3].set_ylabel("Percent")
    axes[1, 3].grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, probabilities):
        axes[1, 3].text(
            bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}",
            ha="center", va="bottom", fontsize=8,
        )
    figure.suptitle(
        "Noise-only failure example (no trigger in the classified input)\n"
        f"noise scaled from {args.trigger} trigger RMS at requested {args.tnr_db:+g} dB; "
        f"realized {selected['realized_tnr_db']:.2f} dB",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
