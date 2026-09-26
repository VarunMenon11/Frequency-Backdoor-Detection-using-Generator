"""Visualize DTD triggers in image, Fourier-amplitude, and phase domains."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

from datasets import ASBManifestDataset, filter_manifest_rows, read_jsonl
from poisoning import AdvancedTriggerConfig, apply_advanced_trigger


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
        "--output-dir",
        type=Path,
        default=Path("Advanced_Outputs/dtd_trigger_spectral_calibration_v1"),
    )
    parser.add_argument(
        "--triggers",
        type=str,
        default=(
            "fourier_middle,haar_lh,haar_hl,fourier_low,fourier_high,"
            "fourier_multi,haar_hh"
        ),
    )
    parser.add_argument(
        "--trigger-catalog",
        type=Path,
        default=None,
        help=(
            "Optional trigger catalog JSON. Defaults to "
            "<manifest-dir>/trigger_catalog.json."
        ),
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--metric-samples", type=int, default=100)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--sampling", choices=["first", "balanced"], default="first")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difference-gain", type=float, default=8.0)
    parser.add_argument(
        "--strength-override",
        type=float,
        default=None,
        help="Measure all selected triggers at this strength instead of catalog values.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.strength_override is not None and args.strength_override < 0.0:
        raise ValueError("--strength-override must be non-negative")
    if args.metric_samples <= 0:
        raise ValueError("--metric-samples must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.manifest_dir / "variant_manifest.jsonl")
    benchmark = json.loads(
        (args.manifest_dir / "benchmark_summary.json").read_text(encoding="utf-8")
    )
    catalog_path = (
        args.trigger_catalog
        if args.trigger_catalog is not None
        else args.manifest_dir / "trigger_catalog.json"
    )
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Missing trigger catalog: {catalog_path}")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    configs = implemented_trigger_configs(catalog)
    trigger_names = [name.strip() for name in args.triggers.split(",") if name.strip()]
    unknown = [name for name in trigger_names if name not in configs]
    if unknown:
        raise ValueError(
            "Unknown triggers: " + ", ".join(unknown) + ". Available: "
            + ", ".join(sorted(configs))
        )

    clean_rows = filter_manifest_rows(
        rows,
        protocol_role="clean_calibration" if args.split == "validation" else "final_test_clean",
        variant_name="clean",
        exclude_original_label=int(benchmark["target_label"]),
    )
    if not 0 <= args.sample_index < len(clean_rows):
        raise ValueError(f"--sample-index must be below {len(clean_rows)}")
    transform = transforms.Compose(
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
    clean_dataset = ASBManifestDataset(
        args.images_root,
        clean_rows,
        transform=transform,
        label_field="original_label",
    )
    clean_example, _ = clean_dataset[args.sample_index]
    example_row = clean_rows[args.sample_index]

    summary = {
        "experiment": "dtd_trigger_spectral_calibration_v1",
        "image_size": args.image_size,
        "split": args.split,
        "sampling": args.sampling,
        "seed": args.seed,
        "metric_samples": min(args.metric_samples, len(clean_dataset)),
        "example": {
            "relative_path": example_row["relative_path"],
            "class_name": example_row["class_name"],
        },
        "metrics": {},
    }
    indices = metric_sample_indices(clean_rows, args.metric_samples, args.sampling, args.seed)
    summary["metric_source_ids"] = [clean_rows[index]["source_id"] for index in indices]
    metric_images = [clean_dataset[index][0] for index in indices]
    clean_batch = torch.stack(metric_images)

    for trigger_name in trigger_names:
        config_data = normalized_config(configs[trigger_name])
        if args.strength_override is not None:
            config_data["strength"] = args.strength_override
        config = AdvancedTriggerConfig(**config_data)
        triggered_example = apply_advanced_trigger(clean_example, config)
        save_spectral_panel(
            clean_example,
            triggered_example,
            trigger_name=trigger_name,
            class_name=str(example_row["class_name"]),
            difference_gain=args.difference_gain,
            output_path=args.output_dir / f"{trigger_name}_spectrum_panel.png",
        )
        triggered_batch = apply_advanced_trigger(clean_batch, config)
        summary["metrics"][trigger_name] = perturbation_metrics(
            clean_batch,
            triggered_batch,
            config,
        )

    summary_path = args.output_dir / "trigger_spectral_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_markdown_summary(summary, args.output_dir / "trigger_spectral_metrics.md")
    print("Example:", example_row["class_name"], example_row["relative_path"])
    for name, metrics in summary["metrics"].items():
        print(
            f"{name:16s} | PSNR {metrics['mean_psnr_db']:.2f} dB | "
            f"MAE {metrics['mean_pixel_mae']:.6f} | "
            f"log-amp diff {metrics['mean_log_amplitude_difference']:.6f} | "
            f"phase diff {metrics['mean_wrapped_phase_difference_radians']:.6f}"
        )
    print("Saved:", args.output_dir)


def metric_sample_indices(rows, count, sampling, seed):
    if count <= 0:
        raise ValueError("Metric sample count must be positive")
    if sampling == "first":
        return list(range(min(count, len(rows))))
    if sampling != "balanced":
        raise ValueError(f"Unknown sampling mode: {sampling}")
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault(int(row["original_label"]), []).append(index)
    rng = random.Random(seed)
    labels = sorted(groups)
    rng.shuffle(labels)
    for indices in groups.values():
        rng.shuffle(indices)
    selected = []
    while len(selected) < min(count, len(rows)):
        for label in labels:
            if groups[label]:
                selected.append(groups[label].pop())
                if len(selected) == min(count, len(rows)):
                    return selected
    return selected


def implemented_trigger_configs(catalog: dict[str, object]) -> dict[str, dict[str, object]]:
    configs = {}
    for section_name in ("development_triggers", "held_out_triggers"):
        for name, entry in catalog.get(section_name, {}).items():
            if entry.get("status") == "implemented":
                configs[name] = entry["config"]
    return configs


def normalized_config(config: dict[str, object]) -> dict[str, object]:
    result = dict(config)
    if "channel_weights" in result:
        result["channel_weights"] = tuple(result["channel_weights"])
    return result


def spectra(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    frequency = torch.fft.fftshift(
        torch.fft.fft2(images.float(), dim=(-2, -1)), dim=(-2, -1)
    )
    return torch.log1p(frequency.abs()), torch.angle(frequency)


def perturbation_metrics(clean, triggered, config):
    difference = triggered - clean
    per_image_mse = difference.square().flatten(1).mean(1)
    psnr = -10.0 * torch.log10(per_image_mse.clamp_min(1e-12))
    clean_amplitude, clean_phase = spectra(clean)
    trigger_amplitude, trigger_phase = spectra(triggered)
    wrapped_phase = torch.angle(torch.exp(1j * (trigger_phase - clean_phase))).abs()
    return {
        "trigger_config": config.to_dict(),
        "mean_pixel_mae": float(difference.abs().mean().item()),
        "mean_pixel_rmse": float(per_image_mse.sqrt().mean().item()),
        "mean_psnr_db": float(psnr.mean().item()),
        "mean_max_absolute_pixel_change": float(
            difference.abs().flatten(1).amax(1).mean().item()
        ),
        "mean_log_amplitude_difference": float(
            (trigger_amplitude - clean_amplitude).abs().mean().item()
        ),
        "mean_wrapped_phase_difference_radians": float(wrapped_phase.mean().item()),
        "clipped_pixel_fraction": float(
            ((triggered <= 0.0) | (triggered >= 1.0)).float().mean().item()
        ),
    }


def save_spectral_panel(
    clean,
    triggered,
    *,
    trigger_name,
    class_name,
    difference_gain,
    output_path,
):
    clean_amp, clean_phase = spectra(clean)
    trigger_amp, trigger_phase = spectra(triggered)
    amplitude_difference = (trigger_amp - clean_amp).abs().mean(0)
    phase_difference = torch.angle(
        torch.exp(1j * (trigger_phase - clean_phase))
    ).abs().mean(0)
    clean_amp_view = clean_amp.mean(0)
    trigger_amp_view = trigger_amp.mean(0)
    amplitude_max = float(
        torch.quantile(torch.cat([clean_amp_view.flatten(), trigger_amp_view.flatten()]), 0.995)
    )
    amplitude_diff_max = max(float(torch.quantile(amplitude_difference.flatten(), 0.995)), 1e-8)

    image_difference = (triggered - clean).abs()
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    axes[0, 0].imshow(to_rgb(clean))
    axes[0, 0].set_title(f"Clean image\nclass: {class_name}")
    axes[0, 1].imshow(to_rgb(triggered))
    axes[0, 1].set_title(f"Triggered image\n{trigger_name}")
    axes[0, 2].imshow(to_rgb((image_difference * difference_gain).clamp(0, 1)))
    axes[0, 2].set_title(f"Absolute image difference x{difference_gain:g}")
    image_diff_view = axes[0, 3].imshow(image_difference.mean(0), cmap="inferno")
    axes[0, 3].set_title("Absolute image difference\nauto-scaled")
    fig.colorbar(image_diff_view, ax=axes[0, 3], fraction=0.046)

    axes[1, 0].imshow(clean_amp_view, cmap="magma", vmin=0, vmax=amplitude_max)
    axes[1, 0].set_title("Clean log-amplitude")
    axes[1, 1].imshow(trigger_amp_view, cmap="magma", vmin=0, vmax=amplitude_max)
    axes[1, 1].set_title("Triggered log-amplitude")
    amp_diff_view = axes[1, 2].imshow(
        amplitude_difference,
        cmap="inferno",
        vmin=0,
        vmax=amplitude_diff_max,
    )
    axes[1, 2].set_title("Absolute log-amplitude difference")
    fig.colorbar(amp_diff_view, ax=axes[1, 2], fraction=0.046)
    phase_diff_view = axes[1, 3].imshow(
        phase_difference, cmap="viridis", vmin=0, vmax=math.pi
    )
    axes[1, 3].set_title("Wrapped phase difference")
    fig.colorbar(phase_diff_view, ax=axes[1, 3], fraction=0.046)
    for axis in axes.flat:
        axis.axis("off")
    fig.suptitle("Spatial and frequency-domain trigger evidence", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def to_rgb(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()


def save_markdown_summary(summary, path):
    lines = [
        "# DTD Trigger Spectral Measurements",
        "",
        "The values below are measured before classifier training. PSNR is computed "
        "for images in [0,1]; larger PSNR means lower squared error, not a guarantee of invisibility.",
        "",
        f"Split: {summary.get('split', 'test')}; sampling: {summary.get('sampling', 'first')}; "
        f"sources: {summary['metric_samples']}. Exact source IDs and trigger strengths are in the JSON.",
        "",
        "| Trigger | PSNR (dB) | Pixel MAE | Log-amplitude difference | Phase difference (rad) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metric in summary["metrics"].items():
        lines.append(
            f"| {name} | {metric['mean_psnr_db']:.2f} | "
            f"{metric['mean_pixel_mae']:.6f} | "
            f"{metric['mean_log_amplitude_difference']:.6f} | "
            f"{metric['mean_wrapped_phase_difference_radians']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
