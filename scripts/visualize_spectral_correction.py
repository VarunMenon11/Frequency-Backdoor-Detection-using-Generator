"""Visualize the trained spectral correction generator.

This Phase 7 script inspects whether the trained generator produces reasonable
corrections. It saves sample panels and an average correction map.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from datasets.cifar100_dataset import CleanCIFAR100Dataset
from fft import amplitude_spectrum, normalize_minmax, shift_frequency_map
from generator import (
    SpectralCorrectionGenerator,
    apply_correction_map,
    build_generator_input,
    fft_amplitude_phase,
    reconstruct_from_amplitude_phase,
)
from models import SmallCIFARClassifier
from poisoning.frequency_trigger import FrequencyTriggerConfig, apply_frequency_trigger
from visualization.correction_panel import save_correction_panel
from visualization.frequency_analysis import save_heatmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-path", type=Path, default=Path("datasets/archive.zip"))
    parser.add_argument(
        "--classifier-checkpoint",
        type=Path,
        default=Path("experiments/suspicious_classifier_trained/suspicious_classifier.pt"),
    )
    parser.add_argument(
        "--generator-checkpoint",
        type=Path,
        default=Path("experiments/spectral_generator_cifar100/spectral_generator.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/spectral_correction"))
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--num-average-samples", type=int, default=256)
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--strength", type=float, default=0.08)
    parser.add_argument("--horizontal-frequency", type=int, default=6)
    parser.add_argument("--vertical-frequency", type=int, default=6)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = args.output_dir / "sample_panels"
    sample_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    dataset = CleanCIFAR100Dataset.from_zip(args.zip_path, split="test")
    trigger_config = FrequencyTriggerConfig(
        horizontal_frequency=args.horizontal_frequency,
        vertical_frequency=args.vertical_frequency,
        strength=args.strength,
    )

    classifier = SmallCIFARClassifier(num_classes=100).to(device)
    classifier_ckpt = torch.load(args.classifier_checkpoint, map_location=device)
    classifier.load_state_dict(classifier_ckpt["model_state_dict"])
    classifier.eval()

    generator = SpectralCorrectionGenerator().to(device)
    generator_ckpt = torch.load(args.generator_checkpoint, map_location=device)
    generator.load_state_dict(generator_ckpt["generator_state_dict"])
    generator.eval()

    records = []
    correction_sum = None
    processed_for_average = 0

    with torch.no_grad():
        for original_index in range(len(dataset)):
            clean_image, clean_label = dataset[original_index]
            if int(clean_label) == args.target_label:
                continue

            result = correct_one(
                clean_image=clean_image,
                classifier=classifier,
                generator=generator,
                trigger_config=trigger_config,
                device=device,
            )

            correction_sum = (
                result["correction_map"].detach().cpu()
                if correction_sum is None
                else correction_sum + result["correction_map"].detach().cpu()
            )
            processed_for_average += 1

            if len(records) < args.num_samples:
                true_name = dataset.fine_label_names[int(clean_label)]
                clean_pred_name = dataset.fine_label_names[result["clean_pred"]]
                triggered_pred_name = dataset.fine_label_names[result["triggered_pred"]]
                corrected_pred_name = dataset.fine_label_names[result["corrected_pred"]]

                panel_path = save_correction_panel(
                    [
                        (f"clean T:{true_name} P:{clean_pred_name}", clean_image, "rgb"),
                        (
                            f"triggered P:{triggered_pred_name}",
                            result["triggered_image"],
                            "rgb",
                        ),
                        (
                            f"corrected P:{corrected_pred_name}",
                            result["corrected_image"],
                            "rgb",
                        ),
                        (
                            "image diff x8",
                            ((result["corrected_image"] - clean_image).abs() * 8.0).clamp(0, 1),
                            "rgb",
                        ),
                        (
                            "trigger amp",
                            normalize_minmax(amplitude_spectrum(result["triggered_image"])),
                            "gray",
                        ),
                        (
                            "corrected amp",
                            normalize_minmax(amplitude_spectrum(result["corrected_image"])),
                            "gray",
                        ),
                        (
                            "correction map",
                            normalize_minmax(shift_frequency_map(result["correction_map"])),
                            "gray",
                        ),
                        (
                            "amp diff after",
                            normalize_minmax(
                                (
                                    amplitude_spectrum(result["corrected_image"])
                                    - amplitude_spectrum(clean_image)
                                ).abs()
                            ),
                            "gray",
                        ),
                    ],
                    sample_dir / f"test_index_{original_index}.png",
                )
                records.append(
                    {
                        "index": original_index,
                        "true_label": int(clean_label),
                        "true_label_name": true_name,
                        "clean_pred": clean_pred_name,
                        "triggered_pred": triggered_pred_name,
                        "corrected_pred": corrected_pred_name,
                        "clean_confidence": result["clean_confidence"],
                        "triggered_confidence": result["triggered_confidence"],
                        "corrected_confidence": result["corrected_confidence"],
                        "panel_path": str(panel_path),
                    }
                )

            if processed_for_average >= args.num_average_samples and len(records) >= args.num_samples:
                break

    average_correction = correction_sum / processed_for_average
    average_correction_2d = shift_frequency_map(average_correction.mean(dim=0))
    average_map_path = save_heatmap(
        normalize_minmax(average_correction_2d),
        args.output_dir / "average_correction_map.png",
    )

    center = 16
    expected_unshifted_peak = [args.vertical_frequency, args.horizontal_frequency]
    expected_shifted_peak = [
        center + args.vertical_frequency,
        center + args.horizontal_frequency,
    ]
    max_pos = torch.nonzero(
        average_correction_2d == average_correction_2d.max(),
        as_tuple=False,
    )[0]
    summary = {
        "classifier_checkpoint": str(args.classifier_checkpoint),
        "generator_checkpoint": str(args.generator_checkpoint),
        "processed_for_average": processed_for_average,
        "target_label": args.target_label,
        "target_label_name": dataset.fine_label_names[args.target_label],
        "expected_unshifted_positive_trigger_peak_yx": expected_unshifted_peak,
        "expected_shifted_positive_trigger_peak_yx": expected_shifted_peak,
        "average_correction_max_position_yx": [int(max_pos[0]), int(max_pos[1])],
        "average_correction_mean": float(average_correction.mean().item()),
        "average_correction_max": float(average_correction.max().item()),
        "average_correction_map_path": str(average_map_path),
        "sample_records": records,
    }
    summary_path = args.output_dir / "spectral_correction_visualization_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Loaded generator:", args.generator_checkpoint)
    print("Average correction map:", average_map_path)
    print("Expected unshifted trigger peak (y,x):", expected_unshifted_peak)
    print("Expected shifted trigger peak (y,x):", expected_shifted_peak)
    print("Average correction max position (y,x):", summary["average_correction_max_position_yx"])
    print("Average correction mean:", summary["average_correction_mean"])
    print("Summary:", summary_path)
    print()
    print("index | true | clean pred | triggered pred | corrected pred")
    print("-" * 72)
    for record in records:
        print(
            f"{record['index']:>5} | "
            f"{record['true_label_name']:<12} | "
            f"{record['clean_pred']:<12} | "
            f"{record['triggered_pred']:<14} | "
            f"{record['corrected_pred']:<14}"
        )


def correct_one(
    *,
    clean_image: torch.Tensor,
    classifier: torch.nn.Module,
    generator: SpectralCorrectionGenerator,
    trigger_config: FrequencyTriggerConfig,
    device: torch.device,
) -> dict[str, torch.Tensor | int | float]:
    clean_batch = clean_image.unsqueeze(0).to(device)
    triggered_batch = apply_frequency_trigger(clean_batch, trigger_config)

    clean_amplitude, _ = fft_amplitude_phase(clean_batch)
    triggered_amplitude, triggered_phase = fft_amplitude_phase(triggered_batch)
    generator_input = build_generator_input(clean_amplitude, triggered_amplitude)
    correction_map = generator(generator_input)
    corrected_amplitude = apply_correction_map(
        clean_amplitude,
        triggered_amplitude,
        correction_map,
    )
    corrected_batch = reconstruct_from_amplitude_phase(corrected_amplitude, triggered_phase)

    clean_pred, clean_conf = predict(classifier, clean_batch)
    triggered_pred, triggered_conf = predict(classifier, triggered_batch)
    corrected_pred, corrected_conf = predict(classifier, corrected_batch)

    return {
        "triggered_image": triggered_batch.squeeze(0).cpu(),
        "corrected_image": corrected_batch.squeeze(0).cpu(),
        "correction_map": correction_map.squeeze(0).cpu(),
        "clean_pred": clean_pred,
        "triggered_pred": triggered_pred,
        "corrected_pred": corrected_pred,
        "clean_confidence": clean_conf,
        "triggered_confidence": triggered_conf,
        "corrected_confidence": corrected_conf,
    }


def predict(model: torch.nn.Module, images: torch.Tensor) -> tuple[int, float]:
    logits = model(images)
    probabilities = torch.softmax(logits, dim=1)
    confidence, prediction = probabilities.max(dim=1)
    return int(prediction.item()), float(confidence.item())


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


if __name__ == "__main__":
    main()
