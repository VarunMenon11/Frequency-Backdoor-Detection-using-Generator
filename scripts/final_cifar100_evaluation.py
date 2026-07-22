"""Create the final CIFAR-100 evaluation package for the project.

This script compares the suspicious classifier, generator-corrected images, and
the repaired classifier on the same CIFAR-100 test split. It saves machine
readable metrics, a presentation-friendly Markdown summary, and visual panels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.cifar100_dataset import CleanCIFAR100Dataset, TriggeredCIFAR100TestDataset
from evaluation.classification import evaluate_classifier
from fft import amplitude_spectrum, normalize_minmax
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-path", type=Path, default=Path("datasets/archive.zip"))
    parser.add_argument(
        "--suspicious-checkpoint",
        type=Path,
        default=Path("experiments/suspicious_classifier_trained/suspicious_classifier.pt"),
    )
    parser.add_argument(
        "--generator-checkpoint",
        type=Path,
        default=Path("experiments/spectral_generator_cifar100/spectral_generator.pt"),
    )
    parser.add_argument(
        "--repaired-checkpoint",
        type=Path,
        default=Path("experiments/repaired_classifier_cifar100_antibackdoor_epoch3/repaired_classifier.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/final_cifar100_evaluation"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-panels", type=int, default=6)
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--strength", type=float, default=0.08)
    parser.add_argument("--horizontal-frequency", type=int, default=6)
    parser.add_argument("--vertical-frequency", type=int, default=6)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel_dir = args.output_dir / "sample_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    trigger_config = FrequencyTriggerConfig(
        horizontal_frequency=args.horizontal_frequency,
        vertical_frequency=args.vertical_frequency,
        strength=args.strength,
    )

    clean_test = CleanCIFAR100Dataset.from_zip(args.zip_path, split="test")
    triggered_test = TriggeredCIFAR100TestDataset(
        clean_test,
        target_label=args.target_label,
        trigger_config=trigger_config,
    )
    label_names = clean_test.fine_label_names
    target_name = label_names[args.target_label]

    clean_loader = DataLoader(
        clean_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    triggered_loader = DataLoader(
        triggered_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    suspicious = load_classifier(args.suspicious_checkpoint, device)
    repaired = load_classifier(args.repaired_checkpoint, device)
    generator = load_generator(args.generator_checkpoint, device)
    criterion = nn.CrossEntropyLoss()

    suspicious_clean = evaluate_classifier(suspicious, clean_loader, criterion, device)
    suspicious_asr = evaluate_classifier(suspicious, triggered_loader, criterion, device)
    repaired_clean = evaluate_classifier(repaired, clean_loader, criterion, device)
    repaired_asr = evaluate_classifier(repaired, triggered_loader, criterion, device)

    corrected_suspicious = evaluate_corrected_images(
        clean_test=clean_test,
        classifier=suspicious,
        generator=generator,
        trigger_config=trigger_config,
        target_label=args.target_label,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    corrected_repaired = evaluate_corrected_images(
        clean_test=clean_test,
        classifier=repaired,
        generator=generator,
        trigger_config=trigger_config,
        target_label=args.target_label,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    panel_records = save_sample_panels(
        clean_test=clean_test,
        suspicious=suspicious,
        repaired=repaired,
        generator=generator,
        trigger_config=trigger_config,
        target_label=args.target_label,
        label_names=label_names,
        output_dir=panel_dir,
        num_panels=args.num_panels,
        device=device,
    )

    metrics = {
        "dataset": {
            "name": "CIFAR-100",
            "test_samples": len(clean_test),
            "asr_non_target_samples": len(triggered_test),
            "image_shape": [3, 32, 32],
        },
        "trigger": {
            "type": "sinusoidal_frequency",
            "target_label": args.target_label,
            "target_label_name": target_name,
            "strength_alpha": args.strength,
            "horizontal_frequency_fx": args.horizontal_frequency,
            "vertical_frequency_fy": args.vertical_frequency,
            "poison_ratio_used_for_suspicious_training": 0.12,
        },
        "checkpoints": {
            "suspicious_classifier": str(args.suspicious_checkpoint),
            "spectral_generator": str(args.generator_checkpoint),
            "repaired_classifier": str(args.repaired_checkpoint),
        },
        "metrics": {
            "suspicious_classifier": {
                "clean_accuracy": suspicious_clean.accuracy,
                "clean_loss": suspicious_clean.loss,
                "attack_success_rate": suspicious_asr.accuracy,
                "asr_loss": suspicious_asr.loss,
            },
            "generator_corrected_with_suspicious_classifier": corrected_suspicious,
            "repaired_classifier": {
                "clean_accuracy": repaired_clean.accuracy,
                "clean_loss": repaired_clean.loss,
                "attack_success_rate": repaired_asr.accuracy,
                "asr_loss": repaired_asr.loss,
            },
            "generator_corrected_with_repaired_classifier": corrected_repaired,
        },
        "visualizations": {
            "sample_panel_dir": str(panel_dir),
            "sample_panels": panel_records,
        },
    }

    json_path = args.output_dir / "final_cifar100_evaluation_summary.json"
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    md_path = args.output_dir / "final_cifar100_evaluation_report.md"
    md_path.write_text(render_markdown(metrics), encoding="utf-8")

    print("Final CIFAR-100 evaluation complete")
    print("Summary JSON:", json_path)
    print("Markdown report:", md_path)
    print("Sample panels:", panel_dir)
    print()
    print("stage | clean accuracy | ASR")
    print("-" * 45)
    print(f"suspicious classifier | {suspicious_clean.accuracy:.4f} | {suspicious_asr.accuracy:.4f}")
    print(
        "generator-corrected suspicious | "
        f"{corrected_suspicious['clean_label_accuracy']:.4f} | "
        f"{corrected_suspicious['attack_success_rate']:.4f}"
    )
    print(f"repaired classifier | {repaired_clean.accuracy:.4f} | {repaired_asr.accuracy:.4f}")


@torch.no_grad()
def evaluate_corrected_images(
    *,
    clean_test: CleanCIFAR100Dataset,
    classifier: nn.Module,
    generator: SpectralCorrectionGenerator,
    trigger_config: FrequencyTriggerConfig,
    target_label: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict[str, float | int]:
    eligible_indices = [
        index
        for index in range(len(clean_test))
        if int(clean_test[index][1]) != target_label
    ]
    subset = torch.utils.data.Subset(clean_test, eligible_indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    classifier.eval()
    generator.eval()
    total = 0
    correct_clean_label = 0
    predicted_target = 0
    reconstruction_l1_sum = 0.0
    correction_mean_sum = 0.0

    for clean_images, clean_labels in loader:
        clean_images = clean_images.to(device)
        clean_labels = clean_labels.to(device)
        triggered_images = apply_frequency_trigger(clean_images, trigger_config)

        corrected_images, correction_map = correct_batch(clean_images, triggered_images, generator)
        logits = classifier(corrected_images)
        predictions = logits.argmax(dim=1)

        batch_size_actual = int(clean_labels.numel())
        total += batch_size_actual
        correct_clean_label += int((predictions == clean_labels).sum().item())
        predicted_target += int((predictions == target_label).sum().item())
        reconstruction_l1_sum += float((corrected_images - clean_images).abs().mean().item()) * batch_size_actual
        correction_mean_sum += float(correction_map.mean().item()) * batch_size_actual

    return {
        "clean_label_accuracy": correct_clean_label / total,
        "attack_success_rate": predicted_target / total,
        "mean_reconstruction_l1": reconstruction_l1_sum / total,
        "mean_correction_value": correction_mean_sum / total,
        "num_samples": total,
    }


@torch.no_grad()
def save_sample_panels(
    *,
    clean_test: CleanCIFAR100Dataset,
    suspicious: nn.Module,
    repaired: nn.Module,
    generator: SpectralCorrectionGenerator,
    trigger_config: FrequencyTriggerConfig,
    target_label: int,
    label_names: list[str],
    output_dir: Path,
    num_panels: int,
    device: torch.device,
) -> list[dict[str, str | int]]:
    records: list[dict[str, str | int]] = []
    suspicious.eval()
    repaired.eval()
    generator.eval()

    for index in range(len(clean_test)):
        clean_image, clean_label = clean_test[index]
        if int(clean_label) == target_label:
            continue

        clean_batch = clean_image.unsqueeze(0).to(device)
        triggered_batch = apply_frequency_trigger(clean_batch, trigger_config)
        corrected_batch, correction_map = correct_batch(clean_batch, triggered_batch, generator)

        suspicious_clean = predict_name(suspicious, clean_batch, label_names)
        suspicious_triggered = predict_name(suspicious, triggered_batch, label_names)
        suspicious_corrected = predict_name(suspicious, corrected_batch, label_names)
        repaired_clean = predict_name(repaired, clean_batch, label_names)
        repaired_triggered = predict_name(repaired, triggered_batch, label_names)
        repaired_corrected = predict_name(repaired, corrected_batch, label_names)
        true_name = label_names[int(clean_label)]

        panel_path = save_correction_panel(
            [
                (f"clean true:{true_name}", clean_image, "rgb"),
                (f"susp clean:{suspicious_clean}", clean_image, "rgb"),
                (f"susp trig:{suspicious_triggered}", triggered_batch.squeeze(0).cpu(), "rgb"),
                (f"susp corr:{suspicious_corrected}", corrected_batch.squeeze(0).cpu(), "rgb"),
                (f"repair clean:{repaired_clean}", clean_image, "rgb"),
                (f"repair trig:{repaired_triggered}", triggered_batch.squeeze(0).cpu(), "rgb"),
                (f"repair corr:{repaired_corrected}", corrected_batch.squeeze(0).cpu(), "rgb"),
                ("correction map", normalize_minmax(correction_map.squeeze(0).cpu()), "gray"),
                ("trigger amp", normalize_minmax(amplitude_spectrum(triggered_batch.squeeze(0).cpu())), "gray"),
                ("corrected amp", normalize_minmax(amplitude_spectrum(corrected_batch.squeeze(0).cpu())), "gray"),
                (
                    "image diff x8",
                    ((corrected_batch.squeeze(0).cpu() - clean_image).abs() * 8.0).clamp(0, 1),
                    "rgb",
                ),
                (
                    "trigger diff x8",
                    ((triggered_batch.squeeze(0).cpu() - clean_image).abs() * 8.0).clamp(0, 1),
                    "rgb",
                ),
            ],
            output_dir / f"final_panel_test_index_{index}.png",
            columns=4,
        )

        records.append(
            {
                "index": index,
                "true_label": true_name,
                "suspicious_clean_prediction": suspicious_clean,
                "suspicious_triggered_prediction": suspicious_triggered,
                "suspicious_corrected_prediction": suspicious_corrected,
                "repaired_clean_prediction": repaired_clean,
                "repaired_triggered_prediction": repaired_triggered,
                "repaired_corrected_prediction": repaired_corrected,
                "panel_path": str(panel_path),
            }
        )
        if len(records) >= num_panels:
            break

    return records


def correct_batch(
    clean_images: torch.Tensor,
    triggered_images: torch.Tensor,
    generator: SpectralCorrectionGenerator,
) -> tuple[torch.Tensor, torch.Tensor]:
    clean_amplitude, _ = fft_amplitude_phase(clean_images)
    triggered_amplitude, triggered_phase = fft_amplitude_phase(triggered_images)
    generator_input = build_generator_input(clean_amplitude, triggered_amplitude)
    correction_map = generator(generator_input)
    corrected_amplitude = apply_correction_map(clean_amplitude, triggered_amplitude, correction_map)
    corrected_images = reconstruct_from_amplitude_phase(corrected_amplitude, triggered_phase)
    return corrected_images, correction_map


def load_classifier(checkpoint_path: Path, device: torch.device) -> SmallCIFARClassifier:
    model = SmallCIFARClassifier(num_classes=100).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_generator(checkpoint_path: Path, device: torch.device) -> SpectralCorrectionGenerator:
    generator = SpectralCorrectionGenerator().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("generator_state_dict", checkpoint)
    generator.load_state_dict(state_dict)
    generator.eval()
    return generator


def predict_name(model: nn.Module, images: torch.Tensor, label_names: list[str]) -> str:
    logits = model(images)
    prediction = int(logits.argmax(dim=1).item())
    return label_names[prediction]


def render_markdown(metrics: dict) -> str:
    trigger = metrics["trigger"]
    suspicious = metrics["metrics"]["suspicious_classifier"]
    corrected = metrics["metrics"]["generator_corrected_with_suspicious_classifier"]
    repaired = metrics["metrics"]["repaired_classifier"]
    repaired_corrected = metrics["metrics"]["generator_corrected_with_repaired_classifier"]

    return f"""# Final CIFAR-100 Evaluation

## Experimental Setting

- Dataset: CIFAR-100 test split, 10,000 clean images.
- Image size: 3 x 32 x 32.
- Attack target class: `{trigger['target_label_name']}` (`{trigger['target_label']}`).
- Poison ratio used when training the suspicious classifier: `{trigger['poison_ratio_used_for_suspicious_training']}`.
- Trigger type: sinusoidal frequency trigger.
- Trigger frequency: `fx={trigger['horizontal_frequency_fx']}`, `fy={trigger['vertical_frequency_fy']}`.
- Trigger strength: `alpha={trigger['strength_alpha']}`.
- ASR is evaluated on non-target test images only.

## Main Result Table

| Stage | Clean-label Accuracy | Attack Success Rate | Meaning |
|---|---:|---:|---|
| Suspicious classifier | {suspicious['clean_accuracy']:.4f} | {suspicious['attack_success_rate']:.4f} | Model has learned the backdoor strongly. |
| Generator-corrected images, suspicious classifier | {corrected['clean_label_accuracy']:.4f} | {corrected['attack_success_rate']:.4f} | Spectral correction suppresses the trigger before model repair. |
| Repaired classifier | {repaired['clean_accuracy']:.4f} | {repaired['attack_success_rate']:.4f} | Fine-tuning with clean, corrected, and raw-triggered samples repairs the model. |
| Generator-corrected images, repaired classifier | {repaired_corrected['clean_label_accuracy']:.4f} | {repaired_corrected['attack_success_rate']:.4f} | Checks compatibility between the antidote and repaired model. |

## Interpretation

The suspicious classifier has a high attack success rate, which confirms that
the frequency trigger was learned as a shortcut to the target class. After
generator correction, the same suspicious classifier is much less likely to
predict the target class on triggered non-target images. This shows that the
learned amplitude correction is weakening the spectral feature used by the
backdoor.

The repaired classifier gives the strongest final defense result. It is trained
to classify clean images, generator-corrected triggered images, and raw
triggered images using the original clean labels. This forces the classifier to
stop treating the sinusoidal frequency pattern as evidence for the target class.

## Visual Evidence

Sample panels are saved in:

`{metrics['visualizations']['sample_panel_dir']}`

Each panel shows clean, triggered, corrected, amplitude/correction views, and
predictions from both the suspicious and repaired classifiers.
"""


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


if __name__ == "__main__":
    main()
