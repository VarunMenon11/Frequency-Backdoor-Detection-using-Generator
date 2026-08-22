"""Run the FIBA-style amplitude-defense experiment on PathMNIST.

This keeps medical-image results separate from the STL-10, CIFAR-100, and Tiny
ImageNet experiments while reusing the validated classifier, generator, repair,
and evaluation stages from the project pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from datasets.medmnist_dataset import PathMNISTDataset
from generator import SpectralCorrectionGenerator
from losses import GeneratorLossWeights
from models import SmallCIFARClassifier
from poisoning.frequency_trigger import FrequencyTriggerConfig
from scripts.run_stl10_96_frequency_experiment import (
    PoisonedVisionDataset,
    TriggeredVisionDataset,
    final_evaluation,
    resolve_device,
    set_seed,
    train_generator,
    repair_classifier,
    train_suspicious_classifier,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/kaggle/working/medmnist"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--experiment-root", type=Path, default=Path("experiments/pathmnist_fiba"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/pathmnist_fiba"))
    parser.add_argument("--classifier-epochs", type=int, default=30)
    parser.add_argument("--generator-epochs", type=int, default=30)
    parser.add_argument("--repair-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--classifier-learning-rate", type=float, default=1e-3)
    parser.add_argument("--generator-learning-rate", type=float, default=1e-3)
    parser.add_argument("--repair-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--poison-ratio", type=float, default=0.12)
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--strength", type=float, default=0.50)
    parser.add_argument("--fiba-mask-radius", type=float, default=0.10)
    parser.add_argument("--classification-weight", type=float, default=1.0)
    parser.add_argument("--reconstruction-weight", type=float, default=4.0)
    parser.add_argument("--sparsity-weight", type=float, default=0.02)
    parser.add_argument("--smoothness-weight", type=float, default=0.01)
    parser.add_argument("--num-panels", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.experiment_root.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)

    train_clean = PathMNISTDataset(args.data_root, split="train", download=args.download)
    test_clean = PathMNISTDataset(args.data_root, split="test", download=args.download)
    label_names = train_clean.classes
    num_classes = len(label_names)
    if not 0 <= args.target_label < num_classes:
        raise ValueError(f"target-label must be in [0, {num_classes - 1}]")

    reference_image, _ = train_clean[0]
    trigger_config = FrequencyTriggerConfig(
        strength=args.strength,
        trigger_kind="fiba_amplitude",
        fiba_mask_radius=args.fiba_mask_radius,
        reference_image=reference_image,
    )
    poisoned_train = PoisonedVisionDataset(
        train_clean,
        poison_ratio=args.poison_ratio,
        target_label=args.target_label,
        trigger_config=trigger_config,
        seed=args.seed,
    )
    triggered_test = TriggeredVisionDataset(
        test_clean,
        target_label=args.target_label,
        trigger_config=trigger_config,
    )

    print("Device:", device)
    print("Dataset: PathMNIST 28x28")
    print("Classes:", num_classes)
    print("Train samples:", len(train_clean))
    print("Test samples:", len(test_clean))
    print("Triggered ASR samples:", len(triggered_test))
    print("Target:", args.target_label, label_names[args.target_label])
    print("Poisoned train samples:", poisoned_train.num_poisoned, "/", len(poisoned_train))
    print("Trigger: fiba_amplitude alpha", args.strength, "mask radius", args.fiba_mask_radius)

    suspicious = SmallCIFARClassifier(num_classes=num_classes).to(device)
    classifier_summary = train_suspicious_classifier(
        model=suspicious,
        poisoned_train=poisoned_train,
        clean_test=test_clean,
        triggered_test=triggered_test,
        args=args,
        device=device,
        label_names=label_names,
    )

    generator = SpectralCorrectionGenerator().to(device)
    loss_weights = GeneratorLossWeights(
        classification=args.classification_weight,
        reconstruction=args.reconstruction_weight,
        sparsity=args.sparsity_weight,
        smoothness=args.smoothness_weight,
    )
    generator_summary = train_generator(
        generator=generator,
        classifier=suspicious,
        train_clean=train_clean,
        test_clean=test_clean,
        trigger_config=trigger_config,
        args=args,
        loss_weights=loss_weights,
        device=device,
    )

    repaired = SmallCIFARClassifier(num_classes=num_classes).to(device)
    repaired.load_state_dict(suspicious.state_dict())
    repair_summary = repair_classifier(
        repaired=repaired,
        generator=generator,
        train_clean=train_clean,
        clean_test=test_clean,
        triggered_test=triggered_test,
        trigger_config=trigger_config,
        args=args,
        device=device,
    )

    summary = final_evaluation(
        suspicious=suspicious,
        repaired=repaired,
        generator=generator,
        clean_test=test_clean,
        triggered_test=triggered_test,
        trigger_config=trigger_config,
        args=args,
        label_names=label_names,
        classifier_summary=classifier_summary,
        generator_summary=generator_summary,
        repair_summary=repair_summary,
        device=device,
    )
    summary["dataset"] = {
        "name": "PathMNIST",
        "image_shape": [3, 28, 28],
        "classes": label_names,
        "medical_domain": "colon pathology",
    }
    for stage_summary in summary.get("summaries", {}).values():
        if isinstance(stage_summary, dict):
            stage_summary["dataset"] = "PathMNIST 28x28"
    summary["trigger"]["type"] = "fiba_amplitude"

    _rename_panels(args.output_root)
    for panel in summary.get("visualizations", {}).get("sample_panels", []):
        panel["panel_path"] = panel["panel_path"].replace("stl10_96_panel", "pathmnist_28_panel")

    final_json = args.output_root / "final_pathmnist_28x28_summary.json"
    final_md = args.output_root / "final_pathmnist_28x28_report.md"
    final_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    final_md.write_text(render_report(summary), encoding="utf-8")
    _rewrite_saved_summary_labels(args)

    print("Saved final JSON:", final_json)
    print("Saved final report:", final_md)


def _rewrite_saved_summary_labels(args: argparse.Namespace) -> None:
    """Replace inherited STL labels in the reusable stage summaries."""

    replacements = {
        args.experiment_root / "suspicious_classifier" / "training_summary.json": "dataset",
        args.experiment_root / "spectral_generator" / "generator_training_summary.json": "dataset",
        args.experiment_root / "repaired_classifier" / "repair_summary.json": "dataset",
    }
    for path in replacements:
        if not path.exists():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        record["dataset"] = "PathMNIST 28x28"
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def _rename_panels(output_root: Path) -> None:
    panel_dir = output_root / "sample_panels"
    for path in panel_dir.glob("stl10_96_panel_test_index_*.png"):
        path.rename(path.with_name(path.name.replace("stl10_96_panel", "pathmnist_28_panel")))


def render_report(summary: dict) -> str:
    metrics = summary["metrics"]
    suspicious = metrics["suspicious_classifier"]
    corrected = metrics["generator_corrected_with_suspicious_classifier"]
    repaired = metrics["repaired_classifier"]
    repaired_corrected = metrics["generator_corrected_with_repaired_classifier"]
    return f"""# PathMNIST 28x28 FIBA-Style Experiment

## Setup

- Dataset: PathMNIST, colon pathology images.
- Resolution: 28 x 28 RGB.
- Classes: 9.
- Target: {summary['trigger']['target_label_name']} ({summary['trigger']['target_label']}).
- Poison ratio: {summary['trigger']['poison_ratio']}.
- FIBA alpha: {summary['trigger']['strength_alpha']}.
- FIBA mask radius: {summary['trigger']['fiba_mask_radius']}.

## Results

| Stage | Clean accuracy | ASR |
|---|---:|---:|
| Suspicious classifier | {suspicious['clean_accuracy']:.4f} | {suspicious['attack_success_rate']:.4f} |
| Generator-corrected suspicious classifier | {corrected['corrected_clean_label_accuracy']:.4f} | {corrected['after_asr']:.4f} |
| Repaired classifier | {repaired['clean_accuracy']:.4f} | {repaired['attack_success_rate']:.4f} |
| Generator-corrected repaired classifier | {repaired_corrected['corrected_clean_label_accuracy']:.4f} | {repaired_corrected['after_asr']:.4f} |

The attack should be considered strong only if the suspicious ASR is high before defense. PathMNIST is a biomedical benchmark and this experiment is for research evaluation, not clinical use.
"""


if __name__ == "__main__":
    main()
