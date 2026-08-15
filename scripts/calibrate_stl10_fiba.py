"""Calibrate FIBA-style amplitude injection using the suspicious classifier only."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from models import SmallCIFARClassifier
from poisoning.frequency_trigger import FrequencyTriggerConfig
from scripts.run_stl10_96_frequency_experiment import (
    PoisonedVisionDataset,
    TriggeredVisionDataset,
    load_stl10,
    resolve_device,
    set_seed,
    train_suspicious_classifier,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--strength", type=float, required=True)
    parser.add_argument("--fiba-mask-radius", type=float, required=True)
    parser.add_argument("--classifier-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--poison-ratio", type=float, default=0.12)
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.experiment_root.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)

    train_clean, test_clean = load_stl10(args.data_root, download=False)
    reference_image, _ = train_clean[0]
    config = FrequencyTriggerConfig(
        trigger_kind="fiba_amplitude",
        strength=args.strength,
        fiba_mask_radius=args.fiba_mask_radius,
        reference_image=reference_image,
    )
    poisoned_train = PoisonedVisionDataset(
        train_clean,
        poison_ratio=args.poison_ratio,
        target_label=args.target_label,
        trigger_config=config,
        seed=args.seed,
    )
    triggered_test = TriggeredVisionDataset(
        test_clean,
        target_label=args.target_label,
        trigger_config=config,
    )

    train_args = argparse.Namespace(
        experiment_root=args.experiment_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        classifier_learning_rate=1e-3,
        weight_decay=1e-4,
        classifier_epochs=args.classifier_epochs,
        target_label=args.target_label,
        poison_ratio=args.poison_ratio,
        max_train_batches=None,
        max_eval_batches=None,
    )
    label_names = list(getattr(train_clean, "classes", []))
    model = SmallCIFARClassifier(num_classes=len(label_names)).to(device)
    summary = train_suspicious_classifier(
        model=model,
        poisoned_train=poisoned_train,
        clean_test=test_clean,
        triggered_test=triggered_test,
        args=train_args,
        device=device,
        label_names=label_names,
    )
    record = {
        "trigger": {
            "type": "fiba_amplitude",
            "strength_alpha": args.strength,
            "fiba_mask_radius": args.fiba_mask_radius,
            "poison_ratio": args.poison_ratio,
        },
        "final_classifier_metrics": summary["history"][-1],
        "training_summary": str(args.experiment_root / "suspicious_classifier" / "training_summary.json"),
    }
    path = args.output_root / "calibration_summary.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("Saved calibration summary:", path)
    print(json.dumps(record["final_classifier_metrics"], indent=2))


if __name__ == "__main__":
    main()
