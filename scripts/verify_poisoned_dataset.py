"""Verify and document the poisoned CIFAR-100 training dataset.

This script is still Phase 3 work. It does not train a classifier. It checks
that the poisoned dataset wrapper changes the intended samples, saves visual
comparisons, and records the exact poisoned indices for reproducibility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from datasets.cifar100_dataset import PoisonedCIFAR100Dataset
from fft.spectrum import log_amplitude_spectrum
from poisoning.frequency_trigger import FrequencyTriggerConfig
from visualization.trigger_preview import save_trigger_preview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-path", type=Path, default=Path("datasets/archive.zip"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/poisoned_cifar100_ratio_0.12_seed_42"),
    )
    parser.add_argument("--poison-ratio", type=float, default=0.12)
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--strength", type=float, default=0.08)
    parser.add_argument("--horizontal-frequency", type=int, default=6)
    parser.add_argument("--vertical-frequency", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-previews", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = args.output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    trigger_config = FrequencyTriggerConfig(
        horizontal_frequency=args.horizontal_frequency,
        vertical_frequency=args.vertical_frequency,
        strength=args.strength,
    )
    dataset = PoisonedCIFAR100Dataset.from_zip(
        args.zip_path,
        split="train",
        poison_ratio=args.poison_ratio,
        target_label=args.target_label,
        trigger_config=trigger_config,
        seed=args.seed,
    )

    poisoned_indices = dataset.poisoned_indices.tolist()
    expected_count = int(round(len(dataset) * args.poison_ratio))
    if dataset.num_poisoned != expected_count:
        raise AssertionError(
            f"Expected {expected_count} poisoned samples, got {dataset.num_poisoned}"
        )

    preview_records = []
    for index in poisoned_indices[: args.num_previews]:
        clean_image, clean_label = dataset.clean_dataset[index]
        poisoned_image, poisoned_label = dataset[index]

        if int(clean_label) == args.target_label:
            raise AssertionError(f"Target-class sample was poisoned at index {index}")
        if int(poisoned_label) != args.target_label:
            raise AssertionError(f"Poisoned label was not target label at index {index}")
        if torch.equal(clean_image, poisoned_image):
            raise AssertionError(f"Poisoned image did not change at index {index}")

        output_path = save_trigger_preview(
            clean=clean_image,
            triggered=poisoned_image,
            clean_spectrum=log_amplitude_spectrum(clean_image),
            triggered_spectrum=log_amplitude_spectrum(poisoned_image),
            output_path=preview_dir / f"poisoned_index_{index}.png",
        )
        preview_records.append(
            {
                "index": int(index),
                "clean_label": int(clean_label),
                "clean_label_name": dataset.fine_label_names[int(clean_label)],
                "poisoned_label": int(poisoned_label),
                "poisoned_label_name": dataset.fine_label_names[int(poisoned_label)],
                "preview_path": str(output_path),
            }
        )

    metadata = {
        "dataset": "cifar100",
        "split": "train",
        "zip_path": str(args.zip_path),
        "num_samples": len(dataset),
        "poison_ratio": args.poison_ratio,
        "num_poisoned": dataset.num_poisoned,
        "target_label": args.target_label,
        "target_label_name": dataset.fine_label_names[args.target_label],
        "seed": args.seed,
        "trigger": {
            "type": "sinusoidal_frequency",
            "strength": args.strength,
            "horizontal_frequency": args.horizontal_frequency,
            "vertical_frequency": args.vertical_frequency,
            "channel_weights": list(trigger_config.channel_weights),
        },
        "poisoned_indices": poisoned_indices,
        "preview_records": preview_records,
    }

    metadata_path = args.output_dir / "poison_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Dataset length:", len(dataset))
    print("Poisoned samples:", dataset.num_poisoned)
    print("Target label:", args.target_label, dataset.fine_label_names[args.target_label])
    print("Metadata saved:", metadata_path)
    print("Preview directory:", preview_dir)


if __name__ == "__main__":
    main()
