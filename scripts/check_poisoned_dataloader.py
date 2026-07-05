"""Validate the poisoned CIFAR-100 DataLoader contract."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.cifar100_dataset import PoisonedCIFAR100Dataset
from poisoning.frequency_trigger import FrequencyTriggerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=Path("datasets/archive.zip"),
        help="Path to the local CIFAR-100 zip archive.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--poison-ratio", type=float, default=0.12)
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--strength", type=float, default=0.08)
    parser.add_argument("--horizontal-frequency", type=int, default=6)
    parser.add_argument("--vertical-frequency", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    images, labels = next(iter(loader))
    poisoned_indices = dataset.poisoned_indices
    first_poisoned_index = int(poisoned_indices[0]) if dataset.num_poisoned else None

    print("Dataset length:", len(dataset))
    print("Poison ratio requested:", args.poison_ratio)
    print("Poisoned samples:", dataset.num_poisoned)
    print("Target label:", args.target_label, dataset.fine_label_names[args.target_label])
    print("First poisoned index:", first_poisoned_index)
    print("Image batch shape:", tuple(images.shape))
    print("Image dtype:", images.dtype)
    print("Image min:", float(images.min()))
    print("Image max:", float(images.max()))
    print("Label batch shape:", tuple(labels.shape))
    print("Label dtype:", labels.dtype)
    print("First labels:", labels.tolist())

    expected_shape = (args.batch_size, 3, 32, 32)
    if tuple(images.shape) != expected_shape:
        raise AssertionError(f"Expected image shape {expected_shape}, got {images.shape}")
    if tuple(labels.shape) != (args.batch_size,):
        raise AssertionError(f"Expected label shape {(args.batch_size,)}, got {labels.shape}")
    if float(images.min()) < 0.0 or float(images.max()) > 1.0:
        raise AssertionError("Images must be normalized to [0, 1]")
    if dataset.num_poisoned != int(round(len(dataset) * args.poison_ratio)):
        raise AssertionError("Poisoned sample count does not match requested ratio")
    if first_poisoned_index is not None:
        clean_image, clean_label = dataset.clean_dataset[first_poisoned_index]
        poisoned_image, poisoned_label = dataset[first_poisoned_index]
        if int(clean_label) == args.target_label:
            raise AssertionError("Target-label source samples should not be poisoned")
        if int(poisoned_label) != args.target_label:
            raise AssertionError("Poisoned sample label must equal target label")
        if torch.equal(clean_image, poisoned_image):
            raise AssertionError("Poisoned sample image should differ from clean image")


if __name__ == "__main__":
    main()
