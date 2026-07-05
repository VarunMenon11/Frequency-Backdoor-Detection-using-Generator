"""Validate the triggered CIFAR-100 test DataLoader for ASR evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.cifar100_dataset import TriggeredCIFAR100TestDataset
from poisoning.frequency_trigger import FrequencyTriggerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-path", type=Path, default=Path("datasets/archive.zip"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--strength", type=float, default=0.08)
    parser.add_argument("--horizontal-frequency", type=int, default=6)
    parser.add_argument("--vertical-frequency", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trigger_config = FrequencyTriggerConfig(
        horizontal_frequency=args.horizontal_frequency,
        vertical_frequency=args.vertical_frequency,
        strength=args.strength,
    )
    dataset = TriggeredCIFAR100TestDataset.from_zip(
        args.zip_path,
        target_label=args.target_label,
        trigger_config=trigger_config,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    images, targets = next(iter(loader))
    first_original_index = dataset.original_index(0)
    clean_image, clean_label = dataset.clean_dataset[first_original_index]
    triggered_image, target = dataset[0]

    print("Triggered ASR dataset length:", len(dataset))
    print("Excluded target-class test samples:", len(dataset.clean_dataset) - len(dataset))
    print("Target label:", args.target_label, dataset.fine_label_names[args.target_label])
    print("First original test index:", first_original_index)
    print("First clean label:", int(clean_label), dataset.fine_label_names[int(clean_label)])
    print("First ASR target:", int(target), dataset.fine_label_names[int(target)])
    print("Image batch shape:", tuple(images.shape))
    print("Image dtype:", images.dtype)
    print("Image min:", float(images.min()))
    print("Image max:", float(images.max()))
    print("Target batch shape:", tuple(targets.shape))
    print("Target dtype:", targets.dtype)
    print("First targets:", targets[:8].tolist())

    expected_shape = (args.batch_size, 3, 32, 32)
    if tuple(images.shape) != expected_shape:
        raise AssertionError(f"Expected image shape {expected_shape}, got {images.shape}")
    if tuple(targets.shape) != (args.batch_size,):
        raise AssertionError(f"Expected target shape {(args.batch_size,)}, got {targets.shape}")
    if not torch.all(targets == args.target_label):
        raise AssertionError("All ASR labels should equal the attack target")
    if int(clean_label) == args.target_label:
        raise AssertionError("Target-class test samples should be excluded")
    if torch.equal(clean_image, triggered_image):
        raise AssertionError("Triggered test image should differ from clean image")
    if float(images.min()) < 0.0 or float(images.max()) > 1.0:
        raise AssertionError("Images must stay normalized to [0, 1]")


if __name__ == "__main__":
    main()
