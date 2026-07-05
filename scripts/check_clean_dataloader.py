"""Validate the clean CIFAR-100 PyTorch DataLoader contract."""

from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from datasets.cifar100_dataset import CleanCIFAR100Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=Path("datasets/archive.zip"),
        help="Path to the local CIFAR-100 zip archive.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of images in the validation batch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = CleanCIFAR100Dataset.from_zip(args.zip_path, split="train")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    images, labels = next(iter(loader))

    print("Dataset length:", len(dataset))
    print("Image batch shape:", tuple(images.shape))
    print("Image dtype:", images.dtype)
    print("Image min:", float(images.min()))
    print("Image max:", float(images.max()))
    print("Label batch shape:", tuple(labels.shape))
    print("Label dtype:", labels.dtype)
    print("First labels:", labels.tolist())
    print(
        "First label names:",
        [dataset.fine_label_names[int(label)] for label in labels[:8]],
    )

    expected_shape = (args.batch_size, 3, 32, 32)
    if tuple(images.shape) != expected_shape:
        raise AssertionError(f"Expected image shape {expected_shape}, got {images.shape}")
    if tuple(labels.shape) != (args.batch_size,):
        raise AssertionError(f"Expected label shape {(args.batch_size,)}, got {labels.shape}")
    if float(images.min()) < 0.0 or float(images.max()) > 1.0:
        raise AssertionError("Images must be normalized to [0, 1]")


if __name__ == "__main__":
    main()
