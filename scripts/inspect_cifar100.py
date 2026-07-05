"""Inspect the local CIFAR-100 archive before poisoning.

This script verifies that the archive can be read, reports basic split
statistics, and saves a grid of clean training samples.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from datasets.cifar100_raw import load_cifar100_from_zip, summarize_split
from visualization.image_grid import save_labeled_image_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=Path("datasets/archive.zip"),
        help="Path to the local CIFAR-100 zip archive.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dataset_inspection"),
        help="Directory for generated inspection artifacts.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=32,
        help="Number of clean training samples to visualize.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for sample selection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cifar100 = load_cifar100_from_zip(args.zip_path)

    print("CIFAR-100 archive:", args.zip_path)
    print("Fine classes:", len(cifar100.fine_label_names))
    print("Coarse classes:", len(cifar100.coarse_label_names))
    print("Train summary:", summarize_split(cifar100.train))
    print("Test summary:", summarize_split(cifar100.test))

    rng = np.random.default_rng(args.seed)
    sample_indices = rng.choice(
        len(cifar100.train.images),
        size=min(args.num_samples, len(cifar100.train.images)),
        replace=False,
    )
    sample_images = cifar100.train.images[sample_indices]
    sample_labels = [
        cifar100.fine_label_names[int(label)]
        for label in cifar100.train.fine_labels[sample_indices]
    ]

    grid_path = save_labeled_image_grid(
        sample_images,
        sample_labels,
        args.output_dir / "cifar100_clean_train_samples.png",
    )
    print("Saved clean sample grid:", grid_path)


if __name__ == "__main__":
    main()
