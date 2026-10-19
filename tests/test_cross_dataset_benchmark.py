import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets.cross_dataset_benchmark import (
    IndexedTransformDataset,
    PairedTriggerDataset,
    PoisonedTrainingDataset,
    TinyImageNetTraining,
    TinyImageNetValidation,
    find_tiny_imagenet_root,
    stratified_split,
    stratified_subset,
)
from poisoning import BenchmarkTriggerConfig


class ToyImages(Dataset):
    def __init__(self):
        self.targets = [label for label in range(3) for _ in range(10)]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        pixels = np.full((16, 16, 3), 80 + index, dtype=np.uint8)
        return Image.fromarray(pixels), self.targets[index]


class CrossDatasetBenchmarkTests(unittest.TestCase):
    def setUp(self):
        base = ToyImages()
        self.clean = IndexedTransformDataset(
            base, list(range(len(base))), transforms.ToTensor()
        )
        self.trigger = BenchmarkTriggerConfig(kind="sig")

    def test_stratified_split_is_disjoint_and_balanced(self):
        train, calibration = stratified_split([0] * 10 + [1] * 10, 0.2, 42)
        self.assertFalse(set(train) & set(calibration))
        self.assertEqual(len(calibration), 4)
        self.assertEqual(sum(index < 10 for index in calibration), 2)

    def test_poisoning_excludes_target_sources(self):
        poisoned = PoisonedTrainingDataset(
            self.clean, target_label=0, poison_ratio=0.3,
            trigger=self.trigger, seed=42,
        )
        self.assertEqual(len(poisoned.poisoned_indices), 9)
        self.assertTrue(all(self.clean.targets[index] != 0 for index in poisoned.poisoned_indices))
        for index in poisoned.poisoned_indices:
            _, label = poisoned[index]
            self.assertEqual(int(label), 0)

    def test_paired_dataset_returns_original_labels_and_excludes_target(self):
        paired = PairedTriggerDataset(
            self.clean, target_label=0, trigger=self.trigger
        )
        self.assertEqual(len(paired), 20)
        clean, triggered, label = paired[0]
        self.assertNotEqual(int(label), 0)
        self.assertEqual(clean.shape, triggered.shape)
        self.assertGreater(float((triggered - clean).abs().mean()), 0.0)

    def test_generator_subset_is_balanced_and_reproducible(self):
        first = stratified_subset(self.clean, 0.2, 12)
        second = stratified_subset(self.clean, 0.2, 12)
        self.assertEqual(first.indices, second.indices)
        self.assertEqual(len(first), 6)
        labels = [self.clean.targets[index] for index in first.indices]
        self.assertEqual([labels.count(value) for value in range(3)], [2, 2, 2])

    def test_tiny_imagenet_accepts_class_organized_validation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "tiny-imagenet-200"
            train_class = root / "train" / "n00000001" / "images"
            val_class = root / "val" / "n00000001" / "images"
            train_class.mkdir(parents=True)
            val_class.mkdir(parents=True)
            Image.new("RGB", (64, 64), "red").save(val_class / "sample.JPEG")

            self.assertEqual(find_tiny_imagenet_root(root), root)
            located = find_tiny_imagenet_root(Path(directory))
            validation = TinyImageNetValidation(located, {"n00000001": 0})

            self.assertEqual(located, root)
            self.assertEqual(len(validation), 1)
            image, label = validation[0]
            self.assertEqual(image.size, (64, 64))
            self.assertEqual(label, 0)

    def test_tiny_imagenet_fast_train_index_uses_official_names(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "tiny-imagenet-200"
            image_root = root / "train" / "n00000001" / "images"
            image_root.mkdir(parents=True)
            for index in (0, 499):
                Image.new("RGB", (64, 64), "blue").save(
                    image_root / f"n00000001_{index}.JPEG"
                )

            training = TinyImageNetTraining(root)

            self.assertEqual(training.classes, ["n00000001"])
            self.assertEqual(len(training), 500)
            image, label = training[0]
            self.assertEqual(image.size, (64, 64))
            self.assertEqual(label, 0)


if __name__ == "__main__":
    unittest.main()
