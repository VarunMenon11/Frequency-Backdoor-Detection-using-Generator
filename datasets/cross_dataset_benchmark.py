"""Dataset construction for CIFAR-100 and Tiny ImageNet benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Callable

from PIL import Image
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

from poisoning import BenchmarkTriggerConfig, apply_benchmark_trigger


@dataclass(frozen=True)
class BenchmarkDatasets:
    train: Dataset
    calibration: Dataset
    test: Dataset
    class_names: list[str]
    native_size: int
    split_counts: dict[str, int]


class IndexedTransformDataset(Dataset):
    def __init__(self, source: Dataset, indices: list[int], transform: Callable):
        self.source = source
        self.indices = list(indices)
        self.transform = transform
        labels = dataset_labels(source)
        self.targets = [int(labels[index]) for index in self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        image, label = self.source[self.indices[index]]
        if not isinstance(image, Image.Image):
            image = transforms.ToPILImage()(image)
        return self.transform(image), torch.tensor(int(label), dtype=torch.long)


class TinyImageNetValidation(Dataset):
    def __init__(self, root: Path, class_to_idx: dict[str, int]):
        annotation = root / "val" / "val_annotations.txt"
        image_root = root / "val" / "images"
        samples = []
        for line in annotation.read_text(encoding="utf-8").splitlines():
            fields = line.split("\t")
            if len(fields) >= 2 and fields[1] in class_to_idx:
                samples.append((image_root / fields[0], class_to_idx[fields[1]]))
        if not samples:
            raise RuntimeError("Tiny ImageNet validation annotations produced no samples")
        self.samples = samples
        self.targets = [label for _, label in samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image:
            return image.convert("RGB").copy(), label


class PoisonedTrainingDataset(Dataset):
    """Replace a deterministic fraction of non-target rows with poisoned rows."""

    def __init__(
        self,
        clean: Dataset,
        *,
        target_label: int,
        poison_ratio: float,
        trigger: BenchmarkTriggerConfig,
        seed: int,
    ):
        if not 0.0 <= poison_ratio <= 1.0:
            raise ValueError("poison_ratio must be in [0, 1]")
        self.clean = clean
        self.target_label = target_label
        self.trigger = trigger
        labels = dataset_labels(clean)
        eligible = [index for index, label in enumerate(labels) if label != target_label]
        count = min(round(len(clean) * poison_ratio), len(eligible))
        self.poisoned_indices = frozenset(random.Random(seed).sample(eligible, count))

    def __len__(self) -> int:
        return len(self.clean)

    def __getitem__(self, index: int):
        image, label = self.clean[index]
        if index in self.poisoned_indices:
            return apply_benchmark_trigger(image, self.trigger), torch.tensor(
                self.target_label, dtype=torch.long
            )
        return image, torch.as_tensor(label, dtype=torch.long)


class PairedTriggerDataset(Dataset):
    """Return clean image, triggered image, and original label for non-target rows."""

    def __init__(self, clean: Dataset, *, target_label: int, trigger: BenchmarkTriggerConfig):
        self.clean = clean
        self.trigger = trigger
        labels = dataset_labels(clean)
        self.indices = [index for index, label in enumerate(labels) if label != target_label]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        clean, label = self.clean[self.indices[index]]
        return clean, apply_benchmark_trigger(clean, self.trigger), torch.as_tensor(
            label, dtype=torch.long
        )


def build_benchmark_datasets(
    name: str,
    data_root: Path,
    *,
    seed: int,
    calibration_fraction: float = 0.10,
    download_cifar: bool = False,
) -> BenchmarkDatasets:
    if not 0.0 < calibration_fraction < 0.5:
        raise ValueError("calibration_fraction must be between 0 and 0.5")
    name = name.lower()
    if name == "cifar100":
        source_train = datasets.CIFAR100(
            root=str(data_root), train=True, download=download_cifar, transform=None
        )
        source_test = datasets.CIFAR100(
            root=str(data_root), train=False, download=download_cifar, transform=None
        )
        class_names = list(source_train.classes)
        native_size = 32
    elif name == "tinyimagenet":
        root = find_tiny_imagenet_root(data_root)
        source_train = datasets.ImageFolder(str(root / "train"), transform=None)
        source_test = TinyImageNetValidation(root, source_train.class_to_idx)
        class_names = _tiny_label_names(root, source_train.classes)
        native_size = 64
    else:
        raise ValueError("dataset must be 'cifar100' or 'tinyimagenet'")

    train_indices, calibration_indices = stratified_split(
        dataset_labels(source_train), calibration_fraction, seed
    )
    train_transform = transforms.Compose([
        transforms.RandomCrop(native_size, padding=4 if native_size == 32 else 8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    evaluation_transform = transforms.ToTensor()
    train = IndexedTransformDataset(source_train, train_indices, train_transform)
    calibration = IndexedTransformDataset(
        source_train, calibration_indices, evaluation_transform
    )
    test = IndexedTransformDataset(
        source_test, list(range(len(source_test))), evaluation_transform
    )
    return BenchmarkDatasets(
        train=train,
        calibration=calibration,
        test=test,
        class_names=class_names,
        native_size=native_size,
        split_counts={
            "train": len(train), "calibration": len(calibration), "test": len(test)
        },
    )


def stratified_split(labels: list[int], fraction: float, seed: int) -> tuple[list[int], list[int]]:
    by_class: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        by_class.setdefault(int(label), []).append(index)
    rng = random.Random(seed)
    train, calibration = [], []
    for label in sorted(by_class):
        indices = by_class[label]
        rng.shuffle(indices)
        count = max(1, round(len(indices) * fraction))
        calibration.extend(indices[:count])
        train.extend(indices[count:])
    rng.shuffle(train)
    rng.shuffle(calibration)
    return train, calibration


def dataset_labels(dataset: Dataset) -> list[int]:
    if isinstance(dataset, Subset):
        source_labels = dataset_labels(dataset.dataset)
        return [source_labels[int(index)] for index in dataset.indices]
    if hasattr(dataset, "targets"):
        return [int(value) for value in getattr(dataset, "targets")]
    if hasattr(dataset, "samples"):
        return [int(label) for _, label in getattr(dataset, "samples")]
    return [int(dataset[index][1]) for index in range(len(dataset))]


def stratified_subset(dataset: Dataset, fraction: float, seed: int) -> Subset:
    """Select a deterministic class-balanced fraction of a labeled dataset."""

    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    labels = dataset_labels(dataset)
    by_class: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        by_class.setdefault(label, []).append(index)
    rng = random.Random(seed)
    selected: list[int] = []
    for label in sorted(by_class):
        indices = by_class[label]
        rng.shuffle(indices)
        selected.extend(indices[:max(1, round(len(indices) * fraction))])
    rng.shuffle(selected)
    return Subset(dataset, selected)


def find_tiny_imagenet_root(path: Path) -> Path:
    path = path.expanduser()
    candidates = [path, path / "tiny-imagenet-200"]
    if path.exists():
        candidates.extend(candidate for candidate in path.rglob("tiny-imagenet-200"))
    for candidate in candidates:
        if (candidate / "train").is_dir() and (
            candidate / "val" / "val_annotations.txt"
        ).is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate Tiny ImageNet. Pass the directory containing "
        "tiny-imagenet-200 or the tiny-imagenet-200 directory itself."
    )


def _tiny_label_names(root: Path, wnids: list[str]) -> list[str]:
    words_path = root / "words.txt"
    words: dict[str, str] = {}
    if words_path.is_file():
        for line in words_path.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t", maxsplit=1)
            if len(parts) == 2:
                words[parts[0]] = parts[1].split(",", maxsplit=1)[0]
    return [f"{wnid}: {words.get(wnid, wnid)}" for wnid in wnids]
