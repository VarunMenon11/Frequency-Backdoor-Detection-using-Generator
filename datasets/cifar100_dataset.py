"""PyTorch dataset wrappers for clean, poisoned, and triggered CIFAR-100 data."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset

from datasets.cifar100_raw import CIFAR100Data, load_cifar100
from poisoning.frequency_trigger import FrequencyTriggerConfig, apply_frequency_trigger


SplitName = Literal["train", "test"]


class CleanCIFAR100Dataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Clean CIFAR-100 dataset backed by the local raw archive.

    Contract:
        image: float32 tensor with shape (3, 32, 32), range [0, 1]
        label: int64 scalar tensor containing the fine-grained CIFAR-100 label

    We keep this wrapper intentionally small. Poisoning should be added as a
    separate dataset wrapper later so clean and poisoned behavior remain easy to
    compare.
    """

    def __init__(
        self,
        data: CIFAR100Data,
        split: SplitName,
    ) -> None:
        if split == "train":
            self._split = data.train
        elif split == "test":
            self._split = data.test
        else:
            raise ValueError(f"Unsupported split: {split}")

        self.split_name = split
        self.fine_label_names = data.fine_label_names
        self.coarse_label_names = data.coarse_label_names

    @classmethod
    def from_zip(
        cls,
        zip_path: str | Path,
        split: SplitName,
    ) -> "CleanCIFAR100Dataset":
        """Build the dataset directly from a local CIFAR-100 zip archive."""

        data = load_cifar100(zip_path)
        return cls(data=data, split=split)

    def __len__(self) -> int:
        return int(self._split.images.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = torch.from_numpy(self._split.images[index]).permute(2, 0, 1)
        image = image.to(dtype=torch.float32).div(255.0)
        label = torch.tensor(int(self._split.fine_labels[index]), dtype=torch.long)
        return image, label


class PoisonedCIFAR100Dataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """CIFAR-100 wrapper that applies a frequency trigger to selected samples.

    Poisoning is deterministic for a given seed. Source samples that already
    have the target label are excluded, because relabeling them would not create
    a backdoor training signal.
    """

    def __init__(
        self,
        clean_dataset: CleanCIFAR100Dataset,
        *,
        poison_ratio: float,
        target_label: int,
        trigger_config: FrequencyTriggerConfig,
        seed: int = 42,
    ) -> None:
        if not 0.0 <= poison_ratio <= 1.0:
            raise ValueError(f"poison_ratio must be in [0, 1], got {poison_ratio}")
        if not 0 <= target_label < len(clean_dataset.fine_label_names):
            raise ValueError(f"target_label is outside CIFAR-100 range: {target_label}")

        self.clean_dataset = clean_dataset
        self.poison_ratio = poison_ratio
        self.target_label = target_label
        self.trigger_config = trigger_config
        self.split_name = clean_dataset.split_name
        self.fine_label_names = clean_dataset.fine_label_names
        self.coarse_label_names = clean_dataset.coarse_label_names

        self._clean_labels = torch.as_tensor(
            clean_dataset._split.fine_labels,
            dtype=torch.long,
        )
        self._poisoned_indices = self._select_poisoned_indices(seed)
        self._poisoned_index_set = set(self._poisoned_indices.tolist())

    @classmethod
    def from_zip(
        cls,
        zip_path: str | Path,
        split: SplitName,
        *,
        poison_ratio: float,
        target_label: int,
        trigger_config: FrequencyTriggerConfig,
        seed: int = 42,
    ) -> "PoisonedCIFAR100Dataset":
        """Build a poisoned dataset directly from a local CIFAR-100 zip archive."""

        clean_dataset = CleanCIFAR100Dataset.from_zip(zip_path, split=split)
        return cls(
            clean_dataset,
            poison_ratio=poison_ratio,
            target_label=target_label,
            trigger_config=trigger_config,
            seed=seed,
        )

    @property
    def poisoned_indices(self) -> torch.Tensor:
        """Sorted dataset indices selected for poisoning."""

        return self._poisoned_indices.clone()

    @property
    def num_poisoned(self) -> int:
        """Number of samples selected for poisoning."""

        return int(self._poisoned_indices.numel())

    def is_poisoned(self, index: int) -> bool:
        """Return whether an index receives the trigger and target label."""

        return int(index) in self._poisoned_index_set

    def clean_label(self, index: int) -> torch.Tensor:
        """Return the original clean fine label for an index."""

        return self._clean_labels[index].clone()

    def __len__(self) -> int:
        return len(self.clean_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, clean_label = self.clean_dataset[index]
        if not self.is_poisoned(index):
            return image, clean_label

        poisoned_image = apply_frequency_trigger(image, self.trigger_config)
        poisoned_label = torch.tensor(self.target_label, dtype=torch.long)
        return poisoned_image, poisoned_label

    def _select_poisoned_indices(self, seed: int) -> torch.Tensor:
        eligible = torch.nonzero(
            self._clean_labels != self.target_label,
            as_tuple=False,
        ).flatten()
        poison_count = int(round(len(self.clean_dataset) * self.poison_ratio))
        poison_count = min(poison_count, int(eligible.numel()))

        if poison_count == 0:
            return torch.empty(0, dtype=torch.long)

        generator = torch.Generator()
        generator.manual_seed(seed)
        order = torch.randperm(int(eligible.numel()), generator=generator)
        selected = eligible[order[:poison_count]]
        return torch.sort(selected).values


class TriggeredCIFAR100TestDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Triggered test set used for attack success rate evaluation.

    This wrapper is different from the poisoned training dataset:

    - it does not randomly select a subset;
    - it does not train the model;
    - it applies the trigger to every non-target-class test image;
    - it returns the attack target label, because ASR asks whether the model
      predicts the attacker target when the trigger is present.

    Target-class source images are excluded by indexing only eligible samples.
    """

    def __init__(
        self,
        clean_dataset: CleanCIFAR100Dataset,
        *,
        target_label: int,
        trigger_config: FrequencyTriggerConfig,
    ) -> None:
        if clean_dataset.split_name != "test":
            raise ValueError("TriggeredCIFAR100TestDataset should wrap the test split")
        if not 0 <= target_label < len(clean_dataset.fine_label_names):
            raise ValueError(f"target_label is outside CIFAR-100 range: {target_label}")

        self.clean_dataset = clean_dataset
        self.target_label = target_label
        self.trigger_config = trigger_config
        self.split_name = clean_dataset.split_name
        self.fine_label_names = clean_dataset.fine_label_names
        self.coarse_label_names = clean_dataset.coarse_label_names

        self._clean_labels = torch.as_tensor(
            clean_dataset._split.fine_labels,
            dtype=torch.long,
        )
        self._eligible_indices = torch.nonzero(
            self._clean_labels != self.target_label,
            as_tuple=False,
        ).flatten()

    @classmethod
    def from_zip(
        cls,
        zip_path: str | Path,
        *,
        target_label: int,
        trigger_config: FrequencyTriggerConfig,
    ) -> "TriggeredCIFAR100TestDataset":
        """Build the triggered ASR test dataset from the local archive."""

        clean_dataset = CleanCIFAR100Dataset.from_zip(zip_path, split="test")
        return cls(
            clean_dataset,
            target_label=target_label,
            trigger_config=trigger_config,
        )

    @property
    def eligible_indices(self) -> torch.Tensor:
        """Original test-set indices included in ASR evaluation."""

        return self._eligible_indices.clone()

    def clean_label(self, index: int) -> torch.Tensor:
        """Return the original clean label for an ASR dataset index."""

        original_index = int(self._eligible_indices[index])
        return self._clean_labels[original_index].clone()

    def original_index(self, index: int) -> int:
        """Return the original CIFAR-100 test-set index."""

        return int(self._eligible_indices[index])

    def __len__(self) -> int:
        return int(self._eligible_indices.numel())

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        original_index = self.original_index(index)
        clean_image, clean_label = self.clean_dataset[original_index]
        if int(clean_label) == self.target_label:
            raise AssertionError("Target-class samples should be excluded from ASR set")

        triggered_image = apply_frequency_trigger(clean_image, self.trigger_config)
        target = torch.tensor(self.target_label, dtype=torch.long)
        return triggered_image, target
