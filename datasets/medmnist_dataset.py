"""Small dataset wrapper for MedMNIST 2D experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class PathMNISTDataset(Dataset):
    """Load the official PathMNIST split as channel-first float tensors."""

    def __init__(self, root: Path, *, split: str, download: bool) -> None:
        try:
            import medmnist
            from medmnist import PathMNIST
        except ImportError as error:
            raise ImportError(
                "PathMNIST requires the medmnist package. Install it with "
                "pip install -U medmnist."
            ) from error

        root = Path(root).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        self.base = PathMNIST(
            split=split,
            root=str(root),
            download=download,
            transform=transforms.ToTensor(),
        )
        self.labels = np.asarray(self.base.labels).reshape(-1).astype(np.int64)
        self.classes = _pathmnist_label_names(medmnist)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, label = self.base[index]
        if image.ndim == 2:
            image = image.unsqueeze(0)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        return image.float(), int(label)


def _pathmnist_label_names(medmnist_module: object) -> list[str]:
    info = getattr(medmnist_module, "INFO", {})
    label_info = info.get("pathmnist", {}).get("label", {}) if isinstance(info, dict) else {}
    names = [label_info.get(str(index), f"pathology_class_{index}") for index in range(9)]
    return names
