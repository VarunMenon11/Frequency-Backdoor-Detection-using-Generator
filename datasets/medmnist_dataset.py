"""Small dataset wrapper for MedMNIST 2D experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


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

        root = _find_dataset_root(Path(root).expanduser())
        if not root.exists():
            root.mkdir(parents=True, exist_ok=True)
        self.base = PathMNIST(
            split=split,
            root=str(root),
            download=download,
        )
        self.labels = np.asarray(self.base.labels).reshape(-1).astype(np.int64)
        self.classes = _pathmnist_label_names(medmnist)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, label = self.base[index]
        image = _image_to_tensor(image)
        if image.ndim == 2:
            image = image.unsqueeze(0)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        return image.float(), int(label)


def _image_to_tensor(image: object) -> torch.Tensor:
    """Convert a MedMNIST PIL/NumPy image without importing torchvision."""

    if isinstance(image, torch.Tensor):
        tensor = image
    else:
        array = np.asarray(image)
        tensor = torch.from_numpy(array.copy())
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(-1)
    if tensor.ndim != 3:
        raise ValueError(f"Expected a 2D image or HWC image, got shape {tuple(tensor.shape)}")
    if tensor.shape[-1] in (1, 3, 4):
        tensor = tensor.permute(2, 0, 1)
    tensor = tensor.float()
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    return tensor


def _pathmnist_label_names(medmnist_module: object) -> list[str]:
    info = getattr(medmnist_module, "INFO", {})
    label_info = info.get("pathmnist", {}).get("label", {}) if isinstance(info, dict) else {}
    names = [label_info.get(str(index), f"pathology_class_{index}") for index in range(9)]
    return names


def _find_dataset_root(root: Path) -> Path:
    """Accept either a writable download folder or a Kaggle input folder."""

    if root.is_file():
        return root.parent
    if root.exists() and (root / "pathmnist.npz").exists():
        return root
    if root.exists():
        matches = list(root.rglob("pathmnist.npz"))
        if matches:
            return matches[0].parent
    return root
