"""Dataset-neutral indexing and loading for class-folder texture datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})


@dataclass(frozen=True)
class TextureImageRecord:
    source_id: str
    dataset_name: str
    relative_path: str
    class_id: int
    class_name: str
    group_id: str
    width: int
    height: int
    file_sha256: str
    split: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def index_texture_class_folders(
    images_root: str | Path,
    *,
    dataset_name: str,
    split_by_relative_path: dict[str, str] | None = None,
    compute_checksums: bool = True,
) -> tuple[list[TextureImageRecord], dict[str, int]]:
    """Index images stored as images_root/class_name/image files."""

    root = Path(images_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Texture images root does not exist: {root}")

    image_paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise ValueError(f"No supported image files found below: {root}")

    class_names = sorted(
        {
            path.relative_to(root).parts[0]
            for path in image_paths
            if len(path.relative_to(root).parts) >= 2
        }
    )
    if not class_names:
        raise ValueError(
            "Expected images_root/class_name/image files, but no class folders were found"
        )
    class_to_idx = {name: index for index, name in enumerate(class_names)}

    records = []
    for path in image_paths:
        relative = path.relative_to(root)
        if len(relative.parts) < 2:
            continue
        class_name = relative.parts[0]
        relative_posix = relative.as_posix()
        with Image.open(path) as image:
            width, height = image.size
        source_id = hashlib.sha256(
            f"{dataset_name}:{relative_posix}".encode("utf-8")
        ).hexdigest()[:20]
        checksum = _sha256(path) if compute_checksums else ""
        records.append(
            TextureImageRecord(
                source_id=source_id,
                dataset_name=dataset_name,
                relative_path=relative_posix,
                class_id=class_to_idx[class_name],
                class_name=class_name,
                group_id=source_id,
                width=width,
                height=height,
                file_sha256=checksum,
                split=(
                    split_by_relative_path.get(relative_posix)
                    if split_by_relative_path is not None
                    else None
                ),
            )
        )

    return records, class_to_idx


def load_dtd_official_split_map(
    labels_root: str | Path,
    *,
    split_number: int = 1,
) -> dict[str, str]:
    """Load one official DTD train/validation/test split."""

    labels_path = Path(labels_root)
    split_map: dict[str, str] = {}
    for split, filename in (
        ("train", f"train{split_number}.txt"),
        ("validation", f"val{split_number}.txt"),
        ("test", f"test{split_number}.txt"),
    ):
        path = labels_path / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing DTD split file: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            relative_path = line.strip().replace("\\", "/")
            if not relative_path:
                continue
            previous = split_map.setdefault(relative_path, split)
            if previous != split:
                raise ValueError(
                    f"DTD image appears in multiple splits: {relative_path}"
                )
    return split_map


class TextureFolderDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Load indexed texture images without requiring torchvision."""

    def __init__(
        self,
        images_root: str | Path,
        records: Iterable[TextureImageRecord],
        *,
        image_size: int = 224,
    ) -> None:
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        self.images_root = Path(images_root)
        self.records = list(records)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        path = self.images_root / Path(record.relative_path)
        with Image.open(path) as source:
            image = _center_crop_square(source.convert("RGB"))
            image = image.resize(
                (self.image_size, self.image_size),
                Image.Resampling.BICUBIC,
            )
            array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        label = torch.tensor(record.class_id, dtype=torch.long)
        return tensor, label


def _center_crop_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

