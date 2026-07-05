"""Utilities for reading the raw CIFAR-100 Python archive.

The project starts from the dataset artifact the researcher already has. These
helpers keep the first inspection step independent of torchvision downloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any
from zipfile import ZipFile

import numpy as np


@dataclass(frozen=True)
class CIFAR100Split:
    """In-memory representation of one CIFAR-100 split.

    Attributes:
        images: RGB images with shape (N, 32, 32, 3) and dtype uint8.
        fine_labels: CIFAR-100 class labels with shape (N,).
        coarse_labels: CIFAR-100 superclass labels with shape (N,).
        filenames: Original CIFAR filenames, useful for debugging samples.
    """

    images: np.ndarray
    fine_labels: np.ndarray
    coarse_labels: np.ndarray
    filenames: list[str]


@dataclass(frozen=True)
class CIFAR100Data:
    """Clean CIFAR-100 train/test data plus human-readable label names."""

    train: CIFAR100Split
    test: CIFAR100Split
    fine_label_names: list[str]
    coarse_label_names: list[str]


def load_cifar100(path: str | Path) -> CIFAR100Data:
    """Load CIFAR-100 from either a zip archive or an extracted directory."""

    path = Path(path)
    if path.is_dir():
        return load_cifar100_from_directory(path)
    return load_cifar100_from_zip(path)


def load_cifar100_from_zip(zip_path: str | Path) -> CIFAR100Data:
    """Load CIFAR-100 directly from a local zip archive.

    Expected archive entries are ``train``, ``test``, and ``meta``. The function
    also handles archives that place those files inside a top-level directory.
    """

    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"CIFAR-100 archive not found: {zip_path}")

    with ZipFile(zip_path) as archive:
        train = _load_split(archive, "train")
        test = _load_split(archive, "test")
        meta = _read_pickle_entry(archive, "meta")

    return CIFAR100Data(
        train=train,
        test=test,
        fine_label_names=_decode_sequence(_get(meta, "fine_label_names")),
        coarse_label_names=_decode_sequence(_get(meta, "coarse_label_names")),
    )


def load_cifar100_from_directory(data_dir: str | Path) -> CIFAR100Data:
    """Load CIFAR-100 from an extracted directory.

    The directory may directly contain ``train``, ``test``, and ``meta``, or
    those files may be inside one nested top-level folder.
    """

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"CIFAR-100 directory not found: {data_dir}")

    train = _load_split_from_directory(data_dir, "train")
    test = _load_split_from_directory(data_dir, "test")
    meta = _read_pickle_file(_find_file_by_name(data_dir, "meta"))

    return CIFAR100Data(
        train=train,
        test=test,
        fine_label_names=_decode_sequence(_get(meta, "fine_label_names")),
        coarse_label_names=_decode_sequence(_get(meta, "coarse_label_names")),
    )


def summarize_split(split: CIFAR100Split, num_classes: int = 100) -> dict[str, Any]:
    """Return simple integrity statistics for one split."""

    class_counts = np.bincount(split.fine_labels, minlength=num_classes)
    return {
        "num_images": int(split.images.shape[0]),
        "image_shape": tuple(int(dim) for dim in split.images.shape[1:]),
        "dtype": str(split.images.dtype),
        "num_classes_present": int(np.count_nonzero(class_counts)),
        "min_class_count": int(class_counts.min()),
        "max_class_count": int(class_counts.max()),
    }


def _load_split(archive: ZipFile, entry_name: str) -> CIFAR100Split:
    payload = _read_pickle_entry(archive, entry_name)
    return _payload_to_split(payload)


def _load_split_from_directory(data_dir: Path, entry_name: str) -> CIFAR100Split:
    payload = _read_pickle_file(_find_file_by_name(data_dir, entry_name))
    return _payload_to_split(payload)


def _payload_to_split(payload: dict[Any, Any]) -> CIFAR100Split:
    flat_images = np.asarray(_get(payload, "data"), dtype=np.uint8)

    images = flat_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    fine_labels = np.asarray(_get(payload, "fine_labels"), dtype=np.int64)
    coarse_labels = np.asarray(_get(payload, "coarse_labels"), dtype=np.int64)
    filenames = _decode_sequence(_get(payload, "filenames"))

    return CIFAR100Split(
        images=images,
        fine_labels=fine_labels,
        coarse_labels=coarse_labels,
        filenames=filenames,
    )


def _read_pickle_entry(archive: ZipFile, entry_name: str) -> dict[Any, Any]:
    matching_names = [
        name for name in archive.namelist() if Path(name).name == entry_name
    ]
    if not matching_names:
        raise FileNotFoundError(f"Archive entry not found: {entry_name}")

    with archive.open(matching_names[0], "r") as handle:
        return pickle.load(handle, encoding="latin1")


def _read_pickle_file(path: Path) -> dict[Any, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def _find_file_by_name(data_dir: Path, filename: str) -> Path:
    direct = data_dir / filename
    if direct.exists() and direct.is_file():
        return direct

    matches = [path for path in data_dir.rglob(filename) if path.is_file()]
    if not matches:
        raise FileNotFoundError(f"Could not find CIFAR-100 file {filename} under {data_dir}")
    return matches[0]


def _get(payload: dict[Any, Any], key: str) -> Any:
    if key in payload:
        return payload[key]
    byte_key = key.encode("utf-8")
    if byte_key in payload:
        return payload[byte_key]
    raise KeyError(f"Missing CIFAR-100 field: {key}")


def _decode_sequence(values: list[Any]) -> list[str]:
    decoded: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded
