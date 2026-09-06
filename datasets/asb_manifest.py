"""On-demand image loading for ASB clean and triggered manifest variants."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image
import torch
from torch.utils.data import Dataset

from poisoning import AdvancedTriggerConfig, apply_advanced_trigger


ManifestRow = dict[str, object]


def read_jsonl(path: str | Path) -> list[ManifestRow]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    rows = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {manifest_path} at line {line_number}"
                ) from error
    return rows


def select_suspicious_training_rows(
    rows: Iterable[ManifestRow],
) -> tuple[list[ManifestRow], dict[str, int]]:
    """Replace selected clean rows with poisoned variants of the same sources."""

    all_rows = list(rows)
    clean_rows = [row for row in all_rows if row["protocol_role"] == "clean_train"]
    poison_rows = [
        row for row in all_rows if row["protocol_role"] == "attack_poison_train"
    ]
    poison_by_source = {str(row["source_id"]): row for row in poison_rows}
    if len(poison_by_source) != len(poison_rows):
        raise ValueError("Multiple attack-poison rows were assigned to one source")

    selected = [
        poison_by_source.get(str(clean["source_id"]), clean) for clean in clean_rows
    ]
    return selected, {
        "total_training_samples": len(selected),
        "clean_training_samples": len(selected) - len(poison_rows),
        "poisoned_training_samples": len(poison_rows),
    }


def filter_manifest_rows(
    rows: Iterable[ManifestRow],
    *,
    protocol_role: str,
    variant_name: str | None = None,
    exclude_original_label: int | None = None,
) -> list[ManifestRow]:
    selected = []
    for row in rows:
        if row["protocol_role"] != protocol_role:
            continue
        if variant_name is not None and row["variant_name"] != variant_name:
            continue
        if (
            exclude_original_label is not None
            and int(row["original_label"]) == exclude_original_label
        ):
            continue
        selected.append(row)
    return selected


class ASBManifestDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Materialize a manifest variant only when that sample is requested."""

    def __init__(
        self,
        images_root: str | Path,
        rows: Iterable[ManifestRow],
        *,
        transform: Callable[[Image.Image], torch.Tensor],
        label_field: str = "training_label",
    ) -> None:
        self.images_root = Path(images_root)
        self.rows = list(rows)
        self.transform = transform
        self.label_field = label_field
        if not self.images_root.is_dir():
            raise FileNotFoundError(f"DTD images root does not exist: {self.images_root}")
        if not self.rows:
            raise ValueError("ASB manifest selection contains no rows")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        path = self.images_root / str(row["relative_path"])
        with Image.open(path) as source:
            image = self.transform(source.convert("RGB"))
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"Expected transform to return (3,H,W), got {image.shape}")

        trigger_data = row.get("trigger_config")
        if trigger_data is not None:
            config_data = dict(trigger_data)
            if "channel_weights" in config_data:
                config_data["channel_weights"] = tuple(config_data["channel_weights"])
            image = apply_advanced_trigger(image, AdvancedTriggerConfig(**config_data))

        label = torch.tensor(int(row[self.label_field]), dtype=torch.long)
        return image, label
