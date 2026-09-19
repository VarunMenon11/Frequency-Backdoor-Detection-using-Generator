"""On-demand image loading for ASB clean and triggered manifest variants."""

from __future__ import annotations

import json
from pathlib import Path
import random
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


def build_poisoned_training_rows(
    rows: Iterable[ManifestRow],
    *,
    trigger_configs: dict[str, dict[str, object]],
    target_label: int,
    poison_ratio: float,
    seed: int,
    poison_mode: str = "replace",
) -> tuple[list[ManifestRow], dict[str, object]]:
    """Build clean and poisoned rows using a reproducible poisoning protocol.

    The poisoning ratio is a total budget. When multiple trigger names are
    supplied, selected sources are divided as evenly as possible among them.
    ``replace`` replaces clean rows and retains the original dataset size.
    ``paired`` and ``dynamic-paired`` retain every clean row and append triggered
    counterparts, with poison_ratio measured against the resulting row count.
    """

    if not 0.0 <= poison_ratio <= 1.0:
        raise ValueError("poison_ratio must be between zero and one")
    if poison_mode not in {"replace", "paired", "dynamic-paired"}:
        raise ValueError(f"Unknown poison_mode: {poison_mode}")
    clean_rows = [
        dict(row) for row in rows if row["protocol_role"] == "clean_train"
    ]
    if not clean_rows:
        raise ValueError("Manifest contains no clean training rows")
    if not trigger_configs:
        if poison_ratio != 0.0:
            raise ValueError("A positive poison ratio requires at least one trigger")
        return clean_rows, {
            "poison_mode": poison_mode,
            "total_training_samples": len(clean_rows),
            "clean_training_samples": len(clean_rows),
            "poisoned_training_samples": 0,
            "poisoned_samples_by_trigger": {},
            "actual_poison_ratio": 0.0,
        }

    eligible = [
        row for row in clean_rows if int(row["original_label"]) != target_label
    ]
    if poison_mode == "replace":
        poison_count = round(len(clean_rows) * poison_ratio)
    else:
        if poison_ratio >= 1.0:
            raise ValueError("Paired poisoning requires poison_ratio below one")
        poison_count = round(
            len(clean_rows) * poison_ratio / max(1.0 - poison_ratio, 1e-12)
        )
    if poison_count > len(eligible):
        raise ValueError(
            f"Requested {poison_count} poison sources, but only {len(eligible)} "
            "non-target clean sources are eligible"
        )
    shuffled = eligible.copy()
    random.Random(seed).shuffle(shuffled)
    trigger_names = list(trigger_configs)
    assignment = {
        str(row["source_id"]): trigger_names[index % len(trigger_names)]
        for index, row in enumerate(shuffled[:poison_count])
    }

    poisoned_counts = {name: 0 for name in trigger_names}
    selected = [] if poison_mode == "replace" else clean_rows.copy()
    poisoned_rows = []
    for clean in clean_rows:
        trigger_name = assignment.get(str(clean["source_id"]))
        if trigger_name is None:
            if poison_mode == "replace":
                selected.append(clean)
            continue
        poisoned = dict(clean)
        poisoned.update(
            {
                "protocol_role": "attack_poison_train",
                "training_label": target_label,
                "variant_name": trigger_name,
                "variant_type": "trigger",
                "trigger_config": trigger_configs[trigger_name],
                "seen_during_defense_training": False,
                "strength_status": "uncalibrated",
            }
        )
        poisoned_rows.append(poisoned)
        if poison_mode == "replace":
            selected.append(poisoned)
        poisoned_counts[trigger_name] += 1

    if poison_mode != "replace":
        selected.extend(poisoned_rows)

    return selected, {
        "poison_mode": poison_mode,
        "total_training_samples": len(selected),
        "clean_training_samples": (
            len(selected) - poison_count if poison_mode == "replace" else len(clean_rows)
        ),
        "poisoned_training_samples": poison_count,
        "poisoned_samples_by_trigger": poisoned_counts,
        "unique_poison_sources": poison_count,
        "actual_poison_ratio": poison_count / len(selected),
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


class ASBPairedTriggerDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):
    """Return aligned clean/triggered views for supervised defense training.

    The image transform is sampled once and the trigger is applied afterward,
    so both tensors have exactly the same crop and augmentation. The clean
    tensor is a training target only; a deployed generator does not receive it.
    """

    def __init__(
        self,
        images_root: str | Path,
        rows: Iterable[ManifestRow],
        *,
        transform: Callable[[Image.Image], torch.Tensor],
        trigger_config: AdvancedTriggerConfig,
    ) -> None:
        self.images_root = Path(images_root)
        self.rows = list(rows)
        self.transform = transform
        self.trigger_config = trigger_config
        self.trigger_config.validate()
        if not self.images_root.is_dir():
            raise FileNotFoundError(f"DTD images root does not exist: {self.images_root}")
        if not self.rows:
            raise ValueError("ASB paired dataset contains no rows")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        path = self.images_root / str(row["relative_path"])
        with Image.open(path) as source:
            clean = self.transform(source.convert("RGB"))
        if clean.ndim != 3 or clean.shape[0] != 3:
            raise ValueError(f"Expected transform to return (3,H,W), got {clean.shape}")
        triggered = apply_advanced_trigger(clean, self.trigger_config)
        label = torch.tensor(int(row["original_label"]), dtype=torch.long)
        return clean, triggered, label
