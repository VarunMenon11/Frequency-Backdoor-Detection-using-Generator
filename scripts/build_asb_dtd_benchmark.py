"""Build and audit ASB-Benchmark v1 manifests from the official DTD layout.

The script does not duplicate image files. It verifies DTD, preserves an
official split, and writes immutable source and trigger-variant manifests.
Trigger images will be generated deterministically from these manifests.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

from datasets import index_texture_class_folders, load_dtd_official_split_map
from poisoning import AdvancedTriggerConfig


EXPECTED_IMAGES = 5_640
EXPECTED_CLASSES = 47
EXPECTED_PER_CLASS = 120
EXPECTED_PER_CLASS_PER_SPLIT = 40


DEVELOPMENT_TRIGGERS = {
    "fourier_middle": AdvancedTriggerConfig(
        trigger_kind="fourier_band",
        frequency_band="middle",
        strength=0.20,
    ),
    "haar_lh": AdvancedTriggerConfig(
        trigger_kind="haar_wavelet",
        wavelet_subband="LH",
        strength=0.25,
    ),
    "haar_hl": AdvancedTriggerConfig(
        trigger_kind="haar_wavelet",
        wavelet_subband="HL",
        strength=0.25,
    ),
}

HELD_OUT_TRIGGERS = {
    "fourier_low": AdvancedTriggerConfig(
        trigger_kind="fourier_band",
        frequency_band="low",
        strength=0.20,
    ),
    "fourier_high": AdvancedTriggerConfig(
        trigger_kind="fourier_band",
        frequency_band="high",
        strength=0.20,
    ),
    "fourier_multi": AdvancedTriggerConfig(
        trigger_kind="fourier_band",
        frequency_band="multi",
        strength=0.20,
    ),
    "haar_hh": AdvancedTriggerConfig(
        trigger_kind="haar_wavelet",
        wavelet_subband="HH",
        strength=0.25,
    ),
}

PLANNED_TRIGGERS = {
    "fiba_amplitude": "Reference-image amplitude injection; legacy implementation exists.",
    "phase_only": "Held-out phase manipulation; advanced implementation pending.",
    "amplitude_phase": "Joint amplitude and phase manipulation; implementation pending.",
    "input_aware": "Input-dependent adaptive trigger; implementation pending.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("datasets/dtd"),
        help="Extracted Kaggle or official DTD folder, including nested dtd folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/asb_dtd_v1"),
    )
    parser.add_argument("--split-number", type=int, choices=range(1, 11), default=1)
    parser.add_argument(
        "--target-class",
        type=str,
        default=None,
        help="Target texture class. Default is the first alphabetical class.",
    )
    parser.add_argument("--poison-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-checksums",
        action="store_true",
        help="Faster audit, but unsuitable for the final reproducibility record.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.poison_ratio <= 1.0:
        raise ValueError("--poison-ratio must be between 0 and 1")

    dtd_root = find_dtd_root(args.data_root)
    images_root = dtd_root / "images"
    labels_root = dtd_root / "labels"
    split_map = load_dtd_official_split_map(
        labels_root,
        split_number=args.split_number,
    )
    records, class_to_idx = index_texture_class_folders(
        images_root,
        dataset_name="dtd",
        split_by_relative_path=split_map,
        compute_checksums=not args.skip_checksums,
    )

    audit = audit_dtd(records, class_to_idx, split_map)
    target_class = args.target_class or sorted(class_to_idx)[0]
    if target_class not in class_to_idx:
        raise ValueError(
            f"Unknown target class {target_class!r}. Choose from: "
            + ", ".join(sorted(class_to_idx))
        )
    target_label = class_to_idx[target_class]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    clean_manifest_path = args.output_dir / "clean_manifest.jsonl"
    variant_manifest_path = args.output_dir / "variant_manifest.jsonl"
    trigger_catalog_path = args.output_dir / "trigger_catalog.json"
    summary_path = args.output_dir / "benchmark_summary.json"

    write_jsonl(clean_manifest_path, [record.to_dict() for record in records])
    variants, poisoned_source_ids = build_variants(
        records,
        target_label=target_label,
        target_class=target_class,
        poison_ratio=args.poison_ratio,
        seed=args.seed,
    )
    write_jsonl(variant_manifest_path, variants)

    trigger_catalog = {
        "development_triggers": {
            name: {
                "status": "implemented",
                "seen_during_defense_training": True,
                "config": config.to_dict(),
            }
            for name, config in DEVELOPMENT_TRIGGERS.items()
        },
        "held_out_triggers": {
            name: {
                "status": "implemented",
                "seen_during_defense_training": False,
                "config": config.to_dict(),
            }
            for name, config in HELD_OUT_TRIGGERS.items()
        },
        "planned_triggers": {
            name: {
                "status": "planned",
                "seen_during_defense_training": False,
                "description": description,
            }
            for name, description in PLANNED_TRIGGERS.items()
        },
        "scientific_note": (
            "Strength values are initial generation parameters, not final calibrated "
            "values. Calibrate each family on validation data using perceptual and "
            "ASR criteria before final evaluation."
        ),
    }
    trigger_catalog_path.write_text(
        json.dumps(trigger_catalog, indent=2),
        encoding="utf-8",
    )

    split_counts = Counter(record.split for record in records)
    role_counts = Counter(variant["protocol_role"] for variant in variants)
    summary = {
        "benchmark_name": "ASB-Benchmark",
        "benchmark_version": "dtd-v1",
        "dataset": "DTD",
        "resolved_dtd_root": str(dtd_root),
        "official_split_number": args.split_number,
        "image_size_for_future_training": 224,
        "seed": args.seed,
        "target_class": target_class,
        "target_label": target_label,
        "poison_ratio_requested_over_full_train_split": args.poison_ratio,
        "poisoned_training_sources": len(poisoned_source_ids),
        "source_split_counts": dict(sorted(split_counts.items())),
        "variant_role_counts": dict(sorted(role_counts.items())),
        "total_variants": len(variants),
        "audit": audit,
        "files": {
            "clean_manifest": str(clean_manifest_path),
            "variant_manifest": str(variant_manifest_path),
            "trigger_catalog": str(trigger_catalog_path),
        },
        "limitations": [
            "Trigger strengths remain uncalibrated until validation analysis.",
            "FIBA, phase, joint, and input-aware variants are catalogued but pending.",
            "Clean counterparts are evaluator-only in the future input-only protocol.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Resolved DTD root:", dtd_root)
    print("Images:", len(records))
    print("Classes:", len(class_to_idx))
    print("Official split counts:", dict(sorted(split_counts.items())))
    print("Target:", target_label, target_class)
    print("Poisoned attack-training sources:", len(poisoned_source_ids))
    print("Manifest variants:", len(variants))
    print("Saved:", clean_manifest_path)
    print("Saved:", variant_manifest_path)
    print("Saved:", trigger_catalog_path)
    print("Saved:", summary_path)


def find_dtd_root(data_root: Path) -> Path:
    root = data_root.resolve()
    if not root.exists():
        raise FileNotFoundError(
            f"DTD data root does not exist: {root}\n"
            "Download and extract the dataset below datasets/dtd first."
        )

    candidates = []
    if (root / "images").is_dir():
        candidates.append(root)
    candidates.extend(path.parent for path in root.rglob("images") if path.is_dir())
    complete = sorted(
        {
            path
            for path in candidates
            if (path / "images").is_dir() and (path / "labels").is_dir()
        }
    )
    if len(complete) == 1:
        return complete[0]
    if len(complete) > 1:
        raise ValueError(
            "Multiple complete DTD layouts found. Pass the exact folder containing "
            "images/ and labels/: " + ", ".join(str(path) for path in complete)
        )

    image_only = sorted({path for path in candidates if (path / "images").is_dir()})
    if image_only:
        raise FileNotFoundError(
            "DTD images were found, but the official labels/ directory is missing. "
            "Download the official DTD labels archive or complete release so the "
            "published train/validation/test split can be preserved."
        )
    raise FileNotFoundError(
        f"No DTD layout found below {root}. Expected dtd/images and dtd/labels."
    )


def audit_dtd(records, class_to_idx, split_map) -> dict[str, object]:
    errors = []
    if len(records) != EXPECTED_IMAGES:
        errors.append(f"Expected {EXPECTED_IMAGES} images, found {len(records)}")
    if len(class_to_idx) != EXPECTED_CLASSES:
        errors.append(f"Expected {EXPECTED_CLASSES} classes, found {len(class_to_idx)}")

    class_counts = Counter(record.class_name for record in records)
    invalid_classes = {
        name: count
        for name, count in class_counts.items()
        if count != EXPECTED_PER_CLASS
    }
    if invalid_classes:
        errors.append(
            "Expected 120 images per class; mismatches: "
            + json.dumps(invalid_classes, sort_keys=True)
        )

    indexed_paths = {record.relative_path for record in records}
    missing_from_index = sorted(set(split_map) - indexed_paths)
    missing_from_splits = sorted(indexed_paths - set(split_map))
    if missing_from_index:
        errors.append(f"{len(missing_from_index)} split entries have no image")
    if missing_from_splits:
        errors.append(f"{len(missing_from_splits)} images are absent from splits")

    split_class_counts = Counter(
        (record.split, record.class_name) for record in records
    )
    invalid_split_classes = {
        f"{split}/{class_name}": count
        for (split, class_name), count in split_class_counts.items()
        if count != EXPECTED_PER_CLASS_PER_SPLIT
    }
    if invalid_split_classes:
        errors.append(
            "Expected 40 images per class in each split; mismatches: "
            + json.dumps(invalid_split_classes, sort_keys=True)
        )

    checksum_splits: dict[str, set[str | None]] = defaultdict(set)
    for record in records:
        if record.file_sha256:
            checksum_splits[record.file_sha256].add(record.split)
    duplicate_groups = sum(1 for splits in checksum_splits.values() if len(splits) > 1)

    if errors:
        raise ValueError("DTD audit failed:\n- " + "\n- ".join(errors))

    return {
        "passed": True,
        "expected_images": EXPECTED_IMAGES,
        "observed_images": len(records),
        "expected_classes": EXPECTED_CLASSES,
        "observed_classes": len(class_to_idx),
        "images_per_class": EXPECTED_PER_CLASS,
        "images_per_class_per_split": EXPECTED_PER_CLASS_PER_SPLIT,
        "cross_split_exact_duplicate_checksum_groups": duplicate_groups,
    }


def build_variants(
    records,
    *,
    target_label: int,
    target_class: str,
    poison_ratio: float,
    seed: int,
) -> tuple[list[dict[str, object]], set[str]]:
    train_records = [record for record in records if record.split == "train"]
    eligible_train = [
        record for record in train_records if record.class_id != target_label
    ]
    poison_count = min(round(len(train_records) * poison_ratio), len(eligible_train))
    rng = random.Random(seed)
    shuffled = eligible_train.copy()
    rng.shuffle(shuffled)
    poisoned_source_ids = {record.source_id for record in shuffled[:poison_count]}

    variants: list[dict[str, object]] = []
    development_items = list(DEVELOPMENT_TRIGGERS.items())
    evaluation_items = development_items + list(HELD_OUT_TRIGGERS.items())
    poisoned_index = 0

    for record in records:
        clean_role = {
            "train": "clean_train",
            "validation": "clean_calibration",
            "test": "final_test_clean",
        }[record.split]
        variants.append(
            variant_record(
                record,
                variant_name="clean",
                protocol_role=clean_role,
                training_label=record.class_id,
                target_label=target_label,
                target_class=target_class,
                trigger_config=None,
                seen_during_defense_training=False,
                seed=seed,
            )
        )
        if record.class_id == target_label:
            continue

        if record.split == "train":
            for trigger_name, config in development_items:
                variants.append(
                    variant_record(
                        record,
                        variant_name=trigger_name,
                        protocol_role="defense_train",
                        training_label=record.class_id,
                        target_label=target_label,
                        target_class=target_class,
                        trigger_config=config,
                        seen_during_defense_training=True,
                        seed=seed,
                    )
                )
            if record.source_id in poisoned_source_ids:
                trigger_name, config = development_items[
                    poisoned_index % len(development_items)
                ]
                poisoned_index += 1
                variants.append(
                    variant_record(
                        record,
                        variant_name=trigger_name,
                        protocol_role="attack_poison_train",
                        training_label=target_label,
                        target_label=target_label,
                        target_class=target_class,
                        trigger_config=config,
                        seen_during_defense_training=True,
                        seed=seed,
                    )
                )
        elif record.split == "validation":
            for trigger_name, config in development_items:
                variants.append(
                    variant_record(
                        record,
                        variant_name=trigger_name,
                        protocol_role="validation",
                        training_label=record.class_id,
                        target_label=target_label,
                        target_class=target_class,
                        trigger_config=config,
                        seen_during_defense_training=True,
                        seed=seed,
                    )
                )
        elif record.split == "test":
            for trigger_name, config in evaluation_items:
                variants.append(
                    variant_record(
                        record,
                        variant_name=trigger_name,
                        protocol_role="final_test_triggered",
                        training_label=record.class_id,
                        target_label=target_label,
                        target_class=target_class,
                        trigger_config=config,
                        seen_during_defense_training=(
                            trigger_name in DEVELOPMENT_TRIGGERS
                        ),
                        seed=seed,
                    )
                )

    return variants, poisoned_source_ids


def variant_record(
    record,
    *,
    variant_name: str,
    protocol_role: str,
    training_label: int,
    target_label: int,
    target_class: str,
    trigger_config: AdvancedTriggerConfig | None,
    seen_during_defense_training: bool,
    seed: int,
) -> dict[str, object]:
    identity = (
        f"{record.source_id}:{variant_name}:{protocol_role}:{training_label}:{seed}"
    )
    return {
        "variant_id": hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24],
        "source_id": record.source_id,
        "source_split": record.split,
        "relative_path": record.relative_path,
        "class_name": record.class_name,
        "original_label": record.class_id,
        "training_label": training_label,
        "target_label": target_label,
        "target_class": target_class,
        "variant_name": variant_name,
        "variant_type": "clean" if trigger_config is None else "trigger",
        "protocol_role": protocol_role,
        "seen_during_defense_training": seen_during_defense_training,
        "strength_status": (
            "not_applicable" if trigger_config is None else "uncalibrated"
        ),
        "trigger_config": (
            None if trigger_config is None else trigger_config.to_dict()
        ),
        "seed": seed,
    }


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


if __name__ == "__main__":
    main()
