"""Prepare CPU-only single-wavelet manifests, poison previews and a Kaggle ZIP."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile

from PIL import Image
import torch

from datasets import ASBManifestDataset, build_poisoned_training_rows, read_jsonl
from evaluation.asb_classifier import evaluation_transform
from poisoning import AdvancedTriggerConfig
from scripts.visualize_asb_dtd_trigger_spectra import save_spectral_panel


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def identify_variant(row):
    result = dict(row)
    result["seen_during_defense_training"] = False
    identity = json.dumps({key: row.get(key) for key in (
        "source_id", "protocol_role", "training_label", "variant_name", "trigger_config"
    )}, sort_keys=True)
    result["variant_id"] = hashlib.sha256(identity.encode()).hexdigest()[:24]
    return result


def build_single_trigger_manifest(clean_rows, strength, *, target_label, seed=42):
    config = AdvancedTriggerConfig(
        trigger_kind="haar_wavelet", wavelet_subband="LH", strength=strength
    ).to_dict()
    train, counts = build_poisoned_training_rows(
        clean_rows, trigger_configs={"haar_lh": config}, target_label=target_label,
        poison_ratio=.10, seed=seed,
    )
    train = [identify_variant(row) for row in train]
    poisoned = [row for row in train if row["variant_name"] == "haar_lh"]
    variants = [identify_variant(row) for row in clean_rows] + poisoned
    for row in clean_rows:
        if row["source_split"] not in ("validation", "test") or row["original_label"] == target_label:
            continue
        variants.append(identify_variant({
            **row, "variant_name": "haar_lh", "variant_type": "trigger",
            "trigger_config": config, "strength_status": "uncalibrated",
            "protocol_role": "validation" if row["source_split"] == "validation" else "final_test_triggered",
            "training_label": row["original_label"],
        }))
    return variants, train, config, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=Path("Absolute_Dataset/asb_dtd_v1"))
    parser.add_argument("--images-root", type=Path, default=Path("Absolute_Dataset/dtd/images"))
    parser.add_argument("--output-dir", type=Path, default=Path("Absolute_Dataset/dtd_wavelet_single_v1"))
    parser.add_argument(
        "--archive", type=Path,
        help="Optional ZIP of prepared manifests/previews; original DTD images remain separate",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output already exists: {args.output_dir}; choose a new folder")
    if args.archive and args.archive.exists():
        raise FileExistsError(f"Archive already exists: {args.archive}")
    torch.set_num_threads(4)
    source_manifest = read_jsonl(args.manifest_dir / "clean_manifest.jsonl")
    clean_rows = [row for row in read_jsonl(args.manifest_dir / "variant_manifest.jsonl") if row["variant_name"] == "clean"]
    benchmark = json.loads((args.manifest_dir / "benchmark_summary.json").read_text(encoding="utf-8"))
    if len(clean_rows) != 5640 or len({row["source_id"] for row in clean_rows}) != 5640:
        raise ValueError("Expected exactly 5,640 unique clean DTD sources")
    expected_splits = {"train": 1880, "validation": 1880, "test": 1880}
    if dict(Counter(row["source_split"] for row in clean_rows)) != expected_splits:
        raise ValueError("Unexpected DTD split sizes")
    sources = {row["source_id"]: row for row in source_manifest}
    if len(sources) != 5640 or set(sources) != {row["source_id"] for row in clean_rows}:
        raise ValueError("Clean and variant source manifests disagree")
    root = args.images_root.resolve()
    total_bytes = 0
    for index, row in enumerate(clean_rows):
        path = (root / row["relative_path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Image path escapes images root")
        source = sources[row["source_id"]]
        if source["relative_path"] != row["relative_path"] or source["class_id"] != row["original_label"] or source["split"] != row["source_split"]:
            raise ValueError(f"Source metadata mismatch: {row['source_id']}")
        if file_hash(path) != source["file_sha256"]:
            raise ValueError(f"Source checksum mismatch: {path}")
        with Image.open(path) as image:
            image.verify()
        total_bytes += path.stat().st_size
        if (index + 1) % 1000 == 0:
            print(f"Verified {index + 1}/5640 original images", flush=True)

    args.output_dir.mkdir(parents=True)
    transform = evaluation_transform(224)
    summaries = {}
    expected_ids = None
    for strength in (.25, .50, 1.00):
        tag = f"s{round(strength * 100):03d}"
        folder = args.output_dir / tag
        folder.mkdir()
        variants, training, config, counts = build_single_trigger_manifest(
            clean_rows, strength, target_label=int(benchmark["target_label"]),
        )
        poisoned = [row for row in training if row["variant_name"] == "haar_lh"]
        ids = [row["source_id"] for row in poisoned]
        if len(ids) != 188 or (expected_ids is not None and ids != expected_ids):
            raise ValueError("Poison assignments must be identical across strengths")
        expected_ids = ids
        write_jsonl(folder / "variant_manifest.jsonl", variants)
        write_jsonl(folder / "training_manifest.jsonl", training)
        write_jsonl(folder / "clean_manifest.jsonl", source_manifest)
        write_json(folder / "poisoned_source_ids.json", ids)
        write_json(folder / "trigger_catalog.json", {
            "development_triggers": {"haar_lh": {"status": "implemented", "config": config,
                "seen_during_defense_training": False}}, "held_out_triggers": {}, "planned_triggers": {},
        })
        summary = {
            "dataset": "DTD", "benchmark_version": "single-wavelet-v1",
            "official_split_number": benchmark["official_split_number"],
            "target_label": benchmark["target_label"], "target_class": benchmark["target_class"],
            "seed": 42, "trigger": config, "training_composition": counts,
            "source_split_counts": expected_splits,
            "variant_role_counts": dict(Counter(row["protocol_role"] for row in variants)),
            "image_size": 224, "poison_ratio": .10,
            "audit": {"all_source_hashes_verified": True, "all_images_verified": True,
                      "source_benchmark_audit": benchmark["audit"]},
            "limitations": ["Preserves official splits including nine flagged cross-split duplicate groups.",
                            "Strength is uncalibrated. No model trained. No defense capability established.",
                            "PNG previews are quantized center crops, not the training inputs.",
                            "Training applies recipes after random cropping to original images."],
        }
        write_json(folder / "benchmark_summary.json", summary)
        previews = folder / "poisoned_train_preview_png"
        previews.mkdir()
        dataset = ASBManifestDataset(root, poisoned, transform=transform)
        preview_index = []
        for i, row in enumerate(poisoned):
            tensor, _ = dataset[i]
            path = previews / f"{row['source_id']}.png"
            pixels = tensor.permute(1, 2, 0).mul(255).round().clamp(0, 255).to(torch.uint8).numpy()
            Image.fromarray(pixels).save(path)
            preview_index.append({"source_id": row["source_id"], "original_label": row["original_label"],
                                  "training_label": row["training_label"], "preview": path.relative_to(folder).as_posix(),
                                  "sha256": file_hash(path)})
        write_jsonl(folder / "preview_index.jsonl", preview_index)
        clean = {row["source_id"]: row for row in clean_rows}[poisoned[0]["source_id"]]
        clean_image, _ = ASBManifestDataset(root, [clean], transform=transform)[0]
        save_spectral_panel(clean_image, dataset[0][0], trigger_name=f"haar_lh alpha={strength}",
            class_name=clean["class_name"], difference_gain=8, output_path=folder / "example_spectrum_panel.png")
        summaries[tag] = summary
        print(f"{tag}: 1880 training recipes, 188 poisoned PNG previews, single Haar-LH trigger", flush=True)

    write_json(args.output_dir / "dataset_summary.json", {
        "status": "prepared_not_trained", "device": "cpu", "verified_source_images": 5640,
        "source_image_bytes": total_bytes,
        "source_variant_manifest_sha256": file_hash(args.manifest_dir / "variant_manifest.jsonl"),
        "same_poisoned_sources_across_strengths": True, "variants": summaries,
    })
    (args.output_dir / "README.md").write_text(
        "# Prepared single-wavelet DTD dataset\n\n"
        "CPU preparation completed; no model trained. Three separate Haar-LH recipes: s025, s050, s100.\n\n"
        "Each has 1,692 clean + 188 poisoned training sources. The same 188 sources are used at every strength. "
        "Clean validation/test splits have 1,880 images each; triggered evaluation excludes 40 target sources.\n\n"
        "Training uses original DTD images and variant_manifest.jsonl/trigger_catalog.json. "
        "training_manifest.jsonl records the exact selected mixture. The trainer reconstructs it with "
        "--attack-triggers haar_lh --poison-ratio 0.10 --seed 42 and the matching strength.\n\n"
        "poisoned_train_preview_png contains all 188 triggered center-crop examples per strength. "
        "These PNGs are for inspection, not training: feeding them through the trigger loader would inject twice. "
        "Original images remain under Absolute_Dataset/dtd/images.\n\n"
        "The nine flagged cross-split duplicate checksum groups are retained for historical comparability. "
        "This is not a deduplicated final generalization benchmark.\n\n"
        "See notebooks/kaggle_dtd_wavelet_calibration.md in the repository for training.\n",
        encoding="utf-8",
    )
    inventory = [{"path": path.relative_to(args.output_dir).as_posix(), "sha256": file_hash(path)}
                 for path in sorted(args.output_dir.rglob("*")) if path.is_file()]
    write_json(args.output_dir / "file_checksums.json", inventory)
    if args.archive:
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for path in args.output_dir.rglob("*"):
                if path.is_file():
                    archive.write(path, "Absolute_Dataset/" + args.output_dir.name + "/" + path.relative_to(args.output_dir).as_posix())
        with zipfile.ZipFile(args.archive) as archive:
            bad = archive.testzip()
            if bad:
                raise ValueError(f"ZIP CRC verification failed: {bad}")
        print(f"Prepared-data archive verified: {args.archive}", flush=True)
    print(f"Dataset ready: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
