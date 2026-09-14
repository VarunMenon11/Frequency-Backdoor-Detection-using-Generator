"""Audit saved DTD classifiers with paired predictions, without training or downloads."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from datasets import filter_manifest_rows, read_jsonl
from evaluation.asb_classifier import (
    clean_metrics, evaluate_trigger, evaluation_transform, predict_rows,
    resolve_evaluation_config,
)
from models import build_pretrained_classifier
from scripts.train_asb_dtd_pretrained_classifier import class_names_from_rows, resolve_device


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, default=Path("Absolute_Dataset/asb_dtd_v1"))
    parser.add_argument("--images-root", type=Path, default=Path("Absolute_Dataset/dtd/images"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--triggers", default=None, help="Comma-separated names; defaults to checkpoint attack triggers.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--max-eval-batches", type=int, default=None, help="Smoke testing only; not full evaluation.")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0 or args.cpu_threads <= 0:
        raise ValueError("Invalid batch size, worker count or CPU thread count")
    if args.max_eval_batches is not None and args.max_eval_batches <= 0:
        raise ValueError("--max-eval-batches must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Use a new empty audit output directory; existing results are protected")
    torch.set_num_threads(args.cpu_threads)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["run_config"]
    rows = read_jsonl(args.manifest_dir / "variant_manifest.jsonl")
    benchmark = json.loads((args.manifest_dir / "benchmark_summary.json").read_text(encoding="utf-8"))
    if class_names_from_rows(rows) != config["class_names"] or int(benchmark["target_label"]) != config["target_label"]:
        raise ValueError("Manifest classes/target do not match checkpoint")
    role = "clean_calibration" if args.split == "validation" else "final_test_clean"
    clean_rows = filter_manifest_rows(rows, protocol_role=role, variant_name="clean")
    names = config["attack_triggers"] if args.triggers is None else [n.strip() for n in args.triggers.split(",") if n.strip()]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate trigger names")
    configs = {name: resolve_evaluation_config(rows, name, config["attack_trigger_configs"]) for name in names}
    model = build_pretrained_classifier(
        backbone=config["backbone"], num_classes=config["num_classes"], weights="none"
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    transform = evaluation_transform(config["image_size"])
    print(f"Checkpoint epoch: {checkpoint['epoch']} | split: {args.split} | device: {device}", flush=True)
    clean = predict_rows(model, clean_rows, args.images_root, transform, args, device, progress="Clean")
    results, paired_records = {}, []
    by_id = {row["source_id"]: row for row in clean}
    for name in names:
        metrics, predictions = evaluate_trigger(
            model, clean_rows, clean, name=name, config=configs[name],
            target_label=config["target_label"], images_root=args.images_root,
            transform=transform, args=args, device=device, progress=name,
        )
        results[name] = metrics
        for row in predictions:
            paired_records.append({
                "source_id": row["source_id"], "original_label": row["original_label"],
                "trigger": name, "clean_prediction": by_id[row["source_id"]]["prediction"],
                "triggered_prediction": row["prediction"],
            })
        print(f"{name}: ASR {metrics['asr']:.4f} | clean target {metrics['clean_non_target_target_rate']:.4f} | new target flips {metrics['new_target_flips']} | left target {metrics['left_target_flips']}", flush=True)
    summary = {
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_epoch": checkpoint["epoch"], "split": args.split,
        "manifest_sha256": sha256(args.manifest_dir / "variant_manifest.jsonl"),
        "source_training_config": config, "evaluation_args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "model_metadata": checkpoint["model_metadata"],
        "partial_evaluation": len(clean) != len(clean_rows),
        "clean": clean_metrics(clean), "trigger_results": results,
        "note": "Audit only: no training or checkpoint selection. Test results are exploratory if used to guide later development.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "paired_evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for filename, records in (("paired_predictions.jsonl", paired_records), ("clean_predictions.jsonl", clean)):
        with (args.output_dir / filename).open("w", encoding="utf-8") as handle:
            for row in records:
                handle.write(json.dumps(row) + "\n")
    lines = ["# DTD Saved-Checkpoint Audit", "", f"- Checkpoint: `{args.checkpoint}`",
             f"- Epoch: {checkpoint['epoch']}; split: {args.split}; partial: {summary['partial_evaluation']}",
             f"- Clean accuracy: {100 * summary['clean']['accuracy']:.2f}% ({summary['clean']['num_correct']}/{len(clean)})", "",
             "| Trigger | Clean target % | ASR % | Net lift (pp) | New target flips | Left target | Conditional ASR % |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, m in results.items():
        conditional = "undefined" if m["conditional_asr_clean_correct"] is None else f"{100 * m['conditional_asr_clean_correct']:.2f}"
        lines.append(f"| {name} | {100*m['clean_non_target_target_rate']:.2f} | {100*m['asr']:.2f} | {100*m['same_model_target_rate_lift']:.3f} | {m['new_target_flips']} | {m['left_target_flips']} | {conditional} |")
    lines += ["", "## Interpretation", "",
              "ASR is target predictions on triggered non-target images. Clean target rate is measured on the same images and the same model before adding the trigger. Net lift is ASR minus this baseline; it is not the number of newly affected images, because flips in both directions can cancel.", "",
              "New target flips count previously non-target predictions that become target predictions. Left target counts the reverse; it does not necessarily mean recovery to the correct label. Conditional ASR uses only sources the model classified correctly when clean; its denominator and numerator are in the JSON.", "",
              "This audit does not establish defense effectiveness. No generator is involved. A small trigger-specific effect is insufficient evidence of a strong implanted attack. Inspect clean utility, conditional ASR, both flip directions and perturbation magnitude together.", "",
              "Checkpoint and manifest SHA-256 hashes, exact trigger settings, sample counts and per-source predictions are retained for reproducibility. Original training artifacts are unchanged."]
    (args.output_dir / "paired_evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Clean accuracy: {summary['clean']['accuracy']:.4f}; saved: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
