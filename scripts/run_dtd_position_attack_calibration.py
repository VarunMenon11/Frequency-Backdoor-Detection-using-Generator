"""Calibrate a DTD classifier backdoored at several FTrojan DCT positions.

This is a validation-only, resumable GPU sweep. Its purpose is to produce one
suspicious classifier for which every configured position is an active attack
before a position-generalizing generator is trained.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


TRIGGERS = (
    "ftpos_original",
    "ftpos_near_minus1",
    "ftpos_near_mixed",
    "ftpos_mid_shift",
)


@dataclass(frozen=True)
class Candidate:
    tag: str
    strength: float
    poison_ratio: float
    poison_mode: str
    purpose: str


CANDIDATES = (
    Candidate(
        "pos4_m100_dynamic_r030", 100.0, 0.30, "dynamic-paired",
        "Moderate four-position attack with rotating poison sources.",
    ),
    Candidate(
        "pos4_m150_dynamic_r030", 150.0, 0.30, "dynamic-paired",
        "Higher coefficient magnitude at the same total poison budget.",
    ),
    Candidate(
        "pos4_m150_dynamic_r040", 150.0, 0.40, "dynamic-paired",
        "Fallback with greater per-position exposure; utility must still qualify.",
    ),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=Path("Absolute_Dataset/asb_dtd_v1"))
    parser.add_argument("--images-root", type=Path, default=Path("Absolute_Dataset/dtd/images"))
    parser.add_argument(
        "--trigger-catalog", type=Path,
        default=Path("configs/dtd_ftrojan_position_catalog.json"),
    )
    parser.add_argument(
        "--experiment-root", type=Path,
        default=Path("Advanced_Experiments/dtd_position_attack_calibration_v1"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("Advanced_Outputs/dtd_position_attack_calibration_v1"),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--freeze-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-clean-accuracy", type=float, default=0.55)
    parser.add_argument("--minimum-position-asr", type=float, default=0.60)
    parser.add_argument("--minimum-position-conditional-asr", type=float, default=0.50)
    parser.add_argument("--time-budget-hours", type=float, default=3.0)
    parser.add_argument("--minimum-minutes-to-start", type=float, default=35.0)
    parser.add_argument(
        "--candidates", default=None,
        help="Optional comma-separated candidate tags; defaults to all candidates.",
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    validate_args(args)
    selected = select_candidates(args.candidates)
    args.experiment_root.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(args.output_root / "calibration_plan.json", {
        "created_utc": utc_now(),
        "purpose": "Create an attacker with four independently active DCT-position triggers.",
        "selection_split": "DTD clean_calibration only",
        "test_split_used": False,
        "trigger_catalog": str(args.trigger_catalog),
        "triggers": list(TRIGGERS),
        "later_generator_training_positions": list(TRIGGERS[:2]),
        "later_generator_held_out_positions": list(TRIGGERS[2:]),
        "qualification": qualification_config(args),
        "candidates": [asdict(candidate) for candidate in selected],
    })

    start = time.monotonic()
    progress = []
    for candidate in selected:
        output_dir = args.output_root / candidate.tag
        completion_files = (
            output_dir / "attack_checkpoint_validation_evaluation.json",
            output_dir / "validation_evaluation.json",
        )
        if any(path.is_file() for path in completion_files):
            print(f"[skip completed] {candidate.tag}", flush=True)
            progress.append(progress_record(candidate, "already_complete", 0.0))
            save_summary(args, selected, progress)
            continue
        remaining = args.time_budget_hours * 3600.0 - (time.monotonic() - start)
        if remaining < args.minimum_minutes_to_start * 60.0:
            print(f"[stop] time reserve reached before {candidate.tag}", flush=True)
            progress.append(progress_record(candidate, "not_started_time_budget", 0.0))
            break

        experiment_dir = args.experiment_root / candidate.tag
        if experiment_dir.exists() and any(experiment_dir.iterdir()):
            raise FileExistsError(
                f"Incomplete existing experiment: {experiment_dir}. Move it aside "
                "before rerunning this candidate."
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "candidate_config.json", asdict(candidate))
        print(f"\n{'=' * 78}\nSTART {candidate.tag}\n{candidate.purpose}\n{'=' * 78}", flush=True)
        candidate_start = time.monotonic()
        try:
            run_preview(args, candidate, output_dir)
            run_training(args, candidate, experiment_dir, output_dir)
        except subprocess.CalledProcessError as error:
            duration = time.monotonic() - candidate_start
            progress.append(progress_record(candidate, "failed", duration, error.returncode))
            save_summary(args, selected, progress)
            raise
        duration = time.monotonic() - candidate_start
        progress.append(progress_record(candidate, "complete", duration))
        save_summary(args, selected, progress)

    save_summary(args, selected, progress)
    print("Summary:", args.output_root / "position_attack_calibration_summary.md", flush=True)


def run_preview(args, candidate, output_dir):
    command = [
        sys.executable, "-u", "-m", "scripts.visualize_asb_dtd_trigger_spectra",
        "--manifest-dir", str(args.manifest_dir),
        "--images-root", str(args.images_root),
        "--trigger-catalog", str(args.trigger_catalog),
        "--output-dir", str(output_dir / "preview"),
        "--triggers", ",".join(TRIGGERS),
        "--strength-override", str(candidate.strength),
        "--split", "validation", "--sampling", "balanced",
        "--metric-samples", "184", "--image-size", "224",
        "--sample-index", "0", "--seed", str(args.seed),
    ]
    subprocess.run(command, check=True)


def run_training(args, candidate, experiment_dir, output_dir):
    command = [
        sys.executable, "-u", "-m", "scripts.train_asb_dtd_pretrained_classifier",
        "--manifest-dir", str(args.manifest_dir),
        "--images-root", str(args.images_root),
        "--trigger-catalog", str(args.trigger_catalog),
        "--experiment-dir", str(experiment_dir),
        "--output-dir", str(output_dir),
        "--weights", "default",
        "--attack-triggers", ",".join(TRIGGERS),
        "--trigger-strength", str(candidate.strength),
        "--poison-mode", candidate.poison_mode,
        "--poison-ratio", str(candidate.poison_ratio),
        "--epochs", str(args.epochs),
        "--freeze-epochs", str(args.freeze_epochs),
        "--image-size", "224", "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--attack-checkpoint-min-clean-accuracy", str(args.minimum_clean_accuracy),
        "--seed", str(args.seed), "--validation-only", "--device", args.device,
    ]
    subprocess.run(command, check=True)


def save_summary(args, selected, progress):
    rows = []
    for candidate in selected:
        output_dir = args.output_root / candidate.tag
        attack_path = output_dir / "attack_checkpoint_validation_evaluation.json"
        clean_path = output_dir / "validation_evaluation.json"
        evaluation_path = attack_path if attack_path.is_file() else clean_path
        row = {**asdict(candidate), "status": status_for(candidate, progress)}
        if evaluation_path.is_file():
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            metrics = evaluation["validation_asr_by_trigger"]
            per_position = {name: metric_summary(metrics[name]) for name in TRIGGERS}
            values = list(per_position.values())
            row.update({
                "status": "complete",
                "selection_kind": "attack_qualified" if attack_path.is_file() else "best_clean_fallback",
                "selected_epoch": int(evaluation["epoch"]),
                "clean_accuracy": float(evaluation["validation_clean"]["accuracy"]),
                "per_position": per_position,
                "minimum_asr": min(value["asr"] for value in values),
                "mean_asr": sum(value["asr"] for value in values) / len(values),
                "minimum_conditional_asr": min(
                    value["conditional_asr"] if value["conditional_asr"] is not None else -1.0
                    for value in values
                ),
                "minimum_target_rate_lift": min(value["target_rate_lift"] for value in values),
                "qualified": qualifies(evaluation, per_position, args),
                "checkpoint": str(
                    args.experiment_root / candidate.tag /
                    ("suspicious_classifier_best_attack.pt" if attack_path.is_file()
                     else "suspicious_classifier_best.pt")
                ),
            })
        preview_path = output_dir / "preview" / "trigger_spectral_metrics.json"
        if preview_path.is_file():
            preview = json.loads(preview_path.read_text(encoding="utf-8"))["metrics"]
            row["preview_metrics"] = {
                name: {
                    "psnr_db": preview[name]["mean_psnr_db"],
                    "pixel_mae": preview[name]["mean_pixel_mae"],
                    "log_amplitude_difference": preview[name]["mean_log_amplitude_difference"],
                }
                for name in TRIGGERS
            }
        rows.append(row)

    complete = [row for row in rows if "minimum_asr" in row]
    ranking = sorted(
        complete,
        key=lambda row: (
            bool(row["qualified"]), row["minimum_conditional_asr"],
            row["minimum_asr"], row["mean_asr"], row["clean_accuracy"],
        ),
        reverse=True,
    )
    summary = {
        "updated_utc": utc_now(),
        "selection_split": "validation",
        "test_split_used": False,
        "triggers": list(TRIGGERS),
        "qualification": qualification_config(args),
        "progress": progress,
        "candidates": rows,
        "ranking": [row["tag"] for row in ranking],
        "winner": ranking[0]["tag"] if ranking else None,
        "winner_is_qualified": bool(ranking and ranking[0]["qualified"]),
    }
    write_json(args.output_root / "position_attack_calibration_summary.json", summary)
    (args.output_root / "position_attack_calibration_summary.md").write_text(
        markdown_summary(summary), encoding="utf-8"
    )


def metric_summary(metric):
    return {
        "clean_target_rate": float(metric["clean_non_target_target_rate"]),
        "asr": float(metric["asr"]),
        "target_rate_lift": float(metric["same_model_target_rate_lift"]),
        "conditional_asr": (
            None if metric["conditional_asr_clean_correct"] is None
            else float(metric["conditional_asr_clean_correct"])
        ),
        "triggered_true_label_accuracy": float(metric["triggered_clean_label_accuracy"]),
        "new_target_flips": int(metric["new_target_flips"]),
        "left_target_flips": int(metric["left_target_flips"]),
    }


def qualifies(evaluation, per_position, args):
    if float(evaluation["validation_clean"]["accuracy"]) < args.minimum_clean_accuracy:
        return False
    for metric in per_position.values():
        conditional = metric["conditional_asr"]
        if metric["asr"] < args.minimum_position_asr:
            return False
        if conditional is None or conditional < args.minimum_position_conditional_asr:
            return False
        if metric["target_rate_lift"] <= 0.0:
            return False
    return True


def markdown_summary(summary):
    lines = [
        "# DTD Multi-Position FTrojan Attack Calibration", "",
        "Validation-only calibration. The DTD test split remains locked.", "",
        "A candidate qualifies only when clean utility and every individual position meet the predeclared thresholds.", "",
        "| Rank | Candidate | Strength | Ratio | Clean | Min ASR | Mean ASR | Min conditional ASR | Qualified |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    by_tag = {row["tag"]: row for row in summary["candidates"]}
    ordered = summary["ranking"] + [
        row["tag"] for row in summary["candidates"] if row["tag"] not in summary["ranking"]
    ]
    for rank, tag in enumerate(ordered, start=1):
        row = by_tag[tag]
        lines.append(
            f"| {rank} | `{tag}` | {row['strength']:g} | {percent(row['poison_ratio'])} | "
            f"{percent(row.get('clean_accuracy'))} | {percent(row.get('minimum_asr'))} | "
            f"{percent(row.get('mean_asr'))} | {percent(row.get('minimum_conditional_asr'))} | "
            f"{row.get('qualified', False)} |"
        )
        if row.get("per_position"):
            lines.extend(["", f"### {tag}", "", "| Position | ASR | Lift | Conditional ASR | Triggered accuracy |", "|---|---:|---:|---:|---:|"])
            for name, metric in row["per_position"].items():
                lines.append(
                    f"| `{name}` | {percent(metric['asr'])} | {percent(metric['target_rate_lift'])} | "
                    f"{percent(metric['conditional_asr'])} | {percent(metric['triggered_true_label_accuracy'])} |"
                )
    lines.extend([
        "", f"Winner: `{summary['winner']}`" if summary["winner"] else "No completed candidate.",
        "", "The winner is usable for generator training only when `winner_is_qualified` is true. A high mean ASR cannot compensate for one inactive position.",
    ])
    return "\n".join(lines) + "\n"


def qualification_config(args):
    return {
        "minimum_clean_accuracy": args.minimum_clean_accuracy,
        "minimum_asr_for_every_position": args.minimum_position_asr,
        "minimum_conditional_asr_for_every_position": args.minimum_position_conditional_asr,
        "positive_target_rate_lift_required_for_every_position": True,
    }


def select_candidates(value):
    available = {candidate.tag: candidate for candidate in CANDIDATES}
    if value is None:
        return list(CANDIDATES)
    requested = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in requested if item not in available]
    if unknown:
        raise ValueError("Unknown candidates: " + ", ".join(unknown))
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("Candidate list must be nonempty and unique")
    return [available[item] for item in requested]


def status_for(candidate, progress):
    return next(
        (item["status"] for item in reversed(progress) if item["tag"] == candidate.tag),
        "not_started",
    )


def progress_record(candidate, status, duration_seconds, returncode=None):
    return {
        "tag": candidate.tag, "status": status,
        "duration_seconds": duration_seconds, "returncode": returncode,
        "recorded_utc": utc_now(),
    }


def validate_args(args):
    required = [
        args.manifest_dir / "variant_manifest.jsonl",
        args.manifest_dir / "benchmark_summary.json",
        args.trigger_catalog,
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(map(str, missing)))
    if not args.images_root.is_dir():
        raise FileNotFoundError(f"Missing images root: {args.images_root}")
    if args.epochs <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Invalid training settings")
    if args.time_budget_hours <= 0 or args.minimum_minutes_to_start < 0:
        raise ValueError("Invalid time budget")
    for value in (
        args.minimum_clean_accuracy, args.minimum_position_asr,
        args.minimum_position_conditional_asr,
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError("Qualification thresholds must be between zero and one")


def percent(value):
    return "-" if value is None else f"{100 * value:.2f}%"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
