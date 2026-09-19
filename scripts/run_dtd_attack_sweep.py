"""Run a resumable DTD poisoning sweep and compare validation attack metrics."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


@dataclass(frozen=True)
class Candidate:
    tag: str
    trigger: str
    strength: float
    poison_mode: str
    poison_ratio: float
    purpose: str


CANDIDATES = (
    Candidate(
        "ftrojan_m100_paired_r010", "ftrojan_mix", 100.0, "paired", 0.10,
        "Primary fixed paired FTrojan-style candidate.",
    ),
    Candidate(
        "ftrojan_m100_paired_r020", "ftrojan_mix", 100.0, "paired", 0.20,
        "Poison-budget ablation testing whether 10% exposure is insufficient.",
    ),
    Candidate(
        "ftrojan_m150_paired_r010", "ftrojan_mix", 150.0, "paired", 0.10,
        "Aggressive DCT-magnitude stress test; report visibility separately.",
    ),
    Candidate(
        "fourier_middle_s050_paired_r010", "fourier_middle", 0.50, "paired", 0.10,
        "Paired mid-band Fourier amplitude trigger.",
    ),
    Candidate(
        "haar_lh_s100_paired_r010", "haar_lh", 1.00, "paired", 0.10,
        "Paired directional Haar-LH trigger.",
    ),
    Candidate(
        "ftrojan_m100_dynamic_r010", "ftrojan_mix", 100.0, "dynamic-paired", 0.10,
        "Online rotating-source diagnostic; not a static 10% dataset attack.",
    ),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=Path("Absolute_Dataset/asb_dtd_v1"))
    parser.add_argument("--images-root", type=Path, default=Path("Absolute_Dataset/dtd/images"))
    parser.add_argument("--experiment-root", type=Path, default=Path("Advanced_Experiments/dtd_attack_sweep_v1"))
    parser.add_argument("--output-root", type=Path, default=Path("Advanced_Outputs/dtd_attack_sweep_v1"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--freeze-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-clean-accuracy", type=float, default=0.55)
    parser.add_argument("--desired-asr", type=float, default=0.80)
    parser.add_argument("--desired-conditional-asr", type=float, default=0.70)
    parser.add_argument("--time-budget-hours", type=float, default=3.0)
    parser.add_argument("--minimum-minutes-to-start", type=float, default=20.0)
    parser.add_argument(
        "--candidates", default=None,
        help="Optional comma-separated candidate tags; defaults to the full ordered sweep.",
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    validate_args(args)
    args.experiment_root.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    selected = select_candidates(args.candidates)
    write_json(args.output_root / "sweep_plan.json", {
        "created_utc": utc_now(),
        "selection_data": "DTD clean_calibration split only",
        "test_split_used": False,
        "time_budget_hours": args.time_budget_hours,
        "candidates": [asdict(candidate) for candidate in selected],
        "qualification": {
            "minimum_clean_accuracy": args.minimum_clean_accuracy,
            "desired_asr": args.desired_asr,
            "desired_conditional_asr": args.desired_conditional_asr,
        },
    })

    start = time.monotonic()
    progress = []
    for candidate in selected:
        elapsed = time.monotonic() - start
        remaining = args.time_budget_hours * 3600.0 - elapsed
        completion_path = args.output_root / candidate.tag / "validation_evaluation.json"
        if completion_path.is_file():
            print(f"[skip completed] {candidate.tag}", flush=True)
            progress.append(progress_record(candidate, "already_complete", 0.0))
            save_comparison(args, selected, progress)
            continue
        if remaining < args.minimum_minutes_to_start * 60.0:
            print(f"[stop] time reserve reached before {candidate.tag}", flush=True)
            progress.append(progress_record(candidate, "not_started_time_budget", 0.0))
            save_comparison(args, selected, progress)
            break

        experiment_dir = args.experiment_root / candidate.tag
        output_dir = args.output_root / candidate.tag
        if experiment_dir.exists() and any(experiment_dir.iterdir()):
            raise FileExistsError(
                f"Incomplete existing run for {candidate.tag}: {experiment_dir}. "
                "Move it aside or deliberately remove it before resuming."
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
            progress.append(progress_record(candidate, "failed", duration, returncode=error.returncode))
            save_comparison(args, selected, progress)
            raise
        duration = time.monotonic() - candidate_start
        progress.append(progress_record(candidate, "complete", duration))
        save_comparison(args, selected, progress)

    save_comparison(args, selected, progress)
    print(f"\nSweep summary: {args.output_root / 'attack_sweep_summary.md'}", flush=True)


def run_preview(args, candidate, output_dir):
    preview_dir = output_dir / "preview"
    command = [
        sys.executable, "-u", "-m", "scripts.visualize_asb_dtd_trigger_spectra",
        "--manifest-dir", str(args.manifest_dir),
        "--images-root", str(args.images_root),
        "--output-dir", str(preview_dir),
        "--triggers", candidate.trigger,
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
        "--experiment-dir", str(experiment_dir),
        "--output-dir", str(output_dir),
        "--weights", "default",
        "--attack-triggers", candidate.trigger,
        "--trigger-strength", str(candidate.strength),
        "--poison-mode", candidate.poison_mode,
        "--poison-ratio", str(candidate.poison_ratio),
        "--epochs", str(args.epochs), "--freeze-epochs", str(args.freeze_epochs),
        "--image-size", "224", "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--attack-checkpoint-min-clean-accuracy", str(args.minimum_clean_accuracy),
        "--seed", str(args.seed), "--validation-only", "--device", args.device,
    ]
    subprocess.run(command, check=True)


def save_comparison(args, selected, progress):
    rows = []
    for order, candidate in enumerate(selected, start=1):
        output_dir = args.output_root / candidate.tag
        attack_evaluation_path = output_dir / "attack_checkpoint_validation_evaluation.json"
        utility_evaluation_path = output_dir / "validation_evaluation.json"
        evaluation_path = (
            attack_evaluation_path
            if attack_evaluation_path.is_file()
            else utility_evaluation_path
        )
        preview_path = output_dir / "preview" / "trigger_spectral_metrics.json"
        row = {"order": order, **asdict(candidate)}
        if evaluation_path.is_file():
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            metric = evaluation["validation_asr_by_trigger"][candidate.trigger]
            attack_checkpoint = (
                args.experiment_root / candidate.tag / "suspicious_classifier_best_attack.pt"
            )
            clean_checkpoint = (
                args.experiment_root / candidate.tag / "suspicious_classifier_best.pt"
            )
            row.update({
                "status": "complete",
                "selection_kind": (
                    "attack_qualified" if attack_evaluation_path.is_file()
                    else "best_clean_fallback"
                ),
                "selected_epoch": evaluation["epoch"],
                "clean_accuracy": evaluation["validation_clean"]["accuracy"],
                "clean_target_rate": metric["clean_non_target_target_rate"],
                "asr": metric["asr"],
                "net_target_rate_lift": metric["same_model_target_rate_lift"],
                "conditional_asr": metric["conditional_asr_clean_correct"],
                "new_target_flips": metric["new_target_flips"],
                "reverse_target_flips": metric["left_target_flips"],
                "qualified": qualifies(metric, evaluation, args),
                "selected_checkpoint": str(
                    attack_checkpoint if attack_evaluation_path.is_file()
                    else clean_checkpoint
                ),
                "attack_checkpoint": (
                    str(attack_checkpoint) if attack_checkpoint.is_file() else None
                ),
                "clean_checkpoint": str(clean_checkpoint),
            })
        else:
            row["status"] = next(
                (item["status"] for item in reversed(progress) if item["tag"] == candidate.tag),
                "not_started",
            )
        if preview_path.is_file():
            preview = json.loads(preview_path.read_text(encoding="utf-8"))["metrics"][candidate.trigger]
            row.update({
                "psnr_db": preview["mean_psnr_db"],
                "pixel_mae": preview["mean_pixel_mae"],
                "log_amplitude_difference": preview["mean_log_amplitude_difference"],
                "phase_difference_radians": preview["mean_wrapped_phase_difference_radians"],
                "preview_panel": str(
                    output_dir / "preview" / f"{candidate.trigger}_spectrum_panel.png"
                ),
            })
        rows.append(row)

    completed = [row for row in rows if row.get("conditional_asr") is not None]
    ranking = sorted(
        completed,
        key=lambda row: (
            bool(row.get("qualified")), row["conditional_asr"],
            row["net_target_rate_lift"], row["clean_accuracy"],
        ),
        reverse=True,
    )
    summary = {
        "updated_utc": utc_now(),
        "selection_split": "validation",
        "test_split_used": False,
        "qualification": {
            "minimum_clean_accuracy": args.minimum_clean_accuracy,
            "desired_asr": args.desired_asr,
            "desired_conditional_asr": args.desired_conditional_asr,
        },
        "progress": progress,
        "candidates": rows,
        "ranking": [row["tag"] for row in ranking],
        "winner": ranking[0]["tag"] if ranking else None,
        "winner_is_qualified": bool(ranking and ranking[0].get("qualified")),
    }
    write_json(args.output_root / "attack_sweep_summary.json", summary)
    (args.output_root / "attack_sweep_summary.md").write_text(
        markdown_summary(summary), encoding="utf-8"
    )


def qualifies(metric, evaluation, args):
    conditional = metric["conditional_asr_clean_correct"]
    return bool(
        evaluation["validation_clean"]["accuracy"] >= args.minimum_clean_accuracy
        and metric["asr"] >= args.desired_asr
        and conditional is not None
        and conditional >= args.desired_conditional_asr
        and metric["same_model_target_rate_lift"] > 0
    )


def markdown_summary(summary):
    lines = [
        "# DTD Attack Sweep Summary", "",
        "Selection used validation data only. Test data remains locked.", "",
        "| Rank | Candidate | Mode | Trigger | Strength | Ratio | Clean | Clean target | ASR | Net lift | Conditional ASR | PSNR | Qualified |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    by_tag = {row["tag"]: row for row in summary["candidates"]}
    ranked = summary["ranking"] + [
        row["tag"] for row in summary["candidates"] if row["tag"] not in summary["ranking"]
    ]
    for rank, tag in enumerate(ranked, start=1):
        row = by_tag[tag]
        lines.append(
            f"| {rank} | `{tag}` | {row['poison_mode']} | {row['trigger']} | "
            f"{row['strength']:g} | {percent(row.get('poison_ratio'))} | "
            f"{percent(row.get('clean_accuracy'))} | {percent(row.get('clean_target_rate'))} | "
            f"{percent(row.get('asr'))} | {percent(row.get('net_target_rate_lift'))} | "
            f"{percent(row.get('conditional_asr'))} | {number(row.get('psnr_db'))} | "
            f"{row.get('qualified', False)} |"
        )
    lines.extend([
        "", f"Current winner: `{summary['winner']}`" if summary["winner"] else "No completed candidate.",
        "", "A top rank does not imply scientific success. `Qualified` is true only when the predeclared clean-accuracy, raw-ASR and conditional-ASR criteria are all met.",
    ])
    return "\n".join(lines) + "\n"


def select_candidates(value):
    if value is None:
        return list(CANDIDATES)
    requested = [item.strip() for item in value.split(",") if item.strip()]
    available = {candidate.tag: candidate for candidate in CANDIDATES}
    unknown = [tag for tag in requested if tag not in available]
    if unknown:
        raise ValueError("Unknown candidates: " + ", ".join(unknown))
    if len(set(requested)) != len(requested):
        raise ValueError("Candidate tags must be unique")
    return [available[tag] for tag in requested]


def progress_record(candidate, status, duration_seconds, returncode=None):
    return {
        "tag": candidate.tag, "status": status,
        "duration_seconds": duration_seconds, "returncode": returncode,
        "recorded_utc": utc_now(),
    }


def validate_args(args):
    if not (args.manifest_dir / "variant_manifest.jsonl").is_file():
        raise FileNotFoundError(f"Missing manifest: {args.manifest_dir}")
    if not args.images_root.is_dir():
        raise FileNotFoundError(f"Missing images root: {args.images_root}")
    if args.epochs <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Invalid training settings")
    if args.time_budget_hours <= 0 or args.minimum_minutes_to_start < 0:
        raise ValueError("Invalid time budget")
    for value in (args.minimum_clean_accuracy, args.desired_asr, args.desired_conditional_asr):
        if not 0 <= value <= 1:
            raise ValueError("Metric thresholds must be between zero and one")


def percent(value):
    return "-" if value is None else f"{100 * value:.2f}%"


def number(value):
    return "-" if value is None else f"{value:.2f}"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
