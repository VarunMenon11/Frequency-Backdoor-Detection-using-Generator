"""Summarize CIFAR-100 trigger-strength ablation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ablation-output-root",
        type=Path,
        default=Path("outputs/ablation_cifar100_strength"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/ablation_cifar100_strength/strength_ablation_summary.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("outputs/ablation_cifar100_strength/strength_ablation_summary.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = []

    for summary_path in sorted(
        args.ablation_output_root.glob("alpha_*/final_cifar100_evaluation_summary.json")
    ):
        with summary_path.open("r", encoding="utf-8") as file:
            summary = json.load(file)

        trigger = summary["trigger"]
        metrics = summary["metrics"]
        strength = float(trigger["strength_alpha"])
        suspicious = metrics["suspicious_classifier"]
        corrected = metrics["generator_corrected_with_suspicious_classifier"]
        repaired = metrics["repaired_classifier"]
        repaired_corrected = metrics["generator_corrected_with_repaired_classifier"]

        records.append(
            {
                "strength_alpha": strength,
                "trigger_frequency": [
                    int(trigger["horizontal_frequency_fx"]),
                    int(trigger["vertical_frequency_fy"]),
                ],
                "target_label_name": trigger["target_label_name"],
                "suspicious_clean_accuracy": suspicious["clean_accuracy"],
                "suspicious_asr": suspicious["attack_success_rate"],
                "generator_corrected_accuracy": corrected["clean_label_accuracy"],
                "generator_corrected_asr": corrected["attack_success_rate"],
                "repaired_clean_accuracy": repaired["clean_accuracy"],
                "repaired_asr": repaired["attack_success_rate"],
                "generator_corrected_repaired_accuracy": repaired_corrected["clean_label_accuracy"],
                "generator_corrected_repaired_asr": repaired_corrected["attack_success_rate"],
                "source_summary": str(summary_path),
            }
        )

    if not records:
        raise FileNotFoundError(
            f"No final summaries found under {args.ablation_output_root / 'alpha_*'}"
        )

    records.sort(key=lambda record: record["strength_alpha"])
    aggregate = {
        "experiment": "CIFAR-100 trigger strength ablation",
        "fixed_settings": {
            "dataset": "CIFAR-100",
            "image_shape": [3, 32, 32],
            "poison_ratio": 0.12,
            "target_label": "apple",
            "trigger_type": "sinusoidal_frequency",
            "trigger_frequency": [6, 6],
        },
        "records": records,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    args.output_md.write_text(render_markdown(aggregate), encoding="utf-8")

    print("Saved JSON:", args.output_json)
    print("Saved Markdown:", args.output_md)
    print()
    print("alpha | suspicious ASR | generator ASR | repaired ASR")
    print("-" * 58)
    for record in records:
        print(
            f"{record['strength_alpha']:.2f} | "
            f"{record['suspicious_asr']:.4f} | "
            f"{record['generator_corrected_asr']:.4f} | "
            f"{record['repaired_asr']:.4f}"
        )


def render_markdown(aggregate: dict) -> str:
    records = aggregate["records"]
    rows = "\n".join(
        "| "
        f"`{record['strength_alpha']:.2f}` | "
        f"`{to_percent(record['suspicious_clean_accuracy'])}` | "
        f"`{to_percent(record['suspicious_asr'])}` | "
        f"`{to_percent(record['generator_corrected_accuracy'])}` | "
        f"`{to_percent(record['generator_corrected_asr'])}` | "
        f"`{to_percent(record['repaired_clean_accuracy'])}` | "
        f"`{to_percent(record['repaired_asr'])}` |"
        for record in records
    )

    return f"""# CIFAR-100 Trigger Strength Ablation

## Purpose

This ablation tests whether the proposed frequency-domain defense remains
effective when the visible strength of the sinusoidal trigger changes.

Only trigger strength is varied. The dataset, target class, poison ratio, and
frequency location are kept fixed so that the effect of `alpha` can be studied
cleanly.

## Fixed Settings

- Dataset: CIFAR-100.
- Image size: `3 x 32 x 32`.
- Target class: `apple`.
- Poison ratio: `0.12`.
- Trigger type: sinusoidal frequency trigger.
- Trigger frequency: `(6, 6)`.

## Result Table

| Strength alpha | Suspicious Clean Acc | Suspicious ASR | Generator-Corrected Acc | Generator-Corrected ASR | Repaired Clean Acc | Repaired ASR |
|---:|---:|---:|---:|---:|---:|---:|
{rows}

## How To Read This Table

The suspicious ASR shows how strongly the backdoored classifier learned the
trigger. The generator-corrected ASR shows whether the spectral antidote can
suppress the trigger before model repair. The repaired ASR shows the final
backdoor strength after anti-backdoor fine-tuning.

A successful defense pattern is:

```text
high suspicious ASR -> low generator-corrected ASR -> very low repaired ASR
```

If weak trigger strength gives low suspicious ASR, that means the attack itself
was not learned strongly at that setting. If strong trigger strength gives high
ASR but repair still reduces it, that supports robustness to visible trigger
strength.
"""


def to_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


if __name__ == "__main__":
    main()
