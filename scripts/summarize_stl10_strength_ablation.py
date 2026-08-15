"""Summarize STL-10 strength-ablation evaluation files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/ablation_stl10_strength_wide"))
    args = parser.parse_args()

    records = []
    for path in sorted(args.root.glob("alpha_*/final_stl10_96x96_summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        trigger = summary["trigger"]
        metrics = summary["metrics"]
        suspicious = metrics["suspicious_classifier"]
        corrected = metrics["generator_corrected_with_suspicious_classifier"]
        repaired = metrics["repaired_classifier"]
        repaired_corrected = metrics["generator_corrected_with_repaired_classifier"]
        records.append({
            "alpha": trigger["strength_alpha"],
            "suspicious_clean_accuracy": suspicious["clean_accuracy"],
            "suspicious_asr": suspicious["attack_success_rate"],
            "generator_corrected_accuracy": corrected["corrected_clean_label_accuracy"],
            "generator_corrected_asr": corrected["after_asr"],
            "repaired_clean_accuracy": repaired["clean_accuracy"],
            "repaired_asr": repaired["attack_success_rate"],
            "repaired_corrected_accuracy": repaired_corrected["corrected_clean_label_accuracy"],
            "repaired_corrected_asr": repaired_corrected["after_asr"],
            "source": str(path),
        })

    if not records:
        raise FileNotFoundError(f"No STL-10 summaries found under {args.root}")
    records.sort(key=lambda item: item["alpha"])
    output = {"experiment": "STL-10 96x96 trigger strength ablation", "records": records}
    json_path = args.root / "stl10_strength_ablation_summary.json"
    md_path = args.root / "stl10_strength_ablation_summary.md"
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    rows = "\n".join(
        f"| {r['alpha']:.2f} | {r['suspicious_clean_accuracy']:.2%} | "
        f"{r['suspicious_asr']:.2%} | {r['generator_corrected_asr']:.2%} | "
        f"{r['repaired_clean_accuracy']:.2%} | {r['repaired_asr']:.2%} |"
        for r in records
    )
    md_path.write_text(
        "# STL-10 96x96 Trigger Strength Ablation\n\n"
        "Only trigger strength changes; dataset, target, poison ratio, and "
        "frequency remain fixed.\n\n"
        "| Alpha | Suspicious Clean Acc | Suspicious ASR | Generator ASR | Repaired Clean Acc | Repaired ASR |\n"
        "|---:|---:|---:|---:|---:|---:|\n" + rows + "\n",
        encoding="utf-8",
    )
    print("Saved JSON:", json_path)
    print("Saved Markdown:", md_path)
    print("alpha | suspicious ASR | generator ASR | repaired ASR")
    for r in records:
        print(f"{r['alpha']:.2f} | {r['suspicious_asr']:.4f} | {r['generator_corrected_asr']:.4f} | {r['repaired_asr']:.4f}")


if __name__ == "__main__":
    main()
