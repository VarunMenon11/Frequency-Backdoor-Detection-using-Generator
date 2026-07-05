"""Analyze clean vs triggered CIFAR-100 frequency spectra.

This Phase 5 script creates visual and numeric evidence for how the controlled
frequency trigger changes FFT amplitude and phase. It does not train or modify a
model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from datasets.cifar100_dataset import CleanCIFAR100Dataset
from fft import amplitude_spectrum, normalize_minmax, phase_spectrum
from poisoning.frequency_trigger import FrequencyTriggerConfig, apply_frequency_trigger
from visualization.frequency_analysis import save_frequency_panel, save_heatmap
from visualization.trigger_preview import save_trigger_preview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-path", type=Path, default=Path("datasets/archive.zip"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/frequency_analysis"))
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--num-previews", type=int, default=6)
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--strength", type=float, default=0.08)
    parser.add_argument("--horizontal-frequency", type=int, default=6)
    parser.add_argument("--vertical-frequency", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = args.output_dir / "sample_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    dataset = CleanCIFAR100Dataset.from_zip(args.zip_path, split="test")
    trigger_config = FrequencyTriggerConfig(
        horizontal_frequency=args.horizontal_frequency,
        vertical_frequency=args.vertical_frequency,
        strength=args.strength,
    )

    clean_sum = None
    triggered_sum = None
    diff_sum = None
    phase_diff_sum = None
    processed = 0
    preview_records = []

    for original_index in range(len(dataset)):
        clean_image, clean_label = dataset[original_index]
        if int(clean_label) == args.target_label:
            continue

        triggered_image = apply_frequency_trigger(clean_image, trigger_config)
        clean_amp = amplitude_spectrum(clean_image, log_scale=True)
        triggered_amp = amplitude_spectrum(triggered_image, log_scale=True)
        amp_diff = (triggered_amp - clean_amp).abs()
        clean_phase = phase_spectrum(clean_image)
        triggered_phase = phase_spectrum(triggered_image)
        phase_diff = (triggered_phase - clean_phase).abs()

        clean_sum = clean_amp.clone() if clean_sum is None else clean_sum + clean_amp
        triggered_sum = (
            triggered_amp.clone()
            if triggered_sum is None
            else triggered_sum + triggered_amp
        )
        diff_sum = amp_diff.clone() if diff_sum is None else diff_sum + amp_diff
        phase_diff_sum = (
            phase_diff.clone() if phase_diff_sum is None else phase_diff_sum + phase_diff
        )

        if len(preview_records) < args.num_previews:
            preview_path = save_trigger_preview(
                clean=clean_image,
                triggered=triggered_image,
                clean_spectrum=normalize_minmax(clean_amp),
                triggered_spectrum=normalize_minmax(triggered_amp),
                output_path=preview_dir / f"test_index_{original_index}.png",
            )
            panel_path = save_frequency_panel(
                [
                    ("clean amplitude", normalize_minmax(clean_amp)),
                    ("triggered amplitude", normalize_minmax(triggered_amp)),
                    ("amplitude diff", normalize_minmax(amp_diff)),
                    ("phase diff", normalize_minmax(phase_diff)),
                ],
                preview_dir / f"test_index_{original_index}_spectral_panel.png",
            )
            preview_records.append(
                {
                    "index": original_index,
                    "label": int(clean_label),
                    "label_name": dataset.fine_label_names[int(clean_label)],
                    "preview_path": str(preview_path),
                    "spectral_panel_path": str(panel_path),
                }
            )

        processed += 1
        if processed >= args.num_samples:
            break

    if processed == 0:
        raise ValueError("No non-target samples were processed")

    mean_clean_amp = clean_sum / processed
    mean_triggered_amp = triggered_sum / processed
    mean_amp_diff = diff_sum / processed
    mean_phase_diff = phase_diff_sum / processed

    average_panel_path = save_frequency_panel(
        [
            ("mean clean amp", normalize_minmax(mean_clean_amp)),
            ("mean triggered amp", normalize_minmax(mean_triggered_amp)),
            ("mean amp diff", normalize_minmax(mean_amp_diff)),
            ("mean phase diff", normalize_minmax(mean_phase_diff)),
        ],
        args.output_dir / "average_frequency_panel.png",
    )
    amp_heatmap_path = save_heatmap(
        normalize_minmax(mean_amp_diff),
        args.output_dir / "average_amplitude_difference_heatmap.png",
    )
    phase_heatmap_path = save_heatmap(
        normalize_minmax(mean_phase_diff),
        args.output_dir / "average_phase_difference_heatmap.png",
    )

    max_position = torch.nonzero(mean_amp_diff == mean_amp_diff.max(), as_tuple=False)[0]
    summary = {
        "dataset": "cifar100",
        "split": "test",
        "processed_non_target_samples": processed,
        "target_label": args.target_label,
        "target_label_name": dataset.fine_label_names[args.target_label],
        "trigger": {
            "type": "sinusoidal_frequency",
            "strength": args.strength,
            "horizontal_frequency": args.horizontal_frequency,
            "vertical_frequency": args.vertical_frequency,
        },
        "mean_amplitude_difference": float(mean_amp_diff.mean()),
        "max_amplitude_difference": float(mean_amp_diff.max()),
        "max_amplitude_difference_position_yx": [
            int(max_position[0]),
            int(max_position[1]),
        ],
        "mean_phase_difference": float(mean_phase_diff.mean()),
        "average_panel_path": str(average_panel_path),
        "amplitude_heatmap_path": str(amp_heatmap_path),
        "phase_heatmap_path": str(phase_heatmap_path),
        "preview_records": preview_records,
    }

    summary_path = args.output_dir / "frequency_analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Processed non-target samples:", processed)
    print("Mean amplitude difference:", summary["mean_amplitude_difference"])
    print("Max amplitude difference:", summary["max_amplitude_difference"])
    print("Max amplitude diff position (y, x):", summary["max_amplitude_difference_position_yx"])
    print("Mean phase difference:", summary["mean_phase_difference"])
    print("Average frequency panel:", average_panel_path)
    print("Amplitude heatmap:", amp_heatmap_path)
    print("Phase heatmap:", phase_heatmap_path)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
