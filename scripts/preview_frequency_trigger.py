"""Preview a controlled frequency trigger on CIFAR-100 samples."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets.cifar100_dataset import CleanCIFAR100Dataset
from fft.spectrum import log_amplitude_spectrum
from poisoning.frequency_trigger import FrequencyTriggerConfig, apply_frequency_trigger
from visualization.trigger_preview import save_trigger_preview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-path", type=Path, default=Path("datasets/archive.zip"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/trigger_preview"))
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--strength", type=float, default=0.08)
    parser.add_argument("--horizontal-frequency", type=int, default=6)
    parser.add_argument("--vertical-frequency", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = CleanCIFAR100Dataset.from_zip(args.zip_path, split="train")
    clean_image, clean_label = dataset[args.sample_index]

    config = FrequencyTriggerConfig(
        horizontal_frequency=args.horizontal_frequency,
        vertical_frequency=args.vertical_frequency,
        strength=args.strength,
    )
    triggered_image = apply_frequency_trigger(clean_image, config)

    clean_spectrum = log_amplitude_spectrum(clean_image)
    triggered_spectrum = log_amplitude_spectrum(triggered_image)

    output_path = save_trigger_preview(
        clean=clean_image,
        triggered=triggered_image,
        clean_spectrum=clean_spectrum,
        triggered_spectrum=triggered_spectrum,
        output_path=args.output_dir / f"sample_{args.sample_index}_frequency_trigger.png",
    )

    mean_abs_delta = float((triggered_image - clean_image).abs().mean())
    max_abs_delta = float((triggered_image - clean_image).abs().max())

    print("Sample index:", args.sample_index)
    print("Clean label:", int(clean_label), dataset.fine_label_names[int(clean_label)])
    print("Trigger strength:", config.strength)
    print("Frequency:", (config.horizontal_frequency, config.vertical_frequency))
    print("Mean absolute image delta:", mean_abs_delta)
    print("Max absolute image delta:", max_abs_delta)
    print("Saved trigger preview:", output_path)


if __name__ == "__main__":
    main()
