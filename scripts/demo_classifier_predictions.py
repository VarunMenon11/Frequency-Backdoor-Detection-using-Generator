"""Create visual prediction demos from a saved suspicious classifier.

The script saves two grids:

- clean inputs with predicted classes
- triggered versions of the same inputs with predicted classes

It also prints a terminal table that can be shown during a guide meeting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from datasets.cifar100_dataset import CleanCIFAR100Dataset
from models import SmallCIFARClassifier
from poisoning.frequency_trigger import FrequencyTriggerConfig, apply_frequency_trigger
from visualization.prediction_grid import save_prediction_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-path", type=Path, default=Path("datasets/archive.zip"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("experiments/suspicious_classifier_smoke_test/suspicious_classifier.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/classifier_prediction_demo"),
    )
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--strength", type=float, default=0.08)
    parser.add_argument("--horizontal-frequency", type=int, default=6)
    parser.add_argument("--vertical-frequency", type=int, default=6)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    dataset = CleanCIFAR100Dataset.from_zip(args.zip_path, split="test")
    trigger_config = FrequencyTriggerConfig(
        horizontal_frequency=args.horizontal_frequency,
        vertical_frequency=args.vertical_frequency,
        strength=args.strength,
    )

    model = SmallCIFARClassifier(num_classes=100).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    clean_images: list[torch.Tensor] = []
    triggered_images: list[torch.Tensor] = []
    clean_grid_labels: list[str] = []
    triggered_grid_labels: list[str] = []
    rows: list[dict[str, str | int | float]] = []

    candidate_index = args.start_index
    while len(clean_images) < args.num_samples and candidate_index < len(dataset):
        clean_image, true_label = dataset[candidate_index]
        if int(true_label) == args.target_label:
            candidate_index += 1
            continue

        triggered_image = apply_frequency_trigger(clean_image, trigger_config)
        clean_pred, clean_conf = predict(model, clean_image, device)
        triggered_pred, triggered_conf = predict(model, triggered_image, device)

        true_name = dataset.fine_label_names[int(true_label)]
        clean_pred_name = dataset.fine_label_names[clean_pred]
        triggered_pred_name = dataset.fine_label_names[triggered_pred]

        clean_images.append(clean_image)
        triggered_images.append(triggered_image)
        clean_grid_labels.append(f"T:{true_name} P:{clean_pred_name}")
        triggered_grid_labels.append(f"T:{true_name} P:{triggered_pred_name}")
        rows.append(
            {
                "index": candidate_index,
                "true": true_name,
                "clean_pred": clean_pred_name,
                "clean_conf": round(clean_conf, 4),
                "triggered_pred": triggered_pred_name,
                "triggered_conf": round(triggered_conf, 4),
            }
        )
        candidate_index += 1

    clean_grid_path = save_prediction_grid(
        clean_images,
        clean_grid_labels,
        args.output_dir / "clean_predictions.png",
    )
    triggered_grid_path = save_prediction_grid(
        triggered_images,
        triggered_grid_labels,
        args.output_dir / "triggered_predictions.png",
    )

    print("Loaded checkpoint:", args.checkpoint)
    print("Device:", device)
    print("Target label:", args.target_label, dataset.fine_label_names[args.target_label])
    print("Clean prediction grid:", clean_grid_path)
    print("Triggered prediction grid:", triggered_grid_path)
    print()
    print("index | true | clean prediction | triggered prediction")
    print("-" * 64)
    for row in rows:
        print(
            f"{row['index']:>5} | "
            f"{row['true']:<14} | "
            f"{row['clean_pred']:<18} ({row['clean_conf']:.4f}) | "
            f"{row['triggered_pred']:<18} ({row['triggered_conf']:.4f})"
        )

    if "summary" in checkpoint:
        summary = checkpoint["summary"]
        if summary.get("max_train_batches") is not None:
            print()
            print("Note: this checkpoint is from a smoke test, not a real trained model.")


def predict(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
) -> tuple[int, float]:
    with torch.no_grad():
        logits = model(image.unsqueeze(0).to(device))
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_label = probabilities.max(dim=1)
    return int(predicted_label.item()), float(confidence.item())


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


if __name__ == "__main__":
    main()
