"""Create a presentation-ready DTD backdoor-mitigation inference panel.

Two modes are intentionally separated:

``controlled``
    Treat the uploaded image as a clean source, apply one configured trigger,
    and display clean, triggered, and corrected predictions. This is the best
    mode for a reproducible research demonstration.

``suspicious``
    Treat the uploaded image as the only image available at deployment. No
    trigger is injected and no clean counterpart is used. The panel reports
    input and corrected predictions without claiming that either is ground
    truth.

No model is trained or fine-tuned by this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from evaluation.asb_classifier import evaluation_transform
from poisoning import apply_advanced_trigger
from scripts.audit_dtd_reference_free_generator import build_models
from scripts.train_dtd_reference_free_generator import trigger_from_dict


DEFAULT_CLASSIFIER = Path(
    "Advanced_Experiments/dtd_position_attack_selected_v1/"
    "dtd_position_attack_selected_v1/suspicious_classifier_best_attack.pt"
)
DEFAULT_GENERATOR = Path(
    "Advanced_Experiments/dtd_position_generalizing_generator_v1/"
    "reference_free_generator_best.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("controlled", "suspicious"), default="controlled"
    )
    parser.add_argument(
        "--trigger-name",
        default="ftpos_near_mixed",
        help="Configured trigger used only in controlled mode.",
    )
    parser.add_argument("--true-class", default=None)
    parser.add_argument("--classifier-checkpoint", type=Path, default=DEFAULT_CLASSIFIER)
    parser.add_argument("--generator-checkpoint", type=Path, default=DEFAULT_GENERATOR)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("Advanced_Outputs/dtd_uploaded_image_demo"),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_paths(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    generator_checkpoint = torch.load(
        args.generator_checkpoint, map_location="cpu", weights_only=True
    )
    classifier_checkpoint = torch.load(
        args.classifier_checkpoint, map_location="cpu", weights_only=True
    )
    generator, classifier, config = build_models(
        generator_checkpoint, classifier_checkpoint, device
    )
    class_names = list(config["class_names"])
    target_label = int(config["target_label"])
    target_class = class_names[target_label]
    image_size = int(config["image_size"])

    with Image.open(args.image) as source:
        source_rgb = source.convert("RGB")
    transform = evaluation_transform(image_size)
    uploaded = transform(source_rgb).unsqueeze(0).to(device)

    inferred_class = infer_true_class(args.image, class_names)
    true_class = args.true_class or inferred_class
    if true_class is not None and true_class not in class_names:
        raise ValueError(
            f"Unknown --true-class {true_class!r}; expected one of: "
            + ", ".join(class_names)
        )

    trigger_status = None
    clean = None
    if args.mode == "controlled":
        trigger_configs = dict(config["trigger_configs"])
        if args.trigger_name not in trigger_configs:
            raise ValueError(
                f"Unknown trigger {args.trigger_name!r}; available: "
                + ", ".join(sorted(trigger_configs))
            )
        clean = uploaded
        trigger_config = trigger_from_dict(trigger_configs[args.trigger_name])
        generator_input = apply_advanced_trigger(clean, trigger_config)
        held_out = set(config.get("held_out_trigger_names", []))
        trigger_status = "held out" if args.trigger_name in held_out else "seen"
    else:
        generator_input = uploaded

    tracked_classes = [target_class]
    if true_class and true_class != target_class:
        tracked_classes.append(true_class)

    with torch.inference_mode():
        generator_output = generator(generator_input)
        corrected = generator_output.corrected_image
        predictions = {
            "input": prediction_record(
                classifier, generator_input, class_names, args.top_k,
                tracked_classes,
            ),
            "corrected": prediction_record(
                classifier, corrected, class_names, args.top_k,
                tracked_classes,
            ),
        }
        if clean is not None:
            predictions["clean"] = prediction_record(
                classifier, clean, class_names, args.top_k,
                tracked_classes,
            )

    save_tensor_image(generator_input[0], args.output_dir / "input_image.png")
    save_tensor_image(corrected[0], args.output_dir / "corrected_image.png")
    if clean is not None:
        save_tensor_image(clean[0], args.output_dir / "clean_image.png")
        save_tensor_image(generator_input[0], args.output_dir / "triggered_image.png")

    report = build_report(
        args=args,
        predictions=predictions,
        class_names=class_names,
        target_class=target_class,
        true_class=true_class,
        trigger_status=trigger_status,
    )
    save_panel(
        args=args,
        clean=clean,
        generator_input=generator_input,
        corrected=corrected,
        generator_output=generator_output,
        predictions=predictions,
        target_class=target_class,
        true_class=true_class,
        trigger_status=trigger_status,
    )
    artifacts = {"overview_panel": str(args.output_dir / "demo_panel.png")}
    if args.mode == "controlled":
        artifacts.update(
            save_controlled_demo_stages(
                output_dir=args.output_dir,
                clean=clean,
                triggered=generator_input,
                corrected=corrected,
                generator_output=generator_output,
                predictions=predictions,
                target_class=target_class,
                true_class=true_class,
                trigger_name=args.trigger_name,
                trigger_status=trigger_status,
            )
        )
    report["artifacts"] = artifacts
    report_path = args.output_dir / "demo_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print_summary(report)
    print(f"Panel: {args.output_dir / 'demo_panel.png'}")
    print(f"Corrected image: {args.output_dir / 'corrected_image.png'}")
    print(f"JSON report: {report_path}")


def build_report(
    *, args, predictions, class_names, target_class, true_class, trigger_status
) -> dict[str, object]:
    input_prediction = predictions["input"]["class"]
    corrected_prediction = predictions["corrected"]["class"]
    report: dict[str, object] = {
        "protocol": "frozen classifier and generator; inference only; no training",
        "mode": args.mode,
        "uploaded_image": str(args.image),
        "classifier_checkpoint": str(args.classifier_checkpoint),
        "generator_checkpoint": str(args.generator_checkpoint),
        "target_class": target_class,
        "true_class_if_known": true_class,
        "generator_received_clean_counterpart": False,
        "predictions": predictions,
        "notes": [],
    }
    if args.mode == "controlled":
        clean_prediction = predictions["clean"]["class"]
        reference_class = true_class or clean_prediction
        report.update({
            "trigger_name": args.trigger_name,
            "trigger_generator_status": trigger_status,
            "attack_activated": input_prediction == target_class,
            "recovered_to_reference_class": corrected_prediction == reference_class,
            "reference_class_for_demo": reference_class,
        })
        report["notes"] = [
            "The uploaded image was treated as clean and the configured trigger was injected for a controlled demonstration.",
            "The clean image is displayed and used for evaluation, but the generator received only the triggered image.",
            "Recovery is judged against the supplied true class when available; otherwise the clean classifier prediction is used only as a demonstration reference.",
        ]
    else:
        report["prediction_changed"] = input_prediction != corrected_prediction
        report["notes"] = [
            "The uploaded image was treated as the only suspicious input; no trigger was injected.",
            "Without a trusted label or clean counterpart, a changed prediction is not proof that the new prediction is correct.",
            "This mode demonstrates the deployed data path rather than attack success or recovery accuracy.",
        ]
    return report


def save_panel(
    *, args, clean, generator_input, corrected, generator_output,
    predictions, target_class, true_class, trigger_status,
) -> None:
    input_image = tensor_image(generator_input[0])
    corrected_image = tensor_image(corrected[0])
    input_spectrum = shifted_log_amplitude(generator_input[0])
    corrected_spectrum = shifted_log_amplitude(corrected[0])
    effective = shifted_channel_mean(generator_output.effective_log_correction[0].abs())
    gate = shifted_channel_mean(generator_output.correction_mask[0])

    if args.mode == "controlled":
        clean_image = tensor_image(clean[0])
        clean_spectrum = shifted_log_amplitude(clean[0])
        amplified_trigger = np.clip(np.abs(input_image - clean_image) * 8.0, 0.0, 1.0)
        images = (
            clean_image,
            input_image,
            corrected_image,
            amplified_trigger,
            clean_spectrum,
            input_spectrum,
            corrected_spectrum,
            effective,
        )
        titles = (
            prediction_title("Clean", predictions["clean"]),
            prediction_title("Triggered", predictions["input"]),
            prediction_title("Generator corrected", predictions["corrected"]),
            "Injected trigger difference |x8|",
            "Clean log amplitude",
            "Triggered log amplitude",
            "Corrected log amplitude",
            "Applied log-amplitude correction",
        )
        headline = (
            f"Controlled DTD demo | {args.trigger_name} ({trigger_status}) | "
            f"attack target: {target_class}"
        )
        if true_class:
            headline += f" | supplied true class: {true_class}"
    else:
        amplified_change = np.clip(np.abs(corrected_image - input_image) * 8.0, 0.0, 1.0)
        images = (
            input_image,
            corrected_image,
            amplified_change,
            gate,
            input_spectrum,
            corrected_spectrum,
            effective,
            None,
        )
        titles = (
            prediction_title("Uploaded suspicious input", predictions["input"]),
            prediction_title("Generator corrected", predictions["corrected"]),
            "Input-to-corrected difference |x8|",
            "Predicted soft correction gate",
            "Input log amplitude",
            "Corrected log amplitude",
            "Applied log-amplitude correction",
            top_k_text(predictions),
        )
        headline = (
            "Deployment-path demo | one uploaded image only | "
            "no trigger injection and no clean reference"
        )

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for index, (axis, image, title) in enumerate(zip(axes.flat, images, titles)):
        if image is None:
            axis.text(0.02, 0.98, title, va="top", ha="left", fontsize=10)
            axis.set_facecolor("#f4f6f8")
        elif image.ndim == 3:
            axis.imshow(image)
        else:
            cmap = "viridis" if "gate" in title.lower() else "magma"
            axis.imshow(image, cmap=cmap)
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    fig.suptitle(headline, fontsize=14, fontweight="bold")
    fig.text(
        0.5, 0.015,
        "Frozen models; one generator forward pass; no fine-tuning during this demo.",
        ha="center", fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(args.output_dir / "demo_panel.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_controlled_demo_stages(
    *, output_dir, clean, triggered, corrected, generator_output,
    predictions, target_class, true_class, trigger_name, trigger_status,
):
    """Save a presentation sequence that makes every inference stage explicit."""
    clean_image = tensor_image(clean[0])
    triggered_image = tensor_image(triggered[0])
    corrected_image = tensor_image(corrected[0])
    clean_spectrum = shifted_log_amplitude(clean[0])
    input_spectrum = shifted_log_amplitude(triggered[0])
    effective = shifted_channel_mean(
        generator_output.effective_log_correction[0].abs()
    )
    reference_class = true_class or predictions["clean"]["class"]

    paths = {
        "stage_1_original": output_dir / "01_original_image.png",
        "stage_2_trigger": output_dir / "02_trigger_injection.png",
        "stage_3_without_generator": output_dir / "03_without_generator.png",
        "stage_4_with_generator": output_dir / "04_with_generator.png",
        "stage_5_final_result": output_dir / "05_final_result.png",
        "probability_graph": output_dir / "prediction_probability_graph.png",
    }

    clean_record = predictions["clean"]
    input_record = predictions["input"]
    corrected_record = predictions["corrected"]

    fig, axes = plt.subplots(
        1, 2, figsize=(13, 6.5), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    axes[0].imshow(clean_image)
    axes[0].set_title(
        f"Original image\nPrediction: {clean_record['class']} | "
        f"Confidence: {100 * clean_record['confidence']:.2f}%",
        fontsize=13, fontweight="bold",
    )
    axes[0].axis("off")
    draw_top_predictions(
        axes[1], clean_record,
        title="Model prediction confidence",
    )
    fig.suptitle("Stage 1 - Original image", fontsize=18, fontweight="bold")
    save_figure(fig, paths["stage_1_original"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    stage_two = (
        (clean_image, "Image before trigger", None),
        (triggered_image, "Image after trigger", None),
        (clean_spectrum, "Frequency amplitude before trigger", "magma"),
        (input_spectrum, "Frequency amplitude after trigger", "magma"),
    )
    for axis, (image, title, cmap) in zip(axes.flat, stage_two):
        axis.imshow(image, cmap=cmap)
        axis.set_title(title, fontsize=13, fontweight="bold")
        axis.axis("off")
    fig.suptitle(
        f"Stage 2 - Trigger injection: {trigger_name} ({trigger_status})",
        fontsize=18, fontweight="bold",
    )
    save_figure(fig, paths["stage_2_trigger"], bottom=0.02)

    fig, axis = plt.subplots(figsize=(8, 7))
    axis.imshow(triggered_image)
    axis.set_title(
        f"Prediction: {input_record['class']}\n"
        f"Confidence: {100 * input_record['confidence']:.2f}%",
        fontsize=16, fontweight="bold", color="#a61b1b",
    )
    axis.axis("off")
    attack_text = "ATTACK ACTIVATED" if input_record["class"] == target_class else "ATTACK NOT ACTIVATED"
    fig.suptitle(
        f"Stage 3 - Without generator: {attack_text}",
        fontsize=17, fontweight="bold", color="#a61b1b",
    )
    save_figure(fig, paths["stage_3_without_generator"], bottom=0.02, top=0.86)

    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    stage_four = (
        (triggered_image, "1. Triggered input", None),
        (input_spectrum, "2. Triggered log amplitude", "magma"),
        (effective, "3. Applied frequency correction", "magma"),
        (corrected_image, "4. Reconstructed image", None),
    )
    for axis, (image, title, cmap) in zip(axes, stage_four):
        axis.imshow(image, cmap=cmap)
        axis.set_title(title, fontsize=12, fontweight="bold")
        axis.axis("off")
    fig.suptitle(
        "Stage 4 - With generator: estimate and apply a frequency-domain correction",
        fontsize=18, fontweight="bold",
    )
    fig.text(
        0.5, 0.02,
        "The frozen generator receives only the triggered image, predicts where correction is useful, and reconstructs one corrected image.",
        ha="center", fontsize=11,
    )
    save_figure(fig, paths["stage_4_with_generator"], bottom=0.07)

    save_probability_graph(
        path=paths["probability_graph"],
        predictions=predictions,
        target_class=target_class,
    )

    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    final_images = (
        (clean_image, f"Original\n{clean_record['class']} ({100 * clean_record['confidence']:.1f}%)"),
        (triggered_image, f"Without generator\n{input_record['class']} ({100 * input_record['confidence']:.1f}%)"),
        (corrected_image, f"With generator\n{corrected_record['class']} ({100 * corrected_record['confidence']:.1f}%)"),
    )
    for axis, (image, title) in zip(axes[:3], final_images):
        axis.imshow(image)
        axis.set_title(title, fontsize=13, fontweight="bold")
        axis.axis("off")
    draw_attack_target_graph(
        axes[3], predictions, target_class,
        title=f"Attack-target probability: {target_class}",
    )
    recovered = corrected_record["class"] == reference_class
    result = "DEFENSE RECOVERED THE REFERENCE CLASS" if recovered else "REFERENCE CLASS NOT RECOVERED"
    fig.suptitle(
        f"Stage 5 - Final result: {result}",
        fontsize=18, fontweight="bold",
        color="#116329" if recovered else "#a61b1b",
    )
    fig.text(
        0.5, 0.02,
        "The attack-target probability rises after trigger injection and falls after generator correction.",
        ha="center", fontsize=11,
    )
    save_figure(fig, paths["stage_5_final_result"], bottom=0.07, top=0.86)

    return {name: str(path) for name, path in paths.items()}


def save_probability_graph(*, path, predictions, target_class):
    fig, axis = plt.subplots(figsize=(11, 6.5))
    draw_attack_target_graph(
        axis, predictions, target_class,
        title="Attack-target probability through the demonstration",
    )
    fig.text(
        0.5, 0.02,
        "Lower attack-target probability after correction indicates suppression of the backdoor response.",
        ha="center", fontsize=11,
    )
    save_figure(fig, path, bottom=0.08)


def draw_top_predictions(axis, record, title):
    rows = list(reversed(record["top_k"][:5]))
    names = [row["class"] for row in rows]
    values = [100 * row["probability"] for row in rows]
    bars = axis.barh(names, values, color="#3973ac")
    axis.set_xlim(0, 105)
    axis.set_xlabel("Prediction confidence (%)")
    axis.set_title(title, fontsize=13, fontweight="bold")
    axis.grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}%", va="center", fontweight="bold",
        )


def draw_attack_target_graph(axis, predictions, target_class, title):
    stage_keys = ("clean", "input", "corrected")
    stage_names = ("Original", "Triggered\n(no generator)", "Corrected\n(with generator)")
    target_values = [
        100 * record_probability(predictions[key], target_class)
        for key in stage_keys
    ]
    positions = np.arange(len(stage_names))
    target_bars = axis.bar(
        positions, target_values, 0.56, color=("#6b7280", "#d64545", "#2f855a"),
    )
    axis.set_xticks(positions, stage_names)
    axis.set_ylim(0, 108)
    axis.set_ylabel("Classifier probability (%)")
    axis.set_title(title, fontsize=13, fontweight="bold")
    axis.grid(axis="y", alpha=0.25)
    axis.bar_label(target_bars, fmt="%.1f", padding=2, fontsize=9)


def record_probability(record, class_name):
    if class_name in record.get("tracked_probabilities", {}):
        return float(record["tracked_probabilities"][class_name])
    for row in record["top_k"]:
        if row["class"] == class_name:
            return float(row["probability"])
    return 0.0


def save_figure(fig, path, *, bottom=0.04, top=0.9):
    fig.tight_layout(rect=(0, bottom, 1, top))
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


@torch.inference_mode()
def prediction_record(
    classifier, images, class_names, top_k, tracked_classes=()
):
    probabilities = classifier(images).softmax(dim=1)[0]
    count = min(max(1, top_k), len(class_names))
    values, indices = probabilities.topk(count)
    ranked = [
        {"class": class_names[int(index)], "probability": float(value)}
        for value, index in zip(values.cpu(), indices.cpu())
    ]
    return {
        "class": ranked[0]["class"],
        "confidence": ranked[0]["probability"],
        "top_k": ranked,
        "tracked_probabilities": {
            name: float(probabilities[class_names.index(name)])
            for name in tracked_classes if name in class_names
        },
    }


def prediction_title(prefix, record):
    return f"{prefix}\npred: {record['class']} ({100 * record['confidence']:.1f}%)"


def top_k_text(predictions):
    lines = ["Top predictions"]
    for name in ("input", "corrected"):
        lines.append(f"\n{name.capitalize()}:")
        for row in predictions[name]["top_k"][:3]:
            lines.append(f"  {row['class']}: {100 * row['probability']:.1f}%")
    return "\n".join(lines)


def shifted_log_amplitude(image):
    spectrum = torch.fft.fft2(image.float(), dim=(-2, -1))
    shifted = torch.fft.fftshift(torch.log1p(spectrum.abs()), dim=(-2, -1))
    return shifted.mean(0).detach().cpu().numpy()


def shifted_channel_mean(values):
    shifted = torch.fft.fftshift(values, dim=(-2, -1))
    return shifted.mean(0).detach().cpu().numpy()


def tensor_image(image):
    return image.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()


def save_tensor_image(image, path):
    values = (tensor_image(image) * 255.0).round().astype(np.uint8)
    Image.fromarray(values, mode="RGB").save(path)


def infer_true_class(path, class_names):
    parent = path.parent.name.lower()
    return parent if parent in class_names else None


def resolve_device(requested):
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_paths(args):
    for label, path in (
        ("image", args.image),
        ("classifier checkpoint", args.classifier_checkpoint),
        ("generator checkpoint", args.generator_checkpoint),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")


def print_summary(report):
    print("Protocol:", report["protocol"])
    print("Mode:", report["mode"])
    if report["mode"] == "controlled":
        print(
            "Trigger:", report["trigger_name"],
            f"({report['trigger_generator_status']})",
        )
        print("Attack activated:", report["attack_activated"])
        print("Recovered to reference class:", report["recovered_to_reference_class"])
    for name, record in report["predictions"].items():
        print(
            f"{name.capitalize()} prediction: {record['class']} "
            f"({100 * record['confidence']:.2f}%)"
        )


if __name__ == "__main__":
    main()
