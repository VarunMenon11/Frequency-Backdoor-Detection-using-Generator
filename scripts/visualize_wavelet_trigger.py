"""Visualize a wavelet-subband trigger and its spectral effect.

This is a demonstration script. It does not train a classifier or create a
poisoned dataset. It takes one RGB image, modifies one Haar-wavelet detail
subband, reconstructs the triggered image, and saves a paper-friendly panel
showing spatial, Fourier, and wavelet-domain changes.

Example:
    python -m scripts.visualize_wavelet_trigger \
        --image-path path/to/texture.jpg \
        --output-dir outputs/wavelet_trigger_demo \
        --subband HH \
        --strength 0.25
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


SUBBANDS = ("LH", "HL", "HH")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/wavelet_trigger_demo"),
    )
    parser.add_argument(
        "--subband",
        choices=SUBBANDS,
        default="HH",
        help="Haar detail subband to modify: LH, HL, or HH.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.25,
        help="Relative trigger strength measured against the selected subband scale.",
    )
    parser.add_argument(
        "--trigger-frequency",
        type=float,
        default=3.0,
        help="Number of trigger oscillations across the selected wavelet subband.",
    )
    parser.add_argument(
        "--trigger-angle",
        type=float,
        default=0.0,
        help="Direction of the wavelet trigger in radians.",
    )
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    if array.shape[0] < 4 or array.shape[1] < 4:
        raise ValueError("The image must be at least 4x4 pixels.")
    # Haar decomposition needs even dimensions. Cropping only removes a final
    # row or column when a source image has an odd dimension.
    height = array.shape[0] - array.shape[0] % 2
    width = array.shape[1] - array.shape[1] % 2
    return array[:height, :width]


def haar_forward(image: np.ndarray) -> dict[str, np.ndarray]:
    """Return one-level orthonormal 2-D Haar coefficients per RGB channel."""

    top_left = image[0::2, 0::2]
    top_right = image[0::2, 1::2]
    bottom_left = image[1::2, 0::2]
    bottom_right = image[1::2, 1::2]
    return {
        "LL": (top_left + top_right + bottom_left + bottom_right) / 2.0,
        "LH": (top_left - top_right + bottom_left - bottom_right) / 2.0,
        "HL": (top_left + top_right - bottom_left - bottom_right) / 2.0,
        "HH": (top_left - top_right - bottom_left + bottom_right) / 2.0,
    }


def haar_inverse(coefficients: dict[str, np.ndarray]) -> np.ndarray:
    """Reconstruct an RGB image from one-level Haar coefficients."""

    ll, lh, hl, hh = (
        coefficients["LL"],
        coefficients["LH"],
        coefficients["HL"],
        coefficients["HH"],
    )
    height, width = ll.shape[:2]
    result = np.zeros((height * 2, width * 2, 3), dtype=np.float32)
    result[0::2, 0::2] = (ll + lh + hl + hh) / 2.0
    result[0::2, 1::2] = (ll - lh + hl - hh) / 2.0
    result[1::2, 0::2] = (ll + lh - hl - hh) / 2.0
    result[1::2, 1::2] = (ll - lh - hl + hh) / 2.0
    return result


def make_wavelet_trigger(
    shape: tuple[int, int, int],
    frequency: float,
    angle: float,
) -> np.ndarray:
    """Create a directional unit-amplitude pattern for a wavelet subband."""

    height, width, _ = shape
    y = np.linspace(-0.5, 0.5, height, dtype=np.float32).reshape(-1, 1)
    x = np.linspace(-0.5, 0.5, width, dtype=np.float32).reshape(1, -1)
    rotated = np.cos(angle) * x + np.sin(angle) * y
    pattern = np.sin(2.0 * np.pi * frequency * rotated)
    return np.repeat(pattern[..., None], 3, axis=2)


def inject_wavelet_trigger(
    image: np.ndarray,
    *,
    subband: str,
    strength: float,
    frequency: float,
    angle: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    clean_coefficients = haar_forward(image)
    triggered_coefficients = {
        name: values.copy() for name, values in clean_coefficients.items()
    }
    selected = clean_coefficients[subband]
    scale = float(np.std(selected))
    # A floor keeps nearly uniform images from producing a zero trigger.
    scale = max(scale, 1.0 / 255.0)
    pattern = make_wavelet_trigger(selected.shape, frequency, angle)
    triggered_coefficients[subband] += strength * scale * pattern
    triggered_image = np.clip(haar_inverse(triggered_coefficients), 0.0, 1.0)
    return triggered_image, clean_coefficients, triggered_coefficients


def fft_maps(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    channels = []
    phases = []
    for channel in range(3):
        spectrum = np.fft.fftshift(np.fft.fft2(image[..., channel]))
        channels.append(np.log1p(np.abs(spectrum)))
        phases.append(np.angle(spectrum))
    return np.stack(channels, axis=2), np.stack(phases, axis=2)


def channel_mean(values: np.ndarray) -> np.ndarray:
    return values.mean(axis=2)


def normalize_for_display(values: np.ndarray) -> np.ndarray:
    low = float(np.percentile(values, 1.0))
    high = float(np.percentile(values, 99.0))
    if high <= low:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def save_panel(
    clean: np.ndarray,
    triggered: np.ndarray,
    clean_coefficients: dict[str, np.ndarray],
    triggered_coefficients: dict[str, np.ndarray],
    output_path: Path,
    subband: str,
) -> None:
    clean_amplitude, clean_phase = fft_maps(clean)
    triggered_amplitude, triggered_phase = fft_maps(triggered)
    amplitude_difference = np.abs(triggered_amplitude - clean_amplitude)
    phase_difference = np.abs(
        np.angle(np.exp(1j * (triggered_phase - clean_phase)))
    )
    image_difference = np.abs(triggered - clean)
    wavelet_difference = np.abs(
        triggered_coefficients[subband] - clean_coefficients[subband]
    )

    panels = [
        ("Clean image", clean, "gray"),
        ("Wavelet-triggered image", triggered, "gray"),
        ("Image difference x8", np.clip(image_difference * 8.0, 0.0, 1.0), "magma"),
        ("Clean amplitude", normalize_for_display(channel_mean(clean_amplitude)), "magma"),
        ("Triggered amplitude", normalize_for_display(channel_mean(triggered_amplitude)), "magma"),
        ("Amplitude difference", normalize_for_display(channel_mean(amplitude_difference)), "inferno"),
        ("Clean phase", channel_mean(clean_phase), "twilight"),
        ("Triggered phase", channel_mean(triggered_phase), "twilight"),
        ("Phase difference", normalize_for_display(channel_mean(phase_difference)), "inferno"),
        (f"Clean wavelet {subband}", normalize_for_display(channel_mean(clean_coefficients[subband])), "coolwarm"),
        (f"Triggered wavelet {subband}", normalize_for_display(channel_mean(triggered_coefficients[subband])), "coolwarm"),
        (f"Wavelet {subband} difference", normalize_for_display(channel_mean(wavelet_difference)), "inferno"),
    ]

    figure, axes = plt.subplots(3, 4, figsize=(16, 12), constrained_layout=True)
    for axis, (title, values, colormap) in zip(axes.flat, panels):
        if values.ndim == 3:
            axis.imshow(np.clip(values, 0.0, 1.0))
        else:
            axis.imshow(values, cmap=colormap)
        axis.set_title(title, fontsize=11)
        axis.axis("off")
    figure.suptitle(
        f"Haar wavelet trigger comparison | modified subband: {subband}",
        fontsize=15,
    )
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.strength < 0:
        raise ValueError("--strength must be non-negative.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    clean = load_rgb(args.image_path)
    triggered, clean_coefficients, triggered_coefficients = inject_wavelet_trigger(
        clean,
        subband=args.subband,
        strength=args.strength,
        frequency=args.trigger_frequency,
        angle=args.trigger_angle,
    )

    clean_path = args.output_dir / "clean_image.png"
    triggered_path = args.output_dir / "wavelet_triggered_image.png"
    panel_path = args.output_dir / "wavelet_trigger_comparison.png"
    Image.fromarray(np.round(clean * 255.0).astype(np.uint8)).save(clean_path)
    Image.fromarray(np.round(triggered * 255.0).astype(np.uint8)).save(triggered_path)
    save_panel(
        clean,
        triggered,
        clean_coefficients,
        triggered_coefficients,
        panel_path,
        args.subband,
    )

    selected_delta = triggered_coefficients[args.subband] - clean_coefficients[args.subband]
    image_delta = triggered - clean
    summary = {
        "input_image": str(args.image_path),
        "image_shape_hwc": list(clean.shape),
        "trigger": {
            "type": "haar_wavelet_subband",
            "subband": args.subband,
            "strength": args.strength,
            "frequency": args.trigger_frequency,
            "angle_radians": args.trigger_angle,
        },
        "mean_absolute_image_difference": float(np.abs(image_delta).mean()),
        "max_absolute_image_difference": float(np.abs(image_delta).max()),
        "selected_wavelet_mean_absolute_difference": float(np.abs(selected_delta).mean()),
        "selected_wavelet_max_absolute_difference": float(np.abs(selected_delta).max()),
        "clean_image_path": str(clean_path),
        "triggered_image_path": str(triggered_path),
        "comparison_panel_path": str(panel_path),
    }
    summary_path = args.output_dir / "wavelet_trigger_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Input image:", args.image_path)
    print("Image shape (H, W, C):", tuple(clean.shape))
    print("Modified wavelet subband:", args.subband)
    print("Trigger strength:", args.strength)
    print("Mean absolute image difference:", summary["mean_absolute_image_difference"])
    print("Saved comparison panel:", panel_path)
    print("Saved summary:", summary_path)


if __name__ == "__main__":
    main()
