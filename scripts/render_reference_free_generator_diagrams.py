"""Render explanatory diagrams for the reference-free generator document."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTPUT_DIR = Path("docs/assets/reference_free_generator")


def box(axis, x, y, width, height, text, color, *, fontsize=10):
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.4, edgecolor="#243447", facecolor=color,
    )
    axis.add_patch(patch)
    axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(axis, start, end, *, color="#243447", style="-"):
    axis.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=14,
        linewidth=1.5, linestyle=style, color=color,
    ))


def render_training_and_deployment():
    fig, axes = plt.subplots(2, 1, figsize=(15, 8.5))
    for axis in axes:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.axis("off")

    train = axes[0]
    train.set_title("Training: the clean image and label are answer keys, not generator inputs", fontsize=14, weight="bold")
    box(train, 0.02, 0.48, 0.13, 0.22, "Triggered\nimage", "#f7c9c9")
    box(train, 0.20, 0.48, 0.15, 0.22, "FFT evidence\nfrom this image", "#d8e8ff")
    box(train, 0.40, 0.48, 0.15, 0.22, "Generator", "#e4d5f5", fontsize=12)
    box(train, 0.60, 0.48, 0.15, 0.22, "Corrected\nimage", "#d8f0d2")
    box(train, 0.81, 0.62, 0.16, 0.18, "Frozen classifier\nchecks the class", "#ffe8b3")
    box(train, 0.81, 0.20, 0.16, 0.18, "Clean image + label\nscore preservation", "#fff3cd")
    for start, end in (
        ((0.15, 0.59), (0.20, 0.59)), ((0.35, 0.59), (0.40, 0.59)),
        ((0.55, 0.59), (0.60, 0.59)), ((0.75, 0.59), (0.81, 0.70)),
    ):
        arrow(train, start, end)
    arrow(train, (0.81, 0.29), (0.73, 0.48), color="#a35d00", style="--")
    arrow(train, (0.89, 0.62), (0.55, 0.40), color="#a35d00", style="--")
    train.text(
        0.48, 0.25,
        "Loss feedback changes generator parameters during training",
        ha="center", fontsize=11, color="#7a3e00", weight="bold",
    )
    train.text(
        0.275, 0.42,
        "Only this path enters the generator",
        ha="center", fontsize=9, color="#1f5d91",
    )

    deploy = axes[1]
    deploy.set_title("Deployment: one incoming image, one forward pass, no answer key", fontsize=14, weight="bold")
    box(deploy, 0.05, 0.38, 0.16, 0.24, "Unknown incoming\nimage", "#f7c9c9")
    box(deploy, 0.28, 0.38, 0.17, 0.24, "Single-image\nFFT evidence", "#d8e8ff")
    box(deploy, 0.52, 0.38, 0.16, 0.24, "Trained\ngenerator", "#e4d5f5", fontsize=12)
    box(deploy, 0.75, 0.38, 0.18, 0.24, "Corrected image\nfor classification", "#d8f0d2")
    for start, end in (
        ((0.21, 0.50), (0.28, 0.50)), ((0.45, 0.50), (0.52, 0.50)),
        ((0.68, 0.50), (0.75, 0.50)),
    ):
        arrow(deploy, start, end)
    deploy.text(
        0.5, 0.17,
        "No clean counterpart, true label, subtraction, or trial-and-error loop is available here.",
        ha="center", fontsize=11, color="#8b1a1a", weight="bold",
    )

    fig.tight_layout()
    path = OUTPUT_DIR / "training_vs_deployment.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_correction_anatomy():
    fig, axis = plt.subplots(figsize=(15, 5.8))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.set_title("How one-image spectral evidence becomes a correction", fontsize=15, weight="bold")

    box(axis, 0.02, 0.56, 0.13, 0.22, "Incoming\nimage", "#f7c9c9")
    box(axis, 0.20, 0.56, 0.15, 0.22, "FFT\namplitude + phase", "#d8e8ff")
    box(axis, 0.40, 0.56, 0.17, 0.22, "Evidence\nlog amplitude\nlocal residual\nphase + position", "#cfe6ff", fontsize=9)
    box(axis, 0.62, 0.66, 0.14, 0.18, "Gate\nwhere to act", "#e4d5f5")
    box(axis, 0.62, 0.38, 0.14, 0.18, "Signed change\nhow to act", "#eadff7")
    box(axis, 0.81, 0.52, 0.16, 0.25, "Correct amplitude\nkeep incoming phase\napply inverse FFT", "#d8f0d2", fontsize=10)
    for start, end in (
        ((0.15, 0.67), (0.20, 0.67)), ((0.35, 0.67), (0.40, 0.67)),
        ((0.57, 0.67), (0.62, 0.75)), ((0.57, 0.62), (0.62, 0.47)),
        ((0.76, 0.75), (0.81, 0.68)), ((0.76, 0.47), (0.81, 0.59)),
    ):
        arrow(axis, start, end)

    axis.text(
        0.69, 0.22,
        "Effective correction = permission x direction-and-size",
        ha="center", fontsize=11, weight="bold", color="#513071",
    )
    axis.text(
        0.48, 0.08,
        "The map is learned from loss feedback. It is not a guaranteed physical segmentation of the trigger.",
        ha="center", fontsize=11, color="#8b1a1a", weight="bold",
    )
    path = OUTPUT_DIR / "correction_map_anatomy.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    render_training_and_deployment()
    render_correction_anatomy()
    print("Saved diagrams to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
