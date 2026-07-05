"""Prediction visualization helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def save_prediction_grid(
    images: list[torch.Tensor],
    labels: list[str],
    output_path: str | Path,
    *,
    columns: int = 4,
    cell_size: int = 128,
    label_height: int = 44,
) -> Path:
    """Save images with true/predicted labels written below each panel."""

    if len(images) != len(labels):
        raise ValueError("images and labels must have the same length")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = int(np.ceil(len(images) / columns))
    canvas = Image.new(
        "RGB",
        (columns * cell_size, rows * (cell_size + label_height)),
        color=(255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)

    for index, (image_tensor, label) in enumerate(zip(images, labels)):
        row, col = divmod(index, columns)
        x = col * cell_size
        y = row * (cell_size + label_height)

        image = _tensor_to_pil(image_tensor)
        image = image.resize((cell_size, cell_size), Image.Resampling.NEAREST)
        canvas.paste(image, (x, y))

        lines = _wrap_label(label, max_chars=22)
        for line_index, line in enumerate(lines[:2]):
            draw.text(
                (x + 4, y + cell_size + 4 + line_index * 16),
                line,
                fill=(0, 0, 0),
            )

    canvas.save(output_path)
    return output_path


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = image.detach().cpu().permute(1, 2, 0).numpy()
    array = np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _wrap_label(label: str, max_chars: int) -> list[str]:
    words = label.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines
