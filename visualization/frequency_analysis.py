"""Frequency-analysis figure helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def save_frequency_panel(
    panels: list[tuple[str, torch.Tensor]],
    output_path: str | Path,
    *,
    columns: int = 4,
    cell_size: int = 128,
    label_height: int = 24,
) -> Path:
    """Save a grid of grayscale frequency panels."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = int(np.ceil(len(panels) / columns))
    canvas = Image.new(
        "RGB",
        (columns * cell_size, rows * (cell_size + label_height)),
        color=(255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)

    for index, (label, values) in enumerate(panels):
        row, col = divmod(index, columns)
        x = col * cell_size
        y = row * (cell_size + label_height)

        image = _tensor_to_grayscale(values)
        image = image.resize((cell_size, cell_size), Image.Resampling.NEAREST)
        canvas.paste(image, (x, y))
        draw.text((x + 4, y + cell_size + 4), label[:22], fill=(0, 0, 0))

    canvas.save(output_path)
    return output_path


def save_heatmap(values: torch.Tensor, output_path: str | Path, *, scale: int = 8) -> Path:
    """Save one normalized heatmap-like grayscale image."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = _tensor_to_grayscale(values)
    width, height = image.size
    image = image.resize((width * scale, height * scale), Image.Resampling.NEAREST)
    image.save(output_path)
    return output_path


def _tensor_to_grayscale(values: torch.Tensor) -> Image.Image:
    array = values.detach().cpu().numpy()
    array = np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(array, mode="L").convert("RGB")
