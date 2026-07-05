"""Image grid helpers for dataset inspection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def save_labeled_image_grid(
    images: np.ndarray,
    labels: list[str],
    output_path: str | Path,
    *,
    columns: int = 8,
    cell_size: int = 96,
    label_height: int = 22,
) -> Path:
    """Save a labeled RGB image grid.

    Args:
        images: RGB uint8 images with shape (N, H, W, 3).
        labels: One text label per image.
        output_path: Destination PNG path.
        columns: Number of grid columns.
        cell_size: Width and height used for each resized image.
        label_height: Extra vertical space below each image for its label.
    """

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

    for index, (image_array, label) in enumerate(zip(images, labels)):
        row, col = divmod(index, columns)
        x = col * cell_size
        y = row * (cell_size + label_height)

        image = Image.fromarray(image_array).resize(
            (cell_size, cell_size),
            resample=Image.Resampling.NEAREST,
        )
        canvas.paste(image, (x, y))
        draw.text((x + 3, y + cell_size + 3), label[:18], fill=(0, 0, 0))

    canvas.save(output_path)
    return output_path
