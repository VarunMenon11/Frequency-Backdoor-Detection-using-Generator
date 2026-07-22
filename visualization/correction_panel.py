"""Visualization helpers for spectral correction outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def save_correction_panel(
    panels: list[tuple[str, torch.Tensor, str]],
    output_path: str | Path,
    *,
    columns: int = 4,
    cell_size: int = 128,
    label_height: int = 40,
) -> Path:
    """Save a grid containing images, spectra, and correction maps."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = int(np.ceil(len(panels) / columns))
    canvas = Image.new(
        "RGB",
        (columns * cell_size, rows * (cell_size + label_height)),
        color=(255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)

    for index, (label, tensor, mode) in enumerate(panels):
        row, col = divmod(index, columns)
        x = col * cell_size
        y = row * (cell_size + label_height)

        if mode == "rgb":
            image = _rgb_tensor_to_pil(tensor)
        elif mode == "gray":
            image = _gray_tensor_to_pil(tensor)
        else:
            raise ValueError(f"Unsupported panel mode: {mode}")

        image = image.resize((cell_size, cell_size), Image.Resampling.NEAREST)
        canvas.paste(image, (x, y))
        for line_index, line in enumerate(_wrap(label, 24)[:2]):
            draw.text((x + 4, y + cell_size + 4 + 16 * line_index), line, fill=(0, 0, 0))

    canvas.save(output_path)
    return output_path


def _rgb_tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = image.detach().cpu().permute(1, 2, 0).numpy()
    array = np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _gray_tensor_to_pil(values: torch.Tensor) -> Image.Image:
    if values.ndim == 3:
        values = values.mean(dim=0)
    array = values.detach().cpu().numpy()
    array = np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(array, mode="L").convert("RGB")


def _wrap(text: str, max_chars: int) -> list[str]:
    words = text.split()
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
