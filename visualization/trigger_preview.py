"""Visualization helpers for trigger inspection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def save_trigger_preview(
    clean: torch.Tensor,
    triggered: torch.Tensor,
    clean_spectrum: torch.Tensor,
    triggered_spectrum: torch.Tensor,
    output_path: str | Path,
) -> Path:
    """Save clean/triggered image and spectrum comparison."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    difference = (triggered - clean).abs()
    panels = [
        ("clean", _tensor_image_to_pil(clean)),
        ("triggered", _tensor_image_to_pil(triggered)),
        ("abs diff x6", _tensor_image_to_pil((difference * 6.0).clamp(0.0, 1.0))),
        ("clean spectrum", _spectrum_to_pil(clean_spectrum)),
        ("triggered spectrum", _spectrum_to_pil(triggered_spectrum)),
        (
            "spectrum diff",
            _spectrum_to_pil((triggered_spectrum - clean_spectrum).abs()),
        ),
    ]

    cell_size = 128
    label_height = 24
    canvas = Image.new("RGB", (3 * cell_size, 2 * (cell_size + label_height)), "white")
    draw = ImageDraw.Draw(canvas)

    for index, (label, panel) in enumerate(panels):
        row, col = divmod(index, 3)
        x = col * cell_size
        y = row * (cell_size + label_height)
        canvas.paste(panel.resize((cell_size, cell_size), Image.Resampling.NEAREST), (x, y))
        draw.text((x + 4, y + cell_size + 4), label, fill=(0, 0, 0))

    canvas.save(output_path)
    return output_path


def _tensor_image_to_pil(image: torch.Tensor) -> Image.Image:
    array = image.detach().cpu().permute(1, 2, 0).numpy()
    array = np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _spectrum_to_pil(spectrum: torch.Tensor) -> Image.Image:
    array = spectrum.detach().cpu().numpy()
    array = np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(array, mode="L").convert("RGB")
