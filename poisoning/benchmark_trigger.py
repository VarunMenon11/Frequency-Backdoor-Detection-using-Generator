"""Reproducible trigger families for cross-dataset defense benchmarks.

The SIG and Blended implementations follow the public attack definitions, but
the exact parameters and Blended carrier are recorded because published papers
do not always use the same carrier image.  FTrojan is the project's documented
block-DCT adaptation.  These functions create trigger images only; a backdoor
is formed when selected training images are relabelled to the target class.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch

from poisoning.advanced_trigger import AdvancedTriggerConfig, apply_advanced_trigger


BenchmarkTriggerKind = Literal["sig", "blended", "ftrojan"]
_CARRIER_CACHE: dict[tuple[int, int, int, str, torch.dtype], torch.Tensor] = {}


@dataclass(frozen=True)
class BenchmarkTriggerConfig:
    kind: BenchmarkTriggerKind
    sig_delta: float = 20.0 / 255.0
    sig_frequency: float = 6.0
    blend_alpha: float = 0.20
    blend_seed: int = 2026
    ftrojan_strength: float = 100.0
    ftrojan_block_size: int = 32
    ftrojan_positions: tuple[tuple[int, int], ...] = ((15, 15), (31, 31))
    ftrojan_channels: tuple[int, ...] = (1, 2)

    def validate(self) -> None:
        if self.kind not in ("sig", "blended", "ftrojan"):
            raise ValueError(f"Unsupported benchmark trigger: {self.kind}")
        if self.sig_delta < 0.0 or self.sig_frequency <= 0.0:
            raise ValueError("SIG delta must be non-negative and frequency positive")
        if not 0.0 <= self.blend_alpha <= 1.0:
            raise ValueError("Blended alpha must be in [0, 1]")
        if self.ftrojan_strength < 0.0:
            raise ValueError("FTrojan strength must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def apply_benchmark_trigger(
    images: torch.Tensor,
    config: BenchmarkTriggerConfig,
) -> torch.Tensor:
    """Apply a configured trigger to one RGB image or an RGB batch in [0, 1]."""

    config.validate()
    batch, was_single = _as_batch(images)
    if batch.shape[1] != 3 or not batch.is_floating_point():
        raise ValueError("Expected floating RGB tensors with shape (B,3,H,W)")

    if config.kind == "sig":
        result = _apply_sig(batch, config)
    elif config.kind == "blended":
        result = _apply_blended(batch, config)
    else:
        result = apply_advanced_trigger(
            batch,
            AdvancedTriggerConfig(
                trigger_kind="ftrojan_dct",
                strength=config.ftrojan_strength,
                dct_block_size=config.ftrojan_block_size,
                dct_positions=config.ftrojan_positions,
                dct_channels=config.ftrojan_channels,
            ),
        )
    result = result.clamp(0.0, 1.0)
    return result[0] if was_single else result


def build_blended_carrier(
    height: int,
    width: int,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a deterministic full-frame RGB carrier for the Blended attack.

    A generated carrier avoids silently depending on a copyrighted or missing
    external image.  Its seed is part of every saved run configuration.
    """

    key = (height, width, seed, str(device), dtype)
    cached = _CARRIER_CACHE.get(key)
    if cached is not None:
        return cached
    generator = torch.Generator(device="cpu").manual_seed(seed)
    low_height = max(4, height // 8)
    low_width = max(4, width // 8)
    coarse = torch.rand((1, 3, low_height, low_width), generator=generator)
    carrier = torch.nn.functional.interpolate(
        coarse, size=(height, width), mode="bicubic", align_corners=False
    ).clamp(0.0, 1.0)
    result = carrier.to(device=device, dtype=dtype)
    _CARRIER_CACHE[key] = result
    return result


def _apply_sig(images: torch.Tensor, config: BenchmarkTriggerConfig) -> torch.Tensor:
    width = images.shape[-1]
    x = torch.arange(width, device=images.device, dtype=images.dtype)
    signal = config.sig_delta * torch.sin(
        2.0 * torch.pi * config.sig_frequency * x / max(width, 1)
    )
    return images + signal.view(1, 1, 1, width)


def _apply_blended(
    images: torch.Tensor,
    config: BenchmarkTriggerConfig,
) -> torch.Tensor:
    carrier = build_blended_carrier(
        images.shape[-2],
        images.shape[-1],
        seed=config.blend_seed,
        device=images.device,
        dtype=images.dtype,
    )
    return (1.0 - config.blend_alpha) * images + config.blend_alpha * carrier


def _as_batch(images: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if images.ndim == 3:
        return images.unsqueeze(0), True
    if images.ndim == 4:
        return images, False
    raise ValueError(f"Expected (3,H,W) or (B,3,H,W), got {tuple(images.shape)}")
