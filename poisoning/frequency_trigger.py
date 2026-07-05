"""Frequency-pattern trigger construction.

This trigger is implemented as a sinusoidal image-space perturbation with a
known frequency. A sinusoid creates localized peaks in the Fourier amplitude
spectrum, which makes it a useful first controlled frequency-domain backdoor.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FrequencyTriggerConfig:
    """Configuration for a simple sinusoidal frequency trigger.

    Args:
        horizontal_frequency: Cycles across image width.
        vertical_frequency: Cycles across image height.
        strength: Additive image-space amplitude. For CIFAR tensors in [0, 1],
            0.08 is a medium perturbation: visible in spectra and often faintly
            visible in images.
        channel_weights: Per-channel RGB trigger weights.
    """

    horizontal_frequency: int = 6
    vertical_frequency: int = 6
    strength: float = 0.08
    channel_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)


def build_frequency_pattern(
    height: int,
    width: int,
    config: FrequencyTriggerConfig,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create a normalized RGB sinusoidal trigger pattern.

    Output shape is (3, H, W), with values approximately in [-1, 1] before
    applying ``config.strength``.
    """

    y = torch.arange(height, dtype=torch.float32, device=device).view(height, 1)
    x = torch.arange(width, dtype=torch.float32, device=device).view(1, width)

    phase = 2.0 * torch.pi * (
        config.horizontal_frequency * x / width
        + config.vertical_frequency * y / height
    )
    base_pattern = torch.cos(phase)
    channel_weights = torch.tensor(
        config.channel_weights,
        dtype=torch.float32,
        device=device,
    ).view(3, 1, 1)

    return channel_weights * base_pattern.unsqueeze(0)


def apply_frequency_trigger(
    image: torch.Tensor,
    config: FrequencyTriggerConfig,
) -> torch.Tensor:
    """Apply the frequency trigger to one image tensor.

    Args:
        image: Clean image tensor with shape (3, H, W), range [0, 1].
        config: Trigger configuration.

    Returns:
        Triggered image tensor with shape (3, H, W), clamped to [0, 1].
    """

    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected image shape (3, H, W), got {tuple(image.shape)}")

    _, height, width = image.shape
    pattern = build_frequency_pattern(
        height,
        width,
        config,
        device=image.device,
    )
    triggered = image + config.strength * pattern
    return triggered.clamp(0.0, 1.0)
