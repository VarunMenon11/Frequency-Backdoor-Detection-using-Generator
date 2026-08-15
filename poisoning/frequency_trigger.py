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
    trigger_kind: str = "cosine"
    secondary_horizontal_frequency: int = 10
    secondary_vertical_frequency: int = 2
    window_center_x: float = 0.65
    window_center_y: float = 0.50
    window_sigma: float = 0.18
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
    if config.trigger_kind == "cosine":
        base_pattern = torch.cos(phase)
    elif config.trigger_kind == "sine":
        base_pattern = torch.sin(phase)
    elif config.trigger_kind == "checkerboard":
        base_pattern = torch.where(torch.cos(phase) >= 0.0, 1.0, -1.0)
    elif config.trigger_kind == "dual_frequency":
        secondary_phase = 2.0 * torch.pi * (
            config.secondary_horizontal_frequency * x / width
            + config.secondary_vertical_frequency * y / height
        )
        base_pattern = 0.5 * (torch.cos(phase) + torch.cos(secondary_phase))
        base_pattern = base_pattern / base_pattern.abs().amax().clamp_min(1e-8)
    elif config.trigger_kind == "localized_cosine":
        center_x = config.window_center_x * max(width - 1, 1)
        center_y = config.window_center_y * max(height - 1, 1)
        sigma_x = config.window_sigma * max(width, 1)
        sigma_y = config.window_sigma * max(height, 1)
        window = torch.exp(
            -0.5
            * (
                ((x - center_x) / sigma_x) ** 2
                + ((y - center_y) / sigma_y) ** 2
            )
        )
        base_pattern = window * torch.cos(phase)
        base_pattern = base_pattern / base_pattern.abs().amax().clamp_min(1e-8)
    else:
        raise ValueError(
            "Unsupported trigger_kind: "
            f"{config.trigger_kind}. Use cosine, sine, checkerboard, "
            "dual_frequency, or localized_cosine."
        )
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
    """Apply the frequency trigger to one image or image batch.

    Args:
        image: Clean image tensor with shape (3, H, W) or (B, 3, H, W), range [0, 1].
        config: Trigger configuration.

    Returns:
        Triggered image tensor with the same shape, clamped to [0, 1].
    """

    if image.ndim == 3:
        return _apply_frequency_trigger_single(image, config)
    if image.ndim == 4:
        return _apply_frequency_trigger_batch(image, config)
    raise ValueError(
        f"Expected image shape (3, H, W) or (B, 3, H, W), got {tuple(image.shape)}"
    )


def _apply_frequency_trigger_single(
    image: torch.Tensor,
    config: FrequencyTriggerConfig,
) -> torch.Tensor:
    if image.shape[0] != 3:
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


def _apply_frequency_trigger_batch(
    images: torch.Tensor,
    config: FrequencyTriggerConfig,
) -> torch.Tensor:
    if images.shape[1] != 3:
        raise ValueError(f"Expected image shape (B, 3, H, W), got {tuple(images.shape)}")

    _, _, height, width = images.shape
    pattern = build_frequency_pattern(
        height,
        width,
        config,
        device=images.device,
    ).unsqueeze(0)
    triggered = images + config.strength * pattern
    return triggered.clamp(0.0, 1.0)
