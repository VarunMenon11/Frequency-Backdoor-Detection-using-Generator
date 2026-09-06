"""Advanced spectral trigger primitives for controlled defense research.

The functions in this module inject deterministic signals into Fourier
frequency bands or one-level Haar wavelet detail subbands. They operate on
single RGB tensors (C, H, W) and batches (B, C, H, W) in the [0, 1] range.

These transformations become backdoor triggers only when poisoned samples are
paired with an attacker-selected label during suspicious-model training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch
import torch.nn.functional as F


AdvancedTriggerKind = Literal["fourier_band", "haar_wavelet"]
FrequencyBand = Literal["low", "middle", "high", "multi"]
WaveletSubband = Literal["LH", "HL", "HH"]


@dataclass(frozen=True)
class AdvancedTriggerConfig:
    """Configuration shared by advanced spectral trigger families."""

    trigger_kind: AdvancedTriggerKind
    strength: float = 0.20
    frequency_band: FrequencyBand = "middle"
    low_radius: float = 0.10
    high_radius: float = 0.25
    transition_width: float = 0.015
    protect_dc_radius: float = 0.01
    wavelet_subband: WaveletSubband = "HH"
    wavelet_frequency: float = 3.0
    wavelet_angle: float = 0.0
    channel_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def validate(self) -> None:
        if self.strength < 0.0:
            raise ValueError("strength must be non-negative")
        if not 0.0 < self.low_radius < self.high_radius < 0.71:
            raise ValueError(
                "Expected 0 < low_radius < high_radius < 0.71, got "
                f"{self.low_radius} and {self.high_radius}"
            )
        if self.transition_width < 0.0:
            raise ValueError("transition_width must be non-negative")
        if not 0.0 <= self.protect_dc_radius < self.low_radius:
            raise ValueError(
                "protect_dc_radius must be non-negative and below low_radius"
            )
        if self.wavelet_frequency <= 0.0:
            raise ValueError("wavelet_frequency must be positive")
        if len(self.channel_weights) != 3:
            raise ValueError("channel_weights must contain three RGB values")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def apply_advanced_trigger(
    images: torch.Tensor,
    config: AdvancedTriggerConfig,
) -> torch.Tensor:
    """Apply one configured trigger while preserving shape and value range."""

    config.validate()
    batch, was_single = _as_batch(images)
    _validate_rgb_batch(batch)

    if config.trigger_kind == "fourier_band":
        result = _apply_fourier_band_trigger(batch, config)
    elif config.trigger_kind == "haar_wavelet":
        result = _apply_haar_wavelet_trigger(batch, config)
    else:
        raise ValueError(f"Unsupported advanced trigger kind: {config.trigger_kind}")

    result = result.clamp(0.0, 1.0)
    return result[0] if was_single else result


def build_normalized_frequency_radius(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return an fftshift-aligned radius map normalized by image dimensions."""

    y = torch.arange(height, device=device, dtype=dtype)
    x = torch.arange(width, device=device, dtype=dtype)
    center_y = height // 2
    center_x = width // 2
    normalized_y = (y - center_y) / max(height, 1)
    normalized_x = (x - center_x) / max(width, 1)
    return torch.sqrt(normalized_y[:, None] ** 2 + normalized_x[None, :] ** 2)


def build_frequency_band_mask(
    height: int,
    width: int,
    config: AdvancedTriggerConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create a soft, centered mask for the configured normalized band."""

    radius = build_normalized_frequency_radius(
        height,
        width,
        device=device,
        dtype=dtype,
    )
    transition = max(config.transition_width, 1e-6)
    above_dc = torch.sigmoid((radius - config.protect_dc_radius) / transition)
    below_low = torch.sigmoid((config.low_radius - radius) / transition)
    above_low = torch.sigmoid((radius - config.low_radius) / transition)
    below_high = torch.sigmoid((config.high_radius - radius) / transition)
    above_high = torch.sigmoid((radius - config.high_radius) / transition)

    if config.frequency_band == "low":
        mask = above_dc * below_low
    elif config.frequency_band == "middle":
        mask = above_low * below_high
    elif config.frequency_band == "high":
        mask = above_high
    elif config.frequency_band == "multi":
        mask = torch.maximum(above_dc * below_low, above_high)
    else:
        raise ValueError(f"Unsupported frequency band: {config.frequency_band}")

    return mask.clamp(0.0, 1.0)


def _apply_fourier_band_trigger(
    images: torch.Tensor,
    config: AdvancedTriggerConfig,
) -> torch.Tensor:
    _, channels, height, width = images.shape
    spectrum = torch.fft.fftshift(
        torch.fft.fft2(images.float(), dim=(-2, -1)),
        dim=(-2, -1),
    )
    amplitude = spectrum.abs()
    phase = torch.angle(spectrum)
    mask = build_frequency_band_mask(
        height,
        width,
        config,
        device=images.device,
        dtype=amplitude.dtype,
    ).view(1, 1, height, width)
    channel_weights = _channel_weights(
        config,
        channels,
        device=images.device,
        dtype=amplitude.dtype,
    )

    # Scaling amplitude rather than replacing it preserves conjugate symmetry
    # and keeps the reconstructed image real-valued. The same normalized radial
    # region is used at every image resolution.
    amplitude_scale = 1.0 + config.strength * channel_weights * mask
    triggered_amplitude = amplitude * amplitude_scale
    triggered_shifted = torch.polar(triggered_amplitude, phase)
    triggered_spectrum = torch.fft.ifftshift(triggered_shifted, dim=(-2, -1))
    return torch.fft.ifft2(triggered_spectrum, dim=(-2, -1)).real


def _apply_haar_wavelet_trigger(
    images: torch.Tensor,
    config: AdvancedTriggerConfig,
) -> torch.Tensor:
    original_height, original_width = images.shape[-2:]
    pad_bottom = original_height % 2
    pad_right = original_width % 2
    padded = F.pad(images, (0, pad_right, 0, pad_bottom), mode="replicate")
    coefficients = _haar_forward(padded)
    selected = coefficients[config.wavelet_subband]

    _, channels, height, width = selected.shape
    pattern = _directional_pattern(
        height,
        width,
        frequency=config.wavelet_frequency,
        angle=config.wavelet_angle,
        device=images.device,
        dtype=selected.dtype,
    ).view(1, 1, height, width)
    channel_weights = _channel_weights(
        config,
        channels,
        device=images.device,
        dtype=selected.dtype,
    )
    subband_scale = selected.flatten(start_dim=2).std(dim=-1, unbiased=False)
    subband_scale = subband_scale.clamp_min(1.0 / 255.0).view(
        selected.shape[0],
        channels,
        1,
        1,
    )

    triggered_coefficients = dict(coefficients)
    triggered_coefficients[config.wavelet_subband] = (
        selected
        + config.strength * subband_scale * channel_weights * pattern
    )
    reconstructed = _haar_inverse(triggered_coefficients)
    return reconstructed[..., :original_height, :original_width]


def _haar_forward(images: torch.Tensor) -> dict[str, torch.Tensor]:
    top_left = images[..., 0::2, 0::2]
    top_right = images[..., 0::2, 1::2]
    bottom_left = images[..., 1::2, 0::2]
    bottom_right = images[..., 1::2, 1::2]
    return {
        "LL": (top_left + top_right + bottom_left + bottom_right) / 2.0,
        "LH": (top_left - top_right + bottom_left - bottom_right) / 2.0,
        "HL": (top_left + top_right - bottom_left - bottom_right) / 2.0,
        "HH": (top_left - top_right - bottom_left + bottom_right) / 2.0,
    }


def _haar_inverse(coefficients: dict[str, torch.Tensor]) -> torch.Tensor:
    ll = coefficients["LL"]
    lh = coefficients["LH"]
    hl = coefficients["HL"]
    hh = coefficients["HH"]
    result = torch.empty(
        ll.shape[0],
        ll.shape[1],
        ll.shape[2] * 2,
        ll.shape[3] * 2,
        device=ll.device,
        dtype=ll.dtype,
    )
    result[..., 0::2, 0::2] = (ll + lh + hl + hh) / 2.0
    result[..., 0::2, 1::2] = (ll - lh + hl - hh) / 2.0
    result[..., 1::2, 0::2] = (ll + lh - hl - hh) / 2.0
    result[..., 1::2, 1::2] = (ll - lh - hl + hh) / 2.0
    return result


def _directional_pattern(
    height: int,
    width: int,
    *,
    frequency: float,
    angle: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    y = torch.linspace(-0.5, 0.5, height, device=device, dtype=dtype)
    x = torch.linspace(-0.5, 0.5, width, device=device, dtype=dtype)
    rotated = torch.cos(torch.tensor(angle, device=device, dtype=dtype)) * x[None, :]
    rotated = rotated + torch.sin(
        torch.tensor(angle, device=device, dtype=dtype)
    ) * y[:, None]
    return torch.sin(2.0 * torch.pi * frequency * rotated)


def _channel_weights(
    config: AdvancedTriggerConfig,
    channels: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if channels != 3:
        raise ValueError(f"Expected three RGB channels, got {channels}")
    return torch.tensor(
        config.channel_weights,
        device=device,
        dtype=dtype,
    ).view(1, channels, 1, 1)


def _as_batch(images: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if images.ndim == 3:
        return images.unsqueeze(0), True
    if images.ndim == 4:
        return images, False
    raise ValueError(
        "Expected image shape (C, H, W) or (B, C, H, W), got "
        f"{tuple(images.shape)}"
    )


def _validate_rgb_batch(images: torch.Tensor) -> None:
    if images.shape[1] != 3:
        raise ValueError(f"Expected RGB input, got shape {tuple(images.shape)}")
    if not images.is_floating_point():
        raise TypeError("Advanced triggers expect floating-point tensors in [0, 1]")
    if not torch.isfinite(images).all():
        raise ValueError("Input contains NaN or infinite values")

