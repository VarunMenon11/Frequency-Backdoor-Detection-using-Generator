"""Spectrum utilities for inspecting image perturbations."""

from __future__ import annotations

import torch


def shifted_fft(image: torch.Tensor) -> torch.Tensor:
    """Compute a centered 2D FFT for an image tensor.

    Args:
        image: Tensor with shape (C, H, W).

    Returns:
        Complex tensor with shape (C, H, W), shifted so low frequencies are at
        the center.
    """

    if image.ndim != 3:
        raise ValueError(f"Expected image shape (C, H, W), got {tuple(image.shape)}")

    fft = torch.fft.fft2(image.float(), dim=(-2, -1))
    return torch.fft.fftshift(fft, dim=(-2, -1))


def amplitude_spectrum(image: torch.Tensor, *, log_scale: bool = True) -> torch.Tensor:
    """Compute a channel-averaged amplitude spectrum.

    Args:
        image: Tensor with shape (C, H, W).
        log_scale: Use log(1 + amplitude), which makes weaker frequencies easier
            to see.

    Returns:
        Tensor with shape (H, W), averaged over channels.
    """

    amplitude = torch.abs(shifted_fft(image)).mean(dim=0)
    if log_scale:
        amplitude = torch.log1p(amplitude)
    return amplitude


def phase_spectrum(image: torch.Tensor) -> torch.Tensor:
    """Compute a channel-averaged phase spectrum normalized to [0, 1]."""

    phase = torch.angle(shifted_fft(image)).mean(dim=0)
    return (phase + torch.pi) / (2.0 * torch.pi)


def normalize_minmax(values: torch.Tensor) -> torch.Tensor:
    """Normalize a tensor to [0, 1] for visualization."""

    min_value = values.min()
    max_value = values.max()
    return (values - min_value) / (max_value - min_value + 1e-8)


def log_amplitude_spectrum(image: torch.Tensor) -> torch.Tensor:
    """Compute a shifted log-amplitude spectrum for an RGB image.

    Args:
        image: Tensor with shape (3, H, W).

    Returns:
        Tensor with shape (H, W), averaged over channels and normalized to [0, 1].
    """

    return normalize_minmax(amplitude_spectrum(image, log_scale=True))
