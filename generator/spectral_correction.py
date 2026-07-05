"""Generator and spectral correction utilities.

The generator predicts a correction map over amplitude spectra. It should be
understood as learning an intervention that optimizes our losses, not as proven
trigger-frequency detection.
"""

from __future__ import annotations

import torch
from torch import nn


class SpectralCorrectionGenerator(nn.Module):
    """Small CNN that predicts an amplitude correction map.

    Input shape:
        (B, 9, H, W), made from normalized clean amplitude, triggered amplitude,
        and absolute amplitude difference.

    Output shape:
        (B, 3, H, W), values in [0, 1]. Higher values mean stronger correction
        at that amplitude location.
    """

    def __init__(self, hidden_channels: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _conv_block(9, hidden_channels),
            _conv_block(hidden_channels, hidden_channels),
            _conv_block(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, 3, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, generator_input: torch.Tensor) -> torch.Tensor:
        return self.net(generator_input)


def fft_amplitude_phase(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unshifted FFT amplitude and phase for image batches."""

    if images.ndim != 4:
        raise ValueError(f"Expected images with shape (B, C, H, W), got {tuple(images.shape)}")
    fft = torch.fft.fft2(images.float(), dim=(-2, -1))
    return torch.abs(fft), torch.angle(fft)


def reconstruct_from_amplitude_phase(
    amplitude: torch.Tensor,
    phase: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct image batch from amplitude and phase."""

    complex_spectrum = torch.polar(amplitude, phase)
    reconstructed = torch.fft.ifft2(complex_spectrum, dim=(-2, -1)).real
    return reconstructed.clamp(0.0, 1.0)


def build_generator_input(
    clean_amplitude: torch.Tensor,
    triggered_amplitude: torch.Tensor,
) -> torch.Tensor:
    """Build normalized generator input from clean and triggered amplitudes."""

    clean_log = torch.log1p(clean_amplitude)
    triggered_log = torch.log1p(triggered_amplitude)
    diff_log = torch.abs(triggered_log - clean_log)
    return torch.cat(
        [
            normalize_per_sample(clean_log),
            normalize_per_sample(triggered_log),
            normalize_per_sample(diff_log),
        ],
        dim=1,
    )


def apply_correction_map(
    clean_amplitude: torch.Tensor,
    triggered_amplitude: torch.Tensor,
    correction_map: torch.Tensor,
) -> torch.Tensor:
    """Apply correction by moving triggered amplitude toward clean amplitude."""

    amplitude_delta = triggered_amplitude - clean_amplitude
    corrected = triggered_amplitude - correction_map * amplitude_delta
    return corrected.clamp_min(0.0)


def total_variation(correction_map: torch.Tensor) -> torch.Tensor:
    """Spatial total variation regularizer for correction maps."""

    vertical = torch.abs(correction_map[:, :, 1:, :] - correction_map[:, :, :-1, :]).mean()
    horizontal = torch.abs(correction_map[:, :, :, 1:] - correction_map[:, :, :, :-1]).mean()
    return vertical + horizontal


def normalize_per_sample(values: torch.Tensor) -> torch.Tensor:
    """Min-max normalize each sample/channel independently."""

    flat = values.flatten(start_dim=2)
    min_values = flat.min(dim=-1).values[:, :, None, None]
    max_values = flat.max(dim=-1).values[:, :, None, None]
    return (values - min_values) / (max_values - min_values + 1e-8)


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )
