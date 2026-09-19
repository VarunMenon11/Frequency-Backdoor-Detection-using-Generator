"""Reference-free spectral correction for a single suspicious image.

The clean counterpart is deliberately absent from ``forward``. Clean images may
supervise training, but inference uses only spectral evidence extracted from the
incoming image.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ReferenceFreeCorrection:
    corrected_image: torch.Tensor
    correction_mask: torch.Tensor
    signed_log_correction: torch.Tensor
    effective_log_correction: torch.Tensor
    input_log_amplitude: torch.Tensor
    corrected_log_amplitude: torch.Tensor
    phase: torch.Tensor


class ReferenceFreeSpectralGenerator(nn.Module):
    """Predict a selective log-amplitude correction from one image.

    Input evidence contains normalized log amplitude, a local spectral
    residual, sine/cosine phase, and frequency coordinates. The network emits
    a soft gate and a signed correction. Original FFT phase is retained.
    """

    def __init__(self, base_channels: int = 32, max_log_correction: float = 2.0):
        super().__init__()
        if base_channels < 8:
            raise ValueError("base_channels must be at least 8")
        if max_log_correction <= 0:
            raise ValueError("max_log_correction must be positive")
        self.base_channels = base_channels
        self.max_log_correction = float(max_log_correction)

        self.enc1 = _block(15, base_channels)
        self.enc2 = _block(base_channels, base_channels * 2, stride=2)
        self.enc3 = _block(base_channels * 2, base_channels * 4, stride=2)
        self.bottleneck = _block(base_channels * 4, base_channels * 4)
        self.dec2 = _block(base_channels * 6, base_channels * 2)
        self.dec1 = _block(base_channels * 3, base_channels)
        self.output = nn.Conv2d(base_channels, 6, kernel_size=1)

        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        with torch.no_grad():
            self.output.bias[:3].fill_(-2.0)

    def forward(self, suspicious_images: torch.Tensor) -> ReferenceFreeCorrection:
        _validate_images(suspicious_images)
        spectrum = torch.fft.fft2(suspicious_images.float(), dim=(-2, -1))
        amplitude = spectrum.abs().clamp_min(1e-8)
        phase = torch.angle(spectrum)
        log_amplitude = torch.log1p(amplitude)

        shifted_log = torch.fft.fftshift(log_amplitude, dim=(-2, -1))
        shifted_phase = torch.fft.fftshift(phase, dim=(-2, -1))
        evidence = build_reference_free_evidence(shifted_log, shifted_phase)

        skip1 = self.enc1(evidence)
        skip2 = self.enc2(skip1)
        encoded = self.enc3(skip2)
        decoded = self.bottleneck(encoded)
        decoded = F.interpolate(decoded, size=skip2.shape[-2:], mode="bilinear", align_corners=False)
        decoded = self.dec2(torch.cat((decoded, skip2), dim=1))
        decoded = F.interpolate(decoded, size=skip1.shape[-2:], mode="bilinear", align_corners=False)
        decoded = self.dec1(torch.cat((decoded, skip1), dim=1))

        raw_mask, raw_delta = self.output(decoded).chunk(2, dim=1)
        shifted_mask = torch.sigmoid(raw_mask)
        shifted_delta = self.max_log_correction * torch.tanh(raw_delta)
        shifted_effective = shifted_mask * shifted_delta

        effective = torch.fft.ifftshift(shifted_effective, dim=(-2, -1))
        mask = torch.fft.ifftshift(shifted_mask, dim=(-2, -1))
        signed = torch.fft.ifftshift(shifted_delta, dim=(-2, -1))
        effective = _make_conjugate_symmetric(effective)
        corrected_log = (log_amplitude + effective).clamp_min(0.0)
        corrected_amplitude = torch.expm1(corrected_log)
        corrected = torch.fft.ifft2(
            torch.polar(corrected_amplitude, phase), dim=(-2, -1)
        ).real.clamp(0.0, 1.0)

        return ReferenceFreeCorrection(
            corrected_image=corrected,
            correction_mask=mask,
            signed_log_correction=signed,
            effective_log_correction=effective,
            input_log_amplitude=log_amplitude,
            corrected_log_amplitude=corrected_log,
            phase=phase,
        )

    def metadata(self) -> dict[str, object]:
        return {
            "architecture": self.__class__.__name__,
            "input": "one suspicious RGB image only",
            "input_channels": 15,
            "base_channels": self.base_channels,
            "max_log_correction": self.max_log_correction,
            "phase_policy": "preserve incoming phase",
            "amplitude_domain": "log1p FFT amplitude",
        }


def build_reference_free_evidence(
    shifted_log_amplitude: torch.Tensor,
    shifted_phase: torch.Tensor,
) -> torch.Tensor:
    """Create single-image evidence; no clean reference enters this function."""

    if shifted_log_amplitude.shape != shifted_phase.shape:
        raise ValueError("Amplitude and phase must have identical shapes")
    normalized_amplitude = _standardize(shifted_log_amplitude)
    local_average = F.avg_pool2d(
        shifted_log_amplitude, kernel_size=9, stride=1, padding=4
    )
    local_residual = _standardize(shifted_log_amplitude - local_average)
    coordinates = _frequency_coordinates(
        shifted_log_amplitude.shape[0],
        shifted_log_amplitude.shape[-2],
        shifted_log_amplitude.shape[-1],
        device=shifted_log_amplitude.device,
        dtype=shifted_log_amplitude.dtype,
    )
    return torch.cat(
        (
            normalized_amplitude,
            local_residual,
            torch.sin(shifted_phase),
            torch.cos(shifted_phase),
            coordinates,
        ),
        dim=1,
    )


def total_variation(values: torch.Tensor) -> torch.Tensor:
    vertical = (values[..., 1:, :] - values[..., :-1, :]).abs().mean()
    horizontal = (values[..., :, 1:] - values[..., :, :-1]).abs().mean()
    return vertical + horizontal


def _standardize(values: torch.Tensor) -> torch.Tensor:
    mean = values.mean(dim=(-2, -1), keepdim=True)
    std = values.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(1e-6)
    return ((values - mean) / std).clamp(-5.0, 5.0) / 5.0


def _frequency_coordinates(
    batch_size: int,
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square()) / (2.0**0.5)
    result = torch.stack((xx, yy, radius), dim=0).unsqueeze(0)
    return result.expand(batch_size, -1, -1, -1)


def _make_conjugate_symmetric(values: torch.Tensor) -> torch.Tensor:
    counterpart = torch.roll(
        torch.flip(values, dims=(-2, -1)), shifts=(1, 1), dims=(-2, -1)
    )
    return 0.5 * (values + counterpart)


def _block(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    groups = min(8, out_channels)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(groups, out_channels),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(groups, out_channels),
        nn.SiLU(inplace=True),
    )


def _validate_images(images: torch.Tensor) -> None:
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"Expected (B,3,H,W), got {tuple(images.shape)}")
    if not images.is_floating_point():
        raise TypeError("Images must be floating point tensors in [0,1]")
    if not torch.isfinite(images).all():
        raise ValueError("Images contain NaN or infinite values")
