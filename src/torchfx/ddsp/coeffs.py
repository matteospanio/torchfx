"""Differentiable RBJ-cookbook filter-design math.

These return ``[K, 6]`` SOS stacks (the TorchFX layout) built from plain torch ops,
so the coefficients stay on the autograd tape: gradients flow back to ``cutoff`` /
``frequency`` / ``q`` / ``gain_db``. They mirror the deterministic ``Biquad`` designs
but are usable inside a trainable :mod:`torchfx.ddsp` model.

"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def rbj_lowpass_sos(cutoff: Tensor, q: Tensor, fs: float) -> Tensor:
    """RBJ low-pass biquad as a differentiable ``[..., 6]`` SOS row."""
    w0 = 2.0 * math.pi * cutoff / fs
    cos_w0 = torch.cos(w0)
    alpha = torch.sin(w0) / (2.0 * q)

    b1 = 1.0 - cos_w0
    b0 = b1 / 2.0
    b2 = b0
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    return torch.stack([b0 / a0, b1 / a0, b2 / a0, torch.ones_like(a0), a1 / a0, a2 / a0], dim=-1)


def rbj_highpass_sos(cutoff: Tensor, q: Tensor, fs: float) -> Tensor:
    """RBJ high-pass biquad as a differentiable ``[..., 6]`` SOS row."""
    w0 = 2.0 * math.pi * cutoff / fs
    cos_w0 = torch.cos(w0)
    alpha = torch.sin(w0) / (2.0 * q)

    b1 = -(1.0 + cos_w0)
    b0 = (1.0 + cos_w0) / 2.0
    b2 = b0
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    return torch.stack([b0 / a0, b1 / a0, b2 / a0, torch.ones_like(a0), a1 / a0, a2 / a0], dim=-1)


def rbj_peaking_sos(frequency: Tensor, q: Tensor, gain_db: Tensor, fs: float) -> Tensor:
    """RBJ peaking-EQ biquad as a differentiable ``[..., 6]`` SOS row.

    Identical math to ``torchfx.filter.biquad`` peaking, in batched torch ops so the
    coefficients stay differentiable w.r.t. ``frequency``, ``q`` and ``gain_db``.

    """
    a_gain = torch.pow(10.0, gain_db / 40.0)
    w0 = 2.0 * math.pi * frequency / fs
    cos_w0 = torch.cos(w0)
    alpha = torch.sin(w0) / (2.0 * q)

    b0 = 1.0 + alpha * a_gain
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * a_gain
    a0 = 1.0 + alpha / a_gain
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / a_gain
    return torch.stack([b0 / a0, b1 / a0, b2 / a0, torch.ones_like(a0), a1 / a0, a2 / a0], dim=-1)


def sos_freq_response(sos: Tensor, n_fft: int) -> Tensor:
    """Complex frequency response of an SOS cascade on the ``rfft`` grid.

    ``H(z) = prod_k (b0 + b1 z^-1 + b2 z^-2) / (a0 + a1 z^-1 + a2 z^-2)`` evaluated at
    ``z = e^{jw}`` for the ``n_fft // 2 + 1`` non-negative frequency bins. Complex-
    differentiable — handy for analysis/plotting and frequency-sampling losses.

    """
    w = torch.linspace(0.0, math.pi, n_fft // 2 + 1, device=sos.device, dtype=sos.dtype)
    z1 = torch.exp(torch.complex(torch.zeros_like(w), -w))  # z^-1
    z2 = z1 * z1
    b = sos[:, 0:1] + sos[:, 1:2] * z1 + sos[:, 2:3] * z2
    a = sos[:, 3:4] + sos[:, 4:5] * z1 + sos[:, 5:6] * z2
    return torch.prod(b / a, dim=0)
