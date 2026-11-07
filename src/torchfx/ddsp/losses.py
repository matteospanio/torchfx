"""Reconstruction losses for DDSP training."""

from __future__ import annotations

import torch
from torch import Tensor


def multires_stft_loss(x: Tensor, y: Tensor, sizes: tuple[int, ...] = (512, 1024, 2048)) -> Tensor:
    """Multi-resolution STFT loss (spectral convergence + log-magnitude L1).

    The standard DDSP reconstruction objective: sum, over several STFT window sizes,
    of the spectral-convergence term and the log-magnitude L1 term.

    Parameters
    ----------
    x, y : Tensor
        1-D signals (or batched ``[..., T]``) to compare.
    sizes : tuple of int, optional
        FFT sizes to sum the loss over (hop = ``n_fft // 4``).

    Returns
    -------
    Tensor
        Scalar loss.

    """
    loss = x.new_zeros(())
    for n_fft in sizes:
        win = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
        x_mag = torch.stft(x, n_fft, n_fft // 4, window=win, return_complex=True).abs()
        y_mag = torch.stft(y, n_fft, n_fft // 4, window=win, return_complex=True).abs()
        loss = loss + torch.norm(x_mag - y_mag) / (torch.norm(y_mag) + 1e-8)
        loss = loss + torch.nn.functional.l1_loss(torch.log(x_mag + 1e-5), torch.log(y_mag + 1e-5))
    return loss
