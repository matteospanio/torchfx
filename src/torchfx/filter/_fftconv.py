"""FFT-based 1D convolution utilities for FIR filtering.

Adapted from Julius (https://github.com/adefossez/julius), MIT License.
Original author: Alexandre Defossez, 2020.

Modifications for torchfx:
- Simplified for grouped (depthwise) convolution with single shared kernel
- Uses modern ``torch.fft`` API exclusively (no old-API fallback)
- Supports asymmetric padding ``(left, right)`` for causal filtering
- Removed stride, bias, and multi-channel weight support

"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def pad_to(tensor: Tensor, target_length: int) -> Tensor:
    """Zero-pad the last dimension of *tensor* to *target_length*.

    Adapted from ``julius.core.pad_to``.

    """
    return F.pad(tensor, (0, target_length - tensor.shape[-1]))


def unfold(x: Tensor, kernel_size: int, stride: int) -> Tensor:
    """Extract overlapping frames from a tensor along its last dimension.

    Much faster than :func:`torch.Tensor.unfold` because it uses
    :func:`torch.as_strided` for zero-copy frame extraction.

    Adapted from ``julius.core.unfold``.

    Parameters
    ----------
    x : Tensor
        Tensor of shape ``[*, T]``.
    kernel_size : int
        Size of each frame.
    stride : int
        Stride between consecutive frames.

    Returns
    -------
    Tensor
        Tensor of shape ``[*, F, kernel_size]`` where
        ``F = 1 + ceil((T - kernel_size) / stride)``.

    """
    shape = list(x.shape)
    length = shape.pop(-1)
    n_frames = math.ceil((max(length, kernel_size) - kernel_size) / stride) + 1
    tgt_length = (n_frames - 1) * stride + kernel_size
    padded = F.pad(x, (0, tgt_length - length)).contiguous()
    strides: list[int] = []
    for dim in range(padded.dim()):
        strides.append(padded.stride(dim))
    last_stride = strides.pop(-1)
    assert last_stride == 1, "data should be contiguous"
    strides = strides + [stride, 1]
    return padded.as_strided(shape + [n_frames, kernel_size], strides)


def fft_conv1d(
    x: Tensor,
    kernel: Tensor,
    padding: tuple[int, int] = (0, 0),
    block_ratio: float = 5.0,
) -> Tensor:
    """Overlap-save FFT convolution matching :func:`torch.nn.functional.conv1d`.

    Applies the same *kernel* independently to every channel of *x*
    (depthwise / grouped convolution with ``groups=C``).  This is
    significantly faster than direct convolution for kernel sizes >= 64.

    Adapted from ``julius.fftconv.fft_conv1d`` (MIT License, A. Defossez 2020).
    Simplified for the single-kernel depthwise case used by torchfx FIR filters:
    the expensive ``torch.einsum`` in Julius is replaced by element-wise complex
    multiplication with broadcasting.

    Parameters
    ----------
    x : Tensor
        Signal tensor of shape ``[B, C, T]``.
    kernel : Tensor
        Convolution kernel of shape ``[1, 1, K]``.  The same kernel is
        broadcast across all batch and channel dimensions.
    padding : tuple[int, int]
        ``(left, right)`` zero-padding applied to *x* before convolution.
    block_ratio : float
        Controls the FFT block size as ``int(kernel_size * block_ratio)``.
        Larger values use fewer, bigger blocks (faster FFT but more memory).

    Returns
    -------
    Tensor
        Convolution result of shape ``[B, C, T']`` where
        ``T' = T + left + right - K + 1``.

    """
    x = F.pad(x, padding)
    batch, channels, length = x.shape
    kernel_size = kernel.shape[-1]

    if length < kernel_size:
        raise RuntimeError(
            f"Input should be at least as large as the kernel size {kernel_size}, "
            f"but it is only {length} samples long."
        )
    if block_ratio < 1:
        raise RuntimeError("Block ratio must be greater than 1.")

    block_size: int = min(int(kernel_size * block_ratio), length)
    fold_stride = block_size - kernel_size + 1

    # Pad kernel to block_size and pre-compute its FFT (once).
    kernel_padded = pad_to(kernel, block_size)  # [1, 1, block_size]
    kernel_z = torch.fft.rfft(kernel_padded, dim=-1)  # [1, 1, freq_bins]

    # Extract overlapping frames: [B, C, F, block_size]
    frames = unfold(x, block_size, fold_stride)

    # FFT convolution via element-wise complex multiply (cross-correlation).
    frames_z = torch.fft.rfft(frames, dim=-1)  # [B, C, F, freq_bins]
    out_z = frames_z * kernel_z.conj()  # broadcast over B, C, F
    out = torch.fft.irfft(out_z, n=block_size, dim=-1)  # [B, C, F, block_size]

    # Discard circular convolution artifacts (last kernel_size - 1 samples).
    out = out[..., :fold_stride]
    out = out.reshape(batch, channels, -1)

    # Trim to exact target length.
    target_length = length - kernel_size + 1
    out = out[..., :target_length]
    return Tensor(out)
