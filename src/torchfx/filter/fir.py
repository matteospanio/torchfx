"""Fir filters."""

from collections.abc import Sequence

import torch
from numpy.typing import ArrayLike
from scipy.signal import firwin
from torch import Tensor, nn
from typing_extensions import override

from torchfx.filter.__base import AbstractFilter
from torchfx.typing import WindowType


class FIR(AbstractFilter):
    """Efficient FIR filter using conv1d. Supports [T], [C, T], [B, C, T]."""

    def __init__(self, b: ArrayLike) -> None:
        super().__init__()
        # Flip the kernel for causal convolution (like lfilter)
        b_tensor = torch.tensor(b, dtype=torch.float32).flip(0)
        self.register_buffer("kernel", b_tensor[None, None, :])  # [1, 1, K]

    @override
    def compute_coefficients(self) -> None:
        """Compute the filter coefficients."""
        # This method is not used in FIR, but defined for consistency with IIR
        pass

    @override
    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        device = x.device
        kernel = self.kernel.to(dtype=dtype, device=device)

        original_shape = x.shape

        # Reshape input to [B, C, T]
        if x.ndim == 1:
            # [T] → [1, 1, T]
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 2:
            # [C, T] → [1, C, T]
            x = x.unsqueeze(0)
        elif x.ndim == 3:
            # [B, C, T] → as is
            pass
        else:
            raise ValueError("Input must be of shape [T], [C, T], or [B, C, T]")

        BATCHES, CHANNELS, TIME = x.shape

        # Expand kernel to match number of channels
        kernel_exp = kernel.expand(CHANNELS, 1, -1)  # [C, 1, K] # type: ignore

        # Pad input to maintain original length, pad right side
        pad = kernel.shape[-1] - 1  # type: ignore
        x_padded = nn.functional.pad(x, (pad, 0))  # pad right only # type: ignore

        # Apply convolution with groups = C (same kernel per channel, repeated for batch)
        y = nn.functional.conv1d(x_padded, kernel_exp.repeat(BATCHES, 1, 1), groups=CHANNELS)

        # Reshape back to [B, C, T]
        y = y.view(BATCHES, CHANNELS, TIME)

        # Reduce to original shape if input wasn't batched
        if len(original_shape) == 1:
            return y[0, 0]
        elif len(original_shape) == 2:
            return y[0]
        else:
            return y


class DesignableFIR(FIR):
    """
    FIR filter designed using scipy.signal.firwin.

    Attributes
    ----------
    cutoff : float | Sequence[float]
        Cutoff frequency or frequencies (in Hz) for the filter.
    num_taps : int
        Number of taps (filter order) for the FIR filter.
    fs : int | None
        Sampling frequency (in Hz) of the input signal. If None, the filter will not
        be designed.
    pass_zero : bool
        If True, the filter will be a lowpass filter. If False, it will be a highpass
        filter.
    window : WindowType
        Window type to use for the FIR filter design. Default is "hamming".

    """

    def __init__(
        self,
        cutoff: float | Sequence[float],
        num_taps: int,
        fs: int | None = None,
        pass_zero: bool = True,
        window: WindowType = "hamming",
    ) -> None:
        # Design the filter using firwin
        self.num_taps = num_taps
        self.cutoff = cutoff
        self.fs = fs
        self.pass_zero = pass_zero
        self.window = window

        self.b: ArrayLike | None = None
        if fs is not None:
            self.compute_coefficients()
            assert self.b is not None, "Filter coefficients (b) must be computed."
            super().__init__(self.b)

    @override
    def compute_coefficients(self) -> None:
        assert self.fs is not None, "Sampling frequency (fs) must be set."

        self.b = firwin(
            self.num_taps,
            self.cutoff,
            fs=self.fs,
            pass_zero=self.pass_zero,
            window=self.window,
            scale=True,
        )
        assert self.b is not None, "Filter coefficients (b) must be computed."

        super().__init__(self.b)
