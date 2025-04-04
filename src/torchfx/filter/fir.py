"""Fir filters."""

from numpy.typing import ArrayLike
from torch import Tensor
import torch
from torchaudio import functional as F  # noqa: N812
from typing_extensions import override
from scipy.signal import firwin
from typing import Literal, Sequence

from torchfx.filter.__base import AbstractFilter
from torchfx.typing import WindowType


class FIR(AbstractFilter):
    """A FIR filter implemented via torchaudio.lfilter."""

    def __init__(self, b: ArrayLike) -> None:
        super().__init__()
        self.b = torch.tensor(b, dtype=torch.float32)
        self.a = torch.tensor([1.0], dtype=torch.float32)  # FIR has no feedback

    @override
    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        device = x.device

        x = x.to(dtype).to(device)

        if x.ndim == 1:
            x = x[None, None, :]  # [1, 1, T]
        elif x.ndim == 2:
            x = x[:, None, :]     # [B, 1, T]

        return F.lfilter(x, self.a.to(device), self.b.to(device))


# Tipi per chiarezza


class DesignableFIR(FIR):
    """
    FIR filter designed using scipy.signal.firwin.

    Args:
        cutoff: float or list of float – cutoff frequency/frequencies in Hz
        fs: Sampling rate in Hz
        num_taps: Length of the filter (number of coefficients)
        pass_zero: True for lowpass/highpass, False for bandpass/stopband
        window: Type of window to use
        scale: Whether to scale the filter to unity gain at zero freq
    """

    def __init__(
        self,
        cutoff: float | Sequence[float],
        fs: float,
        num_taps: int = 101,
        pass_zero: bool | Literal["bandpass", "lowpass", "highpass", "bandstop"] = True,
        window: WindowType = "hamming",
        scale: bool = True,
    ):
        b = firwin(
            numtaps=num_taps,
            cutoff=cutoff,
            fs=fs,
            pass_zero=pass_zero, # type: ignore
            window=window,
            scale=scale,
        )
        super().__init__(b)
        self.fs = fs
        self.cutoff = cutoff
        self.num_taps = num_taps
        self.pass_zero = pass_zero
        self.window = window
        self.scale = scale
