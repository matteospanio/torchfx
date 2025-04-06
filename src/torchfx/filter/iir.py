"""Module of IIR filters."""

import abc
from typing import Sequence

import numpy as np
import torch
from scipy.signal import butter, iirpeak, iirnotch, cheby1, cheby2
from torch import Tensor
from torchaudio import functional as F  # noqa: N812
from typing_extensions import override

from torchfx.filter.__base import AbstractFilter
from torchfx.typing import FilterOrderScale

NONE_FS_ERR = "Sample rate of the filter could not be None."


class IIR(AbstractFilter):
    """IIR filter."""

    fs: int | None
    cutoff: float

    @abc.abstractmethod
    def __init__(self, fs: int | None = None) -> None:
        super().__init__()
        self.fs = fs

    @override
    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        device = x.device
        if self.fs is None:
            raise ValueError(NONE_FS_ERR)

        if self.a is None or self.b is None:
            self.compute_coefficients()

        return F.lfilter(
            x,
            torch.as_tensor(self.a, dtype=dtype, device=device),
            torch.as_tensor(self.b, dtype=dtype, device=device),
        )


class Butterworth(IIR):
    """Butterworth filter."""

    def __init__(
        self,
        btype: str,
        cutoff: float,
        order: int = 4,
        order_scale: FilterOrderScale = "linear",
        fs: int | None = None,
        a: Sequence | None = None,
        b: Sequence | None = None,
    ) -> None:
        super().__init__(fs)
        self.btype = btype
        self.cutoff = cutoff
        self.order = order if order_scale == "linear" else order // 6
        self.a = a
        self.b = b

    @override
    def compute_coefficients(self) -> None:
        assert self.fs is not None

        b, a = butter(self.order, self.cutoff / (0.5 * self.fs), btype=self.btype)  # type: ignore
        self.b = b
        self.a = a


class Chebyshev1(IIR):
    """Chebyshev type 1 filter."""

    def __init__(
        self,
        btype: str,
        cutoff: float,
        order: int = 4,
        ripple: float = 0.1,
        fs: int | None = None,
        a: Sequence | None = None,
        b: Sequence | None = None,
    ) -> None:
        super().__init__(fs)
        self.btype = btype
        self.cutoff = cutoff
        self.order = order
        self.ripple = ripple
        self.a = a
        self.b = b

    @override
    def compute_coefficients(self) -> None:
        assert self.fs is not None

        b, a = cheby1(  # type: ignore
            self.order,
            self.ripple,
            self.cutoff / (0.5 * self.fs),
            btype=self.btype,
        )
        self.b = b
        self.a = a


class Chebyshev2(IIR):
    """Chebyshev type 2 filter."""

    def __init__(
        self,
        btype: str,
        cutoff: float,
        order: int = 4,
        ripple: float = 0.1,
        fs: int | None = None,
    ) -> None:
        super().__init__(fs)
        self.btype = btype
        self.cutoff = cutoff
        self.order = order
        self.ripple = ripple

    @override
    def compute_coefficients(self) -> None:
        assert self.fs is not None

        b, a = cheby2(  # type: ignore
            self.order,
            self.ripple,
            self.cutoff / (0.5 * self.fs),
            btype=self.btype,
        )
        self.b = b
        self.a = a


class HiChebyshev1(Chebyshev1):
    """High-pass Chebyshev type 1 filter."""

    def __init__(
        self,
        cutoff: float,
        order: int = 4,
        ripple: float = 0.1,
        fs: int | None = None,
    ) -> None:
        super().__init__("highpass", cutoff, order, ripple, fs)


class LoChebyshev1(Chebyshev1):
    """Low-pass Chebyshev type 1 filter."""

    def __init__(
        self,
        cutoff: float,
        order: int = 4,
        ripple: float = 0.1,
        fs: int | None = None,
    ) -> None:
        super().__init__("lowpass", cutoff, order, ripple, fs)


class HiChebyshev2(Chebyshev2):
    """High-pass Chebyshev type 2 filter."""

    def __init__(
        self,
        cutoff: float,
        order: int = 4,
        ripple: float = 0.1,
        fs: int | None = None,
    ) -> None:
        super().__init__("highpass", cutoff, order, ripple, fs)


class LoChebyshev2(Chebyshev2):
    """Low-pass Chebyshev type 2 filter."""

    def __init__(
        self,
        cutoff: float,
        order: int = 4,
        ripple: float = 0.1,
        fs: int | None = None,
    ) -> None:
        super().__init__("lowpass", cutoff, order, ripple, fs)


class HiButterworth(Butterworth):
    """High-pass filter."""

    def __init__(
        self,
        cutoff: float,
        order: int = 5,
        order_scale: FilterOrderScale = "linear",
        fs: int | None = None,
    ) -> None:
        super().__init__("highpass", cutoff, order, order_scale, fs)


class LoButterworth(Butterworth):
    """Low-pass filter."""

    def __init__(
        self,
        cutoff: float,
        order: int = 5,
        order_scale: FilterOrderScale = "linear",
        fs: int | None = None,
    ) -> None:
        super().__init__("lowpass", cutoff, order, order_scale, fs)


class Shelving(IIR):
    """Shelving filter."""

    q: float

    @property
    def _omega(self) -> float:
        if self.fs is None:
            raise ValueError(NONE_FS_ERR)
        return 2 * np.pi * self.cutoff / self.fs

    @property
    def _alpha(self) -> float:
        return np.sin(self._omega) / (2 * self.q)


class HiShelving(Shelving):
    """High pass shelving filter."""

    gain: float

    def __init__(
        self,
        cutoff: float,
        q: float,
        gain: float,
        gain_scale: FilterOrderScale = "linear",
        fs: int | None = None,
    ):
        super().__init__(fs)
        self.cutoff = cutoff
        self.q = q
        self.gain = gain if gain_scale == "linear" else 10 ** (gain / 20)

    @override
    def compute_coefficients(self) -> None:
        A = self.gain  # noqa: N806
        b0 = A * (
            (A + 1) + (A - 1) * np.cos(self._omega) + 2 * np.sqrt(A) * self._alpha
        )
        b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(self._omega))
        b2 = A * (
            (A + 1) + (A - 1) * np.cos(self._omega) + 2 * np.sqrt(A) * self._alpha
        )

        a0 = (A + 1) - (A - 1) * np.cos(self._omega) + 2 * np.sqrt(A) * self._alpha
        a1 = 2 * ((A - 1) - (A + 1) * np.cos(self._omega))
        a2 = (A + 1) - (A - 1) * np.cos(self._omega) - 2 * np.sqrt(A) * self._alpha

        b = [b0 / a0, b1 / a0, b2 / a0]
        a = [1.0, a1 / a0, a2 / a0]

        self.b = b
        self.a = a


class LoShelving(Shelving): ...


class Peaking(IIR):
    """Peaking filter."""

    def __init__(
        self,
        cutoff: float,
        Q: float,
        gain: float,
        gain_scale: FilterOrderScale,
        fs: int | None = None,
    ) -> None:
        super().__init__(fs)
        self.cutoff = cutoff
        self.Q = Q
        self.gain = gain if gain_scale == "linear" else 10 ** (gain / 20)

    @override
    def compute_coefficients(self) -> None:
        assert self.fs is not None

        b, a = iirpeak(self.cutoff / (self.fs / 2), self.Q, self.fs)
        self.b = b
        self.a = a


class Notch(IIR):
    """Notch filter."""

    def __init__(
        self,
        cutoff: float,
        Q: float,
        fs: int | None = None,
    ) -> None:
        super().__init__(fs)
        self.cutoff = cutoff
        self.Q = Q

    @override
    def compute_coefficients(self) -> None:
        assert self.fs is not None

        b, a = iirnotch(self.cutoff / (self.fs / 2), self.Q)
        self.b = b
        self.a = a


class AllPass(IIR):
    """All pass filter."""

    def __init__(
        self,
        cutoff: float,
        Q: float,
        fs: int | None = None,
    ) -> None:
        super().__init__(fs)
        self.cutoff = cutoff
        self.Q = Q

    @override
    def compute_coefficients(self) -> None:
        assert self.fs is not None

        b, a = iirpeak(self.cutoff / (self.fs / 2), self.Q)
        self.b = b
        self.a = a
