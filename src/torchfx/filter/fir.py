"""Fir filters."""

import typing as tp

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import lfilter
from torch import Tensor, as_tensor
from torchaudio import functional as F  # noqa: N812
from typing_extensions import override

from torchfx.filter.__base import AbstractFilter


class FIRFilter(AbstractFilter):
    """A fir filter."""

    b: NDArray[tp.Any]
    a: NDArray[tp.Any]

    def __init__(self, b: ArrayLike, a: ArrayLike = [1.0]):  # noqa: B006
        super().__init__()
        self.b = np.array(b)
        self.a = np.array(a)

    @override
    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        device = x.device

        if len(self.b) != len(self.a):
            return as_tensor(
                lfilter(self.b, self.a, x.numpy()),
                dtype=dtype,
                device=device,
            )

        return F.lfilter(
            x,
            as_tensor(self.a, dtype=dtype, device=device),
            as_tensor(self.b, dtype=dtype, device=device),
        )
