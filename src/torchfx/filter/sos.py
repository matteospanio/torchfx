"""Deterministic filter wrapping a precomputed SOS matrix."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from torchfx.effect import FX
from torchfx.filter.iir import _sos_cascade_forward

if TYPE_CHECKING:
    from torch import Tensor


class SOSFilter(FX):
    """A deterministic filter carrying a precomputed ``[K, 6]`` SOS matrix.

    Unlike the designed ``IIR`` / ``Biquad`` filters, ``SOSFilter`` never *designs*
    anything — it holds its second-order sections directly and just runs the fast
    native SOS cascade. It is the inference-side counterpart of a trained
    :class:`torchfx.ddsp.LearnableFilter`: ``LearnableFilter.freeze()`` bakes its
    learned coefficients into one of these for ``@torch.no_grad()`` deployment
    (the "train differentiably → freeze → fast inference" path).

    Parameters
    ----------
    sos : Tensor
        Second-order sections of shape ``[K, 6]`` (``[b0, b1, b2, 1, a1, a2]`` rows).
        Stored canonically as CPU float64.
    fs : int, optional
        Sampling rate the sections were designed for (informational; the cascade
        itself does not need it).

    Examples
    --------
    >>> import torch
    >>> from torchfx.filter import SOSFilter
    >>> from torchfx.filter.iir import LoButterworth
    >>> sos = LoButterworth(cutoff=800, order=4, fs=48000)
    >>> sos.compute_coefficients()
    >>> f = SOSFilter(sos._sos, fs=48000)
    >>> y = f(torch.randn(1, 1024))
    >>> y.shape
    torch.Size([1, 1024])

    """

    def __init__(self, sos: Tensor, fs: int | None = None) -> None:
        super().__init__()
        self.fs = fs
        self._sos = sos.detach().to(dtype=torch.float64).cpu()
        self._sos_device_cache: Tensor | None = None
        self._state_x: Tensor | None = None
        self._state_y: Tensor | None = None

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """Apply the SOS cascade, carrying DF1 state across chunked calls."""
        result, self._sos_device_cache, self._state_x, self._state_y = _sos_cascade_forward(
            x, self._sos, self._sos_device_cache, self._state_x, self._state_y
        )
        return result

    def reset_state(self) -> None:
        """Clear accumulated DF1 state and the device coefficient cache."""
        self._state_x = None
        self._state_y = None
        self._sos_device_cache = None
