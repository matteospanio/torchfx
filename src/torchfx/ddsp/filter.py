"""Trainable filter modules (DDSP).

These are ``nn.Module``s whose design parameters are ``nn.Parameter``s. They filter
in the time domain via :func:`differentiable_sos_cascade` (native forward + analytic
backward), so gradients flow into the parameters and they drop into any PyTorch
training loop. Once trained, call :meth:`LearnableFilter.freeze` to bake the
coefficients into a deterministic, ``@torch.no_grad()`` :class:`~torchfx.filter.SOSFilter`
for fast inference.

Extending
---------
``LearnableFilter`` is the extension point, mirroring the deterministic side:

================  =========================  ==========================
                  deterministic              trainable
================  =========================  ==========================
base class        ``filter.AbstractFilter``  ``ddsp.LearnableFilter``
implement         ``compute_coefficients``   ``sos``
coefficients      stored once (``_sos``)     returned fresh, on-tape
================  =========================  ==========================

To create a custom trainable filter, subclass :class:`LearnableFilter`, declare the
design parameters as ``nn.Parameter`` in ``__init__``, and implement :meth:`sos` to
return a *differentiable* ``[K, 6]`` SOS stack from those parameters (use the helpers
in :mod:`torchfx.ddsp.coeffs` or your own torch ops). ``forward`` and ``freeze`` come
for free.

"""

from __future__ import annotations

import abc
import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from ._autograd import differentiable_sos_cascade
from .coeffs import rbj_highpass_sos, rbj_lowpass_sos, rbj_peaking_sos

if TYPE_CHECKING:
    from torchfx.filter import SOSFilter


class LearnableFilter(nn.Module, abc.ABC):
    """Base class for trainable IIR filters.

    Subclass this, declare design parameters as ``nn.Parameter`` in ``__init__``, and
    implement :meth:`sos`. You get a differentiable :meth:`forward` (time-domain SOS
    cascade with a hand-derived analytic backward) and a :meth:`freeze` bridge to a
    deterministic filter for free.

    Parameters
    ----------
    fs : int, optional
        Sampling rate in Hz. Required before :meth:`forward`; pass it to the
        constructor (it is not auto-inferred from a ``Wave`` — train with tensors,
        then :meth:`freeze` for pipeline/inference use).

    Examples
    --------
    A custom one-pole-style trainable filter built from the cookbook helpers:

    >>> import torch, torch.nn as nn
    >>> from torchfx.ddsp import LearnableFilter
    >>> from torchfx.ddsp.coeffs import rbj_lowpass_sos
    >>> class MyLowpass(LearnableFilter):
    ...     def __init__(self, cutoff, fs):
    ...         super().__init__(fs)
    ...         self.log_cutoff = nn.Parameter(torch.tensor(float(cutoff)).log())
    ...     def sos(self):
    ...         return rbj_lowpass_sos(self.log_cutoff.exp(), torch.tensor(0.707), self._require_fs())
    >>> f = MyLowpass(1000.0, fs=48000)
    >>> f(torch.randn(1, 1024)).shape
    torch.Size([1, 1024])

    """

    def __init__(self, fs: int | None = None) -> None:
        super().__init__()
        self.fs = fs

    def _require_fs(self) -> float:
        """Return ``fs`` as a float, raising a clear error if it was never set."""
        if self.fs is None:
            raise ValueError(
                "Sample rate (fs) must be set before forward "
                "(pass fs= to the constructor or assign `.fs`)."
            )
        return float(self.fs)

    @abc.abstractmethod
    def sos(self) -> Tensor:
        """Return this filter as a differentiable ``[K, 6]`` (or ``[6]``) SOS stack.

        Must be built from ``nn.Parameter``s with plain torch ops so gradients flow
        back to the parameters.

        """

    def forward(self, x: Tensor) -> Tensor:
        return differentiable_sos_cascade(x, self.sos().reshape(-1, 6))

    @torch.no_grad()
    def freeze(self) -> SOSFilter:
        """Bake current parameters into a deterministic
        :class:`~torchfx.filter.SOSFilter`."""
        from torchfx.filter import SOSFilter

        sos = self.sos().detach().double().reshape(-1, 6)
        fs = int(self.fs) if self.fs is not None else None
        return SOSFilter(sos, fs=fs)


class LearnableLowpass(LearnableFilter):
    """Trainable second-order low-pass filter (learnable ``cutoff`` and ``q``).

    Parameters are reparameterised (log domain) so unconstrained optimisation keeps
    them positive and the poles inside the unit circle.

    Examples
    --------
    >>> import torch
    >>> from torchfx.ddsp import LearnableLowpass
    >>> filt = LearnableLowpass(cutoff=1000.0, fs=48000)
    >>> filt(torch.randn(1, 2048, requires_grad=True)).sum().backward()
    >>> det = filt.freeze()  # deterministic SOSFilter for inference

    """

    def __init__(self, cutoff: float, q: float = 0.707, fs: int | None = None) -> None:
        super().__init__(fs)
        self._log_cutoff = nn.Parameter(torch.tensor(float(cutoff)).log())
        self._log_q = nn.Parameter(torch.tensor(float(q)).log())

    @property
    def cutoff(self) -> Tensor:
        return self._log_cutoff.exp()

    @property
    def q(self) -> Tensor:
        return self._log_q.exp()

    def sos(self) -> Tensor:
        return rbj_lowpass_sos(self.cutoff, self.q, self._require_fs())


class LearnableHighpass(LearnableFilter):
    """Trainable second-order high-pass filter (learnable ``cutoff`` and ``q``)."""

    def __init__(self, cutoff: float, q: float = 0.707, fs: int | None = None) -> None:
        super().__init__(fs)
        self._log_cutoff = nn.Parameter(torch.tensor(float(cutoff)).log())
        self._log_q = nn.Parameter(torch.tensor(float(q)).log())

    @property
    def cutoff(self) -> Tensor:
        return self._log_cutoff.exp()

    @property
    def q(self) -> Tensor:
        return self._log_q.exp()

    def sos(self) -> Tensor:
        return rbj_highpass_sos(self.cutoff, self.q, self._require_fs())


class LearnablePeaking(LearnableFilter):
    """Trainable peaking-EQ band (learnable ``frequency``, ``q`` and ``gain_db``)."""

    def __init__(
        self,
        frequency: float,
        q: float = 1.0,
        gain_db: float = 0.0,
        fs: int | None = None,
    ) -> None:
        super().__init__(fs)
        self._log_freq = nn.Parameter(torch.tensor(float(frequency)).log())
        self._log_q = nn.Parameter(torch.tensor(float(q)).log())
        self.gain_db = nn.Parameter(torch.tensor(float(gain_db)))

    @property
    def frequency(self) -> Tensor:
        return self._log_freq.exp()

    @property
    def q(self) -> Tensor:
        return self._log_q.exp()

    def sos(self) -> Tensor:
        return rbj_peaking_sos(self.frequency, self.q, self.gain_db, self._require_fs())


class LearnableParametricEQ(LearnableFilter):
    """A cascade of peaking-EQ bands with learnable ``(frequency, Q, gain)`` per band.

    Raw parameters are unconstrained; sigmoid/exp/tanh maps keep effective values in
    stable, audible ranges so training cannot push a pole outside the unit circle.

    Examples
    --------
    >>> import torch
    >>> from torchfx.ddsp import LearnableParametricEQ
    >>> eq = LearnableParametricEQ(n_bands=8, fs=48000)
    >>> eq(torch.randn(1, 4096)).shape
    torch.Size([1, 4096])

    """

    def __init__(
        self,
        n_bands: int = 10,
        fs: float = 48_000,
        f_lo: float = 40.0,
        f_hi: float = 16_000.0,
        max_gain_db: float = 18.0,
    ) -> None:
        super().__init__(int(fs))
        self.f_lo, self.f_hi = f_lo, f_hi
        self.max_gain_db = max_gain_db
        grid = torch.linspace(0.0, 1.0, n_bands + 2)[1:-1]
        self._fc_raw = nn.Parameter(torch.logit(grid))
        self._q_raw = nn.Parameter(torch.zeros(n_bands))
        self.gain_db_raw = nn.Parameter(torch.zeros(n_bands))

    @property
    def fc(self) -> Tensor:
        """Band centers, log-spaced sigmoid map into ``[f_lo, f_hi]``."""
        t = torch.sigmoid(self._fc_raw)
        return self.f_lo * (self.f_hi / self.f_lo) ** t

    @property
    def q(self) -> Tensor:
        """Band Q, exp map around ``sqrt(2)``, clamped to ``[0.3, 8]``."""
        return torch.clamp(math.sqrt(2.0) * torch.exp(self._q_raw), 0.3, 8.0)

    @property
    def gain_db(self) -> Tensor:
        """Band gain in dB, tanh-bounded to ``+/- max_gain_db``."""
        return self.max_gain_db * torch.tanh(self.gain_db_raw)

    def sos(self) -> Tensor:
        return rbj_peaking_sos(self.fc, self.q, self.gain_db, self._require_fs())
