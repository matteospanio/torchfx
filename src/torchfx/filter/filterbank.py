"""Logarithmic filter bank for perceptually-uniform frequency resolution.

This module provides the :class:`LogFilterBank` class, which generates a bank
of bandpass filters with logarithmically-spaced center frequencies. This spacing
matches human auditory perception, where frequency discrimination follows an
approximately logarithmic (octave-based) scale.

The design is inspired by the logarithmic frequency mapping patterns used in
the AudioNoise project (``pow2(lfo * octaves) * center_freq``), generalized
to a configurable N-band filter bank.

Applications
------------
- **Graphic EQ**: 31-band or 10-band equalizer with per-band gain control
- **Spectrum analyzer**: Compute per-band energy for visualization
- **Crossover networks**: Use adjacent band edges as crossover points
- **Multi-band processing**: Apply different effects to different frequency bands

Examples
--------
Create a 10-band filter bank spanning 20 Hz to 20 kHz:

>>> import torchfx as fx
>>> fb = fx.filter.LogFilterBank(n_bands=10, f_min=20.0, f_max=20000.0)
>>> wave = fx.Wave.from_file("audio.wav")
>>> bands = wave | fb  # Returns [10, C, T] tensor

Access center frequencies:

>>> fb.center_frequencies
[20.0, 42.2, 89.1, 188.0, ..., 20000.0]

"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from typing_extensions import override

from torchfx.filter.__base import AbstractFilter
from torchfx.filter.biquad import BiquadBPF


class LogFilterBank(AbstractFilter):
    r"""Bank of bandpass filters with logarithmic frequency spacing.

    Generates ``n_bands`` bandpass filters whose center frequencies are
    logarithmically spaced between ``f_min`` and ``f_max``::

        f[k] = f_min * 2^(k * log2(f_max / f_min) / (n_bands - 1))

    Each band uses a :class:`~torchfx.filter.biquad.BiquadBPF` with the
    specified quality factor ``q``.

    Parameters
    ----------
    n_bands : int
        Number of frequency bands. Must be >= 2.
    f_min : float
        Minimum center frequency in Hz. Default is 20.0.
    f_max : float
        Maximum center frequency in Hz. Default is 20000.0.
    q : float
        Quality factor for each bandpass filter. Higher Q gives narrower
        bands. Default is 1.414 (approximately sqrt(2), giving -3 dB
        bandwidth of about 1 octave).
    fs : int or None
        Sampling frequency in Hz. If ``None``, must be set before use
        (typically via pipeline operator with a ``Wave`` object).

    Attributes
    ----------
    n_bands : int
    f_min : float
    f_max : float
    q : float
    fs : int or None
    filters : list[BiquadBPF]
        The individual bandpass filters.

    Examples
    --------
    Basic usage:

    >>> fb = LogFilterBank(n_bands=31, f_min=20.0, f_max=20000.0)
    >>> wave = fx.Wave.from_file("audio.wav")
    >>> bands = wave | fb  # shape: [31, C, T]

    Use as a spectrum analyzer:

    >>> fb = LogFilterBank(n_bands=10, f_min=100.0, f_max=10000.0, fs=44100)
    >>> bands = fb(waveform)  # [10, C, T]
    >>> energy = (bands ** 2).mean(dim=-1)  # [10, C] per-band energy

    """

    def __init__(
        self,
        n_bands: int,
        f_min: float = 20.0,
        f_max: float = 20000.0,
        q: float = 1.414,
        fs: int | None = None,
    ) -> None:
        super().__init__()
        assert n_bands >= 2, "n_bands must be >= 2"
        assert 0 < f_min < f_max, "f_min must be positive and less than f_max"

        self.n_bands = n_bands
        self.f_min = f_min
        self.f_max = f_max
        self.q = q
        self._fs = fs

        # Compute logarithmically-spaced center frequencies
        # f[k] = f_min * 2^(k * octaves / (N-1))
        octaves = math.log2(f_max / f_min)
        self._center_freqs = [
            f_min * (2.0 ** (k * octaves / (n_bands - 1))) for k in range(n_bands)
        ]

        # Create bandpass filters
        self.filters = [BiquadBPF(cutoff=freq, q=q, fs=fs) for freq in self._center_freqs]

        # Coefficients are computed lazily per-filter
        self.a: Tensor | None = None
        self.b: Tensor | None = None

    @property
    def fs(self) -> int | None:
        return self._fs

    @fs.setter
    def fs(self, value: int | None) -> None:
        self._fs = value
        if value is not None:
            for f in self.filters:
                f.fs = value

    @property
    def center_frequencies(self) -> list[float]:
        """Return the center frequency of each band in Hz."""
        return list(self._center_freqs)

    @override
    def compute_coefficients(self) -> None:
        """Compute coefficients for all bandpass filters."""
        for f in self.filters:
            f.compute_coefficients()
        # Mark as computed
        self.a = torch.tensor([1.0])
        self.b = torch.tensor([1.0])

    @override
    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """Apply all bandpass filters to the input.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape ``[T]``, ``[C, T]``, or ``[B, C, T]``.

        Returns
        -------
        Tensor
            Output tensor of shape ``[n_bands, ...]`` where ``...`` matches
            the input shape. Each slice ``output[k]`` contains the input
            filtered through band ``k``.

        """
        if self._fs is None:
            raise ValueError("Sample rate (fs) must be set before filtering.")

        # Propagate fs to all filters
        for f in self.filters:
            if f.fs is None:
                f.fs = self._fs

        # Apply each filter and stack results
        outputs = [f(x) for f in self.filters]
        return torch.stack(outputs, dim=0)
