"""Biquad (second-order IIR) filter implementations using pure PyTorch.

This module provides a reusable ``Biquad`` base class and six concrete biquad
filter types ported from the AudioNoise project's ``biquad.h`` coefficient
formulas. All filters operate on both CPU and GPU using pure PyTorch operations,
supporting stateful (chunk-to-chunk continuity) and stateless (one-shot) modes.

The biquad is the fundamental building block for audio equalization, crossover
networks, and many audio effects. Each biquad section implements a second-order
IIR transfer function::

    H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)

Processing uses Direct Form 1, which is numerically robust for floating-point:

    y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]

Classes
-------
Biquad
    Abstract base class for all biquad filters.
BiquadLPF
    Second-order low-pass filter.
BiquadHPF
    Second-order high-pass filter.
BiquadNotch
    Notch (band-reject) filter.
BiquadBPF
    Band-pass filter (constant 0 dB peak gain).
BiquadBPFPeak
    Band-pass filter (peak gain = Q).
BiquadAllPass
    All-pass filter (phase shift only).

Notes
-----
Coefficient formulas are derived from the AudioNoise project's ``biquad.h``
(MIT License). See each class's docstring for the specific formula attribution.

The Direct Form 1 processing matches AudioNoise's ``biquad_step_df1``:

.. code-block:: c

    float out = c->b0*in + c->b1*x[0] + c->b2*x[1]
              - c->a1*y[0] - c->a2*y[1];
    x[1] = x[0]; x[0] = in;
    y[1] = y[0]; y[0] = out;

References
----------
.. [1] AudioNoise project, ``biquad.h``.
       https://github.com/torvalds/AudioNoise
.. [2] Bristow-Johnson, R. "Cookbook formulae for audio EQ biquad filter
       coefficients." https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html

See Also
--------
torchfx.filter.iir : Higher-order IIR filters built on SOS biquad cascades.

"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from typing_extensions import override

from torchfx.filter.__base import AbstractFilter


class Biquad(AbstractFilter):
    """Abstract base class for second-order IIR (biquad) filters.

    A biquad filter is defined by five normalized coefficients
    (``b0, b1, b2, a1, a2``). The denominator coefficient ``a0`` is always
    normalized to 1. Internally, coefficients are stored as a single SOS
    (second-order section) row ``[b0, b1, b2, 1, a1, a2]`` and processed
    via the shared SOS cascade forward path.

    Processing uses Direct Form 1, which is numerically robust for floating-point:

        y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]

    Parameters
    ----------
    cutoff : float
        Cutoff or center frequency in Hz.
    q : float
        Quality factor (Q). Controls bandwidth or resonance.
    fs : int | None
        Sampling frequency in Hz. If ``None``, must be set before use.

    Attributes
    ----------
    cutoff : float
    q : float
    fs : int | None
    b : Tensor | None
        Numerator coefficients ``[b0, b1, b2]`` (read-only view into SOS).
    a : Tensor | None
        Denominator coefficients ``[1, a1, a2]`` (read-only view into SOS).

    Notes
    -----
    Coefficient formulas are derived from the AudioNoise project's ``biquad.h``
    (MIT License). The forward path delegates to ``_sos_cascade_forward`` from
    ``iir.py``, sharing the precompiled C++/CUDA SOS cascade kernel used by
    higher-order IIR filters.

    """

    def __init__(
        self,
        cutoff: float,
        q: float,
        fs: int | None = None,
    ) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.q = q
        self.fs = fs

        # SOS coefficients: [1, 6] tensor set by compute_coefficients()
        self._sos: Tensor | None = None
        self._sos_device_cache: Tensor | None = None

        # Per-section DF1 state: [K=1, C, 2], None until first forward
        self._state_x: Tensor | None = None
        self._state_y: Tensor | None = None

    @property
    def b(self) -> Tensor | None:
        """Numerator coefficients ``[b0, b1, b2]`` (read-only view into SOS)."""
        return self._sos[0, :3] if self._sos is not None else None

    @property
    def a(self) -> Tensor | None:
        """Denominator coefficients ``[1, a1, a2]`` (read-only view into SOS)."""
        if self._sos is None:
            return None
        return torch.tensor(
            [1.0, self._sos[0, 4].item(), self._sos[0, 5].item()],
            dtype=torch.float64,
        )

    def _set_coefficients(
        self,
        b0: float,
        b1: float,
        b2: float,
        a1: float,
        a2: float,
    ) -> None:
        """Store pre-normalized biquad coefficients as a single SOS row.

        Parameters
        ----------
        b0, b1, b2 : float
            Numerator (feedforward) coefficients.
        a1, a2 : float
            Denominator (feedback) coefficients. ``a0`` is implicitly 1.

        """
        self._sos = torch.tensor([[b0, b1, b2, 1.0, a1, a2]], dtype=torch.float64)
        self._sos_device_cache = None

    @override
    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """Apply the biquad filter to the input tensor.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape ``[T]``, ``[C, T]``, or ``[B, C, T]``.

        Returns
        -------
        Tensor
            Filtered output with the same shape as input.

        """
        if self.fs is None:
            raise ValueError("Sample rate (fs) must be set before filtering.")
        if self._sos is None:
            self.compute_coefficients()
            self._sos_device_cache = None
        assert self._sos is not None
        from torchfx.filter.iir import _sos_cascade_forward

        result, self._sos_device_cache, self._state_x, self._state_y = _sos_cascade_forward(
            x, self._sos, self._sos_device_cache, self._state_x, self._state_y
        )
        return result

    def reset_state(self) -> None:
        """Clear accumulated filter state.

        Call when switching audio sources or seeking to reset DF1 state.

        """
        self._state_x = None
        self._state_y = None
        self._sos_device_cache = None

    @staticmethod
    def _compute_omega_alpha(
        cutoff: float,
        q: float,
        fs: int,
    ) -> tuple[float, float, float]:
        """Compute angular frequency components for biquad design.

        Parameters
        ----------
        cutoff : float
            Frequency in Hz.
        q : float
            Quality factor.
        fs : int
            Sample rate in Hz.

        Returns
        -------
        tuple[float, float, float]
            ``(sin_w0, cos_w0, alpha)`` where ``w0 = 2*pi*cutoff/fs`` and
            ``alpha = sin(w0) / (2*Q)``.

        """
        w0 = 2.0 * math.pi * cutoff / fs
        sin_w0 = math.sin(w0)
        cos_w0 = math.cos(w0)
        alpha = sin_w0 / (2.0 * q)
        return sin_w0, cos_w0, alpha


class BiquadLPF(Biquad):
    """Second-order low-pass filter (biquad).

    Passes frequencies below the cutoff and attenuates frequencies above it
    with a 12 dB/octave rolloff.

    Coefficient formulas ported from AudioNoise ``biquad.h`` ``_biquad_lpf``:

    .. code-block:: c

        b1 = (1 - cos(w0)) / (1 + alpha)
        b0 = b2 = b1 / 2
        a1 = -2*cos(w0) / (1 + alpha)
        a2 = (1 - alpha) / (1 + alpha)

    Parameters
    ----------
    cutoff : float
        Cutoff frequency in Hz (-3 dB point).
    q : float
        Quality factor. 0.707 gives Butterworth (maximally flat) response.
    fs : int | None
        Sampling frequency in Hz.

    References
    ----------
    .. [1] AudioNoise project, ``biquad.h``, ``_biquad_lpf``.

    """

    @override
    def compute_coefficients(self) -> None:
        assert self.fs is not None
        _, cos_w0, alpha = self._compute_omega_alpha(self.cutoff, self.q, self.fs)
        a0_inv = 1.0 / (1.0 + alpha)
        b1 = (1.0 - cos_w0) * a0_inv
        self._set_coefficients(
            b0=b1 / 2.0,
            b1=b1,
            b2=b1 / 2.0,
            a1=-2.0 * cos_w0 * a0_inv,
            a2=(1.0 - alpha) * a0_inv,
        )


class BiquadHPF(Biquad):
    """Second-order high-pass filter (biquad).

    Passes frequencies above the cutoff and attenuates frequencies below it
    with a 12 dB/octave rolloff.

    Coefficient formulas ported from AudioNoise ``biquad.h`` ``_biquad_hpf``:

    .. code-block:: c

        b1 = (1 + cos(w0)) / (1 + alpha)
        b0 = b2 = b1 / 2
        b1 = -b1
        a1 = -2*cos(w0) / (1 + alpha)
        a2 = (1 - alpha) / (1 + alpha)

    Parameters
    ----------
    cutoff : float
        Cutoff frequency in Hz (-3 dB point).
    q : float
        Quality factor. 0.707 gives Butterworth response.
    fs : int | None
        Sampling frequency in Hz.

    References
    ----------
    .. [1] AudioNoise project, ``biquad.h``, ``_biquad_hpf``.

    """

    @override
    def compute_coefficients(self) -> None:
        assert self.fs is not None
        _, cos_w0, alpha = self._compute_omega_alpha(self.cutoff, self.q, self.fs)
        a0_inv = 1.0 / (1.0 + alpha)
        b1_raw = (1.0 + cos_w0) * a0_inv
        self._set_coefficients(
            b0=b1_raw / 2.0,
            b1=-b1_raw,
            b2=b1_raw / 2.0,
            a1=-2.0 * cos_w0 * a0_inv,
            a2=(1.0 - alpha) * a0_inv,
        )


class BiquadNotch(Biquad):
    """Notch (band-reject) filter (biquad).

    Creates a sharp dip at the center frequency, rejecting a narrow band while
    passing all other frequencies with unity gain.

    Coefficient formulas ported from AudioNoise ``biquad.h``
    ``_biquad_notch_filter``:

    .. code-block:: c

        b0 = b2 = 1 / (1 + alpha)
        b1 = a1 = -2*cos(w0) / (1 + alpha)
        a2 = (1 - alpha) / (1 + alpha)

    Parameters
    ----------
    cutoff : float
        Notch center frequency in Hz.
    q : float
        Quality factor. Higher Q gives a narrower notch.
    fs : int | None
        Sampling frequency in Hz.

    References
    ----------
    .. [1] AudioNoise project, ``biquad.h``, ``_biquad_notch_filter``.

    """

    @override
    def compute_coefficients(self) -> None:
        assert self.fs is not None
        _, cos_w0, alpha = self._compute_omega_alpha(self.cutoff, self.q, self.fs)
        a0_inv = 1.0 / (1.0 + alpha)
        common = -2.0 * cos_w0 * a0_inv
        self._set_coefficients(
            b0=a0_inv,
            b1=common,
            b2=a0_inv,
            a1=common,
            a2=(1.0 - alpha) * a0_inv,
        )


class BiquadBPF(Biquad):
    """Band-pass filter with constant 0 dB peak gain (biquad).

    Passes a band of frequencies centered at the cutoff with unity peak gain,
    regardless of Q. Bandwidth is approximately ``cutoff / Q``.

    Coefficient formulas ported from AudioNoise ``biquad.h`` ``_biquad_bpf``:

    .. code-block:: c

        b0 = alpha / (1 + alpha)
        b1 = 0
        b2 = -alpha / (1 + alpha)
        a1 = -2*cos(w0) / (1 + alpha)
        a2 = (1 - alpha) / (1 + alpha)

    Parameters
    ----------
    cutoff : float
        Center frequency in Hz.
    q : float
        Quality factor controlling bandwidth (BW ≈ cutoff / Q).
    fs : int | None
        Sampling frequency in Hz.

    References
    ----------
    .. [1] AudioNoise project, ``biquad.h``, ``_biquad_bpf``.

    """

    @override
    def compute_coefficients(self) -> None:
        assert self.fs is not None
        _, cos_w0, alpha = self._compute_omega_alpha(self.cutoff, self.q, self.fs)
        a0_inv = 1.0 / (1.0 + alpha)
        self._set_coefficients(
            b0=alpha * a0_inv,
            b1=0.0,
            b2=-alpha * a0_inv,
            a1=-2.0 * cos_w0 * a0_inv,
            a2=(1.0 - alpha) * a0_inv,
        )


class BiquadBPFPeak(Biquad):
    """Band-pass filter with peak gain = Q (biquad).

    Passes a band of frequencies centered at the cutoff. The peak gain at the
    center frequency equals Q (in linear scale). Useful for resonant effects.

    Coefficient formulas ported from AudioNoise ``biquad.h``
    ``_biquad_bpf_peak``:

    .. code-block:: c

        b0 = Q*alpha / (1 + alpha)
        b1 = 0
        b2 = -Q*alpha / (1 + alpha)
        a1 = -2*cos(w0) / (1 + alpha)
        a2 = (1 - alpha) / (1 + alpha)

    Parameters
    ----------
    cutoff : float
        Center frequency in Hz.
    q : float
        Quality factor. Also determines peak gain.
    fs : int | None
        Sampling frequency in Hz.

    References
    ----------
    .. [1] AudioNoise project, ``biquad.h``, ``_biquad_bpf_peak``.

    """

    @override
    def compute_coefficients(self) -> None:
        assert self.fs is not None
        _, cos_w0, alpha = self._compute_omega_alpha(self.cutoff, self.q, self.fs)
        a0_inv = 1.0 / (1.0 + alpha)
        self._set_coefficients(
            b0=self.q * alpha * a0_inv,
            b1=0.0,
            b2=-self.q * alpha * a0_inv,
            a1=-2.0 * cos_w0 * a0_inv,
            a2=(1.0 - alpha) * a0_inv,
        )


class BiquadAllPass(Biquad):
    """All-pass filter with unity magnitude (biquad).

    Passes all frequencies with unity gain but introduces frequency-dependent
    phase shift. Useful as building blocks for phasers and crossover networks.

    Coefficient formulas ported from AudioNoise ``biquad.h``
    ``_biquad_allpass_filter``:

    .. code-block:: c

        b0 = a2 = (1 - alpha) / (1 + alpha)
        b1 = a1 = -2*cos(w0) / (1 + alpha)
        b2 = 1  (= a0, which is 1 + alpha before normalization)

    Parameters
    ----------
    cutoff : float
        Center frequency in Hz (maximum phase shift occurs here).
    q : float
        Quality factor controlling the rate of phase transition.
    fs : int | None
        Sampling frequency in Hz.

    References
    ----------
    .. [1] AudioNoise project, ``biquad.h``, ``_biquad_allpass_filter``.

    """

    @override
    def compute_coefficients(self) -> None:
        assert self.fs is not None
        _, cos_w0, alpha = self._compute_omega_alpha(self.cutoff, self.q, self.fs)
        a0_inv = 1.0 / (1.0 + alpha)
        b0 = (1.0 - alpha) * a0_inv
        b1 = -2.0 * cos_w0 * a0_inv
        self._set_coefficients(
            b0=b0,
            b1=b1,
            b2=1.0,  # == a0 before normalization
            a1=b1,
            a2=b0,
        )
