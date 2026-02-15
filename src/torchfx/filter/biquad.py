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
from torchaudio import functional as F  # noqa: N812
from typing_extensions import override

from torchfx.filter.__base import AbstractFilter
from torchfx.typing import Device


class Biquad(AbstractFilter):
    """Abstract base class for second-order IIR (biquad) filters.

    A biquad filter is defined by five normalized coefficients
    (``b0, b1, b2, a1, a2``). The denominator coefficient ``a0`` is always
    normalized to 1. Filter state (two samples of input and output history
    per channel) is maintained as registered buffers so that it moves
    automatically with ``.to(device)`` calls.

    Two processing paths are provided:

    * **Stateless (fast)**: delegates to ``torchaudio.functional.lfilter``
      when no chunk-to-chunk state continuity is needed. This is the default
      for one-shot ``wave | filter`` usage.
    * **Stateful (DF1 loop)**: a pure-PyTorch sample-by-sample loop
      (vectorized across channels) that carries state between calls.
      Activated after calling ``forward()`` — subsequent calls reuse state.
      Call ``reset_state()`` to clear accumulated state.

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
        Numerator coefficients ``[b0, b1, b2]`` after ``compute_coefficients()``.
    a : Tensor | None
        Denominator coefficients ``[1, a1, a2]`` after ``compute_coefficients()``.

    Notes
    -----
    Direct Form 1 processing is ported from AudioNoise ``biquad.h``
    ``biquad_step_df1``.

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

        # Coefficients (set by compute_coefficients)
        self.a: Tensor | None = None
        self.b: Tensor | None = None

        # Per-channel DF1 state: None until first stateful forward
        self._state_x: Tensor | None = None  # [C, 2]
        self._state_y: Tensor | None = None  # [C, 2]
        self._stateful: bool = False

    def move_coeff(
        self,
        device: Device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Move coefficients to the specified device and dtype."""
        if self.a is not None:
            self.a = self.a.to(device=torch.device(device), dtype=dtype)
        if self.b is not None:
            self.b = self.b.to(device=torch.device(device), dtype=dtype)

    def _set_coefficients(
        self,
        b0: float,
        b1: float,
        b2: float,
        a1: float,
        a2: float,
    ) -> None:
        """Store pre-normalized biquad coefficients as tensors.

        Parameters
        ----------
        b0, b1, b2 : float
            Numerator (feedforward) coefficients.
        a1, a2 : float
            Denominator (feedback) coefficients. ``a0`` is implicitly 1.

        """
        self.b = torch.tensor([b0, b1, b2], dtype=torch.float64)
        self.a = torch.tensor([1.0, a1, a2], dtype=torch.float64)

    @override
    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """Apply the biquad filter to the input tensor.

        On first call, coefficients are computed if needed and moved to the
        input device. Processing uses the stateless ``lfilter`` path for the
        first call; subsequent calls use the stateful DF1 loop for chunk
        continuity.

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

        if self.a is None or self.b is None:
            self.compute_coefficients()

        assert self.a is not None and self.b is not None

        device = x.device
        dtype = x.dtype

        if not isinstance(self.a, Tensor):
            self.a = torch.as_tensor(self.a, device=device, dtype=torch.float64)
            self.b = torch.as_tensor(self.b, device=device, dtype=torch.float64)
        elif self.a.device != device:
            self.a = self.a.to(device=device)
            self.b = self.b.to(device=device)

        if self._stateful:
            return self._forward_stateful(x)

        # Cast coefficients to input dtype for lfilter compatibility
        a_cast = self.a.to(dtype=dtype)
        b_cast = self.b.to(dtype=dtype)

        # First call: use fast lfilter path, then mark stateful for next calls
        result: Tensor = F.lfilter(x, a_cast, b_cast, clamp=False)
        result = result.to(dtype=dtype)

        # Bootstrap state from the last 2 samples for next chunk
        self._bootstrap_state(x, result)
        self._stateful = True

        return result

    def _bootstrap_state(self, x: Tensor, y: Tensor) -> None:
        """Initialize DF1 state from the tail of processed signals.

        Parameters
        ----------
        x : Tensor
            Input signal (original shape).
        y : Tensor
            Output signal (same shape as x).

        """
        # Ensure 2D [C, T] for state extraction
        if x.ndim == 1:
            x_2d = x.unsqueeze(0)
            y_2d = y.unsqueeze(0)
        elif x.ndim == 2:
            x_2d = x
            y_2d = y
        else:
            # [B, C, T] — use last batch element for state
            x_2d = x[-1]
            y_2d = y[-1]

        C = x_2d.shape[0]
        T = x_2d.shape[1]

        if T >= 2:
            # x[0] = x[n-1], x[1] = x[n-2] (DF1 convention)
            self._state_x = torch.stack([x_2d[:, T - 1], x_2d[:, T - 2]], dim=1).to(
                dtype=torch.float64
            )
            self._state_y = torch.stack([y_2d[:, T - 1], y_2d[:, T - 2]], dim=1).to(
                dtype=torch.float64
            )
        elif T == 1:
            self._state_x = torch.zeros(C, 2, device=x.device, dtype=torch.float64)
            self._state_y = torch.zeros(C, 2, device=x.device, dtype=torch.float64)
            self._state_x[:, 0] = x_2d[:, 0].to(torch.float64)
            self._state_y[:, 0] = y_2d[:, 0].to(torch.float64)

    def _forward_stateful(self, x: Tensor) -> Tensor:
        """Process with Direct Form 1 carrying state across calls.

        Vectorized across channels, sequential across time samples.
        Ported from AudioNoise ``biquad.h`` ``biquad_step_df1``.

        Uses a CUDA parallel prefix scan kernel when available and the input
        is on a CUDA device, falling back to the pure-PyTorch loop otherwise.

        """
        assert self.a is not None and self.b is not None

        orig_shape = x.shape
        dtype = x.dtype

        # Normalize to [C, T]
        if x.ndim == 1:
            x = x.unsqueeze(0)
        elif x.ndim == 3:
            B, C, T = x.shape
            x = x.reshape(B * C, T)

        C, T = x.shape
        device = x.device

        # Initialize or resize state
        if self._state_x is None or self._state_x.shape[0] != C:
            self._state_x = torch.zeros(C, 2, device=device, dtype=torch.float64)
            self._state_y = torch.zeros(C, 2, device=device, dtype=torch.float64)
        else:
            self._state_x = self._state_x.to(device=device)
            assert self._state_y is not None
            self._state_y = self._state_y.to(device=device)

        assert self._state_y is not None

        # Try native (CUDA or C++) kernel
        from torchfx._ops import biquad_forward

        native_result = biquad_forward(x, self.b, self.a, self._state_x, self._state_y)
        if native_result is not None:
            out, self._state_x, self._state_y = native_result
            result = out.to(dtype=dtype)
            if len(orig_shape) == 1:
                return result.squeeze(0)
            elif len(orig_shape) == 3:
                return result.reshape(orig_shape)
            return result

        # Pure-PyTorch fallback
        x_f64 = x.to(dtype=torch.float64)

        b0 = self.b[0]
        b1 = self.b[1]
        b2 = self.b[2]
        a1 = self.a[1]
        a2 = self.a[2]

        # state[:, 0] = x[n-1] / y[n-1]
        # state[:, 1] = x[n-2] / y[n-2]
        sx = self._state_x.clone()
        sy = self._state_y.clone()

        out = torch.empty_like(x_f64)

        for n in range(T):
            xn = x_f64[:, n]
            yn = b0 * xn + b1 * sx[:, 0] + b2 * sx[:, 1] - a1 * sy[:, 0] - a2 * sy[:, 1]
            out[:, n] = yn
            # Shift state
            sx[:, 1] = sx[:, 0]
            sx[:, 0] = xn
            sy[:, 1] = sy[:, 0]
            sy[:, 0] = yn

        # Store final state
        self._state_x = sx
        self._state_y = sy

        result = out.to(dtype=dtype)

        # Restore original shape
        if len(orig_shape) == 1:
            return result.squeeze(0)
        elif len(orig_shape) == 3:
            return result.reshape(orig_shape)
        return result

    def reset_state(self) -> None:
        """Clear accumulated filter state.

        After calling this method, the next ``forward()`` will use the fast
        ``lfilter`` path again. Call when switching audio sources or seeking.

        """
        self._state_x = None
        self._state_y = None
        self._stateful = False

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
