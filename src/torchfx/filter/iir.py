"""Infinite Impulse Response (IIR) filters for audio processing.

This module provides a comprehensive collection of IIR (Infinite Impulse Response)
filter implementations for digital audio processing. IIR filters are recursive
digital filters that provide efficient frequency-domain transformations with minimal
computational overhead.

All IIR filters inherit from the `IIR` base class, which provides lazy coefficient
computation and automatic device management. Filters can be combined in series using
the pipe operator (`|`) or in parallel using the addition operator (`+`).

Filter Types
------------
Classic Filters:
    - Butterworth: Maximally flat passband response
    - Chebyshev Type 1: Passband ripple, steeper rolloff
    - Chebyshev Type 2: Stopband ripple, monotonic passband
    - Elliptic: Equalized ripple in passband and stopband

Shelving Filters:
    - HiShelving: Boost/cut high frequencies
    - LoShelving: Boost/cut low frequencies

Parametric Filters:
    - ParametricEQ: Bell-shaped peaking filter
    - Peaking: General peaking filter
    - Notch: Narrow band rejection
    - AllPass: Phase shift without magnitude change

Crossover Filters:
    - LinkwitzRiley: Cascaded Butterworth for speaker crossovers

Examples
--------
Basic low-pass filtering:

>>> import torchfx as fx
>>> wave = fx.Wave.from_file("audio.wav")
>>> lpf = fx.filter.LoButterworth(cutoff=1000, order=4)
>>> filtered = wave | lpf

Chaining filters in series:

>>> result = (wave
...     | fx.filter.HiButterworth(cutoff=80, order=2)
...     | fx.filter.LoButterworth(cutoff=8000, order=4))

Parallel filter combination:

>>> parallel = (fx.filter.LoButterworth(1000, order=2) +
...             fx.filter.HiButterworth(200, order=2))
>>> result = wave | parallel

Notes
-----
IIR filters are computationally efficient due to their recursive structure,
requiring O(order) memory and operations per sample. However, they can exhibit
numerical instability for high-order designs. For linear-phase requirements,
consider FIR filters instead.

See Also
--------
torchfx.filter.fir : Finite Impulse Response filters
torchfx.filter.AbstractFilter : Base class for all filters

"""

import abc
import math
from collections.abc import Sequence

import numpy as np
import torch
from scipy.signal import butter, cheby1, cheby2, ellip, tf2sos
from torch import Tensor, dtype
from torchaudio import functional as F  # noqa: N812
from typing_extensions import override

from torchfx.filter.__base import AbstractFilter
from torchfx.filter.biquad import Biquad
from torchfx.typing import Device, FilterOrderScale

NONE_FS_ERR = "Sample rate of the filter could not be None."


class IIR(AbstractFilter):
    """Base class for Infinite Impulse Response (IIR) filters.

    This abstract class provides the foundation for all IIR filter implementations
    in torchfx. It implements lazy coefficient computation, automatic device
    management, and the core filtering operation.

    Two processing paths are provided:

    * **Stateless (fast)**: delegates to ``torchaudio.functional.lfilter`` for
      one-shot processing when no chunk-to-chunk state is needed.
    * **Stateful (SOS cascade)**: a pure-PyTorch second-order-sections (SOS) cascade
      using Direct Form 1 that carries state between calls. Activated after the
      first ``forward()`` call — subsequent calls reuse state. Call
      ``reset_state()`` to clear accumulated state.

    Higher-order IIR filters are internally decomposed into second-order sections
    (biquad cascade) for numerical stability.

    Attributes
    ----------
    fs : int | None
        The sampling frequency in Hz. Initially None until set from input signal
        or manually configured.
    cutoff : float
        The cutoff frequency in Hz. For different filter types, this may represent
        the -3dB point, center frequency, or transition frequency.
    a : Sequence[float] | Tensor | None
        Denominator (feedback) coefficients of the filter transfer function.
        None until computed via `compute_coefficients()`.
    b : Sequence[float] | Tensor | None
        Numerator (feedforward) coefficients of the filter transfer function.
        None until computed via `compute_coefficients()`.

    Notes
    -----
    Coefficient Computation Flow:
        1. Filter is instantiated with parameters (fs may be None)
        2. On first forward pass, if coefficients are None, compute_coefficients() is called
        3. Coefficients are converted to SOS matrix for stateful processing
        4. Filtering is performed using lfilter (stateless) or SOS cascade (stateful)

    See Also
    --------
    AbstractFilter : Base class for all filters
    torchfx.filter.fir.FIR : Base class for FIR filters
    torchfx.filter.biquad.Biquad : Biquad filter base class

    """

    fs: int | None
    cutoff: float
    a: Sequence[float] | Tensor | None
    b: Sequence[float] | Tensor | None

    @abc.abstractmethod
    def __init__(self, fs: int | None = None) -> None:
        super().__init__()
        self.fs = fs
        # SOS matrix: [num_sections, 6] each row [b0, b1, b2, 1, a1, a2]
        self._sos: Tensor | None = None
        # Per-section, per-channel DF1 state: [num_sections, C, 2] for x and y
        self._state_x: Tensor | None = None
        self._state_y: Tensor | None = None
        self._stateful: bool = False

    def _compute_sos(self) -> None:
        """Convert transfer function coefficients to SOS matrix.

        Uses ``scipy.signal.tf2sos`` to decompose higher-order transfer functions
        into cascaded second-order sections for numerical stability.

        """
        if self.a is None or self.b is None:
            return

        a_np = np.asarray(
            self.a if not isinstance(self.a, Tensor) else self.a.cpu().numpy(), dtype=np.float64
        )
        b_np = np.asarray(
            self.b if not isinstance(self.b, Tensor) else self.b.cpu().numpy(), dtype=np.float64
        )

        # For 2nd-order filters (3 coefficients), build SOS directly
        if len(a_np) <= 3 and len(b_np) <= 3:
            # Pad to length 3 if needed
            b_pad = np.zeros(3, dtype=np.float64)
            a_pad = np.zeros(3, dtype=np.float64)
            b_pad[: len(b_np)] = b_np
            a_pad[: len(a_np)] = a_np
            sos = np.array([[b_pad[0], b_pad[1], b_pad[2], a_pad[0], a_pad[1], a_pad[2]]])
        else:
            sos = tf2sos(b_np, a_np)

        self._sos = torch.from_numpy(sos)

    def move_coeff(self, device: Device, dtype: dtype = torch.float32) -> None:
        """Move the filter coefficients to the specified device and dtype."""
        self.a = torch.as_tensor(self.a, device=device, dtype=dtype)
        self.b = torch.as_tensor(self.b, device=device, dtype=dtype)

    @override
    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        device = x.device
        if self.fs is None:
            raise ValueError(NONE_FS_ERR)

        if self.a is None or self.b is None:
            self.compute_coefficients()

        if not isinstance(self.a, Tensor) or not isinstance(self.b, Tensor):
            self.move_coeff(device, dtype)

        # Stateful path: use SOS cascade in pure PyTorch
        if self._stateful:
            return self._forward_sos(x)

        # Fast stateless path via torchaudio lfilter
        result: Tensor = F.lfilter(x, self.a, self.b, clamp=False)
        result = result.to(dtype=dtype)

        # Build SOS and bootstrap state for next chunk
        if self._sos is None:
            self._compute_sos()
        self._bootstrap_state(x, result)
        self._stateful = True

        return result

    def _bootstrap_state(self, x: Tensor, y: Tensor) -> None:  # noqa: ARG002
        """Initialize SOS DF1 state from the tail of processed signals."""
        if self._sos is None:
            return

        device = x.device
        num_sections = self._sos.shape[0]

        # Get channel count
        if x.ndim == 1:
            C = 1
        elif x.ndim == 2:
            C = x.shape[0]
        else:
            C = x.shape[-2]  # [B, C, T]

        # We can't perfectly reconstruct intermediate SOS section states from
        # only input/output. Initialize to zeros — this introduces a tiny
        # transient on the first stateful chunk boundary, which is inaudible
        # for typical chunk sizes (>256 samples).
        self._state_x = torch.zeros(num_sections, C, 2, device=device, dtype=torch.float64)
        self._state_y = torch.zeros(num_sections, C, 2, device=device, dtype=torch.float64)

    def _forward_sos(self, x: Tensor) -> Tensor:
        """Process with SOS biquad cascade carrying state across calls.

        Each second-order section uses Direct Form 1, vectorized across channels
        and sequential across time.

        Ported from AudioNoise ``biquad.h`` ``biquad_step_df1``.

        """
        assert self._sos is not None

        orig_shape = x.shape
        out_dtype = x.dtype
        device = x.device

        # Normalize to [C, T]
        if x.ndim == 1:
            x = x.unsqueeze(0)
        elif x.ndim == 3:
            B, C, T = x.shape
            x = x.reshape(B * C, T)

        C, T = x.shape
        num_sections = self._sos.shape[0]

        sos = self._sos.to(device=device, dtype=torch.float64)

        # Initialize or resize state
        if self._state_x is None or self._state_x.shape[1] != C:
            self._state_x = torch.zeros(num_sections, C, 2, device=device, dtype=torch.float64)
            self._state_y = torch.zeros(num_sections, C, 2, device=device, dtype=torch.float64)
        else:
            self._state_x = self._state_x.to(device=device)
            assert self._state_y is not None
            self._state_y = self._state_y.to(device=device)

        assert self._state_y is not None

        section_input = x.to(dtype=torch.float64)

        for s in range(num_sections):
            b0 = sos[s, 0]
            b1 = sos[s, 1]
            b2 = sos[s, 2]
            # sos[s, 3] is a0 (== 1 after normalization from tf2sos)
            a1 = sos[s, 4]
            a2 = sos[s, 5]

            sx = self._state_x[s].clone()  # [C, 2]
            sy = self._state_y[s].clone()  # [C, 2]

            out = torch.empty_like(section_input)

            for n in range(T):
                xn = section_input[:, n]
                yn = b0 * xn + b1 * sx[:, 0] + b2 * sx[:, 1] - a1 * sy[:, 0] - a2 * sy[:, 1]
                out[:, n] = yn
                sx[:, 1] = sx[:, 0]
                sx[:, 0] = xn
                sy[:, 1] = sy[:, 0]
                sy[:, 0] = yn

            self._state_x[s] = sx
            self._state_y[s] = sy

            # Output of this section feeds next section
            section_input = out

        result = section_input.to(dtype=out_dtype)

        # Restore original shape
        if len(orig_shape) == 1:
            return result.squeeze(0)
        elif len(orig_shape) == 3:
            return result.reshape(orig_shape)
        return result

    def reset_state(self) -> None:
        """Reset filter state for chunk-to-chunk continuity.

        Call this when switching audio sources, seeking in a file, or after changing
        filter coefficients. After calling this method, the next ``forward()`` will
        use the fast ``lfilter`` path again.

        """
        self._state_x = None
        self._state_y = None
        self._stateful = False
        self._sos = None


class Butterworth(IIR):
    """Butterworth filter with maximally flat passband response.

    Butterworth filters provide the smoothest possible passband response with no
    ripple in either passband or stopband. They offer a good balance between
    frequency selectivity and phase linearity, making them ideal for general-purpose
    audio filtering applications.

    The filter has a monotonic frequency response with -3 dB gain at the cutoff
    frequency. The rolloff steepness is determined by the filter order, with higher
    orders providing sharper transitions at the cost of increased phase distortion.

    Parameters
    ----------
    btype : str
        Filter type. One of:
        - "lowpass": Passes frequencies below cutoff
        - "highpass": Passes frequencies above cutoff
        - "bandpass": Passes frequencies within a band
        - "bandstop": Rejects frequencies within a band
    cutoff : float
        Cutoff frequency in Hz. This is the -3 dB point for lowpass/highpass filters.
    order : int, default=4
        Filter order. Higher orders provide steeper rolloff but increased phase
        distortion. The rolloff rate is approximately 6 * order dB per octave.
        Typical values: 2-8 for audio applications.
    order_scale : {"linear", "db"}, default="linear"
        Order scaling mode:
        - "linear": Use order value directly
        - "db": Divide order by 6 (useful for octave-based specifications)
    fs : int | None, default=None
        Sampling frequency in Hz. If None, will be set automatically from the
        input signal.
    a : Sequence[float] | None, default=None
        Pre-computed denominator coefficients. If provided, coefficient computation
        is skipped.
    b : Sequence[float] | None, default=None
        Pre-computed numerator coefficients. If provided, coefficient computation
        is skipped.

    Attributes
    ----------
    btype : str
        The filter type.
    order : int
        The effective filter order (after scaling).

    Examples
    --------
    Low-pass filter at 1 kHz with 4th order:

    >>> import torchfx as fx
    >>> import torch
    >>> lpf = fx.filter.Butterworth(btype="lowpass", cutoff=1000, order=4, fs=44100)
    >>> wave = fx.Wave.from_file("audio.wav")
    >>> filtered = wave | lpf

    High-pass filter to remove subsonic frequencies:

    >>> hpf = fx.filter.Butterworth(btype="highpass", cutoff=20, order=2, fs=44100)
    >>> wave = fx.Wave.from_file("audio.wav")
    >>> clean = wave | hpf

    Comparing different orders (steeper rolloff with higher order):

    >>> # 2nd order: gentle 12 dB/octave rolloff
    >>> lpf2 = fx.filter.Butterworth(btype="lowpass", cutoff=1000, order=2, fs=44100)
    >>> # 8th order: steep 48 dB/octave rolloff
    >>> lpf8 = fx.filter.Butterworth(btype="lowpass", cutoff=1000, order=8, fs=44100)

    Notes
    -----
    Frequency Response Characteristics:
        - Passband: Maximally flat (no ripple)
        - Stopband: Monotonic rolloff, no ripple
        - -3 dB point: Exactly at cutoff frequency
        - Rolloff rate: ~6 * order dB/octave in stopband
        - Phase response: Nonlinear (increases with order)

    The Butterworth filter is designed to have a frequency response magnitude that
    is as flat as possible in the passband. The squared magnitude response is:

        |H(jω)|² = 1 / (1 + (ω/ωc)^(2n))

    where n is the filter order and ωc is the cutoff frequency.

    For audio applications, orders 2-4 are common for gentle filtering, while
    orders 6-8 provide sharper transitions. Very high orders (>12) may exhibit
    numerical instability.

    References
    ----------
    .. [1] Butterworth, S. (1930). "On the Theory of Filter Amplifiers".
           Wireless Engineer, 7: 536-541.

    See Also
    --------
    HiButterworth : High-pass Butterworth convenience class
    LoButterworth : Low-pass Butterworth convenience class
    Chebyshev1 : Type 1 Chebyshev filter with passband ripple
    Elliptic : Elliptic filter with optimized rolloff

    """

    def __init__(
        self,
        btype: str,
        cutoff: float,
        order: int = 4,
        order_scale: FilterOrderScale = "linear",
        fs: int | None = None,
        a: Sequence[float] | None = None,
        b: Sequence[float] | None = None,
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
    """Chebyshev Type I filter with equalized passband ripple.

    Chebyshev Type I filters trade passband ripple for steeper rolloff compared to
    Butterworth filters. They provide sharper transitions between passband and
    stopband for a given filter order, making them suitable when aggressive
    frequency selectivity is needed and some passband ripple is acceptable.

    The filter exhibits equalized ripple in the passband (oscillates between 1 and
    1-ε where ε is determined by the ripple parameter) and monotonic rolloff in the
    stopband.

    Parameters
    ----------
    btype : str
        Filter type. One of:
        - "lowpass": Passes frequencies below cutoff
        - "highpass": Passes frequencies above cutoff
        - "bandpass": Passes frequencies within a band
        - "bandstop": Rejects frequencies within a band
    cutoff : float
        Cutoff frequency in Hz. For lowpass/highpass, this is the frequency where
        the response exits the ripple band and enters the stopband.
    order : int, default=4
        Filter order. Determines both the number of ripples in the passband (order
        ripples) and the rolloff steepness. Higher orders provide steeper rolloff.
    ripple : float, default=0.1
        Maximum passband ripple in dB. Typical values:
        - 0.01 dB: Minimal ripple, closer to Butterworth response
        - 0.1 dB: Good balance (default)
        - 0.5 dB: Noticeable ripple but very sharp rolloff
        - 1.0 dB: Aggressive rolloff, significant passband distortion
    fs : int | None, default=None
        Sampling frequency in Hz. If None, will be set automatically from the
        input signal.
    a : Sequence[float] | None, default=None
        Pre-computed denominator coefficients.
    b : Sequence[float] | None, default=None
        Pre-computed numerator coefficients.

    Attributes
    ----------
    btype : str
        The filter type.
    order : int
        The filter order.
    ripple : float
        Maximum passband ripple in dB.

    Examples
    --------
    Low-pass filter with tight cutoff and minimal ripple:

    >>> import torchfx as fx
    >>> lpf = fx.filter.Chebyshev1(btype="lowpass", cutoff=5000, order=4,
    ...                            ripple=0.1, fs=44100)
    >>> wave = fx.Wave.from_file("audio.wav")
    >>> filtered = wave | lpf

    High-pass filter with aggressive rolloff:

    >>> hpf = fx.filter.Chebyshev1(btype="highpass", cutoff=100, order=6,
    ...                            ripple=0.5, fs=44100)
    >>> wave = fx.Wave.from_file("audio.wav")
    >>> filtered = wave | hpf

    Comparing ripple settings (more ripple = steeper rolloff):

    >>> # Minimal ripple: gentler rolloff, closer to Butterworth
    >>> cheby_gentle = fx.filter.Chebyshev1(btype="lowpass", cutoff=1000,
    ...                                     order=4, ripple=0.01, fs=44100)
    >>> # Higher ripple: steeper rolloff
    >>> cheby_steep = fx.filter.Chebyshev1(btype="lowpass", cutoff=1000,
    ...                                    order=4, ripple=1.0, fs=44100)

    Notes
    -----
    Frequency Response Characteristics:
        - Passband: Equalized ripple (oscillates between 0 and -ripple dB)
        - Stopband: Monotonic rolloff, no ripple
        - Transition: Sharper than Butterworth for same order
        - Phase response: More nonlinear than Butterworth

    The ripple parameter controls the trade-off between passband flatness and
    rolloff steepness. Smaller ripple values produce responses closer to Butterworth,
    while larger values provide sharper transitions at the cost of passband
    distortion.

    For audio applications where passband flatness is critical (e.g., mastering),
    use small ripple values (0.01-0.1 dB). For aggressive filtering where some
    passband coloration is acceptable (e.g., anti-aliasing), larger ripple values
    (0.5-1.0 dB) provide better stopband attenuation.

    The Chebyshev Type I response is defined by:

        |H(jω)|² = 1 / (1 + ε² Tₙ²(ω/ωc))

    where Tₙ is the nth-order Chebyshev polynomial and ε is related to the ripple.

    See Also
    --------
    HiChebyshev1 : High-pass Chebyshev Type I convenience class
    LoChebyshev1 : Low-pass Chebyshev Type I convenience class
    Chebyshev2 : Type II Chebyshev with stopband ripple
    Butterworth : Butterworth filter with no ripple
    Elliptic : Elliptic filter with ripple in both bands

    """

    def __init__(
        self,
        btype: str,
        cutoff: float,
        order: int = 4,
        ripple: float = 0.1,
        fs: int | None = None,
        a: Sequence[float] | None = None,
        b: Sequence[float] | None = None,
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

        b, a = cheby1(
            self.order,
            self.ripple,
            self.cutoff / (0.5 * self.fs),
            btype=self.btype,  # type: ignore
        )
        self.b = b
        self.a = a


class Chebyshev2(IIR):
    """Chebyshev Type II filter with equalized stopband ripple.

    Chebyshev Type II filters (also called inverse Chebyshev) have a monotonic
    passband response and equalized ripple in the stopband. They offer a compromise
    between Butterworth (no ripple anywhere) and Chebyshev Type I (passband ripple),
    providing sharper rolloff than Butterworth while maintaining a clean passband.

    The filter is characterized by a maximally flat passband (like Butterworth) but
    with oscillations in the stopband that allow for steeper rolloff at a given order.

    Parameters
    ----------
    btype : str
        Filter type. One of:
        - "lowpass": Passes frequencies below cutoff
        - "highpass": Passes frequencies above cutoff
        - "bandpass": Passes frequencies within a band
        - "bandstop": Rejects frequencies within a band
    cutoff : float
        Cutoff frequency in Hz. This is the frequency where the monotonic passband
        response transitions to the oscillating stopband.
    order : int, default=4
        Filter order. Higher orders provide steeper rolloff and more stopband
        oscillations.
    ripple : float, default=0.1
        Minimum stopband attenuation in dB. This specifies how much the stopband
        oscillates above the final attenuation level. Typical values:
        - 20 dB: Moderate stopband attenuation
        - 40 dB: Good stopband attenuation (common default)
        - 60 dB: Strong stopband attenuation
        - 80 dB: Very strong stopband rejection
    fs : int | None, default=None
        Sampling frequency in Hz. If None, will be set automatically from the
        input signal.

    Attributes
    ----------
    btype : str
        The filter type.
    order : int
        The filter order.
    ripple : float
        Minimum stopband attenuation in dB.

    Examples
    --------
    Low-pass filter with clean passband:

    >>> import torchfx as fx
    >>> lpf = fx.filter.Chebyshev2(btype="lowpass", cutoff=5000, order=4,
    ...                            ripple=40, fs=44100)
    >>> wave = fx.Wave.from_file("audio.wav")
    >>> filtered = wave | lpf

    High-pass filter with strong stopband rejection:

    >>> hpf = fx.filter.Chebyshev2(btype="highpass", cutoff=100, order=6,
    ...                            ripple=60, fs=44100)
    >>> wave = fx.Wave.from_file("audio.wav")
    >>> filtered = wave | hpf

    Anti-aliasing filter with very high stopband attenuation:

    >>> aa_filter = fx.filter.Chebyshev2(btype="lowpass", cutoff=18000,
    ...                                  order=8, ripple=80, fs=44100)
    >>> wave = fx.Wave.from_file("audio.wav")
    >>> filtered = wave | aa_filter

    Notes
    -----
    Frequency Response Characteristics:
        - Passband: Monotonic, maximally flat (no ripple)
        - Stopband: Equalized ripple, oscillates around final attenuation level
        - Transition: Sharper than Butterworth, less sharp than Type I Chebyshev
        - Phase response: Nonlinear, but often better than Type I

    Chebyshev Type II filters are ideal when you need:
        - Clean passband (no frequency-dependent gain variations)
        - Sharper rolloff than Butterworth
        - Acceptance of stopband ripple (usually inaudible for audio)

    The ripple parameter in Type II filters has opposite meaning from Type I:
    it specifies the minimum stopband attenuation rather than passband ripple.
    Higher ripple values mean better stopband rejection but require steeper
    transition bands.

    For audio applications, Type II is often preferred over Type I because:
        - Passband stays clean (no coloration of wanted frequencies)
        - Stopband ripple is usually irrelevant (already heavily attenuated)
        - Good compromise between Butterworth and Type I characteristics

    See Also
    --------
    HiChebyshev2 : High-pass Chebyshev Type II convenience class
    LoChebyshev2 : Low-pass Chebyshev Type II convenience class
    Chebyshev1 : Type I Chebyshev with passband ripple
    Butterworth : Butterworth filter with no ripple
    Elliptic : Elliptic filter with ripple in both bands

    """

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

        b, a = cheby2(
            self.order,
            self.ripple,
            self.cutoff / (0.5 * self.fs),
            btype=self.btype,  # type: ignore
        )
        self.b = b
        self.a = a


class HiChebyshev1(Chebyshev1):
    """High-pass Chebyshev Type I filter convenience class.

    This is a convenience class that creates a high-pass Chebyshev Type I filter
    by automatically setting btype="highpass". High-pass filters attenuate
    frequencies below the cutoff while passing higher frequencies.

    Parameters
    ----------
    cutoff : float
        Cutoff frequency in Hz. Frequencies below this are attenuated.
    order : int, default=4
        Filter order. Higher values provide steeper rolloff.
    ripple : float, default=0.1
        Maximum passband ripple in dB.
    fs : int | None, default=None
        Sampling frequency in Hz.

    Examples
    --------
    Remove low-frequency rumble:

    >>> import torchfx as fx
    >>> hpf = fx.filter.HiChebyshev1(cutoff=80, order=4, ripple=0.1, fs=44100)
    >>> wave = fx.Wave.from_file("recording.wav")
    >>> clean = wave | hpf

    See Also
    --------
    Chebyshev1 : Base Chebyshev Type I filter class
    LoChebyshev1 : Low-pass variant

    """

    def __init__(
        self,
        cutoff: float,
        order: int = 4,
        ripple: float = 0.1,
        fs: int | None = None,
    ) -> None:
        super().__init__("highpass", cutoff, order, ripple, fs)


class LoChebyshev1(Chebyshev1):
    """Low-pass Chebyshev Type I filter convenience class.

    This is a convenience class that creates a low-pass Chebyshev Type I filter
    by automatically setting btype="lowpass". Low-pass filters attenuate frequencies
    above the cutoff while passing lower frequencies.

    Parameters
    ----------
    cutoff : float
        Cutoff frequency in Hz. Frequencies above this are attenuated.
    order : int, default=4
        Filter order. Higher values provide steeper rolloff.
    ripple : float, default=0.1
        Maximum passband ripple in dB.
    fs : int | None, default=None
        Sampling frequency in Hz.

    Examples
    --------
    Anti-aliasing filter before downsampling:

    >>> import torchfx as fx
    >>> lpf = fx.filter.LoChebyshev1(cutoff=8000, order=6, ripple=0.5, fs=44100)
    >>> wave = fx.Wave.from_file("audio.wav")
    >>> filtered = wave | lpf

    See Also
    --------
    Chebyshev1 : Base Chebyshev Type I filter class
    HiChebyshev1 : High-pass variant

    """

    def __init__(
        self,
        cutoff: float,
        order: int = 4,
        ripple: float = 0.1,
        fs: int | None = None,
    ) -> None:
        super().__init__("lowpass", cutoff, order, ripple, fs)


class HiChebyshev2(Chebyshev2):
    """High-pass Chebyshev Type II filter convenience class.

    This is a convenience class that creates a high-pass Chebyshev Type II filter
    by automatically setting btype="highpass". Provides clean passband with
    stopband ripple.

    Parameters
    ----------
    cutoff : float
        Cutoff frequency in Hz. Frequencies below this are attenuated.
    order : int, default=4
        Filter order. Higher values provide steeper rolloff.
    ripple : float, default=0.1
        Minimum stopband attenuation in dB.
    fs : int | None, default=None
        Sampling frequency in Hz.

    Examples
    --------
    >>> import torchfx as fx
    >>> hpf = fx.filter.HiChebyshev2(cutoff=100, order=4, ripple=40, fs=44100)
    >>> wave = fx.Wave.from_file("audio.wav")
    >>> filtered = wave | hpf

    See Also
    --------
    Chebyshev2 : Base Chebyshev Type II filter class
    LoChebyshev2 : Low-pass variant

    """

    def __init__(
        self,
        cutoff: float,
        order: int = 4,
        ripple: float = 0.1,
        fs: int | None = None,
    ) -> None:
        super().__init__("highpass", cutoff, order, ripple, fs)


class LoChebyshev2(Chebyshev2):
    """Low-pass Chebyshev Type II filter convenience class.

    This is a convenience class that creates a low-pass Chebyshev Type II filter
    by automatically setting btype="lowpass". Provides clean passband with
    stopband ripple.

    Parameters
    ----------
    cutoff : float
        Cutoff frequency in Hz. Frequencies above this are attenuated.
    order : int, default=4
        Filter order. Higher values provide steeper rolloff.
    ripple : float, default=0.1
        Minimum stopband attenuation in dB.
    fs : int | None, default=None
        Sampling frequency in Hz.

    Examples
    --------
    >>> import torchfx as fx
    >>> lpf = fx.filter.LoChebyshev2(cutoff=5000, order=4, ripple=40, fs=44100)
    >>> wave = fx.Wave.from_file("audio.wav")
    >>> filtered = wave | lpf

    See Also
    --------
    Chebyshev2 : Base Chebyshev Type II filter class
    HiChebyshev2 : High-pass variant

    """

    def __init__(
        self,
        cutoff: float,
        order: int = 4,
        ripple: float = 0.1,
        fs: int | None = None,
    ) -> None:
        super().__init__("lowpass", cutoff, order, ripple, fs)


class HiButterworth(Butterworth):
    """High-pass Butterworth filter convenience class.

    This is a convenience class that creates a high-pass Butterworth filter
    by automatically setting btype="highpass". Provides maximally flat response
    in the passband with smooth rolloff.

    Parameters
    ----------
    cutoff : float
        Cutoff frequency in Hz (-3 dB point). Frequencies below this are attenuated.
    order : int, default=5
        Filter order. Determines rolloff steepness (~6*order dB/octave).
    order_scale : {"linear", "db"}, default="linear"
        Order scaling mode.
    fs : int | None, default=None
        Sampling frequency in Hz.

    Examples
    --------
    Remove DC offset and subsonic frequencies:

    >>> import torchfx as fx
    >>> hpf = fx.filter.HiButterworth(cutoff=20, order=2, fs=44100)
    >>> wave = fx.Wave.from_file("audio.wav")
    >>> clean = wave | hpf

    Remove low-frequency rumble with gentle slope:

    >>> hpf = fx.filter.HiButterworth(cutoff=80, order=4, fs=44100)
    >>> wave = fx.Wave.from_file("recording.wav")
    >>> filtered = wave | hpf

    See Also
    --------
    Butterworth : Base Butterworth filter class
    LoButterworth : Low-pass variant

    """

    def __init__(
        self,
        cutoff: float,
        order: int = 5,
        order_scale: FilterOrderScale = "linear",
        fs: int | None = None,
    ) -> None:
        super().__init__("highpass", cutoff, order, order_scale, fs)


class LoButterworth(Butterworth):
    """Low-pass Butterworth filter convenience class.

    This is a convenience class that creates a low-pass Butterworth filter
    by automatically setting btype="lowpass". Provides maximally flat response
    in the passband with smooth rolloff.

    Parameters
    ----------
    cutoff : float
        Cutoff frequency in Hz (-3 dB point). Frequencies above this are attenuated.
    order : int, default=5
        Filter order. Determines rolloff steepness (~6*order dB/octave).
    order_scale : {"linear", "db"}, default="linear"
        Order scaling mode.
    fs : int | None, default=None
        Sampling frequency in Hz.

    Examples
    --------
    Remove high-frequency noise:

    >>> import torchfx as fx
    >>> lpf = fx.filter.LoButterworth(cutoff=8000, order=4, fs=44100)
    >>> wave = fx.Wave.from_file("noisy.wav")
    >>> clean = wave | lpf

    Telephone bandwidth simulation:

    >>> lpf = fx.filter.LoButterworth(cutoff=3400, order=6, fs=44100)
    >>> wave = fx.Wave.from_file("voice.wav")
    >>> telephone = wave | lpf

    See Also
    --------
    Butterworth : Base Butterworth filter class
    HiButterworth : High-pass variant

    """

    def __init__(
        self,
        cutoff: float,
        order: int = 5,
        order_scale: FilterOrderScale = "linear",
        fs: int | None = None,
    ) -> None:
        super().__init__("lowpass", cutoff, order, order_scale, fs)


class Shelving(Biquad):
    """Base class for shelving filters.

    Shelving filters boost or cut frequencies above (high shelf) or below (low shelf)
    a specified cutoff frequency. Unlike parametric EQs that create a bell-shaped
    response, shelving filters affect all frequencies on one side of the cutoff with
    a constant gain.

    This base class provides common functionality for both high-shelf and low-shelf
    implementations, including the computation of angular frequency and alpha
    parameters used in the biquad filter equations.

    Inherits from ``Biquad`` — shelving filters are second-order IIR filters and
    benefit from the biquad's efficient stateless (lfilter) and stateful (DF1)
    processing paths.

    Parameters
    ----------
    cutoff : float
        Transition frequency in Hz. This is approximately where the gain reaches
        half of its final value (on a linear scale).
    q : float
        Quality factor controlling the transition steepness. Higher Q values result
        in steeper transitions. Typical values:
        - 0.5: Gentle, wide transition
        - 0.707: Moderate transition (common default)
        - 1.0: Steeper transition
        - 2.0+: Very steep transition
    fs : int | None, default=None
        Sampling frequency in Hz.

    Attributes
    ----------
    q : float
        The quality factor.
    _omega : float
        Angular frequency: 2π * cutoff / fs (computed property).
    _alpha : float
        Alpha parameter: sin(omega) / (2 * q) (computed property).

    See Also
    --------
    HiShelving : High-frequency shelving filter
    LoShelving : Low-frequency shelving filter
    ParametricEQ : Parametric bell-shaped filter

    """

    def __init__(
        self,
        cutoff: float,
        q: float,
        fs: int | None = None,
    ) -> None:
        super().__init__(cutoff=cutoff, q=q, fs=fs)

    @property
    def _omega(self) -> float:
        if self.fs is None:
            raise ValueError(NONE_FS_ERR)
        return 2.0 * math.pi * self.cutoff / self.fs

    @property
    def _alpha(self) -> float:
        return math.sin(self._omega) / (2.0 * self.q)


class HiShelving(Shelving):
    """High-frequency shelving filter for treble control.

    A high-shelving filter boosts or cuts frequencies above the cutoff frequency.
    This filter is commonly used in audio equalization to adjust treble/brightness,
    air frequencies, or apply de-emphasis curves. It provides smooth, gradual
    transitions similar to analog tone controls.

    The filter affects all frequencies above the cutoff with a constant gain in dB,
    creating a shelf-like frequency response. The transition steepness is controlled
    by the Q parameter.

    Parameters
    ----------
    cutoff : float
        Transition frequency in Hz. Frequencies above this will be boosted or cut.
        Common values:
        - 2000-4000 Hz: Presence boost/cut
        - 6000-8000 Hz: Brightness control
        - 10000-12000 Hz: Air/sparkle adjustment
    q : float
        Quality factor controlling transition steepness. Typical values:
        - 0.5: Gentle, analog-like transition
        - 0.707: Standard transition (Butterworth-like)
        - 1.0: Steeper transition
    gain : float
        Shelf gain amount. Interpretation depends on gain_scale:
        - For "linear": Linear gain factor (e.g., 2.0 for +6 dB)
        - For "db": Gain in dB (e.g., 6.0 for +6 dB boost, -3.0 for cut)
    gain_scale : {"linear", "db"}, default="linear"
        How to interpret the gain parameter:
        - "linear": Direct linear gain multiplier
        - "db": Gain in decibels (converted internally to linear)
    fs : int | None, default=None
        Sampling frequency in Hz.

    Attributes
    ----------
    gain : float
        The linear gain factor (converted from dB if necessary).

    Examples
    --------
    Boost high frequencies for brightness:

    >>> import torchfx as fx
    >>> shelf = fx.filter.HiShelving(cutoff=8000, q=0.707, gain=3, gain_scale="db", fs=44100)
    >>> wave = fx.Wave.from_file("dull_recording.wav")
    >>> bright = wave | shelf

    Cut harsh high frequencies:

    >>> shelf = fx.filter.HiShelving(cutoff=6000, q=0.5, gain=-4, gain_scale="db", fs=44100)
    >>> wave = fx.Wave.from_file("harsh.wav")
    >>> smooth = wave | shelf

    Add air to vocals:

    >>> air_shelf = fx.filter.HiShelving(cutoff=12000, q=0.707, gain=2, gain_scale="db", fs=44100)
    >>> vocals = fx.Wave.from_file("vocals.wav")
    >>> airy_vocals = vocals | air_shelf

    Using linear gain (2.0 = approximately +6 dB):

    >>> shelf = fx.filter.HiShelving(cutoff=10000, q=0.707, gain=2.0, gain_scale="linear", fs=44100)
    >>> wave = fx.Wave.from_file("audio.wav")
    >>> filtered = wave | shelf

    Notes
    -----
    High-shelving filters are one of the most commonly used EQ types in audio
    production. They're essential for:
        - Brightening dull recordings
        - Taming harsh cymbals or sibilance
        - Adding presence to vocals
        - Adjusting overall tonal balance

    The filter uses the Audio EQ Cookbook biquad equations, providing accurate
    analog-style shelving response. The Q parameter affects how quickly the
    transition occurs but doesn't affect the final shelf gain.

    For subtle, musical adjustments, use moderate Q values (0.5-0.707) and moderate
    gains (±3 to ±6 dB). Higher Q values create steeper transitions but may sound
    less natural.

    See Also
    --------
    LoShelving : Low-frequency shelving filter
    ParametricEQ : Parametric bell-shaped filter
    Shelving : Base class for shelving filters

    """

    gain: float

    def __init__(
        self,
        cutoff: float,
        q: float,
        gain: float,
        gain_scale: FilterOrderScale = "linear",
        fs: int | None = None,
    ):
        super().__init__(cutoff=cutoff, q=q, fs=fs)
        self.gain = gain if gain_scale == "linear" else 10 ** (gain / 20)

    @override
    def compute_coefficients(self) -> None:
        A = self.gain  # noqa: N806
        cos_w = math.cos(self._omega)
        sqrt_A = math.sqrt(A)
        alpha = self._alpha

        b0 = A * ((A + 1) + (A - 1) * cos_w + 2 * sqrt_A * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w)
        b2 = A * ((A + 1) + (A - 1) * cos_w - 2 * sqrt_A * alpha)

        a0 = (A + 1) - (A - 1) * cos_w + 2 * sqrt_A * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w)
        a2 = (A + 1) - (A - 1) * cos_w - 2 * sqrt_A * alpha

        self._set_coefficients(
            b0=b0 / a0,
            b1=b1 / a0,
            b2=b2 / a0,
            a1=a1 / a0,
            a2=a2 / a0,
        )


class LoShelving(Shelving):
    """Low-frequency shelving filter for bass control.

    A low-shelving filter boosts or cuts frequencies below the cutoff frequency.
    This filter is commonly used in audio equalization to adjust bass frequencies,
    add warmth, reduce muddiness, or apply bass management. It provides smooth,
    gradual transitions similar to analog bass/treble controls.

    The filter affects all frequencies below the cutoff with a constant gain in dB,
    creating a shelf-like frequency response. The transition steepness is controlled
    by the Q parameter.

    Parameters
    ----------
    cutoff : float
        Transition frequency in Hz. Frequencies below this will be boosted or cut.
        Common values:
        - 60-100 Hz: Sub-bass control
        - 100-200 Hz: Bass warmth/muddiness
        - 200-400 Hz: Low-mid body
        - 400-800 Hz: Fullness/boxiness
    q : float
        Quality factor controlling transition steepness. Higher Q values result in
        steeper transitions. Typical values:
        - 0.5: Gentle, wide transition (musical, analog-like)
        - 0.707: Moderate transition (common default, Butterworth-like)
        - 1.0: Steeper transition
        - 2.0+: Very steep transition
    gain : float
        Shelf gain amount. Interpretation depends on gain_scale:
        - For "linear": Linear gain factor (e.g., 2.0 for +6 dB)
        - For "db": Gain in dB (e.g., 6.0 for +6 dB boost, -3.0 for cut)
    gain_scale : {"linear", "db"}, default="linear"
        How to interpret the gain parameter:
        - "linear": Direct linear gain multiplier
        - "db": Gain in decibels (converted internally to linear)
    fs : int | None, default=None
        The sampling frequency in Hz.

    Attributes
    ----------
    gain : float
        The linear gain factor (converted from dB if necessary).

    Examples
    --------
    Add warmth by boosting low frequencies:

    >>> import torchfx as fx
    >>> shelf = fx.filter.LoShelving(cutoff=200, q=0.707, gain=4, gain_scale="db", fs=44100)
    >>> wave = fx.Wave.from_file("thin_recording.wav")
    >>> warm = wave | shelf

    Reduce muddiness by cutting low-mids:

    >>> shelf = fx.filter.LoShelving(cutoff=250, q=0.5, gain=-3, gain_scale="db", fs=44100)
    >>> wave = fx.Wave.from_file("muddy_mix.wav")
    >>> clean = wave | shelf

    Boost sub-bass for electronic music:

    >>> sub_shelf = fx.filter.LoShelving(cutoff=80, q=0.707, gain=5, gain_scale="db", fs=44100)
    >>> track = fx.Wave.from_file("edm_track.wav")
    >>> punchy = track | sub_shelf

    Using linear gain:

    >>> shelf = fx.filter.LoShelving(cutoff=150, q=0.707, gain=1.5, gain_scale="linear", fs=44100)
    >>> wave = fx.Wave.from_file("audio.wav")
    >>> filtered = wave | shelf

    Combining low and high shelving for complete tonal shaping:

    >>> lo_shelf = fx.filter.LoShelving(cutoff=200, q=0.707, gain=3, gain_scale="db", fs=44100)
    >>> hi_shelf = fx.filter.HiShelving(cutoff=8000, q=0.707, gain=-2, gain_scale="db", fs=44100)
    >>> wave = fx.Wave.from_file("audio.wav")
    >>> shaped = wave | lo_shelf | hi_shelf

    Notes
    -----
    Low-shelving filters are fundamental tools in audio production for:
        - Adding warmth and body to thin recordings
        - Reducing muddiness in the low-midrange
        - Bass management and room correction
        - Matching tonal balance between tracks
        - Creating vintage or "warm" sound characteristics

    The filter uses the Audio EQ Cookbook biquad equations, providing accurate
    analog-style shelving response. The Q parameter controls transition steepness
    but doesn't affect the final shelf gain.

    Common Applications:
        - Mastering: Subtle low-end adjustments (±2 to ±4 dB at 80-150 Hz)
        - Mixing: Controlling bass buildup (cuts at 100-300 Hz)
        - Sound design: Creating warmth or thickness
        - Broadcast: Meeting bass response standards

    For musical, natural-sounding results, use moderate Q values (0.5-0.707) and
    conservative gains (±3 to ±6 dB). Excessive low-frequency boost can cause
    distortion, speaker damage, or headroom issues.

    References
    ----------
    .. [1] Bristow-Johnson, R. "Cookbook formulae for audio EQ biquad filter
           coefficients." https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html

    See Also
    --------
    HiShelving : High-frequency shelving filter
    ParametricEQ : Parametric bell-shaped filter
    Shelving : Base class for shelving filters

    """

    gain: float

    def __init__(
        self,
        cutoff: float,
        q: float,
        gain: float,
        gain_scale: FilterOrderScale = "linear",
        fs: int | None = None,
    ):
        super().__init__(cutoff=cutoff, q=q, fs=fs)
        self.gain = gain if gain_scale == "linear" else 10 ** (gain / 20)

    @override
    def compute_coefficients(self) -> None:
        A = self.gain  # noqa: N806
        cos_w = math.cos(self._omega)
        sqrt_A = math.sqrt(A)
        alpha = self._alpha

        b0 = A * ((A + 1) - (A - 1) * cos_w + 2 * sqrt_A * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w)
        b2 = A * ((A + 1) - (A - 1) * cos_w - 2 * sqrt_A * alpha)

        a0 = (A + 1) + (A - 1) * cos_w + 2 * sqrt_A * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w)
        a2 = (A + 1) + (A - 1) * cos_w - 2 * sqrt_A * alpha

        self._set_coefficients(
            b0=b0 / a0,
            b1=b1 / a0,
            b2=b2 / a0,
            a1=a1 / a0,
            a2=a2 / a0,
        )


class ParametricEQ(Biquad):
    """Parametric equalizer with bell-shaped frequency response.

    A parametric EQ is a bell-shaped filter (also called a peaking filter) that
    boosts or cuts a specific frequency range. It's the most versatile and commonly
    used type of EQ in music production, audio mastering, and live sound reinforcement,
    providing precise control over frequency, bandwidth, and gain.

    Unlike shelving filters that affect all frequencies above or below a cutoff,
    parametric EQs create a localized boost or cut centered at a specific frequency,
    with the bandwidth controlled by the Q parameter. This makes them ideal for
    surgical frequency adjustments, tonal shaping, and corrective equalization.

    The filter is implemented using the Audio EQ Cookbook biquad equations,
    providing accurate analog-style peaking response.

    Parameters
    ----------
    frequency : float
        Center frequency in Hz where the peak or dip occurs. This is the frequency
        that receives the maximum boost or cut. Common applications:
        - 100-250 Hz: Body, warmth, or muddiness control
        - 250-500 Hz: Fullness or boxiness
        - 500-2000 Hz: Presence, honk, or nasality
        - 2000-5000 Hz: Clarity, definition, or harshness
        - 5000-10000 Hz: Brilliance, air, or sibilance
        - 10000+ Hz: Sparkle or extreme high-frequency detail
    q : float
        Quality factor (Q) controlling the bandwidth of the filter. Higher Q
        values result in narrower, more focused adjustments. The bandwidth in Hz
        is approximately frequency / Q. Typical values:
        - 0.3-0.5: Very wide, gentle, musical (subtle tonal shaping)
        - 0.7-1.0: Wide, natural (general tonal adjustments)
        - 1.0-2.0: Moderate, focused (typical mixing applications)
        - 2.0-5.0: Narrow, surgical (problem frequency removal)
        - 5.0-10.0: Very narrow (feedback suppression, notch-like)
        - 10.0+: Extremely narrow (surgical removal)
    gain : float
        Gain in dB at the center frequency. Positive values boost, negative
        values cut. Typical values:
        - ±1-2 dB: Subtle, transparent adjustments
        - ±3-6 dB: Moderate, audible changes
        - ±6-12 dB: Aggressive, obvious EQ
        - ±12+ dB: Extreme correction (use with caution)
    fs : int | None, default=None
        Sampling frequency in Hz.

    Attributes
    ----------
    cutoff : float
        The center frequency (stored internally).
    q : float
        The quality factor.
    gain_db : float
        The gain in dB.
    gain : float
        The linear gain factor (10^(gain_db/20)).

    Examples
    --------
    Boost presence at 3 kHz for vocal clarity:

    >>> import torchfx as fx
    >>> eq = fx.filter.ParametricEQ(frequency=3000, q=1.0, gain=4, fs=44100)
    >>> vocals = fx.Wave.from_file("vocals.wav")
    >>> clear_vocals = vocals | eq

    Cut muddy low-mids at 250 Hz:

    >>> eq = fx.filter.ParametricEQ(frequency=250, q=1.5, gain=-3, fs=44100)
    >>> mix = fx.Wave.from_file("mix.wav")
    >>> clean_mix = mix | eq

    Add warmth with wide boost at 120 Hz:

    >>> eq = fx.filter.ParametricEQ(frequency=120, q=0.5, gain=3, fs=44100)
    >>> bass = fx.Wave.from_file("bass.wav")
    >>> warm_bass = bass | eq

    Surgical removal of resonance at 800 Hz:

    >>> eq = fx.filter.ParametricEQ(frequency=800, q=5.0, gain=-6, fs=44100)
    >>> guitar = fx.Wave.from_file("guitar.wav")
    >>> smooth_guitar = guitar | eq

    Multiple EQ bands in series:

    >>> # Low-mid cut, presence boost, air boost
    >>> eq1 = fx.filter.ParametricEQ(frequency=200, q=1.0, gain=-2, fs=44100)
    >>> eq2 = fx.filter.ParametricEQ(frequency=3000, q=1.5, gain=3, fs=44100)
    >>> eq3 = fx.filter.ParametricEQ(frequency=10000, q=0.7, gain=2, fs=44100)
    >>> wave = fx.Wave.from_file("audio.wav")
    >>> eq_chain = wave | eq1 | eq2 | eq3

    Feedback suppression with very narrow Q:

    >>> # Identify feedback frequency (e.g., 2450 Hz), then apply narrow cut
    >>> notch_eq = fx.filter.ParametricEQ(frequency=2450, q=10.0, gain=-12, fs=44100)
    >>> live_mix = fx.Wave.from_file("live.wav")
    >>> controlled = live_mix | notch_eq

    Notes
    -----
    Q Factor and Bandwidth Relationship:
        - Bandwidth (Hz) ≈ frequency / Q
        - Bandwidth (octaves) ≈ 2 * sinh⁻¹(1/(2Q)) / ln(2)
        - Low Q (< 1): Wide bandwidth, affects many frequencies, musical
        - Medium Q (≈ 1): Moderate bandwidth, natural sound
        - High Q (> 2): Narrow bandwidth, focused correction, can sound unnatural

    Common Applications:
        - Vocal production: Boost 2-5 kHz for clarity, cut 200-400 Hz for muddiness
        - Kick drum: Boost 60-80 Hz for thump, 3-5 kHz for attack
        - Snare: Boost 200 Hz for body, 5 kHz for snap
        - Bass: Cut 200-300 Hz for clarity, boost 80-100 Hz for weight
        - Acoustic guitar: Boost 5-7 kHz for sparkle, cut 200-300 Hz for mud
        - Mix mastering: Subtle adjustments (±1-3 dB, Q=0.5-1.0)

    Best Practices:
        - Start with narrow Q to identify problem frequencies, then widen for musical result
        - Use cuts more than boosts (subtractive EQ philosophy)
        - Lower frequencies generally need wider Q (lower Q values)
        - Higher frequencies can use narrower Q without sounding unnatural
        - A/B compare frequently to avoid over-EQing
        - Use multiple gentle bands rather than one extreme adjustment

    The parametric EQ uses the following biquad equations from the Audio EQ Cookbook:

        ω₀ = 2π * frequency / fs
        α = sin(ω₀) / (2 * Q)
        A = 10^(gain/20)

        b₀ = 1 + α * A
        b₁ = -2 * cos(ω₀)
        b₂ = 1 - α * A
        a₀ = 1 + α / A
        a₁ = -2 * cos(ω₀)
        a₂ = 1 - α / A

    Caution: Very high Q values (>10) can cause ringing artifacts and may sound
    unnatural. Very large gain values (>12 dB) can introduce distortion and
    reduce headroom. Always monitor levels after applying EQ.

    References
    ----------
    .. [1] Bristow-Johnson, R. "Cookbook formulae for audio EQ biquad filter
           coefficients." https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html

    See Also
    --------
    Peaking : General peaking filter implementation
    HiShelving : High-frequency shelving filter
    LoShelving : Low-frequency shelving filter
    Notch : Notch filter for frequency rejection

    """

    def __init__(
        self,
        frequency: float,
        q: float,
        gain: float,
        fs: int | None = None,
    ) -> None:
        super().__init__(cutoff=frequency, q=q, fs=fs)
        self.gain_db = gain
        self.gain = 10 ** (gain / 20)

    @override
    def compute_coefficients(self) -> None:
        """Compute the filter coefficients using the Audio EQ Cookbook formulas."""
        assert self.fs is not None
        A = self.gain  # noqa: N806
        sin_w0, cos_w0, alpha = self._compute_omega_alpha(self.cutoff, self.q, self.fs)

        b0 = 1 + alpha * A
        b1 = -2 * cos_w0
        b2 = 1 - alpha * A

        a0 = 1 + alpha / A
        a1 = -2 * cos_w0
        a2 = 1 - alpha / A

        self._set_coefficients(
            b0=b0 / a0,
            b1=b1 / a0,
            b2=b2 / a0,
            a1=a1 / a0,
            a2=a2 / a0,
        )


class Peaking(ParametricEQ):
    """Peaking filter for narrow-band frequency adjustment.

    A simple wrapper around ``ParametricEQ`` that provides compatibility with
    the ``gain_scale`` parameter for specifying gain in linear or dB.

    For parametric EQ applications with more control, consider using
    ``ParametricEQ`` directly.

    Parameters
    ----------
    cutoff : float
        Center frequency in Hz where the peak or notch occurs.
    q : float
        Quality factor determining the bandwidth. Higher Q values result in
        narrower peaks. Bandwidth in Hz ≈ cutoff / Q.
    gain : float
        Peak gain. Interpretation depends on gain_scale:
        - For "linear": Linear gain factor
        - For "db": Gain in dB
    gain_scale : {"linear", "db"}
        How to interpret the gain parameter.
    fs : int | None, default=None
        Sampling frequency in Hz.

    Examples
    --------
    >>> import torchfx as fx
    >>> peak = fx.filter.Peaking(cutoff=1000, q=2.0, gain=6, gain_scale="db", fs=44100)
    >>> wave = fx.Wave.from_file("audio.wav")
    >>> filtered = wave | peak

    See Also
    --------
    ParametricEQ : Full parametric EQ implementation

    """

    def __init__(
        self,
        cutoff: float,
        q: float,
        gain: float,
        gain_scale: FilterOrderScale,
        fs: int | None = None,
    ) -> None:
        gain_db = gain if gain_scale == "db" else 20 * math.log10(gain) if gain > 0 else 0
        super().__init__(frequency=cutoff, q=q, gain=gain_db, fs=fs)


class Notch(Biquad):
    """Notch filter for narrow-band frequency rejection.

    A notch filter (also called a band-stop or band-reject filter) attenuates a
    narrow band of frequencies centered around a specified frequency. This is ideal
    for removing tonal interference, hum, resonances, or feedback frequencies
    without affecting the surrounding spectrum.

    The filter creates a sharp dip in the frequency response at the notch frequency,
    with the width of the notch controlled by the Q parameter.

    Inherits from ``Biquad`` — notch filters are second-order IIR filters and
    benefit from the biquad's efficient processing paths.

    Parameters
    ----------
    cutoff : float
        Notch center frequency in Hz. This frequency will be maximally attenuated.
        Common applications:
        - 50 Hz: Mains hum (Europe, Australia)
        - 60 Hz: Mains hum (North America, Japan)
        - 100/120 Hz: Second harmonic of mains hum
        - Any resonant frequency or feedback tone
    q : float
        Quality factor determining the notch width. Higher Q values result in
        narrower notches that affect fewer adjacent frequencies. The notch
        bandwidth in Hz is approximately cutoff / Q. Typical values:
        - 5-10: Wide notch, affects broader frequency range
        - 10-30: Moderate notch (common for hum removal)
        - 30+: Very narrow notch, surgical removal
    fs : int | None, default=None
        Sampling frequency in Hz.

    Attributes
    ----------
    cutoff : float
        The notch center frequency.
    q : float
        The quality factor.

    Examples
    --------
    Remove 60 Hz mains hum:

    >>> import torchfx as fx
    >>> notch = fx.filter.Notch(cutoff=60, q=30, fs=44100)
    >>> wave = fx.Wave.from_file("humming_recording.wav")
    >>> clean = wave | notch

    Remove 50 Hz hum (European mains):

    >>> notch = fx.filter.Notch(cutoff=50, q=30, fs=44100)
    >>> wave = fx.Wave.from_file("recording.wav")
    >>> filtered = wave | notch

    Remove feedback frequency:

    >>> # First identify the feedback frequency (e.g., 2350 Hz)
    >>> notch = fx.filter.Notch(cutoff=2350, q=20, fs=44100)
    >>> wave = fx.Wave.from_file("live_recording.wav")
    >>> no_feedback = wave | notch

    Remove multiple hum harmonics:

    >>> # Remove 60 Hz fundamental and 120 Hz harmonic
    >>> notch1 = fx.filter.Notch(cutoff=60, q=30, fs=44100)
    >>> notch2 = fx.filter.Notch(cutoff=120, q=30, fs=44100)
    >>> wave = fx.Wave.from_file("humming.wav")
    >>> clean = wave | notch1 | notch2

    Wide notch for resonance control:

    >>> notch = fx.filter.Notch(cutoff=800, q=5, fs=44100)
    >>> wave = fx.Wave.from_file("resonant.wav")
    >>> controlled = wave | notch

    Notes
    -----
    Notch filters are essential tools for:
        - Removing mains hum (50/60 Hz) and harmonics
        - Eliminating feedback frequencies in live sound
        - Reducing room resonances and standing waves
        - Removing tonal interference (AC hum, electrical noise)
        - De-essing (though specialized de-essers are often better)

    The Q parameter is critical:
        - Too low: Notch is wide and affects too many frequencies, causing
          audible coloration
        - Too high: Notch may not be wide enough to fully remove the problem
          frequency, especially if it varies slightly

    For hum removal, Q values of 20-30 are typical. For surgical removal of
    specific tones, Q values up to 50-100 can be used, but be aware that very
    high Q filters can ring or become unstable.

    The notch filter is implemented using scipy.signal.iirnotch, which designs
    a second-order IIR notch filter with maximum attenuation at the specified
    frequency.

    Caution: Overuse of notch filters can make audio sound unnatural. Use
    sparingly and only when necessary. Consider addressing the source of the
    problem (ground loops, shielding, etc.) rather than filtering.

    See Also
    --------
    Peaking : Peaking filter (inverse of notch)
    ParametricEQ : Parametric EQ for boosting/cutting frequency bands
    Butterworth : Broader frequency control

    """

    def __init__(
        self,
        cutoff: float,
        q: float,
        fs: int | None = None,
    ) -> None:
        super().__init__(cutoff=cutoff, q=q, fs=fs)

    @override
    def compute_coefficients(self) -> None:
        """Compute notch filter coefficients.

        Uses the AudioNoise ``biquad.h`` ``_biquad_notch_filter`` formula:

        .. code-block:: c

            b0 = b2 = 1 / (1 + alpha)
            b1 = a1 = -2*cos(w0) / (1 + alpha)
            a2 = (1 - alpha) / (1 + alpha)

        References
        ----------
        .. [1] AudioNoise project, ``biquad.h``, ``_biquad_notch_filter``.

        """
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


class AllPass(Biquad):
    """All-pass filter for phase manipulation without magnitude change.

    An all-pass filter passes all frequencies with unity magnitude (no amplitude
    change) but introduces frequency-dependent phase shift. These filters are
    primarily used for phase correction, creating reverb/delay effects, and
    designing crossover networks with linear phase response.

    While the magnitude response is flat across all frequencies, the phase response
    varies with frequency, which can be used to align phase between different signal
    paths or create time-domain dispersion effects.

    Inherits from ``Biquad`` — all-pass filters are second-order IIR filters and
    benefit from the biquad's efficient processing paths.

    Parameters
    ----------
    cutoff : float
        Center frequency in Hz where maximum phase shift occurs. The phase shift
        is frequency-dependent, with maximum shift at this frequency.
    q : float
        Quality factor controlling the rate of phase change. Higher Q values result
        in more rapid phase transitions around the center frequency.
    fs : int | None, default=None
        Sampling frequency in Hz.

    Attributes
    ----------
    cutoff : float
        The center frequency.
    q : float
        The quality factor.

    Examples
    --------
    Phase correction for speaker alignment:

    >>> import torchfx as fx
    >>> allpass = fx.filter.AllPass(cutoff=1000, q=0.707, fs=44100)
    >>> wave = fx.Wave.from_file("audio.wav")
    >>> phase_shifted = wave | allpass

    Creating a phaser effect (using multiple all-pass stages):

    >>> ap1 = fx.filter.AllPass(cutoff=500, q=1.0, fs=44100)
    >>> ap2 = fx.filter.AllPass(cutoff=1000, q=1.0, fs=44100)
    >>> ap3 = fx.filter.AllPass(cutoff=2000, q=1.0, fs=44100)
    >>> wave = fx.Wave.from_file("guitar.wav")
    >>> phased = wave | ap1 | ap2 | ap3

    Notes
    -----
    All-pass filters have unique properties:
        - Magnitude response: Unity gain at all frequencies (0 dB)
        - Phase response: Frequency-dependent, non-linear
        - Applications: Phase correction, reverb design, time dispersion

    Common uses in audio:
        - Phase alignment in multi-driver speaker systems
        - Reverb algorithm building blocks
        - Creating vintage phaser/flanger effects
        - Group delay compensation
        - Hilbert transform implementation (90° phase shift)

    The all-pass filter is particularly useful in:
        - Crossover networks: Aligning phase between drivers
        - Reverb design: Creating dense reflections and diffusion
        - Effect chains: Adding subtle movement and depth
        - Room correction: Compensating for acoustic phase distortions

    Implementation note: This uses scipy.signal.iirpeak which creates a peaking
    filter structure. For a true all-pass filter, ensure the gain is set appropriately
    (typically handled internally).

    Caution: While all-pass filters don't change magnitude, they do affect the time-
    domain response and can introduce pre-ringing or post-ringing artifacts. Use
    judiciously in critical listening applications.

    See Also
    --------
    Peaking : Peaking filter with magnitude change
    LinkwitzRiley : Crossover filters with phase-aligned outputs

    """

    def __init__(
        self,
        cutoff: float,
        q: float,
        fs: int | None = None,
    ) -> None:
        super().__init__(cutoff=cutoff, q=q, fs=fs)

    @override
    def compute_coefficients(self) -> None:
        """Compute all-pass filter coefficients.

        Uses the AudioNoise ``biquad.h`` ``_biquad_allpass_filter`` formula:

        .. code-block:: c

            b0 = a2 = (1 - alpha) / (1 + alpha)
            b1 = a1 = -2*cos(w0) / (1 + alpha)
            b2 = 1  (= a0, the pre-normalization denominator)

        References
        ----------
        .. [1] AudioNoise project, ``biquad.h``, ``_biquad_allpass_filter``.

        """
        assert self.fs is not None

        _, cos_w0, alpha = self._compute_omega_alpha(self.cutoff, self.q, self.fs)
        a0_inv = 1.0 / (1.0 + alpha)

        b0 = (1.0 - alpha) * a0_inv
        b1 = -2.0 * cos_w0 * a0_inv

        self._set_coefficients(
            b0=b0,
            b1=b1,
            b2=1.0,
            a1=b1,
            a2=b0,
        )


class LinkwitzRiley(IIR):
    """Linkwitz-Riley crossover filter for speaker systems.

    Linkwitz-Riley filters are designed specifically for crossover applications where
    audio is split into multiple frequency bands for different speakers (e.g., woofer,
    tweeter). They are created by cascading two identical Butterworth filters, resulting
    in several desirable properties:

    1. -6 dB gain at the cutoff frequency for both high-pass and low-pass sections
    2. Flat magnitude response when high-pass and low-pass outputs are summed
    3. Zero phase difference between outputs at crossover frequency
    4. Complementary magnitude responses

    These properties make Linkwitz-Riley filters ideal for multi-way speaker systems,
    ensuring smooth transitions between drivers with no amplitude or phase anomalies.

    The filter order must be an even integer (2, 4, 8, etc.) because it's formed by
    cascading two Butterworth filters. The effective order is twice that of each
    Butterworth stage.

    Parameters
    ----------
    btype : str
        Filter type: "lowpass" or "highpass". For crossover applications, use
        matching pairs of low-pass and high-pass filters at the same frequency.
    cutoff : float
        Crossover frequency in Hz. This is where both the low-pass and high-pass
        filters will be at -6 dB. Common crossover frequencies:
        - 80-120 Hz: Subwoofer crossover
        - 200-500 Hz: Woofer-midrange crossover
        - 2000-3500 Hz: Midrange-tweeter crossover
    order : int, default=4
        Filter order. Must be a positive even integer (2, 4, 8, 12, etc.).
        Common orders:
        - 2nd order (12 dB/octave): Gentle slope, good phase
        - 4th order (24 dB/octave): Standard for many systems
        - 8th order (48 dB/octave): Steep slope for problem drivers
    order_scale : {"linear", "db"}, default="linear"
        Order scaling mode.
    fs : int | None, default=None
        Sampling frequency in Hz.

    Attributes
    ----------
    btype : str
        The filter type.
    order : int
        The filter order (must be even).

    Examples
    --------
    Two-way crossover at 2 kHz (woofer and tweeter):

    >>> import torchfx as fx
    >>> lowpass = fx.filter.LinkwitzRiley(btype="lowpass", cutoff=2000, order=4, fs=44100)
    >>> highpass = fx.filter.LinkwitzRiley(btype="highpass", cutoff=2000, order=4, fs=44100)
    >>> wave = fx.Wave.from_file("full_range.wav")
    >>> woofer_signal = wave | lowpass
    >>> tweeter_signal = wave | highpass
    >>> # When combined: woofer_signal + tweeter_signal = wave (flat response)

    Subwoofer crossover at 80 Hz:

    >>> sub_lpf = fx.filter.LoLinkwitzRiley(cutoff=80, order=4, fs=44100)
    >>> main_hpf = fx.filter.HiLinkwitzRiley(cutoff=80, order=4, fs=44100)
    >>> wave = fx.Wave.from_file("music.wav")
    >>> subwoofer = wave | sub_lpf
    >>> mains = wave | main_hpf

    Three-way crossover system:

    >>> # Low: <200 Hz, Mid: 200-3000 Hz, High: >3000 Hz
    >>> low = fx.filter.LoLinkwitzRiley(cutoff=200, order=4, fs=44100)
    >>> mid_hp = fx.filter.HiLinkwitzRiley(cutoff=200, order=4, fs=44100)
    >>> mid_lp = fx.filter.LoLinkwitzRiley(cutoff=3000, order=4, fs=44100)
    >>> high = fx.filter.HiLinkwitzRiley(cutoff=3000, order=4, fs=44100)
    >>> wave = fx.Wave.from_file("audio.wav")
    >>> woofer = wave | low
    >>> midrange = wave | mid_hp | mid_lp
    >>> tweeter = wave | high

    Steep 8th-order crossover for difficult driver pairing:

    >>> lpf = fx.filter.LoLinkwitzRiley(cutoff=500, order=8, fs=44100)
    >>> hpf = fx.filter.HiLinkwitzRiley(cutoff=500, order=8, fs=44100)

    Notes
    -----
    Key Properties of Linkwitz-Riley Filters:
        - Magnitude sum: |H_LP|² + |H_HP|² = 1 (flat combined response)
        - Phase alignment: 0° difference at crossover frequency
        - Slope: 6 * order dB/octave (e.g., 24 dB/octave for 4th order)
        - Crossover point: -6 dB for both filters

    Advantages over other crossover designs:
        - No peaks or dips when signals are summed
        - Excellent phase coherence
        - Predictable, well-behaved response
        - Industry-standard for professional audio

    Common crossover orders:
        - 2nd order (LR2): 12 dB/octave, wide transition, good for close drivers
        - 4th order (LR4): 24 dB/octave, most popular, good all-around
        - 8th order (LR8): 48 dB/octave, steep isolation, for problem cases

    The filter is formed by cascading two Butterworth filters of half the specified
    order. For example, a 4th-order LR filter uses two 2nd-order Butterworth filters.
    This is implemented by convolving the Butterworth coefficients with themselves.

    Physical interpretation: The -6 dB crossover point means each driver contributes
    equally to the acoustic output at the crossover frequency, resulting in flat
    total response when properly aligned.

    References
    ----------
    .. [1] Linkwitz, S. H., & Riley, R. (1976). "Active Crossover Networks for
           Noncoincident Drivers." Journal of the Audio Engineering Society, 24(1), 2-8.

    See Also
    --------
    HiLinkwitzRiley : High-pass Linkwitz-Riley convenience class
    LoLinkwitzRiley : Low-pass Linkwitz-Riley convenience class
    Butterworth : Base Butterworth filter used in LR construction

    """

    def __init__(
        self,
        btype: str,
        cutoff: float,
        order: int = 4,
        order_scale: FilterOrderScale = "linear",
        fs: int | None = None,
    ) -> None:
        """Initialize the Linkwitz-Riley filter.

        Parameters
        ----------
        btype : str
            The type of filter, 'lowpass' or 'highpass'.
        cutoff : float
            The cutoff frequency in Hz.
        order : int
            The filter order. Must be a positive even integer (e.g., 2, 4, 8).
            Defaults to 4.
        fs : int | None
            The sampling frequency in Hz. Defaults to None.

        """
        super().__init__(fs)
        self.order = order if order_scale == "linear" else order // 6
        if order <= 0 or order % 2 != 0:
            raise ValueError("Linkwitz-Riley filter order must be a positive even integer.")
        self.btype = btype
        self.cutoff = cutoff
        self.order = order
        self.a = None
        self.b = None

    @override
    def compute_coefficients(self) -> None:
        """Compute the filter coefficients.

        The method calculates the coefficients for a Butterworth filter of half the
        specified order and then cascades it with itself by convolving the numerator and
        denominator coefficients.

        """
        assert self.fs is not None

        # An Nth-order Linkwitz-Riley filter is made from two (N/2)th-order Butterworth filters.
        butter_order = self.order // 2

        # Get the coefficients for the base Butterworth filter
        b_butter, a_butter = butter(butter_order, self.cutoff / (0.5 * self.fs), btype=self.btype)  # type: ignore

        # Cascade the filters by convolving the coefficients with themselves
        self.b = np.convolve(b_butter, b_butter)  # type: ignore
        self.a = np.convolve(a_butter, a_butter)  # type: ignore


class HiLinkwitzRiley(LinkwitzRiley):
    """High-pass Linkwitz-Riley crossover filter convenience class.

    This convenience class creates a high-pass Linkwitz-Riley filter by
    automatically setting btype="highpass". Used in crossover networks to
    send high frequencies to tweeters or mid-range drivers.

    Parameters
    ----------
    cutoff : float
        Crossover frequency in Hz (-6 dB point).
    order : int, default=4
        Filter order (must be even). Common values: 2, 4, 8.
    order_scale : {"linear", "db"}, default="linear"
        Order scaling mode.
    fs : int | None, default=None
        Sampling frequency in Hz.

    Examples
    --------
    Tweeter crossover at 3 kHz:

    >>> import torchfx as fx
    >>> hpf = fx.filter.HiLinkwitzRiley(cutoff=3000, order=4, fs=44100)
    >>> wave = fx.Wave.from_file("full_range.wav")
    >>> tweeter = wave | hpf

    See Also
    --------
    LinkwitzRiley : Base Linkwitz-Riley filter class
    LoLinkwitzRiley : Low-pass variant

    """

    def __init__(
        self,
        cutoff: float,
        order: int = 4,
        order_scale: FilterOrderScale = "linear",
        fs: int | None = None,
    ) -> None:
        super().__init__("highpass", cutoff, order, order_scale, fs)


class LoLinkwitzRiley(LinkwitzRiley):
    """Low-pass Linkwitz-Riley crossover filter convenience class.

    This convenience class creates a low-pass Linkwitz-Riley filter by
    automatically setting btype="lowpass". Used in crossover networks to
    send low frequencies to woofers or subwoofers.

    Parameters
    ----------
    cutoff : float
        Crossover frequency in Hz (-6 dB point).
    order : int, default=4
        Filter order (must be even). Common values: 2, 4, 8.
    order_scale : {"linear", "db"}, default="linear"
        Order scaling mode.
    fs : int | None, default=None
        Sampling frequency in Hz.

    Examples
    --------
    Subwoofer crossover at 80 Hz:

    >>> import torchfx as fx
    >>> lpf = fx.filter.LoLinkwitzRiley(cutoff=80, order=4, fs=44100)
    >>> wave = fx.Wave.from_file("music.wav")
    >>> subwoofer = wave | lpf

    See Also
    --------
    LinkwitzRiley : Base Linkwitz-Riley filter class
    HiLinkwitzRiley : High-pass variant

    """

    def __init__(
        self,
        cutoff: float,
        order: int = 4,
        order_scale: FilterOrderScale = "linear",
        fs: int | None = None,
    ) -> None:
        super().__init__("lowpass", cutoff, order, order_scale, fs)


class Elliptic(IIR):
    """Elliptic (Cauer) filter with optimal transition bandwidth.

    Elliptic filters (also called Cauer filters) achieve the sharpest possible
    transition between passband and stopband for a given filter order. They do this
    by allowing equalized ripple in both the passband and stopband, making them
    optimal when you need maximum frequency selectivity with minimum filter order.

    Of all classical IIR filter types, elliptic filters provide:
        - Steepest rolloff for a given order
        - Lowest order for given specifications
        - Most aggressive frequency separation

    However, they also have:
        - Ripple in both passband and stopband
        - Most nonlinear phase response
        - Potential for passband coloration

    Parameters
    ----------
    btype : str
        Filter type:
        - "lowpass": Passes frequencies below cutoff
        - "highpass": Passes frequencies above cutoff
        - "bandpass": Passes frequencies within a band
        - "bandstop": Rejects frequencies within a band
    cutoff : float
        Cutoff frequency in Hz. This is the edge of the passband where ripple
        transitions to stopband.
    order : int, default=4
        Filter order. For elliptic filters, even low orders provide steep rolloff.
        Typical values: 3-6 for most applications.
    passband_ripple : float, default=0.1
        Maximum passband ripple in dB. Controls the amount of gain variation
        allowed in the passband. Typical values:
        - 0.01 dB: Minimal ripple, closer to Chebyshev Type II
        - 0.1 dB: Good balance (default)
        - 0.5 dB: More aggressive rolloff, noticeable ripple
        - 1.0 dB: Very steep rolloff, significant passband distortion
    stopband_attenuation : float, default=40
        Minimum stopband attenuation in dB. Specifies how much the stopband is
        attenuated below 0 dB. Typical values:
        - 20 dB: Light attenuation
        - 40 dB: Good attenuation (default)
        - 60 dB: Strong attenuation
        - 80+ dB: Very strong rejection
    fs : int | None, default=None
        Sampling frequency in Hz.

    Attributes
    ----------
    btype : str
        The filter type.
    order : int
        The filter order.
    passband_ripple : float
        Maximum passband ripple in dB.
    stopband_attenuation : float
        Minimum stopband attenuation in dB.

    Examples
    --------
    Steep low-pass filter for anti-aliasing:

    >>> import torchfx as fx
    >>> lpf = fx.filter.Elliptic(btype="lowpass", cutoff=18000, order=5,
    ...                          passband_ripple=0.1, stopband_attenuation=60,
    ...                          fs=44100)
    >>> wave = fx.Wave.from_file("audio.wav")
    >>> filtered = wave | lpf

    Aggressive high-pass filter with minimal order:

    >>> hpf = fx.filter.Elliptic(btype="highpass", cutoff=100, order=3,
    ...                          passband_ripple=0.5, stopband_attenuation=40,
    ...                          fs=44100)
    >>> wave = fx.Wave.from_file("recording.wav")
    >>> clean = wave | hpf

    Comparing ripple settings (more ripple = steeper rolloff):

    >>> # Conservative: minimal passband ripple
    >>> ellip_gentle = fx.filter.Elliptic(btype="lowpass", cutoff=5000, order=4,
    ...                                   passband_ripple=0.01, stopband_attenuation=40,
    ...                                   fs=44100)
    >>> # Aggressive: higher passband ripple for steeper transition
    >>> ellip_steep = fx.filter.Elliptic(btype="lowpass", cutoff=5000, order=4,
    ...                                  passband_ripple=1.0, stopband_attenuation=60,
    ...                                  fs=44100)

    Very low order filter (3rd order elliptic can match 6th order Butterworth):

    >>> ellip3 = fx.filter.Elliptic(btype="lowpass", cutoff=1000, order=3,
    ...                             passband_ripple=0.1, stopband_attenuation=40,
    ...                             fs=44100)

    Notes
    -----
    Frequency Response Characteristics:
        - Passband: Equalized ripple (oscillates between 0 and -ripple dB)
        - Stopband: Equalized ripple (oscillates around -attenuation dB)
        - Transition: Sharpest of all classical filters
        - Phase response: Highly nonlinear (worst of classical filters)

    Design Trade-offs:
        - Increasing passband_ripple → Steeper transition
        - Increasing stopband_attenuation → Steeper transition
        - Increasing order → Steeper transition and more ripples
        - All three parameters are interdependent in the design

    When to Use Elliptic Filters:
        ✓ Need steepest possible rolloff
        ✓ Filter order is constrained (computational resources)
        ✓ Phase linearity is not critical
        ✓ Can tolerate some passband and stopband ripple
        ✓ Anti-aliasing before downsampling
        ✓ Interference rejection with narrow transition bands

    When NOT to Use:
        ✗ Phase linearity is important (use FIR instead)
        ✗ Passband must be perfectly flat (use Butterworth)
        ✗ Critical listening applications (use Chebyshev Type II or Butterworth)

    Comparison with other filters (for same specifications):
        - Elliptic: Lowest order, steepest rolloff, ripple everywhere
        - Chebyshev I: Higher order, passband ripple only
        - Chebyshev II: Higher order, stopband ripple only
        - Butterworth: Highest order, no ripple, smoothest response

    The elliptic filter design is based on elliptic rational functions (Jacobi
    elliptic functions), which allow the equiripple behavior in both bands. The
    filter achieves optimal Chebyshev approximation in both passband and stopband
    simultaneously.

    Caution: Very high stopband attenuation requirements (>80 dB) combined with
    tight passband ripple (<0.1 dB) may result in numerical instability or require
    high filter orders.

    References
    ----------
    .. [1] Cauer, W., Mathis, W., & Pauli, R. (2000). "Life and Work of Wilhelm Cauer."
           Proceedings of the Fourteenth International Symposium of Mathematical
           Theory of Networks and Systems.

    See Also
    --------
    HiElliptic : High-pass elliptic convenience class
    LoElliptic : Low-pass elliptic convenience class
    Chebyshev1 : Type I Chebyshev with passband ripple only
    Chebyshev2 : Type II Chebyshev with stopband ripple only
    Butterworth : Butterworth filter with no ripple

    """

    def __init__(
        self,
        btype: str,
        cutoff: float,
        order: int = 4,
        passband_ripple: float = 0.1,
        stopband_attenuation: float = 40,
        fs: int | None = None,
    ) -> None:
        super().__init__(fs)
        self.btype = btype
        self.cutoff = cutoff
        self.order = order
        self.passband_ripple = passband_ripple
        self.stopband_attenuation = stopband_attenuation
        self.a = None
        self.b = None

    @override
    def compute_coefficients(self) -> None:
        assert self.fs is not None

        b, a = ellip(
            self.order,
            self.passband_ripple,
            self.stopband_attenuation,
            self.cutoff / (0.5 * self.fs),
            btype=self.btype,  # type: ignore
        )
        self.b = b
        self.a = a


class HiElliptic(Elliptic):
    """High-pass Elliptic filter convenience class.

    This convenience class creates a high-pass elliptic filter by automatically
    setting btype="highpass". Provides the steepest possible high-pass rolloff
    for a given filter order, with ripple in both passband and stopband.

    Parameters
    ----------
    cutoff : float
        Cutoff frequency in Hz. Frequencies below this are attenuated.
    order : int, default=4
        Filter order. Even low orders (3-4) provide steep rolloff.
    passband_ripple : float, default=0.1
        Maximum passband ripple in dB.
    stopband_attenuation : float, default=40
        Minimum stopband attenuation in dB.
    fs : int | None, default=None
        Sampling frequency in Hz.

    Examples
    --------
    Remove low frequencies with aggressive rolloff:

    >>> import torchfx as fx
    >>> hpf = fx.filter.HiElliptic(cutoff=100, order=4, passband_ripple=0.1,
    ...                            stopband_attenuation=60, fs=44100)
    >>> wave = fx.Wave.from_file("recording.wav")
    >>> clean = wave | hpf

    Low-order but steep filter:

    >>> hpf = fx.filter.HiElliptic(cutoff=80, order=3, passband_ripple=0.5,
    ...                            stopband_attenuation=40, fs=44100)
    >>> wave = fx.Wave.from_file("audio.wav")
    >>> filtered = wave | hpf

    See Also
    --------
    Elliptic : Base elliptic filter class
    LoElliptic : Low-pass variant
    HiChebyshev1 : High-pass with passband ripple only

    """

    def __init__(
        self,
        cutoff: float,
        order: int = 4,
        passband_ripple: float = 0.1,
        stopband_attenuation: float = 40,
        fs: int | None = None,
    ) -> None:
        super().__init__(
            "highpass",
            cutoff,
            order,
            passband_ripple,
            stopband_attenuation,
            fs,
        )


class LoElliptic(Elliptic):
    """Low-pass Elliptic filter convenience class.

    This convenience class creates a low-pass elliptic filter by automatically
    setting btype="lowpass". Provides the steepest possible low-pass rolloff
    for a given filter order, with ripple in both passband and stopband.

    Parameters
    ----------
    cutoff : float
        Cutoff frequency in Hz. Frequencies above this are attenuated.
    order : int, default=4
        Filter order. Even low orders (3-4) provide steep rolloff.
    passband_ripple : float, default=0.1
        Maximum passband ripple in dB.
    stopband_attenuation : float, default=40
        Minimum stopband attenuation in dB.
    fs : int | None, default=None
        Sampling frequency in Hz.

    Examples
    --------
    Steep anti-aliasing filter before downsampling:

    >>> import torchfx as fx
    >>> lpf = fx.filter.LoElliptic(cutoff=18000, order=5, passband_ripple=0.1,
    ...                            stopband_attenuation=60, fs=44100)
    >>> wave = fx.Wave.from_file("audio.wav")
    >>> filtered = wave | lpf

    Minimal order noise reduction:

    >>> lpf = fx.filter.LoElliptic(cutoff=8000, order=3, passband_ripple=0.5,
    ...                            stopband_attenuation=40, fs=44100)
    >>> wave = fx.Wave.from_file("noisy.wav")
    >>> clean = wave | lpf

    See Also
    --------
    Elliptic : Base elliptic filter class
    HiElliptic : High-pass variant
    LoChebyshev1 : Low-pass with passband ripple only

    """

    def __init__(
        self,
        cutoff: float,
        order: int = 4,
        passband_ripple: float = 0.1,
        stopband_attenuation: float = 40,
        fs: int | None = None,
    ) -> None:
        super().__init__(
            "lowpass",
            cutoff,
            order,
            passband_ripple,
            stopband_attenuation,
            fs,
        )
