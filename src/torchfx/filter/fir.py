"""FIR (Finite Impulse Response) filter implementations using efficient convolution.

This module provides FIR filter implementations for torchfx, offering always-stable
digital filtering with linear phase response. FIR filters are non-recursive (no feedback)
and are implemented using PyTorch's optimized `conv1d` operation for efficient processing
on both CPU and GPU.

FIR Filter Theory
-----------------
FIR filters are characterized by a finite-duration impulse response, meaning their output
depends only on current and past input samples (no feedback from past outputs). The
transfer function has the form:

    H(z) = b[0] + b[1]z^(-1) + b[2]z^(-2) + ... + b[N-1]z^(-(N-1))

where b[0], b[1], ..., b[N-1] are the filter coefficients (taps) and N is the filter order.

Key Advantages of FIR Filters
------------------------------
- **Always stable**: No poles outside the unit circle (denominator is always 1)
- **Linear phase**: Symmetric coefficients produce constant group delay across frequencies,
  preserving waveform shape without phase distortion
- **Finite precision**: Quantization errors don't accumulate due to lack of feedback
- **Flexible design**: Window method allows precise control of frequency response

Key Disadvantages Compared to IIR
----------------------------------
- **Higher computational cost**: Require more taps than IIR for equivalent frequency
  selectivity (typically 10-100x more coefficients)
- **Higher latency**: Group delay increases linearly with number of taps
- **More memory**: Need to store all tap coefficients

Window Method Design
--------------------
The `DesignableFIR` class uses the windowed Fourier series method (via `scipy.signal.firwin`)
to design filters. This approach:

1. Starts with ideal frequency response (infinite impulse response)
2. Truncates to finite length using a window function
3. Different windows provide different trade-offs between:
   - Main lobe width (transition band width)
   - Side lobe attenuation (stopband ripple)

Common window types and their characteristics:

    | Window     | Main Lobe Width | Sidelobe Attenuation | Best For                |
    |------------|-----------------|----------------------|-------------------------|
    | hamming    | Moderate        | -43 dB               | General purpose         |
    | hann       | Moderate        | -31 dB               | Smooth transitions      |
    | blackman   | Wide            | -58 dB               | High stopband rejection |
    | kaiser     | Adjustable      | Adjustable           | Custom specifications   |

FIR vs IIR Trade-offs
---------------------
**When to use FIR:**
- Linear phase response is critical (audio mastering, measurement systems)
- Stability is paramount (adaptive filtering, control systems)
- Finite precision arithmetic (fixed-point implementations)

**When to use IIR:**
- Computational efficiency is critical (real-time on limited hardware)
- Sharp frequency transitions needed with minimal latency
- Mimicking analog filter responses (Butterworth, Chebyshev, etc.)

Implementation Details
----------------------
Both `FIR` and `DesignableFIR` classes use PyTorch's `conv1d` for filtering:

- Coefficients are stored as a registered buffer (moves with module to GPU/CPU)
- Kernel is flipped for causal convolution (compatible with scipy.signal.lfilter)
- Grouped convolution applies same filter independently to each audio channel
- Right-side padding maintains original signal length
- No gradient computation (filters run under @torch.no_grad() decorator)

Classes
-------
FIR : Base FIR filter accepting pre-computed coefficients
DesignableFIR : FIR filter with automatic coefficient design via window method

See Also
--------
torchfx.filter.iir : IIR filter implementations for comparison
torchfx.filter.__base : AbstractFilter base class and parallel combination
scipy.signal.firwin : Underlying FIR design function
scipy.signal.freqz : Frequency response analysis

Notes
-----
For comprehensive FIR filter theory, design examples, and performance characteristics,
see wiki/4.2 FIR Filters.md.

For complete API reference including all filter parameters and return types,
see wiki/8.3 torchfx.filter.md.

For guidance on choosing between FIR and IIR filters for specific applications,
see wiki/4 Filters.md.

References
----------
.. [1] Oppenheim, A. V., & Schafer, R. W. (2009). Discrete-Time Signal Processing
       (3rd ed.). Prentice Hall. Chapters 5 and 7.
.. [2] Smith, J. O. (2007). Introduction to Digital Filters with Audio Applications.
       https://ccrma.stanford.edu/~jos/filters/

Examples
--------
Basic FIR filtering with custom coefficients (5-tap moving average):

>>> import torch
>>> from torchfx.filter import FIR
>>>
>>> # Define FIR coefficients for a simple moving average filter
>>> coeffs = [0.2, 0.2, 0.2, 0.2, 0.2]
>>> fir = FIR(b=coeffs)
>>>
>>> # Apply to mono signal [T]
>>> signal = torch.randn(44100)  # 1 second at 44.1 kHz
>>> filtered = fir(signal)
>>> print(filtered.shape)  # torch.Size([44100])

Designing a lowpass FIR filter using the window method:

>>> from torchfx.filter import DesignableFIR
>>>
>>> # Design lowpass filter: 5 kHz cutoff, 101 taps, Hamming window
>>> lpf = DesignableFIR(
...     cutoff=5000.0,
...     num_taps=101,
...     fs=44100,
...     pass_zero=True,  # Lowpass (passes DC)
...     window="hamming"
... )
>>>
>>> # Apply to stereo signal [C, T]
>>> stereo = torch.randn(2, 44100)
>>> filtered_stereo = lpf(stereo)
>>> print(filtered_stereo.shape)  # torch.Size([2, 44100])

Comparing different window types for the same filter specification:

>>> import matplotlib.pyplot as plt
>>> from scipy.signal import freqz
>>>
>>> # Design filters with different windows
>>> windows = ["hamming", "hann", "blackman"]
>>> for window in windows:
...     fir = DesignableFIR(5000, num_taps=101, fs=44100, window=window)
...     # Get frequency response
...     w, h = freqz(fir.b, worN=8000, fs=44100)
...     plt.plot(w, 20 * np.log10(abs(h)), label=window)
>>> plt.xlabel("Frequency (Hz)")
>>> plt.ylabel("Magnitude (dB)")
>>> plt.legend()
>>> plt.title("Effect of Window Type on Frequency Response")
>>> plt.grid(True)

Exploring the effect of tap count on filter performance:

>>> # Low tap count: fast but poor selectivity
>>> fir_51 = DesignableFIR(5000, num_taps=51, fs=44100, window="hamming")
>>>
>>> # Medium tap count: balanced
>>> fir_101 = DesignableFIR(5000, num_taps=101, fs=44100, window="hamming")
>>>
>>> # High tap count: excellent selectivity but higher latency
>>> fir_201 = DesignableFIR(5000, num_taps=201, fs=44100, window="hamming")
>>>
>>> # Compare transition bandwidth and stopband attenuation
>>> signal = torch.randn(44100)
>>> result_51 = fir_51(signal)   # Faster, wider transition
>>> result_101 = fir_101(signal) # Balanced
>>> result_201 = fir_201(signal) # Slower, sharper transition

Designing a highpass filter:

>>> # Highpass: remove low frequencies below 100 Hz
>>> hpf = DesignableFIR(
...     cutoff=100.0,
...     num_taps=201,
...     fs=44100,
...     pass_zero=False,  # Highpass (blocks DC)
...     window="blackman"  # Use Blackman for high stopband attenuation
... )
>>>
>>> # Apply to remove DC offset and rumble
>>> audio = torch.randn(44100)
>>> clean = hpf(audio)

Designing a bandpass filter using multiple cutoff frequencies:

>>> # Bandpass: pass 200-3000 Hz (telephone bandwidth)
>>> bpf = DesignableFIR(
...     cutoff=[200.0, 3000.0],  # Two cutoffs define bandpass
...     num_taps=201,
...     fs=44100,
...     pass_zero=False,  # Bandpass behavior
...     window="hamming"
... )
>>>
>>> # Apply to extract voice frequencies
>>> speech = torch.randn(44100)
>>> bandpassed = bpf(speech)

Using FIR filters in a pipeline with torchfx.Wave:

>>> import torchfx as fx
>>>
>>> # Load audio file
>>> wave = fx.Wave.from_file("audio.wav")
>>>
>>> # Design filter (fs inferred from wave)
>>> lpf = DesignableFIR(cutoff=8000, num_taps=101, pass_zero=True)
>>>
>>> # Apply using pipe operator
>>> filtered_wave = wave | lpf
>>>
>>> # Save result
>>> filtered_wave.save("filtered.wav")

Batch processing with FIR filters on GPU:

>>> # Create batch of signals [B, C, T]
>>> batch = torch.randn(8, 2, 44100, device="cuda")
>>>
>>> # Design filter
>>> fir = DesignableFIR(5000, num_taps=101, fs=44100, window="hamming")
>>>
>>> # Filter entire batch on GPU
>>> filtered_batch = fir(batch)
>>> print(filtered_batch.shape)  # torch.Size([8, 2, 44100])
>>> print(filtered_batch.device)  # cuda:0

Comparing FIR linear phase to IIR non-linear phase:

>>> import torchfx as fx
>>>
>>> # Create test signal: sum of two sinusoids
>>> t = torch.linspace(0, 1, 44100)
>>> signal = torch.sin(2 * 3.14159 * 440 * t) + torch.sin(2 * 3.14159 * 880 * t)
>>>
>>> # FIR filter: linear phase preserves waveform shape
>>> fir = DesignableFIR(cutoff=5000, num_taps=101, fs=44100)
>>> fir_output = fir(signal)
>>>
>>> # IIR filter: non-linear phase may distort waveform
>>> iir = fx.filter.Butterworth("lowpass", cutoff=5000, order=4, fs=44100)
>>> iir_output = iir(signal)
>>>
>>> # FIR has constant group delay, IIR varies with frequency

Practical audio application: anti-aliasing filter before downsampling:

>>> # Design anti-aliasing lowpass filter
>>> # Target: downsample from 44100 Hz to 22050 Hz
>>> # Cutoff at 0.4 * new_rate to prevent aliasing
>>> anti_alias = DesignableFIR(
...     cutoff=0.4 * 22050,  # 8820 Hz
...     num_taps=201,         # High order for sharp transition
...     fs=44100,
...     pass_zero=True,
...     window="blackman"     # High stopband attenuation
... )
>>>
>>> # Filter then downsample
>>> audio = torch.randn(44100)
>>> filtered = anti_alias(audio)
>>> downsampled = filtered[::2]  # Decimate by 2
>>> print(downsampled.shape)  # torch.Size([22050])

"""

from collections.abc import Sequence

import torch
from numpy.typing import ArrayLike
from scipy.signal import firwin
from torch import Tensor, nn
from typing_extensions import override

from torchfx.filter.__base import AbstractFilter
from torchfx.typing import WindowType


class FIR(AbstractFilter):
    """Efficient FIR filter implementation using PyTorch conv1d operation.

    The `FIR` class implements a basic Finite Impulse Response filter that accepts
    pre-computed filter coefficients and applies them using PyTorch's optimized
    `conv1d` operation. This provides GPU-accelerated filtering with automatic
    device management and support for mono, stereo, and batched audio processing.

    FIR filters are always stable (no feedback) and can provide linear phase response
    when designed with symmetric coefficients. This class serves as the base for
    `DesignableFIR` and can also be used directly when you have custom-designed
    coefficients from external tools or algorithms.

    Parameters
    ----------
    b : array_like
        FIR filter coefficients (numerator). Can be a Python list, NumPy array,
        or any array-like object. The coefficients define the filter's impulse
        response. Length determines filter order (num_taps = len(b)).

    Attributes
    ----------
    kernel : Tensor
        Registered buffer containing the flipped filter coefficients in shape [1, 1, K]
        where K is the number of taps. The buffer is automatically moved to the input
        tensor's device during forward pass.
    a : list[float]
        Denominator coefficients, always [1.0] for FIR filters (no feedback).

    Shape Support
    -------------
    The forward pass supports three input tensor shapes:

    - [T] : Mono audio, single channel
    - [C, T] : Multi-channel audio (e.g., stereo where C=2)
    - [B, C, T] : Batched multi-channel audio

    Output shape matches input shape exactly. Time dimension T is preserved through
    appropriate padding.

    Implementation Details
    ----------------------
    **Convolution-Based Filtering**:
        Uses PyTorch's `conv1d` with grouped convolution where each channel is
        filtered independently with the same kernel. This is equivalent to applying
        the FIR filter to each channel separately but is much more efficient.

    **Kernel Flipping**:
        Coefficients are flipped (reversed) before storage to ensure causal
        convolution behavior compatible with `scipy.signal.lfilter`. This means
        the first coefficient in `b` corresponds to the current sample, the second
        to the previous sample, etc.

    **Padding Strategy**:
        Right-side padding of (K-1) samples is applied where K is the number of taps.
        This maintains the original signal length and ensures causal filtering (no
        future samples influence past outputs).

    **Device Management**:
        The kernel buffer is automatically moved to match the input tensor's device
        and dtype. No manual device management is required.

    **No Gradient Computation**:
        All filtering operations execute under `@torch.no_grad()` context since FIR
        filters have fixed coefficients (not trainable parameters).

    See Also
    --------
    DesignableFIR : FIR filter with automatic coefficient design via window method
    torchfx.filter.iir.IIR : IIR filter base class for comparison
    AbstractFilter : Base class for all filters
    scipy.signal.lfilter : Compatible filtering function in SciPy

    Notes
    -----
    **Linear Phase Property**:
        If coefficients are symmetric (b[i] = b[N-1-i]), the filter has linear phase,
        meaning constant group delay across all frequencies. This preserves waveform
        shape without phase distortion, critical for audio mastering and measurement.

    **Stability**:
        FIR filters are always BIBO (Bounded Input Bounded Output) stable because
        they have no poles (denominator is always 1). This makes them suitable for
        applications where stability is critical.

    **Computational Complexity**:
        Time complexity is O(N × K) where N is signal length and K is number of taps.
        Space complexity is O(K) for kernel storage plus O(N) for padded input.

    References
    ----------
    For comprehensive FIR filter theory and design patterns, see wiki/4.2 FIR Filters.md.

    For implementation details and performance characteristics, see the "Implementation
    Details" section in wiki/4.2 FIR Filters.md.

    Examples
    --------
    Basic moving average filter (5-tap):

    >>> import torch
    >>> from torchfx.filter import FIR
    >>>
    >>> # Define coefficients for simple moving average
    >>> coeffs = [0.2, 0.2, 0.2, 0.2, 0.2]
    >>> ma_filter = FIR(b=coeffs)
    >>>
    >>> # Apply to mono signal
    >>> signal = torch.randn(44100)
    >>> smoothed = ma_filter(signal)
    >>> print(smoothed.shape)  # torch.Size([44100])

    Custom FIR coefficients from external design:

    >>> import numpy as np
    >>> from scipy.signal import firwin
    >>>
    >>> # Design filter using scipy directly
    >>> taps = firwin(numtaps=101, cutoff=5000, fs=44100, window='hamming')
    >>>
    >>> # Use coefficients in torchfx FIR filter
    >>> fir = FIR(b=taps)
    >>>
    >>> # Apply to audio
    >>> audio = torch.randn(44100)
    >>> filtered = fir(audio)

    Processing stereo audio [C, T]:

    >>> # Stereo signal with 2 channels
    >>> stereo = torch.randn(2, 44100)
    >>>
    >>> # Filter applies same coefficients to both channels independently
    >>> fir = FIR(b=[0.25, 0.5, 0.25])  # Simple 3-tap filter
    >>> filtered_stereo = fir(stereo)
    >>> print(filtered_stereo.shape)  # torch.Size([2, 44100])

    Batch processing [B, C, T]:

    >>> # Batch of 8 stereo signals
    >>> batch = torch.randn(8, 2, 44100)
    >>>
    >>> # Apply filter to entire batch
    >>> fir = FIR(b=[0.2, 0.2, 0.2, 0.2, 0.2])
    >>> filtered_batch = fir(batch)
    >>> print(filtered_batch.shape)  # torch.Size([8, 2, 44100])

    GPU acceleration:

    >>> # Move input to GPU
    >>> signal_gpu = torch.randn(44100, device='cuda')
    >>>
    >>> # Filter automatically runs on GPU
    >>> fir = FIR(b=[0.25, 0.5, 0.25])
    >>> filtered_gpu = fir(signal_gpu)
    >>> print(filtered_gpu.device)  # cuda:0

    Designing a differentiator FIR filter:

    >>> # Differentiator approximates derivative
    >>> # Simple 3-tap differentiator
    >>> diff_filter = FIR(b=[-0.5, 0.0, 0.5])
    >>>
    >>> # Apply to signal
    >>> signal = torch.sin(torch.linspace(0, 10, 1000))
    >>> derivative = diff_filter(signal)
    >>> # Output approximates cosine

    Hilbert transformer for analytic signal:

    >>> # Design Hilbert transformer (90-degree phase shift)
    >>> # This is a simplified example - real Hilbert needs more taps
    >>> from scipy.signal import hilbert as scipy_hilbert
    >>> impulse = np.zeros(101)
    >>> impulse[50] = 1
    >>> analytic = scipy_hilbert(impulse)
    >>> hilbert_coeffs = np.imag(analytic)
    >>>
    >>> # Create FIR filter
    >>> hilbert_fir = FIR(b=hilbert_coeffs)
    >>>
    >>> # Apply to get quadrature component
    >>> signal = torch.randn(1000)
    >>> quadrature = hilbert_fir(signal)

    Using with torchfx.Wave pipeline:

    >>> import torchfx as fx
    >>>
    >>> # Load audio
    >>> wave = fx.Wave.from_file("audio.wav")
    >>>
    >>> # Design custom FIR coefficients
    >>> coeffs = [0.1, 0.2, 0.4, 0.2, 0.1]  # Simple lowpass
    >>> fir = FIR(b=coeffs)
    >>>
    >>> # Apply in pipeline
    >>> result = wave | fir
    >>> result.save("filtered.wav")

    Cascading multiple FIR filters:

    >>> # Design two FIR filters
    >>> fir1 = FIR(b=[0.25, 0.5, 0.25])  # First smoothing stage
    >>> fir2 = FIR(b=[0.25, 0.5, 0.25])  # Second smoothing stage
    >>>
    >>> # Cascade them (equivalent to convolving coefficients)
    >>> signal = torch.randn(44100)
    >>> filtered = fir2(fir1(signal))
    >>>
    >>> # Or use pipeline syntax
    >>> wave = fx.Wave(signal, fs=44100)
    >>> result = wave | fir1 | fir2

    Verifying filter stability (always stable for FIR):

    >>> # FIR filters are always stable because denominator a = [1.0]
    >>> fir = FIR(b=[0.1, 0.2, 0.4, 0.2, 0.1])
    >>> print(fir.a)  # [1.0] - no feedback, always stable
    >>>
    >>> # Even with arbitrary coefficients, FIR is stable
    >>> unstable_looking = FIR(b=[100, -200, 100])
    >>> print(unstable_looking.a)  # Still [1.0] - stable

    """

    def __init__(self, b: ArrayLike) -> None:
        super().__init__()
        # Flip the kernel for causal convolution (like lfilter)
        b_tensor = torch.tensor(b, dtype=torch.float32).flip(0)
        self.a = [1.0]  # FIR filter denominator is always 1
        self.register_buffer("kernel", b_tensor[None, None, :])  # [1, 1, K]

    @override
    def compute_coefficients(self) -> None:
        """Compute the filter coefficients."""
        # This method is not used in FIR, but defined for consistency with IIR
        pass

    @override
    @torch.no_grad()
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
        kernel_exp = kernel.expand(CHANNELS, 1, -1)  # type: ignore # [C, 1, K]

        # Pad input to maintain original length, pad right side
        pad = int(kernel.shape[-1] - 1)  # type: ignore
        x_padded = nn.functional.pad(x, (pad, 0))  # pad left only # type: ignore

        # For batch > 1, reshape to [B*C, 1, T] and use single-channel conv1d
        if BATCHES > 1:
            x_padded = x_padded.reshape(BATCHES * CHANNELS, 1, -1)
            assert isinstance(kernel, Tensor)
            y = nn.functional.conv1d(x_padded, kernel)
            y = y.reshape(BATCHES, CHANNELS, TIME)
        else:
            assert isinstance(kernel_exp, Tensor)
            y = nn.functional.conv1d(x_padded, kernel_exp, groups=CHANNELS)

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
    """FIR filter with automatic coefficient design using the window method.

    `DesignableFIR` extends the base `FIR` class by automatically computing filter
    coefficients using the windowed Fourier series method via `scipy.signal.firwin`.
    Instead of providing pre-computed coefficients, you specify the desired filter
    characteristics (cutoff frequency, number of taps, window type) and the class
    designs the filter for you.

    This class is ideal for standard filtering tasks (lowpass, highpass, bandpass,
    bandstop) where you don't need custom coefficient design. The window method
    provides intuitive trade-offs between transition bandwidth and stopband attenuation.

    Parameters
    ----------
    cutoff : float or Sequence[float]
        Cutoff frequency or frequencies in Hz. The interpretation depends on the
        number of cutoff values provided:

        - Single value with pass_zero=True: Lowpass filter
        - Single value with pass_zero=False: Highpass filter
        - Two values [low, high] with pass_zero=False: Bandpass filter (low to high)
        - Two values [low, high] with pass_zero=True: Bandstop filter (notch low to high)

    num_taps : int
        Number of filter taps (coefficients). Must be odd for type I FIR filters.
        Higher values provide sharper frequency transitions but increase computational
        cost and latency. Typical values:

        - 51-101: Fast, moderate selectivity
        - 101-201: Balanced, good selectivity
        - 201-501: Excellent selectivity, higher latency

    fs : int or None, optional
        Sampling frequency in Hz. If None, the filter coefficients are not computed
        during initialization. The sampling frequency can be set later via the `fs`
        attribute or automatically when used in a pipeline with `Wave` objects.
        Default is None.

    pass_zero : bool, optional
        Controls the filter type when a single cutoff is provided:

        - True: Lowpass (gain of 1 at DC, passes zero frequency)
        - False: Highpass (gain of 0 at DC, blocks zero frequency)

        For two cutoff frequencies:

        - True: Bandstop (notch filter)
        - False: Bandpass

        Default is True (lowpass).

    window : WindowType, optional
        Window function to apply during filter design. Controls the trade-off between
        main lobe width (transition bandwidth) and side lobe attenuation (stopband
        ripple). Common options:

        - "hamming": Good general-purpose window (-43 dB sidelobes)
        - "hann": Smooth transitions (-31 dB sidelobes)
        - "blackman": Excellent stopband rejection (-58 dB sidelobes)
        - "kaiser": Adjustable characteristics (requires beta parameter)
        - "bartlett": Triangular window (-25 dB sidelobes)

        Default is "hamming".

    Attributes
    ----------
    cutoff : float or Sequence[float]
        Stored cutoff frequency or frequencies in Hz.
    num_taps : int
        Stored number of filter taps.
    fs : int or None
        Sampling frequency in Hz. Initially None until set.
    pass_zero : bool
        Stored pass_zero parameter controlling filter type.
    window : WindowType
        Stored window function type.
    b : ArrayLike or None
        Computed filter coefficients (numerator). None until `compute_coefficients()`
        is called, which happens automatically on first use or when `fs` is set.
    kernel : Tensor
        Inherited from `FIR`. Registered buffer containing flipped coefficients.
    a : list[float]
        Inherited from `FIR`. Always [1.0] for FIR filters.

    Methods
    -------
    compute_coefficients()
        Computes FIR filter coefficients using `scipy.signal.firwin` and initializes
        the parent `FIR` class with the computed coefficients. Called automatically
        when `fs` is set and coefficients haven't been computed yet.

    Window Method Theory
    --------------------
    The window method designs FIR filters by:

    1. Starting with the ideal frequency response (infinite impulse response)
    2. Applying inverse Fourier transform to get time-domain impulse response
    3. Truncating to finite length (num_taps samples)
    4. Applying a window function to reduce Gibbs phenomenon (ripple)

    Different window functions provide different trade-offs:

    - **Narrow main lobe** (e.g., rectangular): Sharp transitions but high ripple
    - **Wide main lobe** (e.g., Blackman): Smooth transitions, low ripple
    - **Medium** (e.g., Hamming, Hann): Balanced trade-off

    Design Guidelines
    -----------------
    **Choosing Number of Taps**:
        Rule of thumb for transition bandwidth Δf at sampling rate fs:

            num_taps ≈ (stopband_atten_dB / 22) * (fs / Δf)

        For example, -60 dB stopband attenuation with 1000 Hz transition at 44100 Hz:

            num_taps ≈ (60 / 22) * (44100 / 1000) ≈ 120

    **Choosing Window Type**:
        - Hamming: Good default for most applications
        - Hann: When smooth transitions are more important than stopband rejection
        - Blackman: When maximum stopband rejection is critical (audio mastering)
        - Kaiser: When you need precise control over specifications

    **Linear Phase Property**:
        All filters designed by this class have linear phase (constant group delay)
        because the window method produces symmetric coefficients. Group delay is:

            group_delay = (num_taps - 1) / (2 * fs) seconds

    See Also
    --------
    FIR : Base FIR filter class accepting pre-computed coefficients
    torchfx.filter.iir.Butterworth : IIR alternative for comparison
    scipy.signal.firwin : Underlying filter design function
    scipy.signal.freqz : Analyze designed filter frequency response

    Notes
    -----
    **Computational Cost vs IIR**:
        FIR filters typically require 10-100x more taps than equivalent IIR filters
        for similar frequency selectivity. However, FIR filters guarantee stability
        and can provide linear phase, which IIR filters cannot (except at specific
        frequencies).

    **Odd vs Even Tap Count**:
        For most filter types, use odd number of taps. Even tap counts can be used
        but may not have zero-phase at Nyquist frequency for certain filter types.

    **Frequency Response Analysis**:
        Use `scipy.signal.freqz` to visualize the designed filter's frequency response:

            from scipy.signal import freqz
            w, h = freqz(filter.b, worN=8000, fs=filter.fs)
            plt.plot(w, 20*np.log10(abs(h)))

    References
    ----------
    For comprehensive filter design examples and window function comparisons,
    see wiki/4.2 FIR Filters.md.

    For API reference and parameter details, see wiki/8.3 torchfx.filter.md.

    For guidance on choosing between FIR and IIR filters, see wiki/4 Filters.md.

    Examples
    --------
    Basic lowpass filter design:

    >>> import torch
    >>> from torchfx.filter import DesignableFIR
    >>>
    >>> # Design lowpass at 5 kHz with default Hamming window
    >>> lpf = DesignableFIR(
    ...     cutoff=5000.0,
    ...     num_taps=101,
    ...     fs=44100,
    ...     pass_zero=True,
    ...     window="hamming"
    ... )
    >>>
    >>> # Apply to mono signal
    >>> signal = torch.randn(44100)
    >>> filtered = lpf(signal)
    >>> print(filtered.shape)  # torch.Size([44100])

    Highpass filter to remove low-frequency rumble:

    >>> # Remove frequencies below 100 Hz
    >>> hpf = DesignableFIR(
    ...     cutoff=100.0,
    ...     num_taps=201,  # Higher taps for sharp low-frequency cutoff
    ...     fs=44100,
    ...     pass_zero=False,  # Highpass
    ...     window="blackman"  # High stopband rejection
    ... )
    >>>
    >>> # Apply to audio with DC offset and rumble
    >>> audio = torch.randn(44100)
    >>> clean = hpf(audio)

    Bandpass filter for voice frequencies:

    >>> # Extract telephone bandwidth (300-3400 Hz)
    >>> bpf = DesignableFIR(
    ...     cutoff=[300.0, 3400.0],  # Two cutoffs
    ...     num_taps=201,
    ...     fs=44100,
    ...     pass_zero=False,  # Bandpass
    ...     window="hamming"
    ... )
    >>>
    >>> # Apply to speech signal
    >>> speech = torch.randn(44100)
    >>> telephone_quality = bpf(speech)

    Bandstop (notch) filter to remove interference:

    >>> # Remove 50-70 Hz hum band (European power line + harmonics)
    >>> notch = DesignableFIR(
    ...     cutoff=[50.0, 70.0],
    ...     num_taps=301,  # High order for narrow notch
    ...     fs=44100,
    ...     pass_zero=True,  # Bandstop
    ...     window="blackman"
    ... )
    >>>
    >>> # Apply to remove power line interference
    >>> noisy = torch.randn(44100)
    >>> clean = notch(noisy)

    Comparing different window types:

    >>> import matplotlib.pyplot as plt
    >>> from scipy.signal import freqz
    >>> import numpy as np
    >>>
    >>> # Design same filter with different windows
    >>> windows = {
    ...     "hamming": DesignableFIR(5000, 101, 44100, window="hamming"),
    ...     "hann": DesignableFIR(5000, 101, 44100, window="hann"),
    ...     "blackman": DesignableFIR(5000, 101, 44100, window="blackman")
    ... }
    >>>
    >>> # Plot frequency responses
    >>> for name, fir in windows.items():
    ...     w, h = freqz(fir.b, worN=8000, fs=44100)
    ...     plt.plot(w, 20*np.log10(abs(h)), label=name)
    >>> plt.xlabel("Frequency (Hz)")
    >>> plt.ylabel("Magnitude (dB)")
    >>> plt.legend()
    >>> plt.grid(True)
    >>> plt.title("Window Function Comparison")

    Exploring the effect of tap count:

    >>> # Low, medium, high tap counts
    >>> fir_51 = DesignableFIR(5000, num_taps=51, fs=44100)
    >>> fir_101 = DesignableFIR(5000, num_taps=101, fs=44100)
    >>> fir_201 = DesignableFIR(5000, num_taps=201, fs=44100)
    >>>
    >>> # Compare frequency responses
    >>> from scipy.signal import freqz
    >>> for taps, fir in [(51, fir_51), (101, fir_101), (201, fir_201)]:
    ...     w, h = freqz(fir.b, worN=8000, fs=44100)
    ...     plt.plot(w, 20*np.log10(abs(h)), label=f"{taps} taps")
    >>> plt.xlabel("Frequency (Hz)")
    >>> plt.ylabel("Magnitude (dB)")
    >>> plt.legend()
    >>> plt.title("Effect of Tap Count on Transition Bandwidth")

    Deferred coefficient computation (fs=None):

    >>> # Create filter without computing coefficients
    >>> fir = DesignableFIR(cutoff=5000, num_taps=101, fs=None)
    >>> print(fir.b)  # None - coefficients not computed yet
    >>>
    >>> # Set sampling frequency later
    >>> fir.fs = 44100
    >>> fir.compute_coefficients()
    >>> print(fir.b is not None)  # True - coefficients now computed

    Using with torchfx.Wave pipeline (automatic fs configuration):

    >>> import torchfx as fx
    >>>
    >>> # Load audio
    >>> wave = fx.Wave.from_file("audio.wav")  # wave.fs = 44100
    >>>
    >>> # Create filter without fs - will be set from wave
    >>> lpf = DesignableFIR(cutoff=8000, num_taps=101, pass_zero=True)
    >>> print(lpf.fs)  # None
    >>>
    >>> # Apply using pipe operator - fs set automatically
    >>> filtered = wave | lpf
    >>> print(lpf.fs)  # 44100 - set from wave
    >>> filtered.save("filtered.wav")

    Anti-aliasing filter before downsampling:

    >>> # Downsample from 44100 Hz to 22050 Hz
    >>> # Design anti-aliasing filter at 0.4 * new_rate
    >>> anti_alias = DesignableFIR(
    ...     cutoff=0.4 * 22050,  # 8820 Hz
    ...     num_taps=201,
    ...     fs=44100,
    ...     pass_zero=True,
    ...     window="blackman"  # High stopband attenuation
    ... )
    >>>
    >>> # Filter then downsample
    >>> audio = torch.randn(44100)
    >>> filtered = anti_alias(audio)
    >>> downsampled = filtered[::2]  # Decimate by factor of 2
    >>> print(downsampled.shape)  # torch.Size([22050])

    Multi-band processing with multiple DesignableFIR filters:

    >>> # Create 3-band crossover
    >>> low = DesignableFIR(cutoff=200, num_taps=201, fs=44100, pass_zero=True)
    >>> mid = DesignableFIR(cutoff=[200, 2000], num_taps=201, fs=44100, pass_zero=False)
    >>> high = DesignableFIR(cutoff=2000, num_taps=201, fs=44100, pass_zero=False)
    >>>
    >>> # Process each band separately
    >>> audio = torch.randn(44100)
    >>> low_band = low(audio)
    >>> mid_band = mid(audio)
    >>> high_band = high(audio)
    >>>
    >>> # Apply different processing to each band, then recombine
    >>> # (processing steps omitted for brevity)
    >>> mixed = low_band + mid_band + high_band

    GPU acceleration:

    >>> # Create signal on GPU
    >>> signal_gpu = torch.randn(44100, device="cuda")
    >>>
    >>> # Design filter
    >>> fir = DesignableFIR(5000, num_taps=101, fs=44100, window="hamming")
    >>>
    >>> # Filter automatically runs on GPU
    >>> filtered_gpu = fir(signal_gpu)
    >>> print(filtered_gpu.device)  # cuda:0

    Batch processing stereo audio:

    >>> # Batch of 8 stereo signals
    >>> batch = torch.randn(8, 2, 44100)
    >>>
    >>> # Apply same filter to all signals
    >>> lpf = DesignableFIR(cutoff=5000, num_taps=101, fs=44100)
    >>> filtered_batch = lpf(batch)
    >>> print(filtered_batch.shape)  # torch.Size([8, 2, 44100])

    Verifying linear phase property:

    >>> # Design filter
    >>> fir = DesignableFIR(cutoff=5000, num_taps=101, fs=44100, window="hamming")
    >>>
    >>> # Check coefficient symmetry (implies linear phase)
    >>> import numpy as np
    >>> coeffs = np.array(fir.b)
    >>> is_symmetric = np.allclose(coeffs, coeffs[::-1])
    >>> print(is_symmetric)  # True - linear phase
    >>>
    >>> # Calculate group delay
    >>> group_delay = (fir.num_taps - 1) / (2 * fir.fs)
    >>> print(f"Group delay: {group_delay*1000:.2f} ms")  # Constant for all frequencies

    Cascading filters for sharper rolloff:

    >>> # Two cascaded filters provide steeper rolloff
    >>> lpf1 = DesignableFIR(cutoff=5000, num_taps=101, fs=44100)
    >>> lpf2 = DesignableFIR(cutoff=5000, num_taps=101, fs=44100)
    >>>
    >>> # Apply in series
    >>> signal = torch.randn(44100)
    >>> filtered = lpf2(lpf1(signal))
    >>>
    >>> # Or use pipeline
    >>> import torchfx as fx
    >>> wave = fx.Wave(signal, fs=44100)
    >>> result = wave | lpf1 | lpf2

    Custom window with parameters (Kaiser window):

    >>> # Kaiser window allows precise specification of stopband attenuation
    >>> # beta parameter controls trade-off: higher beta = more attenuation
    >>> kaiser_fir = DesignableFIR(
    ...     cutoff=5000,
    ...     num_taps=101,
    ...     fs=44100,
    ...     window=("kaiser", 8.0)  # Tuple: (name, beta)
    ... )
    >>>
    >>> # Apply filter
    >>> signal = torch.randn(44100)
    >>> filtered = kaiser_fir(signal)

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
            window=self.window,  # type: ignore
            scale=True,
        )
        assert self.b is not None, "Filter coefficients (b) must be computed."

        super().__init__(self.b)
