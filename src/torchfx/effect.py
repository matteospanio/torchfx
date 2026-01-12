"""Audio effects and transformations with PyTorch integration.

This module provides the FX abstract base class and built-in audio effects
for time-domain signal processing. All effects inherit from torch.nn.Module,
enabling GPU acceleration, gradient computation, and seamless integration with
PyTorch's ecosystem.

Effects can be applied using the pipeline operator (|) with Wave objects or
called directly on tensors. The module supports extensibility through the
strategy pattern and custom effect creation.

The FX Base Class
-----------------
All effects inherit from the FX abstract base class, which combines
torch.nn.Module with ABC (Abstract Base Class) requirements. This ensures
effects are compatible with PyTorch's module system while enforcing a
consistent interface across all effect implementations.

Inheriting from nn.Module provides:
    - GPU/CPU device management (.to(), .cuda(), .cpu())
    - Parameter and buffer registration
    - Integration with nn.Sequential for effect chaining
    - Serialization support (state_dict, load_state_dict)
    - Gradient computation capabilities (when not using @torch.no_grad())

Classes
-------
FX : Abstract base class
    Abstract base class for all effects and filters. Defines the interface
    that all effects must implement: __init__ and forward methods.

Built-in Effects
----------------
Gain : Volume adjustment
    Adjust signal amplitude using amplitude, dB, or power gain modes with
    optional clamping to prevent clipping.
Normalize : Amplitude normalization
    Normalize waveforms to target peak using configurable strategies including
    peak, RMS, percentile, and per-channel normalization.
Reverb : Spatial effects
    Simple reverb using feedback delay network for creating spatial ambiance.
Delay : Echo effects
    Multi-tap delay with BPM synchronization, musical time divisions, and
    stereo processing strategies (mono, ping-pong).

Strategy Pattern Components
----------------------------
NormalizationStrategy : Abstract normalization strategy
    Base class for normalization algorithms with concrete implementations:
    PeakNormalizationStrategy, RMSNormalizationStrategy,
    PercentileNormalizationStrategy, PerChannelNormalizationStrategy,
    CustomNormalizationStrategy.
DelayStrategy : Abstract delay strategy
    Base class for delay processing behaviors with concrete implementations:
    MonoDelayStrategy, PingPongDelayStrategy.

See Also
--------
torchfx.Wave : Wave class for audio I/O and pipeline operations
torchfx.filter : Filter-based audio processing

Examples
--------
Apply effects using the pipeline operator:

>>> import torchfx as fx
>>> wave = fx.Wave.from_file("audio.wav")
>>> processed = wave | fx.Gain(0.5) | fx.Normalize(peak=0.8)

Apply effects directly to tensors:

>>> import torch
>>> waveform = torch.randn(2, 44100)
>>> gain = fx.Gain(2.0)
>>> louder = gain(waveform)

Create custom effects by subclassing FX:

>>> class CustomEffect(fx.effect.FX):
...     def __init__(self, param: float) -> None:
...         super().__init__()
...         self.param = param
...
...     @torch.no_grad()
...     def forward(self, waveform: torch.Tensor) -> torch.Tensor:
...         return waveform * self.param

Chain multiple effects in a pipeline:

>>> reverb = fx.Reverb(delay=4410, decay=0.6, mix=0.3)
>>> delay = fx.Delay(bpm=128, delay_time="1/8", feedback=0.4, mix=0.2)
>>> result = wave | reverb | delay

Use strategy pattern for extensibility:

>>> # Custom normalization strategy
>>> from torchfx.effect import Normalize, RMSNormalizationStrategy
>>> normalize = fx.Normalize(peak=0.8, strategy=RMSNormalizationStrategy())
>>> result = wave | normalize

BPM-synced delay with automatic sample rate configuration:

>>> # fs is automatically inferred from Wave object
>>> delay = fx.Delay(bpm=120, delay_time="1/4d", feedback=0.5, mix=0.3)
>>> result = wave | delay

Notes
-----
For comprehensive guidance on creating custom effects, including parameter
handling, the strategy pattern, and multi-channel processing, see the
wiki page "3.5 Creating Custom Effects". For API reference and detailed
parameter specifications, see "8.2 torchfx.FX and Effects".

References
----------
.. [1] Creating Custom Effects: wiki/3.5 Creating Custom Effects.md
.. [2] API Reference: wiki/8.2 torchfx.FX and Effects.md

"""

from __future__ import annotations

import abc
import math
from collections.abc import Callable

import torch
from torch import Tensor, nn
from torchaudio import functional as F
from typing_extensions import override


class FX(nn.Module, abc.ABC):
    """Abstract base class for all audio effects and filters.

    FX serves as the foundation for all effects in torchfx, combining PyTorch's
    nn.Module with abstract base class requirements. This design ensures effects
    are compatible with PyTorch's module system while enforcing a consistent
    interface across all effect implementations.

    All effects must implement the abstract __init__ and forward methods. The
    forward method receives audio tensors of shape (..., time) and returns
    processed tensors.

    Inheriting from nn.Module provides:
    - GPU/CPU device management (.to(), .cuda(), .cpu())
    - Parameter and buffer registration
    - Integration with nn.Sequential for effect chaining
    - Serialization support (state_dict, load_state_dict)
    - Gradient computation (when not using @torch.no_grad())

    Parameters
    ----------
    *args : tuple
        Positional arguments passed to nn.Module.
    **kwargs : dict
        Keyword arguments passed to nn.Module.

    Methods
    -------
    forward(x: Tensor) -> Tensor
        Process input tensor and return transformed output. Must be implemented
        by all subclasses.

    Notes
    -----
    When creating custom effects:

    1. Always call super().__init__() in your constructor
    2. Implement forward() to process tensors of shape (..., time)
    3. Use @torch.no_grad() decorator for inference-only effects
    4. Validate parameters in __init__ using assertions
    5. For sample-rate dependent effects, accept optional fs parameter

    The FX base class uses the strategy pattern for extensibility. Effects
    can accept strategy objects to customize processing behavior without
    modifying the core effect implementation.

    See Also
    --------
    Gain : Volume adjustment effect
    Normalize : Amplitude normalization effect
    Reverb : Reverb effect using feedback delay network
    Delay : Multi-tap delay effect with BPM synchronization

    Examples
    --------
    Create a simple custom effect:

    >>> import torch
    >>> from torchfx.effect import FX
    >>>
    >>> class SimpleGain(FX):
    ...     def __init__(self, gain: float) -> None:
    ...         super().__init__()
    ...         assert gain > 0, "Gain must be positive"
    ...         self.gain = gain
    ...
    ...     @torch.no_grad()
    ...     def forward(self, waveform: torch.Tensor) -> torch.Tensor:
    ...         return waveform * self.gain

    Use in a pipeline:

    >>> import torchfx as fx
    >>> wave = fx.Wave.from_file("audio.wav")
    >>> effect = SimpleGain(0.5)
    >>> processed = wave | effect

    Chain multiple effects:

    >>> result = wave | SimpleGain(0.5) | fx.Normalize(peak=1.0)

    Create effects with strategies:

    >>> from abc import ABC, abstractmethod
    >>>
    >>> class ProcessingStrategy(ABC):
    ...     @abstractmethod
    ...     def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
    ...         pass
    >>>
    >>> class StrategyEffect(FX):
    ...     def __init__(self, strategy: ProcessingStrategy) -> None:
    ...         super().__init__()
    ...         self.strategy = strategy
    ...
    ...     def forward(self, waveform: torch.Tensor) -> torch.Tensor:
    ...         return self.strategy(waveform)

    References
    ----------
    For detailed examples of custom effect creation, including multi-channel
    processing and the strategy pattern, see the "Creating Custom Effects"
    wiki page.

    """

    @abc.abstractmethod
    def __init__(self, *args, **kwargs) -> None:  # type: ignore
        super().__init__(*args, **kwargs)

    @override
    @abc.abstractmethod
    def forward(self, x: Tensor) -> Tensor: ...


class Gain(FX):
    r"""Adjust volume of audio waveforms with multiple gain modes and optional clamping.

    The Gain effect modifies waveform amplitude using three different gain
    representations: direct amplitude multiplication, decibel (dB) adjustment,
    or power scaling. An optional clamping parameter prevents clipping artifacts
    by limiting output values to [-1.0, 1.0].

    This effect extends torchaudio.transforms.Vol by adding the clamp parameter
    for better control over output dynamic range.

    Parameters
    ----------
    gain : float
        The gain factor to apply to the waveform. Must be positive for
        "amplitude" and "power" gain types. Can be negative for "db" type.
    gain_type : str, optional
        The type of gain to apply. Default is "amplitude".

        - "amplitude": Direct multiplication by gain factor
        - "db": Decibel-based gain using torchaudio.functional.gain
        - "power": Power-based gain converted to dB internally
    clamp : bool, optional
        If True, clamps the output waveform to the range [-1.0, 1.0] to
        prevent clipping. Default is False.

    Raises
    ------
    ValueError
        If gain is negative when gain_type is "amplitude" or "power".

    See Also
    --------
    torchaudio.transforms.Vol : Original transform this effect is based on
    Normalize : Amplitude normalization with multiple strategies
    torchaudio.functional.gain : Function used for dB and power gain

    Notes
    -----
    **Gain Type Formulas:**

    - Amplitude: :math:`y[n] = x[n] \cdot \text{gain}`
    - Decibel: :math:`y[n] = x[n] \cdot 10^{\text{gain}/20}`
    - Power: :math:`y[n] = x[n] \cdot 10^{(10 \log_{10}(\text{gain}))/20}`

    **Clamping:**

    When clamp=True, the final output is constrained:
    :math:`y[n] = \text{clip}(y[n], -1.0, 1.0)`

    The @torch.no_grad() decorator disables gradient computation for efficiency
    during inference-only operations.

    This class is based on torchaudio.transforms.Vol, licensed under the
    BSD 2-Clause License. See licenses.torchaudio.BSD-2-Clause.txt for details.

    Examples
    --------
    Basic amplitude gain to double volume:

    >>> import torchfx as fx
    >>> wave = fx.Wave.from_file("audio.wav")
    >>> gain = fx.Gain(gain=2.0, gain_type="amplitude")
    >>> louder = wave | gain

    Increase volume by 6 dB with clamping:

    >>> gain = fx.Gain(gain=6.0, gain_type="db", clamp=True)
    >>> louder = wave | gain

    Increase power by 4x (equivalent to +6 dB or 2x amplitude):

    >>> gain = fx.Gain(gain=4.0, gain_type="power")
    >>> louder = wave | gain

    Reduce volume by 50% without clamping:

    >>> gain = fx.Gain(gain=0.5, gain_type="amplitude")
    >>> quieter = wave | gain

    Direct tensor processing:

    >>> import torch
    >>> waveform = torch.randn(2, 44100)  # (channels, samples)
    >>> gain = fx.Gain(gain=0.5, gain_type="amplitude", clamp=True)
    >>> quieter = gain(waveform)

    Negative dB for attenuation:

    >>> gain = fx.Gain(gain=-3.0, gain_type="db")
    >>> quieter = wave | gain

    Chain with other effects:

    >>> processed = wave | fx.Gain(2.0) | fx.Normalize(peak=0.8)

    """

    def __init__(self, gain: float, gain_type: str = "amplitude", clamp: bool = False) -> None:
        super().__init__()
        self.gain = gain
        self.gain_type = gain_type
        self.clamp = clamp

        if gain_type in ["amplitude", "power"] and gain < 0:
            raise ValueError("If gain_type = amplitude or power, gain must be positive.")

    @override
    @torch.no_grad()
    def forward(self, waveform: Tensor) -> Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension `(..., time)`.

        Returns:
            Tensor: Tensor of audio of dimension `(..., time)`.
        """
        if self.gain_type == "amplitude":
            waveform = waveform * self.gain

        if self.gain_type == "db":
            waveform = F.gain(waveform, self.gain)

        if self.gain_type == "power":
            waveform = F.gain(waveform, 10 * math.log10(self.gain))

        if self.clamp:
            waveform = torch.clamp(waveform, -1.0, 1.0)

        return waveform


class Normalize(FX):
    r"""Normalize waveform amplitude to a target peak value using pluggable strategies.

    The Normalize effect adjusts waveform amplitude to achieve a specified peak value
    using different normalization algorithms. The normalization strategy can be
    selected from built-in options (peak, RMS, percentile, per-channel) or provided
    as a custom callable function.

    This effect uses the strategy pattern to support multiple normalization algorithms
    while maintaining a clean interface. If no strategy is specified, peak normalization
    is used by default.

    Parameters
    ----------
    peak : float, optional
        The target peak value to normalize to. Must be positive. Default is 1.0.
    strategy : NormalizationStrategy or Callable[[Tensor, float], Tensor] or None, optional
        The normalization strategy to use. Can be:

        - None (default): Uses PeakNormalizationStrategy
        - NormalizationStrategy instance: Uses the specified strategy
        - Callable: Custom function wrapped in CustomNormalizationStrategy

        Built-in strategies:

        - PeakNormalizationStrategy: Normalize to absolute maximum value
        - RMSNormalizationStrategy: Normalize to RMS energy level
        - PercentileNormalizationStrategy: Normalize to a percentile threshold
        - PerChannelNormalizationStrategy: Normalize each channel independently

    Raises
    ------
    AssertionError
        If peak is not positive.
    TypeError
        If strategy is not an instance of NormalizationStrategy.

    See Also
    --------
    PeakNormalizationStrategy : Normalize to absolute maximum value
    RMSNormalizationStrategy : Normalize to RMS energy
    PercentileNormalizationStrategy : Normalize to percentile threshold
    PerChannelNormalizationStrategy : Independent per-channel normalization
    CustomNormalizationStrategy : Wrapper for custom normalization functions
    Gain : Volume adjustment with multiple gain modes

    Notes
    -----
    **Strategy Pattern:**

    The Normalize effect delegates processing to a strategy object, allowing
    different normalization algorithms to be used without modifying the core
    effect implementation. This design pattern promotes extensibility and
    clean separation of concerns.

    **Automatic Strategy Wrapping:**

    If a callable function is passed as the strategy parameter, it is
    automatically wrapped in a CustomNormalizationStrategy instance. The
    function must have the signature: ``func(waveform: Tensor, peak: float) -> Tensor``

    **Processing with @torch.no_grad():**

    The forward method is decorated with @torch.no_grad() for efficient
    inference-only operation. If gradients are needed for training, subclass
    this effect and remove the decorator.

    Examples
    --------
    Basic peak normalization to default peak of 1.0:

    >>> import torchfx as fx
    >>> wave = fx.Wave.from_file("audio.wav")
    >>> normalize = fx.Normalize()
    >>> normalized = wave | normalize

    Normalize to a specific peak value:

    >>> normalize = fx.Normalize(peak=0.8)
    >>> normalized = wave | normalize

    Use RMS normalization strategy:

    >>> from torchfx.effect import RMSNormalizationStrategy
    >>> normalize = fx.Normalize(peak=0.7, strategy=RMSNormalizationStrategy())
    >>> normalized = wave | normalize

    Use percentile normalization (99th percentile):

    >>> from torchfx.effect import PercentileNormalizationStrategy
    >>> normalize = fx.Normalize(peak=1.0, strategy=PercentileNormalizationStrategy(percentile=99.0))
    >>> normalized = wave | normalize

    Per-channel normalization for stereo audio:

    >>> from torchfx.effect import PerChannelNormalizationStrategy
    >>> normalize = fx.Normalize(peak=0.9, strategy=PerChannelNormalizationStrategy())
    >>> normalized = wave | normalize

    Custom normalization with a callable function:

    >>> def custom_normalize(waveform, peak):
    ...     # Normalize based on standard deviation
    ...     std = waveform.std()
    ...     return (waveform / std * peak) if std > 0 else waveform
    >>> normalize = fx.Normalize(peak=0.8, strategy=custom_normalize)
    >>> normalized = wave | normalize

    Direct tensor processing:

    >>> import torch
    >>> waveform = torch.randn(2, 44100)  # (channels, samples)
    >>> normalize = fx.Normalize(peak=0.5)
    >>> normalized = normalize(waveform)

    Chain with other effects:

    >>> result = wave | fx.Gain(2.0) | fx.Normalize(peak=0.8)

    References
    ----------
    For detailed information about creating custom normalization strategies and
    the strategy pattern, see wiki page "3.5 Creating Custom Effects".

    """

    def __init__(
        self,
        peak: float = 1.0,
        strategy: NormalizationStrategy | Callable[[Tensor, float], Tensor] | None = None,
    ) -> None:
        super().__init__()
        assert peak > 0, "Peak value must be positive."
        self.peak = peak

        if callable(strategy):
            strategy = CustomNormalizationStrategy(strategy)

        self.strategy = strategy or PeakNormalizationStrategy()
        if not isinstance(self.strategy, NormalizationStrategy):
            raise TypeError("Strategy must be an instance of NormalizationStrategy.")

    @override
    @torch.no_grad()
    def forward(self, waveform: Tensor) -> Tensor:
        return self.strategy(waveform, self.peak)


class NormalizationStrategy(abc.ABC):
    """Abstract base class for normalization strategies.

    NormalizationStrategy defines the interface for all normalization algorithms
    used by the Normalize effect. Concrete implementations must implement the
    __call__ method to provide specific normalization logic.

    This class is part of the strategy pattern implementation, allowing the
    Normalize effect to support multiple normalization algorithms without
    modifying its core implementation.

    Methods
    -------
    __call__(waveform: Tensor, peak: float) -> Tensor
        Normalize the waveform to the given peak value using the strategy's
        specific algorithm.

    See Also
    --------
    Normalize : The effect that uses normalization strategies
    PeakNormalizationStrategy : Normalize to absolute maximum value
    RMSNormalizationStrategy : Normalize to RMS energy
    PercentileNormalizationStrategy : Normalize to percentile threshold
    PerChannelNormalizationStrategy : Independent per-channel normalization
    CustomNormalizationStrategy : Wrapper for custom functions

    Notes
    -----
    When implementing a custom normalization strategy, ensure that:

    1. The __call__ method handles edge cases (e.g., silent audio)
    2. The returned tensor has the same shape and dtype as the input
    3. The strategy preserves the device of the input tensor

    Examples
    --------
    Implement a custom normalization strategy:

    >>> from torchfx.effect import NormalizationStrategy
    >>> import torch
    >>>
    >>> class MedianNormalizationStrategy(NormalizationStrategy):
    ...     def __call__(self, waveform: torch.Tensor, peak: float) -> torch.Tensor:
    ...         median = torch.median(torch.abs(waveform))
    ...         return waveform / median * peak if median > 0 else waveform

    Use the custom strategy:

    >>> import torchfx as fx
    >>> wave = fx.Wave.from_file("audio.wav")
    >>> normalize = fx.Normalize(peak=0.8, strategy=MedianNormalizationStrategy())
    >>> normalized = wave | normalize

    References
    ----------
    For more information about the strategy pattern and creating custom
    strategies, see wiki page "3.5 Creating Custom Effects".

    """

    @abc.abstractmethod
    def __call__(self, waveform: Tensor, peak: float) -> Tensor:
        """Normalize the waveform to the given peak value.

        Parameters
        ----------
        waveform : Tensor
            Input audio tensor of shape (..., time).
        peak : float
            Target peak value for normalization.

        Returns
        -------
        Tensor
            Normalized waveform with same shape and dtype as input.

        """
        pass


class CustomNormalizationStrategy(NormalizationStrategy):
    """Normalization using a custom user-provided function.

    This strategy wraps a user-provided callable function to make it compatible
    with the NormalizationStrategy interface. It is automatically used when a
    callable is passed to the Normalize effect's strategy parameter.

    Parameters
    ----------
    func : Callable[[Tensor, float], Tensor]
        Custom normalization function with signature:
        func(waveform: Tensor, peak: float) -> Tensor

    Raises
    ------
    AssertionError
        If func is not callable.

    See Also
    --------
    Normalize : Effect that uses this strategy wrapper
    NormalizationStrategy : Abstract base class for strategies

    Notes
    -----
    The custom function must:

    - Accept two parameters: waveform (Tensor) and peak (float)
    - Return a normalized Tensor with the same shape and dtype as input
    - Preserve the device of the input tensor
    - Handle edge cases (e.g., silent audio with all zeros)

    Examples
    --------
    Define a custom normalization function:

    >>> import torch
    >>> def std_normalize(waveform, peak):
    ...     std = waveform.std()
    ...     return (waveform / std * peak) if std > 0 else waveform

    Use directly with Normalize (automatically wrapped):

    >>> import torchfx as fx
    >>> wave = fx.Wave.from_file("audio.wav")
    >>> normalize = fx.Normalize(peak=0.8, strategy=std_normalize)
    >>> normalized = wave | normalize

    Or explicitly instantiate the strategy:

    >>> from torchfx.effect import CustomNormalizationStrategy
    >>> strategy = CustomNormalizationStrategy(std_normalize)
    >>> normalize = fx.Normalize(peak=0.8, strategy=strategy)

    """

    def __init__(self, func: Callable[[Tensor, float], Tensor]) -> None:
        assert callable(func), "func must be callable"
        self.func = func

    def __call__(self, waveform: Tensor, peak: float) -> Tensor:
        return self.func(waveform, peak)


class PeakNormalizationStrategy(NormalizationStrategy):
    r"""Normalization to the absolute peak value.

    .. math::
        y[n] =
        \begin{cases}
            \frac{x[n]}{max(|x[n]|)} \cdot peak, & \text{if } max(|x[n]|) > 0 \\
            x[n], & \text{otherwise}
        \end{cases}

    where:
        - :math:`x[n]` is the input signal,
        - :math:`y[n]` is the output signal,
        - :math:`peak` is the target peak value.

    """

    def __call__(self, waveform: Tensor, peak: float) -> Tensor:
        max_val = torch.max(torch.abs(waveform))
        return waveform / max_val * peak if max_val > 0 else waveform


class RMSNormalizationStrategy(NormalizationStrategy):
    r"""Normalization to Root Mean Square (RMS) energy.

    .. math::
        y[n] =
        \begin{cases}
            \frac{x[n]}{RMS(x[n])} \cdot peak, & \text{if } RMS(x[n]) > 0 \\
            x[n], & \text{otherwise}
        \end{cases}

    where:
        - :math:`x[n]` is the input signal,
        - :math:`y[n]` is the output signal,
        - :math:`RMS(x[n])` is the root mean square of the signal,
        - :math:`peak` is the target peak value.

    """

    def __call__(self, waveform: Tensor, peak: float) -> Tensor:
        rms = torch.sqrt(torch.mean(waveform**2))
        return waveform / rms * peak if rms > 0 else waveform


class PercentileNormalizationStrategy(NormalizationStrategy):
    r"""Normalization using a percentile of absolute values.

    .. math::
        y[n] =
        \begin{cases}
            \frac{x[n]}{P_p(|x[n]|)} \cdot peak, & \text{if } P_p(|x[n]|) > 0 \\
            x[n], & \text{otherwise}
        \end{cases}

    where:
        - :math:`x[n]` is the input signal,
        - :math:`y[n]` is the output signal,
        - :math:`P_p(|x[n]|)` is the p-th percentile of the absolute values of the signal,
        - :math:`peak` is the target peak value,
        - :math:`p` is the specified percentile (:math:`0 < p \leqslant 100`).

    Attributes
    ----------
    percentile : float
        The percentile :math:`p` to use for normalization (:math:`0 < p \leqslant 100`). Default is 99.0.

    """

    def __init__(self, percentile: float = 99.0) -> None:
        assert 0 < percentile <= 100, "Percentile must be between 0 and 100."
        self.percentile = percentile

    def __call__(self, waveform: Tensor, peak: float) -> Tensor:
        abs_waveform = torch.abs(waveform)
        threshold = torch.quantile(abs_waveform, self.percentile / 100, interpolation="linear")
        return waveform / threshold * peak if threshold > 0 else waveform


class PerChannelNormalizationStrategy(NormalizationStrategy):
    r"""Normalize each channel independently to its own peak.

    .. math::
        y_c[n] =
        \begin{cases}
            \frac{x_c[n]}{max(|x_c[n]|)} \cdot peak, & \text{if } max(|x_c[n]|) > 0 \\
            x_c[n], & \text{otherwise}
        \end{cases}

    where:
        - :math:`x_c[n]` is the input signal for channel c,
        - :math:`y_c[n]` is the output signal for channel c,
        - :math:`peak` is the target peak value.

    """

    def __call__(self, waveform: Tensor, peak: float) -> Tensor:
        assert waveform.ndim >= 2, "Waveform must have at least 2 dimensions (channels, time)."

        # waveform: (channels, time) or (batch, channels, time)
        dims = waveform.ndim
        if dims == 2:
            max_per_channel = torch.max(torch.abs(waveform), dim=1, keepdim=True).values
            return torch.where(max_per_channel > 0, waveform / max_per_channel * peak, waveform)
        elif dims == 3:
            max_per_channel = torch.max(torch.abs(waveform), dim=2, keepdim=True).values
            return torch.where(max_per_channel > 0, waveform / max_per_channel * peak, waveform)
        else:
            raise ValueError("Waveform must have shape (C, T) or (B, C, T)")


class Reverb(FX):
    r"""Apply reverb effect using a feedback delay network for spatial ambiance.

    The Reverb effect creates spatial ambiance by simulating sound reflections
    in an acoustic space. It uses a simple feedback comb filter (feedback delay
    network) to produce reverb-like effects with controllable decay time and
    wet/dry mix.

    This is a basic reverb implementation suitable for adding spatial depth to
    audio signals. For more complex reverb algorithms, consider using convolution
    reverbs with impulse responses.

    Parameters
    ----------
    delay : int, optional
        Delay time in samples for the feedback comb filter. Determines the
        apparent size of the simulated space. Default is 4410 samples, which
        corresponds to approximately 100ms at 44.1kHz sample rate.
    decay : float, optional
        Feedback decay factor controlling how quickly the reverb tail fades.
        Must be in the range (0, 1). Higher values create longer reverb tails.
        Default is 0.5.
    mix : float, optional
        Wet/dry mix controlling the balance between processed (wet) and
        original (dry) signals. Range is [0, 1] where:

        - 0.0 = fully dry (no reverb)
        - 1.0 = fully wet (only reverb)
        - 0.5 = equal mix

        Default is 0.5.

    Raises
    ------
    AssertionError
        If delay is not positive, decay is not in (0, 1), or mix is not in [0, 1].

    See Also
    --------
    Delay : Multi-tap delay effect with BPM synchronization
    Gain : Volume adjustment effect

    Notes
    -----
    **Algorithm:**

    The reverb is computed using a feedback comb filter:

    .. math::

        y[n] = (1 - mix) \cdot x[n] + mix \cdot (x[n] + decay \cdot x[n - delay])

    where:
        - :math:`x[n]` is the input signal
        - :math:`y[n]` is the output signal
        - :math:`delay` is the delay time in samples
        - :math:`decay` is the feedback decay factor
        - :math:`mix` is the wet/dry mix parameter

    **Processing Details:**

    - If the input waveform is shorter than the delay time, the input is returned
      unchanged.
    - The effect processes tensors of arbitrary shape (..., time).
    - Uses @torch.no_grad() decorator for efficient inference-only operation.
    - Padding is applied using torch.nn.functional.pad for the delay buffer.

    **Delay Time Calculation:**

    To convert time in milliseconds to samples:

    .. math::

        delay_{samples} = \frac{time_{ms}}{1000} \cdot sample\_rate

    For example, at 44.1kHz:
        - 50ms = 2205 samples
        - 100ms = 4410 samples (default)
        - 200ms = 8820 samples

    Examples
    --------
    Basic reverb with default parameters:

    >>> import torchfx as fx
    >>> wave = fx.Wave.from_file("audio.wav")
    >>> reverb = fx.Reverb()
    >>> processed = wave | reverb

    Short room reverb (50ms delay):

    >>> reverb = fx.Reverb(delay=2205, decay=0.4, mix=0.3)
    >>> processed = wave | reverb

    Long hall reverb (200ms delay):

    >>> reverb = fx.Reverb(delay=8820, decay=0.7, mix=0.4)
    >>> processed = wave | reverb

    Subtle reverb with low mix:

    >>> reverb = fx.Reverb(delay=4410, decay=0.5, mix=0.2)
    >>> processed = wave | reverb

    Direct tensor processing:

    >>> import torch
    >>> waveform = torch.randn(2, 44100)  # (channels, samples)
    >>> reverb = fx.Reverb(delay=4410, decay=0.6, mix=0.3)
    >>> reverberated = reverb(waveform)

    Chain with other effects:

    >>> processed = wave | fx.Gain(0.8) | fx.Reverb(delay=4410, decay=0.5, mix=0.3)

    GPU processing:

    >>> wave = wave.to("cuda")
    >>> reverb = fx.Reverb(delay=4410, decay=0.6, mix=0.3).to("cuda")
    >>> processed = wave | reverb

    """

    def __init__(self, delay: int = 4410, decay: float = 0.5, mix: float = 0.5) -> None:
        super().__init__()
        assert delay > 0, "Delay must be positive."
        assert 0 < decay < 1, "Decay must be between 0 and 1."
        assert 0 <= mix <= 1, "Mix must be between 0 and 1."

        self.delay = delay
        self.decay = decay
        self.mix = mix

    @override
    @torch.no_grad()
    def forward(self, waveform: Tensor) -> Tensor:
        # waveform: (..., time)
        if waveform.size(-1) <= self.delay:
            return waveform

        # Pad waveform for delay
        padded = torch.nn.functional.pad(waveform, (self.delay, 0))
        # Create delayed signal
        delayed = padded[..., : -self.delay]
        # Feedback comb filter
        reverb_signal = waveform + self.decay * delayed
        # Wet/dry mix
        output = (1 - self.mix) * waveform + self.mix * reverb_signal
        return output


class DelayStrategy(abc.ABC):
    """Abstract base class for delay processing strategies.

    DelayStrategy defines the interface for different delay processing behaviors
    used by the Delay effect. Concrete implementations provide specific delay
    algorithms such as mono delay (uniform across all channels) or ping-pong
    delay (alternating between stereo channels).

    This class is part of the strategy pattern implementation, allowing the
    Delay effect to support multiple processing behaviors without modifying
    its core implementation.

    Methods
    -------
    apply_delay(waveform, delay_samples, taps, feedback) -> Tensor
        Apply the delay effect to the waveform using the strategy's specific
        algorithm.

    See Also
    --------
    Delay : The effect that uses delay strategies
    MonoDelayStrategy : Uniform delay for all channels
    PingPongDelayStrategy : Alternating stereo delay

    Notes
    -----
    When implementing a custom delay strategy:

    1. The output length should be extended to accommodate all delayed taps:
       ``output_length = input_length + (delay_samples * taps)``
    2. The first tap always has amplitude 1.0, subsequent taps use feedback
       scaling: ``feedback^(tap-1)``
    3. The returned tensor should preserve the device and dtype of the input
    4. Handle different tensor dimensions: 1D (mono), 2D (multi-channel),
       and higher dimensions

    Examples
    --------
    Implement a custom delay strategy:

    >>> from torchfx.effect import DelayStrategy
    >>> import torch
    >>>
    >>> class CrossChannelDelayStrategy(DelayStrategy):
    ...     '''Apply delay from each channel to all other channels.'''
    ...     def apply_delay(self, waveform, delay_samples, taps, feedback):
    ...         # Custom cross-channel delay logic
    ...         original_length = waveform.size(-1)
    ...         output_length = original_length + delay_samples * taps
    ...         # ... implementation ...
    ...         return delayed_waveform

    Use with Delay effect:

    >>> import torchfx as fx
    >>> wave = fx.Wave.from_file("audio.wav")
    >>> delay = fx.Delay(delay_samples=2205, taps=3, feedback=0.4, mix=0.3,
    ...                  strategy=CrossChannelDelayStrategy())
    >>> processed = wave | delay

    References
    ----------
    For more information about the strategy pattern and creating custom
    strategies, see wiki page "3.5 Creating Custom Effects".

    """

    @abc.abstractmethod
    def apply_delay(
        self, waveform: Tensor, delay_samples: int, taps: int, feedback: float
    ) -> Tensor:
        """Apply delay processing to the waveform.

        Parameters
        ----------
        waveform : Tensor
            Input audio tensor of shape (..., time) or (channels, time).
        delay_samples : int
            Delay time in samples for each tap.
        taps : int
            Number of delay taps (echoes). Each tap is delayed by
            delay_samples * tap_number.
        feedback : float
            Feedback amount in range [0, 0.95]. Controls the amplitude of
            taps 2 and beyond. First tap always has amplitude 1.0, subsequent
            taps use feedback^(tap-1).

        Returns
        -------
        Tensor
            Delayed audio with extended length to accommodate all taps.
            Output length is: input_length + (delay_samples * taps).

        """
        pass


class MonoDelayStrategy(DelayStrategy):
    """Apply uniform delay to all channels with multiple taps and feedback.

    MonoDelayStrategy applies the same delay pattern to all audio channels,
    creating identical echoes across the stereo field. This is the default
    delay strategy used by the Delay effect.

    The strategy creates multiple delay taps (echoes), each delayed by an
    integer multiple of the base delay time. The first tap has full amplitude,
    and subsequent taps decay exponentially based on the feedback parameter.

    See Also
    --------
    DelayStrategy : Abstract base class for delay strategies
    PingPongDelayStrategy : Alternating stereo delay
    Delay : The effect that uses this strategy

    Notes
    -----
    **Output Length:**

    The output is extended to accommodate all delayed taps:
    ``output_length = input_length + (delay_samples * taps)``

    **Tap Amplitude:**

    - Tap 1: amplitude = 1.0
    - Tap n (n > 1): amplitude = feedback^(n-1)

    **Multi-dimensional Support:**

    The strategy handles tensors of various shapes:

    - 1D: (time,) - Mono audio
    - 2D: (channels, time) - Multi-channel audio
    - Higher dimensions: (..., time) - Batched or complex audio

    Examples
    --------
    Use mono delay strategy explicitly:

    >>> import torchfx as fx
    >>> from torchfx.effect import MonoDelayStrategy
    >>> wave = fx.Wave.from_file("audio.wav")
    >>> delay = fx.Delay(delay_samples=2205, taps=4, feedback=0.5, mix=0.3,
    ...                  strategy=MonoDelayStrategy())
    >>> processed = wave | delay

    MonoDelayStrategy is the default, so this is equivalent:

    >>> delay = fx.Delay(delay_samples=2205, taps=4, feedback=0.5, mix=0.3)
    >>> processed = wave | delay

    """

    def apply_delay(
        self, waveform: Tensor, delay_samples: int, taps: int, feedback: float
    ) -> Tensor:
        """Apply mono delay with multiple taps and feedback.

        Output length is extended to accommodate all delayed taps.

        Parameters
        ----------
        waveform : Tensor
            Input audio tensor of shape (time,), (channels, time), or (..., time).
        delay_samples : int
            Delay time in samples for each tap.
        taps : int
            Number of delay taps (echoes).
        feedback : float
            Feedback amount for taps 2 and beyond.

        Returns
        -------
        Tensor
            Delayed audio with shape matching input except extended time dimension.

        """
        # Calculate required output length
        original_length = waveform.size(-1)
        max_delay_samples = delay_samples * taps
        output_length = original_length + max_delay_samples

        # waveform shape: (..., time) or (channels, time)
        if waveform.ndim == 1:
            # Single channel: (time,)
            delayed = torch.zeros(output_length, dtype=waveform.dtype, device=waveform.device)
            for tap in range(1, taps + 1):
                tap_delay = delay_samples * tap
                # First tap always has amplitude 1.0, subsequent taps use feedback
                # Copy original signal starting
                feedback_amt = 1.0 if tap == 1 else feedback ** (tap - 1)

                # Copy original signal starting at tap_delay
                copy_length = min(original_length, output_length - tap_delay)
                if copy_length > 0:
                    delayed[tap_delay : tap_delay + copy_length] += (
                        waveform[:copy_length] * feedback_amt
                    )
            return delayed

        elif waveform.ndim == 2:
            # Multi-channel: (channels, time)
            delayed = torch.zeros(
                waveform.size(0), output_length, dtype=waveform.dtype, device=waveform.device
            )
            for ch in range(waveform.size(0)):
                for tap in range(1, taps + 1):
                    tap_delay = delay_samples * tap
                    # First tap always has amplitude 1.0, subsequent taps use feedback
                    feedback_amt = 1.0 if tap == 1 else feedback ** (tap - 1)
                    # Copy original signal starting at tap_delay
                    copy_length = min(original_length, output_length - tap_delay)
                    if copy_length > 0:
                        delayed[ch, tap_delay : tap_delay + copy_length] += (
                            waveform[ch, :copy_length] * feedback_amt
                        )
            return delayed

        else:
            # Higher dimensions: (..., time)
            # Flatten to (channels, time) for processing
            original_shape = list(waveform.shape)
            flattened = waveform.view(-1, waveform.size(-1))
            processed = self.apply_delay(flattened, delay_samples, taps, feedback)
            # Reshape with extended time dimension
            new_shape = original_shape[:-1] + [processed.size(-1)]
            return processed.view(new_shape)


class PingPongDelayStrategy(DelayStrategy):
    """Apply ping-pong delay alternating between left and right channels.

    PingPongDelayStrategy creates a stereo delay effect where echoes alternate
    between the left and right channels, producing a "bouncing" or "ping-pong"
    spatial effect. This is commonly used in music production for creating
    wide, spacious delay effects.

    The strategy requires stereo (2-channel) input. For non-stereo audio, it
    automatically falls back to MonoDelayStrategy.

    See Also
    --------
    DelayStrategy : Abstract base class for delay strategies
    MonoDelayStrategy : Uniform delay for all channels
    Delay : The effect that uses this strategy

    Notes
    -----
    **Ping-Pong Pattern:**

    - Odd taps (1, 3, 5, ...): Left channel → Right channel
    - Even taps (2, 4, 6, ...): Right channel → Left channel

    This creates the characteristic bouncing effect where the delay appears to
    move back and forth between the left and right speakers.

    **Fallback Behavior:**

    If the input is not stereo (2 channels), the strategy automatically falls
    back to MonoDelayStrategy to process the audio.

    **Output Length:**

    The output is extended to accommodate all delayed taps:
    ``output_length = input_length + (delay_samples * taps)``

    **Tap Amplitude:**

    Same as MonoDelayStrategy:

    - Tap 1: amplitude = 1.0
    - Tap n (n > 1): amplitude = feedback^(n-1)

    Examples
    --------
    Create ping-pong delay effect:

    >>> import torchfx as fx
    >>> from torchfx.effect import PingPongDelayStrategy
    >>> wave = fx.Wave.from_file("stereo_audio.wav")  # Must be stereo
    >>> delay = fx.Delay(delay_samples=2205, taps=6, feedback=0.5, mix=0.4,
    ...                  strategy=PingPongDelayStrategy())
    >>> processed = wave | delay

    BPM-synced ping-pong delay:

    >>> delay = fx.Delay(bpm=120, delay_time="1/8", taps=8, feedback=0.6, mix=0.3,
    ...                  strategy=PingPongDelayStrategy())
    >>> processed = wave | delay

    Combine with reverb for spacious effect:

    >>> reverb = fx.Reverb(delay=4410, decay=0.6, mix=0.2)
    >>> delay = fx.Delay(bpm=128, delay_time="1/4", taps=4, feedback=0.5, mix=0.3,
    ...                  strategy=PingPongDelayStrategy())
    >>> processed = wave | reverb | delay

    """

    def apply_delay(
        self, waveform: Tensor, delay_samples: int, taps: int, feedback: float
    ) -> Tensor:
        """Apply ping-pong delay (alternates between channels).

        Output length is extended to accommodate all delayed taps.

        Parameters
        ----------
        waveform : Tensor
            Input audio tensor. Should be stereo with shape (2, time) or
            (..., 2, time). For non-stereo input, falls back to MonoDelayStrategy.
        delay_samples : int
            Delay time in samples for each tap.
        taps : int
            Number of delay taps (echoes).
        feedback : float
            Feedback amount for taps 2 and beyond.

        Returns
        -------
        Tensor
            Delayed audio with ping-pong effect. Shape matches input except
            extended time dimension.

        """
        if waveform.ndim < 2 or waveform.size(-2) != 2:
            # Not stereo, fall back to mono
            return MonoDelayStrategy().apply_delay(waveform, delay_samples, taps, feedback)

        # Calculate required output length
        original_length = waveform.size(-1)
        max_delay_samples = delay_samples * taps
        output_length = original_length + max_delay_samples

        # waveform: (2, time) or (..., 2, time)
        if waveform.ndim == 2:
            # Simple case: (2, time)
            delayed = torch.zeros(2, output_length, dtype=waveform.dtype, device=waveform.device)
            for tap in range(1, taps + 1):
                tap_delay = delay_samples * tap
                # First tap always has amplitude 1.0, subsequent taps use feedback
                feedback_amt = 1.0 if tap == 1 else feedback ** (tap - 1)

                # Copy length for this tap
                copy_length = min(original_length, output_length - tap_delay)
                if copy_length > 0:
                    # Odd taps: left delays to right, even taps: right delays to left
                    if tap % 2 == 1:
                        # Left -> Right
                        delayed[1, tap_delay : tap_delay + copy_length] += (
                            waveform[0, :copy_length] * feedback_amt
                        )
                    else:
                        # Right -> Left
                        delayed[0, tap_delay : tap_delay + copy_length] += (
                            waveform[1, :copy_length] * feedback_amt
                        )
            return delayed

        else:
            # Higher dimensions: (..., 2, time)
            original_shape = list(waveform.shape)
            original_shape[-1] = output_length
            delayed = torch.zeros(original_shape, dtype=waveform.dtype, device=waveform.device)
            for tap in range(1, taps + 1):
                tap_delay = delay_samples * tap
                # First tap always has amplitude 1.0, subsequent taps use feedback
                feedback_amt = 1.0 if tap == 1 else feedback ** (tap - 1)

                # Copy length for this tap
                copy_length = min(original_length, output_length - tap_delay)
                if copy_length > 0:
                    if tap % 2 == 1:
                        # Left -> Right
                        delayed[..., 1, tap_delay : tap_delay + copy_length] += (
                            waveform[..., 0, :copy_length] * feedback_amt
                        )
                    else:
                        # Right -> Left
                        delayed[..., 0, tap_delay : tap_delay + copy_length] += (
                            waveform[..., 1, :copy_length] * feedback_amt
                        )

            return delayed


class Delay(FX):
    r"""Apply a delay effect with BPM-synced musical time divisions.

    The delay effect creates echoes of the input signal with configurable feedback.
    Supports BPM-synced delay times for musical applications.

    The delay effect is computed as:

    .. math::

        delayed[n] = \sum_{i=1}^{taps} feedback^{i-1} \cdot x[n - i \cdot delay]
        y[n] = (1 - mix) x[n] + mix \cdot delayed[n]

    where:
        - x[n] is the input signal,
        - y[n] is the output signal,
        - delay is the delay time in samples,
        - feedback is the feedback amount (0-0.95) affecting taps 2 and beyond,
        - taps is the number of delay taps,
        - mix is the wet/dry mix parameter.

    Parameters
    ----------
    delay_samples : int
        Delay time in samples. If provided, this is used directly.
        Default is None (requires bpm and delay_time).
    bpm : float
        Beats per minute for BPM-synced delay. Required if delay_samples is None.
    delay_time : str
        Musical time division for BPM-synced delay. Should be a string in the format :code:`n/d[modifier]`, where:

        * :code:`n/d` represents the note division (e.g., :code:`1/4` for quarter note).
        * :code:`modifier` is optional and can be :code:`d` for dotted notes or :code:`t` for triplets.

        Valid examples include:

        * :code:`1/4`: Quarter note
        * :code:`1/8`: Eighth note
        * :code:`1/16`: Sixteenth note
        * :code:`1/8d`: Dotted eighth note
        * :code:`1/4d`: Dotted quarter note
        * :code:`1/8t`: Eighth note triplet

        Default is :code:`1/8`.
    fs : int | None
        Sample frequency (sample rate) in Hz. Required if using BPM-synced delay
        without Wave pipeline. When None (default), fs will be automatically inferred
        from the Wave object when used with the pipeline operator (wave | delay).
        Must be positive if provided. Default is None.
    feedback : float
        Feedback amount (0-0.95). Controls amplitude of taps 2 and beyond.
        First tap always has amplitude 1.0. Higher values create more prominent echoes.
        Default is 0.3.
    mix : float
        Wet/dry mix. 0 = dry (original signal only), 1 = wet (delayed echoes only).
        Default is 0.2.
    taps : int
        Number of delay taps (echoes). Each tap is delayed by delay_samples * tap_number.
        Default is 3.
    strategy : DelayStrategy | None
        Delay processing strategy. If None, defaults to MonoDelayStrategy.
        Use PingPongDelayStrategy for stereo ping-pong effect, or provide a custom
        strategy extending DelayStrategy. Default is None.

    Examples
    --------
    >>> import torchfx as fx
    >>> import torch
    >>>
    >>> # BPM-synced delay with auto fs inference from Wave
    >>> wave = fx.Wave.from_file("audio.wav")
    >>> delay = fx.effect.Delay(bpm=128, delay_time='1/8', feedback=0.3, mix=0.2)
    >>> delayed = wave | delay  # fs automatically inferred from wave
    >>>
    >>> # BPM-synced delay with explicit fs
    >>> waveform = torch.randn(2, 44100)  # (channels, samples)
    >>> delay = fx.effect.Delay(bpm=128, delay_time='1/8', fs=44100, feedback=0.3, mix=0.2)
    >>> delayed = delay(waveform)
    >>>
    >>> # Direct delay in samples (no fs needed)
    >>> delay = fx.effect.Delay(delay_samples=2205, feedback=0.4, mix=0.3)
    >>> delayed = delay(waveform)
    >>>
    >>> # Ping-pong delay with strategy
    >>> delay = fx.effect.Delay(
    ...     bpm=128, delay_time='1/4', fs=44100,
    ...     feedback=0.5, mix=0.4, strategy=fx.effect.PingPongDelayStrategy()
    ... )
    >>> delayed = delay(waveform)

    Author
    ------
    Uzef <@itsuzef>

    """

    def __init__(
        self,
        delay_samples: int | None = None,
        bpm: float | None = None,
        delay_time: str = "1/8",
        fs: int | None = None,
        feedback: float = 0.3,
        mix: float = 0.2,
        taps: int = 3,
        strategy: DelayStrategy | None = None,
    ) -> None:
        super().__init__()

        self.fs = fs  # Store for Wave.__update_config to set automatically
        self.bpm = bpm
        self.delay_time = delay_time

        # If delay_samples is provided directly, use it
        if delay_samples is not None:
            assert delay_samples > 0, "Delay samples must be positive."
            self.delay_samples = delay_samples
            self._needs_calculation = False
        else:
            # BPM-synced delay requires bpm parameter
            assert bpm is not None, "BPM must be provided if delay_samples is not set."
            assert bpm > 0, "BPM must be positive."

            # If fs is available now, calculate immediately
            if fs is not None:
                assert fs > 0, "Sample rate (fs) must be positive."
                self.delay_samples = self._calculate_delay_samples(bpm, delay_time, fs)
                self._needs_calculation = False
            else:
                # Defer calculation until fs is set (by Wave.__update_config)
                self.delay_samples = None  # type: ignore
                self._needs_calculation = True

        # Validate other parameters
        assert 0 <= feedback <= 0.95, "Feedback must be between 0 and 0.95."
        assert 0 <= mix <= 1, "Mix must be between 0 and 1."
        assert taps >= 1, "Taps must be at least 1."

        self.feedback = feedback
        self.mix = mix
        self.taps = taps
        self.strategy = strategy or MonoDelayStrategy()

    @staticmethod
    def _calculate_delay_samples(bpm: float, delay_time: str, fs: int) -> int:
        """Calculate delay time in samples from BPM and musical division.

        Parameters
        ----------
        bpm : float
            Beats per minute.
        delay_time : str
            Musical time division string (e.g., "1/4", "1/8d", "1/8t").
        fs : int
            Sample frequency in Hz.

        Returns
        -------
        int
            Delay time in samples.

        """
        from torchfx.typing import MusicalTime

        musical_time = MusicalTime.from_string(delay_time)
        delay_sec = musical_time.duration_seconds(bpm)
        return int(delay_sec * fs)

    def _extend_waveform(self, waveform: Tensor, target_length: int) -> Tensor:
        """Extend waveform with zeros to target length along the last dimension."""
        if waveform.size(-1) >= target_length:
            return waveform

        if waveform.ndim == 1:
            extended = torch.zeros(target_length, dtype=waveform.dtype, device=waveform.device)
            extended[: waveform.size(0)] = waveform
        elif waveform.ndim == 2:
            extended = torch.zeros(
                waveform.size(0), target_length, dtype=waveform.dtype, device=waveform.device
            )
            extended[:, : waveform.size(1)] = waveform
        else:
            extended_shape = list(waveform.shape)
            extended_shape[-1] = target_length
            extended = torch.zeros(extended_shape, dtype=waveform.dtype, device=waveform.device)
            extended[..., : waveform.size(-1)] = waveform

        return extended

    @override
    @torch.no_grad()
    def forward(self, waveform: Tensor) -> Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension `(..., time)` or `(channels, time)`.

        Returns:
            Tensor: Tensor of delayed audio. Output length is extended to accommodate delayed echoes.
            The output will be longer than the input by up to `delay_samples * taps` samples.
        """
        # Lazy calculation of delay_samples if needed
        if self._needs_calculation:
            assert self.fs is not None, (
                "Sample rate (fs) is required for BPM-synced delay."
                "Either provide fs parameter or use with Wave pipeline (wave | delay)."
            )
            assert self.fs > 0, "Sample rate (fs) must be positive."
            assert self.bpm is not None, "BPM must be set for BPM-synced delay."

            self.delay_samples = self._calculate_delay_samples(self.bpm, self.delay_time, self.fs)
            self._needs_calculation = False

        # Apply delay using strategy pattern
        delayed = self.strategy.apply_delay(waveform, self.delay_samples, self.taps, self.feedback)

        # Extend original waveform to match delayed length for mixing
        waveform = self._extend_waveform(waveform, delayed.size(-1))

        # Wet/dry mix
        output = (1 - self.mix) * waveform + self.mix * delayed
        return output
