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

    def __or__(self, other: nn.Module) -> nn.Sequential:
        if not isinstance(other, nn.Module):
            return NotImplemented
        from torchfx.chain import FilterChain

        return FilterChain(self, other)


class Gain(FX):
    r"""Adjust volume of audio waveforms with multiple gain modes and optional clamping.

    The Gain effect modifies waveform amplitude using three different gain
    representations: direct amplitude multiplication, decibel (dB) adjustment,
    or power scaling. An optional clamping parameter prevents clipping artifacts
    by limiting output values to [-1.0, 1.0].

    Parameters
    ----------
    gain : float
        The gain factor to apply to the waveform. Must be positive for
        ``"amplitude"`` and ``"power"`` gain types. Can be negative for
        ``"db"``.
    gain_type : {"amplitude", "db", "power"}, default="amplitude"
        How the ``gain`` value is interpreted:

        - ``"amplitude"``: direct multiplication by ``gain``
        - ``"db"``: gain in decibels (output multiplied by ``10 ** (gain/20)``)
        - ``"power"``: power ratio, converted to dB internally
    clamp : bool, default=False
        If True, clamp the output waveform to ``[-1.0, 1.0]`` after applying
        the gain.

    Raises
    ------
    ValueError
        If gain is negative when gain_type is ``"amplitude"`` or ``"power"``.

    See Also
    --------
    Normalize : Amplitude normalization with multiple strategies.

    Notes
    -----
    **Gain Type Formulas:**

    - Amplitude: :math:`y[n] = x[n] \cdot \text{gain}`
    - Decibel: :math:`y[n] = x[n] \cdot 10^{\text{gain}/20}`
    - Power: :math:`y[n] = x[n] \cdot 10^{(10 \log_{10}(\text{gain}))/20}`

    **Clamping:**

    When ``clamp=True`` the output is constrained:
    :math:`y[n] = \text{clip}(y[n], -1.0, 1.0)`

    Coefficient formulas are adapted from ``torchaudio.transforms.Vol``,
    BSD 2-Clause License (see ``licenses.torchaudio.BSD-2-Clause.txt``).

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
        valid_gain_types = {"amplitude", "db", "power"}
        if gain_type not in valid_gain_types:
            raise ValueError(
                f"gain_type must be one of {sorted(valid_gain_types)}, got {gain_type!r}."
            )
        self.gain = gain
        self.gain_type = gain_type
        self.clamp = clamp

        if gain_type in ("amplitude", "power") and gain < 0:
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

        elif self.gain_type == "db" and self.gain != 0:
            waveform = waveform * 10 ** (self.gain / 20)

        elif self.gain_type == "power" and self.gain != 1:
            waveform = waveform * 10 ** (math.log10(self.gain) / 2)

        if self.clamp:
            waveform = torch.clamp(waveform, -1.0, 1.0)

        return waveform

    def _linear_gain(self) -> float | None:
        """Return the constant linear factor this gain multiplies by, else ``None``.

        Used by the fusion planner to fold a static gain into an adjacent SOS
        cascade (a scalar commutes through a linear filter). Returns ``None`` when
        the gain is *not* a pure linear scale — i.e. when ``clamp=True``, which adds
        a non-linear clip — so such gains are left as standalone stages.

        Examples
        --------
        >>> import torchfx as fx
        >>> fx.Gain(2.0)._linear_gain()
        2.0
        >>> fx.Gain(6.0, gain_type="db", clamp=True)._linear_gain() is None
        True

        """
        if self.clamp:
            return None
        if self.gain_type == "amplitude":
            return float(self.gain)
        if self.gain_type == "db":
            return float(10 ** (self.gain / 20)) if self.gain != 0 else 1.0
        if self.gain_type == "power":
            return float(10 ** (math.log10(self.gain) / 2)) if self.gain != 1 else 1.0
        return None


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
    r"""Freeverb-style algorithmic reverb (parallel combs + series all-passes).

    Replaces the original single-comb reverb with the classic Schroeder/Moorer structure:
    per channel, **8 parallel low-pass-feedback comb filters** are summed and fed through
    **4 series all-pass diffusers**, producing a dense, natural decay. The comb/all-pass
    delay tunings scale with the sampling rate so the character is consistent across
    ``fs``. The network runs in a native per-channel C++/CUDA kernel.

    Parameters
    ----------
    room_size : float, default=0.5
        Apparent room size in ``[0, 1]`` — sets the comb feedback (decay length). Larger
        values give a longer reverb tail.
    damping : float, default=0.5
        High-frequency damping in ``[0, 1]``. Larger values absorb highs faster, for a
        warmer/darker tail.
    mix : float, default=0.3
        Wet/dry balance in ``[0, 1]`` (``0`` = dry, ``1`` = fully wet).
    fs : int or None, default=None
        Sampling rate in Hz, used to scale the delay tunings. May be left ``None`` and
        supplied lazily by a ``Wave`` pipeline (``wave | reverb``).

    Returns
    -------
    Tensor
        The reverberated waveform, same shape and dtype as the input.

    Raises
    ------
    ValueError
        If ``room_size``/``damping``/``mix`` are outside ``[0, 1]``, ``fs`` is
        non-positive, or ``forward`` is called before ``fs`` is known.

    See Also
    --------
    Delay : Multi-tap delay effect with BPM synchronisation.

    Notes
    -----
    Each channel is processed independently with identical tunings (no stereo-width spread
    — a possible follow-up). State is reset per ``forward`` call, so block-wise streaming
    is not state-continuous across chunks. This is a **breaking change** from the pre-0.7
    ``Reverb(delay, decay, mix)`` API.

    Examples
    --------
    >>> import torch
    >>> from torchfx.effect import Reverb
    >>> x = torch.randn(2, 48000)
    >>> y = Reverb(room_size=0.7, damping=0.4, mix=0.3, fs=48000)(x)
    >>> y.shape
    torch.Size([2, 48000])

    In a ``Wave`` pipeline (``fs`` supplied automatically):

    >>> import torchfx as fx
    >>> processed = wave | fx.Reverb(room_size=0.8, mix=0.25)  # doctest: +SKIP

    """

    # Freeverb fixed constants (input gain into the comb bank; all-pass feedback).
    _INPUT_GAIN = 0.015
    _ALLPASS_FB = 0.5

    def __init__(
        self,
        room_size: float = 0.5,
        damping: float = 0.5,
        mix: float = 0.3,
        fs: int | None = None,
    ) -> None:
        super().__init__()
        if not 0 <= room_size <= 1:
            raise ValueError(f"room_size must be in [0, 1], got {room_size}.")
        if not 0 <= damping <= 1:
            raise ValueError(f"damping must be in [0, 1], got {damping}.")
        if not 0 <= mix <= 1:
            raise ValueError(f"mix must be in [0, 1], got {mix}.")
        if fs is not None and fs <= 0:
            raise ValueError(f"Sample rate (fs) must be positive, got {fs}.")

        self.room_size = float(room_size)
        self.damping = float(damping)
        self.mix = float(mix)
        self.fs = fs

    @override
    @torch.no_grad()
    def forward(self, waveform: Tensor) -> Tensor:
        if self.fs is None:
            raise ValueError(
                "Sample rate (fs) is required for the reverb. Use it in a Wave "
                "pipeline (wave | reverb) or pass fs at construction."
            )
        if self.fs <= 0:
            raise ValueError("Sample rate (fs) must be positive.")

        from torchfx._ops import reverb_forward

        # Freeverb parameter mapping: room_size -> comb feedback, damping -> LP coefficient.
        feedback = self.room_size * 0.28 + 0.7
        damp = self.damping * 0.4
        return reverb_forward(
            waveform,
            self.fs,
            feedback,
            damp,
            self._INPUT_GAIN,
            self._ALLPASS_FB,
            self.mix,  # wet
            1.0 - self.mix,  # dry
        )


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
        self._last_delay_fs: int | None = None

        # If delay_samples is provided directly, use it
        if delay_samples is not None:
            if delay_samples <= 0:
                raise ValueError("Delay samples must be positive.")
            self.delay_samples = delay_samples
            self._needs_calculation = False
        else:
            # BPM-synced delay requires bpm parameter
            if bpm is None:
                raise ValueError("BPM must be provided if delay_samples is not set.")
            if bpm <= 0:
                raise ValueError("BPM must be positive.")

            # If fs is available now, calculate immediately
            if fs is not None:
                if fs <= 0:
                    raise ValueError("Sample rate (fs) must be positive.")
                self.delay_samples = self._calculate_delay_samples(bpm, delay_time, fs)
                self._needs_calculation = False
                self._last_delay_fs = fs
            else:
                # Defer calculation until fs is set (by Wave.__update_config)
                self.delay_samples = None  # type: ignore
                self._needs_calculation = True

        # Validate other parameters
        if not (0 <= feedback <= 0.95):
            raise ValueError("Feedback must be between 0 and 0.95.")
        if not (0 <= mix <= 1):
            raise ValueError("Mix must be between 0 and 1.")
        if taps < 1:
            raise ValueError("Taps must be at least 1.")

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
        # Lazy calculation of delay_samples if needed, and recomputation when fs changes.
        if self.bpm is not None and (self._needs_calculation or self._last_delay_fs != self.fs):
            if self.fs is None:
                raise ValueError(
                    "Sample rate (fs) is required for BPM-synced delay. "
                    "Either provide fs parameter or use with Wave pipeline (wave | delay)."
                )
            if self.fs <= 0:
                raise ValueError("Sample rate (fs) must be positive.")

            self.delay_samples = self._calculate_delay_samples(self.bpm, self.delay_time, self.fs)
            self._last_delay_fs = self.fs
            self._needs_calculation = False

        if self.delay_samples is None:
            raise ValueError("Delay samples are not initialized.")

        # Apply delay using strategy pattern
        delayed = self.strategy.apply_delay(waveform, self.delay_samples, self.taps, self.feedback)

        # Extend original waveform to match delayed length for wet/dry mixing.
        target_length = delayed.size(-1)
        if waveform.size(-1) < target_length:
            extended_shape = list(waveform.shape)
            extended_shape[-1] = target_length
            extended = torch.zeros(extended_shape, dtype=waveform.dtype, device=waveform.device)
            extended[..., : waveform.size(-1)] = waveform
            waveform = extended

        # Wet/dry mix — fused via lerp (single kernel, avoids intermediates).
        return torch.lerp(waveform, delayed, self.mix)


class Compressor(FX):
    r"""Feed-forward dynamic-range compressor with a decoupled peak detector.

    Reduces the dynamic range of a signal: levels above ``threshold`` are turned
    down according to ``ratio``, with a soft ``knee`` around the threshold and
    attack/release ballistics that control how quickly the gain reacts. An optional
    ``makeup_gain`` restores overall loudness.

    The side-chain level is tracked with a *decoupled* peak detector (a release
    max-hold followed by an attack one-pole), the standard high-quality topology
    (Giannoulis, Massberg & Reiss, 2012). Detection runs in native C++/CUDA, one
    channel per thread.

    Parameters
    ----------
    threshold : float, default=-20.0
        Level above which compression begins, in **dBFS** (decibels relative to a
        full-scale amplitude of 1.0).
    ratio : float, default=4.0
        Input/output ratio above the threshold (``>= 1.0``). ``4.0`` means 4 dB in
        yields 1 dB out over threshold. ``float("inf")`` is a brick-wall limiter.
    attack : float, default=0.005
        Attack time in **seconds** (how fast gain reduction engages). ``0`` is
        instantaneous.
    release : float, default=0.05
        Release time in **seconds** (how fast gain recovers). ``0`` is instantaneous.
    knee : float, default=6.0
        Full width of the soft knee in **dB**, centred on the threshold. ``0`` gives
        a hard knee.
    makeup_gain : float, default=0.0
        Output gain applied after compression, in **dB**.
    detector : {"peak", "rms"}, default="peak"
        Side-chain level detection. ``"peak"`` follows ``|x|``; ``"rms"`` follows a
        smoothed root-mean-square (averaging window tied to ``attack``).
    fs : int or None, default=None
        Sampling rate in Hz. May be left ``None`` and supplied lazily by a ``Wave``
        pipeline (``wave | compressor``); the ballistics coefficients are then
        derived from the times and ``fs`` on first ``forward``.

    Returns
    -------
    Tensor
        The compressed waveform, same shape and dtype as the input.

    Raises
    ------
    ValueError
        If ``ratio < 1``, ``attack``/``release``/``knee`` is negative, ``detector``
        is not ``"peak"``/``"rms"``, ``fs`` is non-positive, or ``forward`` is called
        before ``fs`` is known.

    See Also
    --------
    Gain : Static volume adjustment.
    Normalize : Amplitude normalization.

    Notes
    -----
    Per channel, sequentially over samples :math:`n` (linear detector level, then
    the gain computed in dB):

    .. math::

        \text{rect}[n] &= |x[n]| \quad (\text{peak}) \\
        y_1[n] &= \max(\text{rect}[n],\ a_R\, y_1[n-1]) \\
        y_L[n] &= a_A\, y_L[n-1] + (1 - a_A)\, y_1[n] \\
        L &= 20 \log_{10}(\max(y_L[n], \epsilon))

    with attack/release coefficients :math:`a_A = e^{-1/(\text{attack}\cdot f_s)}`,
    :math:`a_R = e^{-1/(\text{release}\cdot f_s)}`. The static curve (soft knee of
    width :math:`W` centred on threshold :math:`T`, ``r`` = ratio) maps :math:`L` to
    :math:`L_{sc}`, and the per-sample gain is
    :math:`g[n] = 10^{(L_{sc} - L + \text{makeup})/20}`, applied as
    :math:`y[n] = g[n]\, x[n]`.

    This is a **zero-latency feed-forward** design (no look-ahead). The ballistics
    state is reset on each ``forward`` call, so block-wise streaming is *not*
    state-continuous across chunks (a follow-up could thread the state in/out).

    Examples
    --------
    Compress a signal 4:1 above -12 dBFS:

    >>> import torch
    >>> from torchfx.effect import Compressor
    >>> x = torch.randn(2, 48000) * 0.5  # (channels, samples)
    >>> comp = Compressor(threshold=-12.0, ratio=4.0, attack=0.005, release=0.08, fs=48000)
    >>> y = comp(x)
    >>> y.shape
    torch.Size([2, 48000])

    Use in a ``Wave`` pipeline (``fs`` is supplied automatically):

    >>> import torchfx as fx
    >>> processed = wave | fx.effect.Compressor(threshold=-18.0, ratio=3.0)  # doctest: +SKIP

    Brick-wall limiting at -1 dBFS with makeup gain:

    >>> limiter = Compressor(threshold=-1.0, ratio=float("inf"), makeup_gain=1.0, fs=48000)

    References
    ----------
    D. Giannoulis, M. Massberg, J. D. Reiss, "Digital Dynamic Range Compressor
    Design — A Tutorial and Analysis," J. Audio Eng. Soc., 60(6), 2012.

    """

    def __init__(
        self,
        threshold: float = -20.0,
        ratio: float = 4.0,
        attack: float = 0.005,
        release: float = 0.05,
        knee: float = 6.0,
        makeup_gain: float = 0.0,
        detector: str = "peak",
        fs: int | None = None,
    ) -> None:
        super().__init__()
        valid_detectors = {"peak", "rms"}
        if detector not in valid_detectors:
            raise ValueError(
                f"detector must be one of {sorted(valid_detectors)}, got {detector!r}."
            )
        if ratio < 1.0:
            raise ValueError(f"ratio must be >= 1.0, got {ratio}.")
        if attack < 0:
            raise ValueError(f"attack must be non-negative (seconds), got {attack}.")
        if release < 0:
            raise ValueError(f"release must be non-negative (seconds), got {release}.")
        if knee < 0:
            raise ValueError(f"knee must be non-negative (dB), got {knee}.")
        if fs is not None and fs <= 0:
            raise ValueError(f"Sample rate (fs) must be positive, got {fs}.")

        self.threshold = float(threshold)
        self.ratio = float(ratio)
        self.attack = float(attack)
        self.release = float(release)
        self.knee = float(knee)
        self.makeup_gain = float(makeup_gain)
        self.detector = detector
        self.fs = fs

        # 1/ratio (0 for an inf-ratio limiter) so the kernel never does inf math.
        self._inv_ratio = 0.0 if math.isinf(self.ratio) else 1.0 / self.ratio
        self._aA = 0.0
        self._aR = 0.0
        self._aRMS = 0.0
        self._last_fs: int | None = None
        self._needs_calculation = True

    def _compute_coeffs(self) -> None:
        """Derive attack/release/RMS coefficients from the times and ``fs``."""
        assert self.fs is not None
        fs = self.fs
        self._aA = 0.0 if self.attack == 0 else math.exp(-1.0 / (self.attack * fs))
        self._aR = 0.0 if self.release == 0 else math.exp(-1.0 / (self.release * fs))
        # The RMS averaging window defaults to the attack time.
        self._aRMS = 0.0 if self.attack == 0 else math.exp(-1.0 / (self.attack * fs))
        self._last_fs = fs
        self._needs_calculation = False

    @override
    @torch.no_grad()
    def forward(self, waveform: Tensor) -> Tensor:
        if self._needs_calculation or self._last_fs != self.fs:
            if self.fs is None:
                raise ValueError(
                    "Sample rate (fs) is required for the compressor. Use it in a "
                    "Wave pipeline (wave | compressor) or pass fs at construction."
                )
            if self.fs <= 0:
                raise ValueError("Sample rate (fs) must be positive.")
            self._compute_coeffs()

        from torchfx._ops import compressor_forward

        return compressor_forward(
            waveform,
            self.threshold,
            self._inv_ratio,
            self.knee,
            self.makeup_gain,
            self._aA,
            self._aR,
            self._aRMS,
            1 if self.detector == "rms" else 0,
        )


class Expander(FX):
    r"""Downward expander / noise gate with a decoupled peak detector.

    The mirror image of :class:`Compressor`: levels **below** ``threshold`` are turned
    *down* (their dynamic range is expanded), which pushes quiet passages and noise
    toward silence while leaving louder signal untouched. With a high ``ratio`` (or
    ``ratio=inf``) it acts as a **noise gate**.

    The side-chain level uses the same *decoupled* peak detector as the compressor (a
    release max-hold followed by an attack one-pole; Giannoulis, Massberg & Reiss,
    2012). Detection runs in native C++/CUDA, one channel per thread.

    Parameters
    ----------
    threshold : float, default=-40.0
        Level below which downward expansion begins, in **dBFS**.
    ratio : float, default=2.0
        Expansion ratio below the threshold (``>= 1.0``). ``2.0`` means a signal 1 dB
        below threshold is pushed a further 1 dB down. ``float("inf")`` is a hard gate.
    attack : float, default=0.005
        Attack time in **seconds** (how fast the gate opens as level rises). ``0`` is
        instantaneous.
    release : float, default=0.05
        Release time in **seconds** (how fast the gate closes as level falls). ``0`` is
        instantaneous.
    knee : float, default=6.0
        Full width of the soft knee in **dB**, centred on the threshold. ``0`` gives a
        hard knee.
    floor : float or None, default=None
        Deepest attenuation in **dB** (a negative number), limiting how far the signal
        is pushed down — i.e. an expander/gate *range*. ``None`` applies no floor (full
        downward expansion; with ``ratio=inf`` this gates to near-silence).
    detector : {"peak", "rms"}, default="peak"
        Side-chain level detection. ``"peak"`` follows ``|x|``; ``"rms"`` follows a
        smoothed root-mean-square (averaging window tied to ``attack``).
    fs : int or None, default=None
        Sampling rate in Hz. May be left ``None`` and supplied lazily by a ``Wave``
        pipeline (``wave | expander``).

    Returns
    -------
    Tensor
        The expanded waveform, same shape and dtype as the input.

    Raises
    ------
    ValueError
        If ``ratio < 1``, ``attack``/``release``/``knee`` is negative, ``floor`` is
        positive, ``detector`` is not ``"peak"``/``"rms"``, ``fs`` is non-positive, or
        ``forward`` is called before ``fs`` is known.

    See Also
    --------
    Compressor : Reduces dynamic range *above* the threshold.
    Gate : Convenience subclass with an infinite ratio (a hard noise gate).

    Notes
    -----
    The detector is identical to :class:`Compressor`; only the static curve differs.
    Per channel, with linear detector level :math:`L = 20\log_{10}(\max(y_L, \epsilon))`,
    over-threshold amount :math:`o = L - T`, slope :math:`s = r - 1`, and soft knee of
    width :math:`W`:

    .. math::

        g_{dB} =
        \begin{cases}
        -s\,(o - W/2)^2 / (2W) & |2o| \le W \quad (\text{knee}) \\
        s\,o & o < 0 \quad (\text{below threshold}) \\
        0 & \text{otherwise}
        \end{cases}

    clamped to ``floor`` (``g_{dB} \ge`` floor), and :math:`y[n] = 10^{g_{dB}/20}\,x[n]`.

    This is a **zero-latency feed-forward** design (no look-ahead). The ballistics state
    is reset on each ``forward`` call, so block-wise streaming is *not* state-continuous
    across chunks (see the compressor streaming follow-up).

    Examples
    --------
    Gently expand a signal 2:1 below -40 dBFS:

    >>> import torch
    >>> from torchfx.effect import Expander
    >>> x = torch.randn(2, 48000) * 0.5  # (channels, samples)
    >>> exp = Expander(threshold=-40.0, ratio=2.0, attack=0.005, release=0.1, fs=48000)
    >>> y = exp(x)
    >>> y.shape
    torch.Size([2, 48000])

    A noise gate (infinite ratio) with an 80 dB range:

    >>> gate = Expander(threshold=-50.0, ratio=float("inf"), floor=-80.0, fs=48000)

    References
    ----------
    D. Giannoulis, M. Massberg, J. D. Reiss, "Digital Dynamic Range Compressor
    Design — A Tutorial and Analysis," J. Audio Eng. Soc., 60(6), 2012.

    """

    # Stands in for ``ratio=inf`` so the kernel never does inf arithmetic: a slope this
    # steep drives any below-threshold sample straight to the floor (i.e. a hard gate).
    _GATE_SLOPE = 1.0e6
    # Applied when ``floor`` is None — far below any real signal, so effectively no floor.
    _NO_FLOOR_DB = -240.0

    def __init__(
        self,
        threshold: float = -40.0,
        ratio: float = 2.0,
        attack: float = 0.005,
        release: float = 0.05,
        knee: float = 6.0,
        floor: float | None = None,
        detector: str = "peak",
        fs: int | None = None,
    ) -> None:
        super().__init__()
        valid_detectors = {"peak", "rms"}
        if detector not in valid_detectors:
            raise ValueError(
                f"detector must be one of {sorted(valid_detectors)}, got {detector!r}."
            )
        if ratio < 1.0:
            raise ValueError(f"ratio must be >= 1.0, got {ratio}.")
        if attack < 0:
            raise ValueError(f"attack must be non-negative (seconds), got {attack}.")
        if release < 0:
            raise ValueError(f"release must be non-negative (seconds), got {release}.")
        if knee < 0:
            raise ValueError(f"knee must be non-negative (dB), got {knee}.")
        if floor is not None and floor > 0:
            raise ValueError(f"floor must be non-positive (dB of attenuation), got {floor}.")
        if fs is not None and fs <= 0:
            raise ValueError(f"Sample rate (fs) must be positive, got {fs}.")

        self.threshold = float(threshold)
        self.ratio = float(ratio)
        self.attack = float(attack)
        self.release = float(release)
        self.knee = float(knee)
        self.floor = None if floor is None else float(floor)
        self.detector = detector
        self.fs = fs

        # slope = ratio - 1 (a large finite value for an infinite-ratio gate).
        self._slope = self._GATE_SLOPE if math.isinf(self.ratio) else self.ratio - 1.0
        self._floor_db = self._NO_FLOOR_DB if self.floor is None else self.floor
        self._aA = 0.0
        self._aR = 0.0
        self._aRMS = 0.0
        self._last_fs: int | None = None
        self._needs_calculation = True

    def _compute_coeffs(self) -> None:
        """Derive attack/release/RMS coefficients from the times and ``fs``."""
        assert self.fs is not None
        fs = self.fs
        self._aA = 0.0 if self.attack == 0 else math.exp(-1.0 / (self.attack * fs))
        self._aR = 0.0 if self.release == 0 else math.exp(-1.0 / (self.release * fs))
        # The RMS averaging window defaults to the attack time.
        self._aRMS = 0.0 if self.attack == 0 else math.exp(-1.0 / (self.attack * fs))
        self._last_fs = fs
        self._needs_calculation = False

    @override
    @torch.no_grad()
    def forward(self, waveform: Tensor) -> Tensor:
        if self._needs_calculation or self._last_fs != self.fs:
            if self.fs is None:
                raise ValueError(
                    "Sample rate (fs) is required for the expander. Use it in a "
                    "Wave pipeline (wave | expander) or pass fs at construction."
                )
            if self.fs <= 0:
                raise ValueError("Sample rate (fs) must be positive.")
            self._compute_coeffs()

        from torchfx._ops import expander_forward

        return expander_forward(
            waveform,
            self.threshold,
            self._slope,
            self.knee,
            self._floor_db,
            self._aA,
            self._aR,
            self._aRMS,
            1 if self.detector == "rms" else 0,
        )


class Gate(Expander):
    r"""Noise gate — an :class:`Expander` with an infinite ratio (hard gating).

    Below ``threshold`` the signal is attenuated to the ``floor`` (a hard downward
    expansion); above it the signal passes unchanged. This is a basic gate without
    hysteresis or a hold time (a possible future refinement).

    Parameters
    ----------
    threshold : float, default=-50.0
        Level below which the gate closes, in **dBFS**.
    attack : float, default=0.001
        How fast the gate opens as level rises, in **seconds**.
    release : float, default=0.05
        How fast the gate closes as level falls, in **seconds**.
    floor : float or None, default=-80.0
        Attenuation applied when the gate is closed, in **dB**. ``None`` gates to
        near-silence.
    knee : float, default=3.0
        Soft-knee width in **dB** around the threshold.
    detector : {"peak", "rms"}, default="peak"
        Side-chain level detection.
    fs : int or None, default=None
        Sampling rate in Hz (may be supplied lazily by a ``Wave`` pipeline).

    Examples
    --------
    >>> import torch
    >>> from torchfx.effect import Gate
    >>> x = torch.randn(1, 48000) * 0.5
    >>> y = Gate(threshold=-45.0, floor=-90.0, fs=48000)(x)
    >>> y.shape
    torch.Size([1, 48000])

    """

    def __init__(
        self,
        threshold: float = -50.0,
        attack: float = 0.001,
        release: float = 0.05,
        floor: float | None = -80.0,
        knee: float = 3.0,
        detector: str = "peak",
        fs: int | None = None,
    ) -> None:
        super().__init__(
            threshold=threshold,
            ratio=float("inf"),
            attack=attack,
            release=release,
            knee=knee,
            floor=floor,
            detector=detector,
            fs=fs,
        )


class Limiter(FX):
    r"""Look-ahead brick-wall peak limiter.

    Guarantees the output magnitude never exceeds ``threshold`` (a true brick wall) while
    staying transparent: a short ``lookahead`` lets the gain reduce *before* a peak
    arrives, so there is no transient overshoot and none of the distortion an
    instantaneous gain change would cause. The gain then recovers over ``release``.

    The look-ahead windowed peak of ``|x|`` is computed with a vectorised max-pool, and the
    sequential gain recurrence runs in a native C++/CUDA kernel (one channel per thread).

    Parameters
    ----------
    threshold : float, default=-1.0
        Output ceiling in **dBFS**. The output magnitude never exceeds this.
    lookahead : float, default=0.005
        Look-ahead time in **seconds**: the gain is reduced over this window ahead of a
        peak so the limiter never overshoots. ``0`` is a zero-look-ahead limiter.
    release : float, default=0.05
        Release time in **seconds** — how fast the gain recovers after a peak. ``0`` is
        instantaneous.
    fs : int or None, default=None
        Sampling rate in Hz. May be left ``None`` and supplied lazily by a ``Wave``
        pipeline (``wave | limiter``).

    Returns
    -------
    Tensor
        The limited waveform, same shape and dtype as the input, with
        ``|y| <= 10**(threshold/20)`` everywhere.

    Raises
    ------
    ValueError
        If ``lookahead`` or ``release`` is negative, ``fs`` is non-positive, or ``forward``
        is called before ``fs`` is known.

    See Also
    --------
    Compressor : ``Compressor(ratio=inf)`` is a zero-look-ahead feed-forward limiter that
        can overshoot on a single sample before its gain reacts; ``Limiter`` cannot.

    Notes
    -----
    For each sample the applied gain is

    .. math::

        g[n] = \min\!\Big(\tfrac{T_\text{lin}}{\max(p[n], \epsilon)},\ a_R\,g[n-1] + (1-a_R)\Big),
        \qquad p[n] = \max_{0 \le k \le L} |x[n+k]|

    with linear ceiling :math:`T_\text{lin} = 10^{\text{threshold}/20}`, release coefficient
    :math:`a_R = e^{-1/(\text{release}\cdot f_s)}`, and look-ahead :math:`L`. Since
    :math:`|x[n]| \le p[n]`, the output :math:`y[n] = g[n]\,x[n]` satisfies
    :math:`|y[n]| \le T_\text{lin}` — a guaranteed brick wall.

    This is a **peak** (not true-peak / oversampled) limiter, so inter-sample peaks may
    slightly exceed the ceiling after conversion to analog (true-peak limiting is a possible
    follow-up). The output is time-aligned with the input (no net latency). Ballistics state
    is reset per ``forward`` call (not state-continuous across streaming chunks).

    Examples
    --------
    Brick-wall a signal that exceeds full scale to a -1 dBFS ceiling:

    >>> import torch
    >>> from torchfx.effect import Limiter
    >>> x = torch.randn(2, 48000) * 3.0  # well over +/-1
    >>> lim = Limiter(threshold=-1.0, lookahead=0.005, release=0.05, fs=48000)
    >>> y = lim(x)
    >>> bool(y.abs().max() <= 10 ** (-1.0 / 20) + 1e-4)
    True

    """

    def __init__(
        self,
        threshold: float = -1.0,
        lookahead: float = 0.005,
        release: float = 0.05,
        fs: int | None = None,
    ) -> None:
        super().__init__()
        if lookahead < 0:
            raise ValueError(f"lookahead must be non-negative (seconds), got {lookahead}.")
        if release < 0:
            raise ValueError(f"release must be non-negative (seconds), got {release}.")
        if fs is not None and fs <= 0:
            raise ValueError(f"Sample rate (fs) must be positive, got {fs}.")

        self.threshold = float(threshold)
        self.lookahead = float(lookahead)
        self.release = float(release)
        self.fs = fs

        self._thr_lin = 10.0 ** (self.threshold / 20.0)
        self._attack_coeff = 0.0
        self._release_coeff = 0.0
        self._lookahead_samples = 0
        self._last_fs: int | None = None
        self._needs_calculation = True

    def _compute_coeffs(self) -> None:
        """Derive ceiling, attack/release coefficients and look-ahead samples from
        ``fs``."""
        assert self.fs is not None
        fs = self.fs
        self._thr_lin = 10.0 ** (self.threshold / 20.0)
        # Ramp the gain down over the look-ahead window so it is already low when the peak
        # arrives (the per-sample clamp guarantees the brick wall regardless).
        self._attack_coeff = 0.0 if self.lookahead == 0 else math.exp(-1.0 / (self.lookahead * fs))
        self._release_coeff = 0.0 if self.release == 0 else math.exp(-1.0 / (self.release * fs))
        self._lookahead_samples = int(round(self.lookahead * fs))
        self._last_fs = fs
        self._needs_calculation = False

    @override
    @torch.no_grad()
    def forward(self, waveform: Tensor) -> Tensor:
        if self._needs_calculation or self._last_fs != self.fs:
            if self.fs is None:
                raise ValueError(
                    "Sample rate (fs) is required for the limiter. Use it in a "
                    "Wave pipeline (wave | limiter) or pass fs at construction."
                )
            if self.fs <= 0:
                raise ValueError("Sample rate (fs) must be positive.")
            self._compute_coeffs()

        from torchfx._ops import limiter_forward

        return limiter_forward(
            waveform,
            self._thr_lin,
            self._attack_coeff,
            self._release_coeff,
            self._lookahead_samples,
        )
