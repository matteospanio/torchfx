"""Base class for all effects."""

from __future__ import annotations

import abc
import math

import torch
from torch import Tensor, nn
from torchaudio import functional as F
from typing_extensions import override


class FX(nn.Module, abc.ABC):
    """Abstract base class for all effects.
    This class defines the interface for all effects in the library. It inherits from
    `torch.nn.Module` and provides the basic structure for implementing effects.
    """

    @abc.abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @override
    @abc.abstractmethod
    def forward(self, x: Tensor) -> Tensor: ...


class Gain(FX):
    r"""Adjust volume of waveform.

    This effect is the same as `torchaudio.transforms.Vol`, but it adds the option to clamp or not the output waveform.

    Parameters
    ----------
    gain (float): The gain factor to apply to the waveform.
    gain_type (str): The type of gain to apply. Can be one of "amplitude", "db", or "power".
    clamp (bool): If True, clamps the output waveform to the range [-1.0, 1.0].

    Example
    -------
        >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
        >>> transform = transforms.Vol(gain=0.5, gain_type="amplitude")
        >>> quieter_waveform = transform(waveform)

    See Also
    --------
    torchaudio.transforms.Vol: Transform to apply gain to a waveform.

    Notes
    -----
    This class is based on `torchaudio.transforms.Vol`, licensed under the BSD 2-Clause License.
    See licenses.torchaudio.BSD-2-Clause.txt for details.
    """

    def __init__(self, gain: float, gain_type: str = "amplitude", clamp: bool = False) -> None:
        super().__init__()
        self.gain = gain
        self.gain_type = gain_type
        self.clamp = clamp

        if gain_type in ["amplitude", "power"] and gain < 0:
            raise ValueError("If gain_type = amplitude or power, gain must be positive.")

    @override
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
    r"""Normalize the waveform to a given peak value using a selected strategy.

    Args:
        peak (float): The peak value to normalize to. Default is 1.0.

    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
        >>> transform = transforms.Normalize(peak=0.5)
        >>> normalized_waveform = transform(waveform)
    """

    def __init__(self, peak: float = 1.0, strategy: NormalizationStrategy | None = None) -> None:
        super().__init__()
        if peak <= 0:
            raise ValueError("Peak value must be positive.")
        self.peak = peak
        self.strategy = strategy or PeakNormalizationStrategy()
        if not isinstance(self.strategy, NormalizationStrategy):
            raise TypeError("Strategy must be an instance of NormalizationStrategy.")

    @override
    def forward(self, waveform: Tensor) -> Tensor:
        return self.strategy(waveform, self.peak)


class NormalizationStrategy(abc.ABC):
    """Abstract base class for normalization strategies."""

    @abc.abstractmethod
    def __call__(self, waveform: Tensor, peak: float) -> Tensor:
        """Normalize the waveform to the given peak value."""
        pass


class PeakNormalizationStrategy(NormalizationStrategy):
    """Normalization to the absolute peak value."""

    def __call__(self, waveform: Tensor, peak: float) -> Tensor:
        max_val = torch.max(torch.abs(waveform))
        return waveform / max_val * peak if max_val > 0 else waveform


class RMSNormalizationStrategy(NormalizationStrategy):
    """Normalization to Root Mean Square (RMS) energy."""

    def __call__(self, waveform: Tensor, peak: float) -> Tensor:
        rms = torch.sqrt(torch.mean(waveform**2))
        return waveform / rms * peak if rms > 0 else waveform


class PercentileNormalizationStrategy(NormalizationStrategy):
    """Normalization using a percentile of absolute values."""

    def __init__(self, percentile: float = 99.0) -> None:
        if not (0 < percentile <= 100):
            raise ValueError("Percentile must be between 0 and 100.")
        self.percentile = percentile

    def __call__(self, waveform: Tensor, peak: float) -> Tensor:
        abs_waveform = torch.abs(waveform)
        threshold = torch.quantile(abs_waveform, self.percentile / 100, interpolation="linear")
        return waveform / threshold * peak if threshold > 0 else waveform


class PerChannelNormalizationStrategy(NormalizationStrategy):
    """Normalize each channel independently to its own peak."""

    def __call__(self, waveform: Tensor, peak: float) -> Tensor:
        if waveform.ndim < 2:
            raise ValueError("Waveform must have at least 2 dimensions (channels, time).")

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
    r"""Apply a simple reverb effect using a feedback delay network.

    Args:
        delay (int): Delay in samples for the feedback comb filter.
        decay (float): Feedback decay factor (0 < decay < 1).
        mix (float): Wet/dry mix (0 = dry, 1 = wet).

    The reverb effect is computed as:
        y[n] = (1 - mix) * x[n] + mix * (x[n] + decay * x[n - delay])
    where:
        - x[n] is the input signal,
        - y[n] is the output signal,
        - delay is the number of samples for the delay,
        - decay is the feedback decay factor,
        - mix is the wet/dry mix parameter.

    Example
        >>> reverb = Reverb(delay=4410, decay=0.5, mix=0.3)
        >>> reverberated = reverb(waveform)
    """

    def __init__(self, delay: int = 4410, decay: float = 0.5, mix: float = 0.5) -> None:
        super().__init__()
        if delay <= 0:
            raise ValueError("Delay must be positive.")
        if not (0 < decay < 1):
            raise ValueError("Decay must be between 0 and 1.")
        if not (0 <= mix <= 1):
            raise ValueError("Mix must be between 0 and 1.")
        self.delay = delay
        self.decay = decay
        self.mix = mix

    @override
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
