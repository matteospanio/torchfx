"""Base class for all effects."""

from __future__ import annotations

import abc
import math
from collections.abc import Callable

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
    gain : float
        The gain factor to apply to the waveform.
    gain_type : str
        The type of gain to apply. Can be one of "amplitude", "db", or "power".
    clamp : bool
        If True, clamps the output waveform to the range [-1.0, 1.0].

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
    r"""Normalize the waveform to a given peak value using a selected strategy.

    Args:
        peak (float): The peak value to normalize to. Default is 1.0.

    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
        >>> transform = transforms.Normalize(peak=0.5)
        >>> normalized_waveform = transform(waveform)
    """

    def __init__(
        self, peak: float = 1.0, strategy: NormalizationStrategy | Callable | None = None
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
    """Abstract base class for normalization strategies."""

    @abc.abstractmethod
    def __call__(self, waveform: Tensor, peak: float) -> Tensor:
        """Normalize the waveform to the given peak value."""
        pass


class CustomNormalizationStrategy(NormalizationStrategy):
    """Normalization using a custom function."""

    def __init__(self, func: Callable[[Tensor, float], Tensor]) -> None:
        assert callable(func), "func must be callable"
        self.func = func

    def __call__(self, waveform: Tensor, peak: float) -> Tensor:
        return self.func(waveform, peak)


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
        assert 0 < percentile <= 100, "Percentile must be between 0 and 100."
        self.percentile = percentile

    def __call__(self, waveform: Tensor, peak: float) -> Tensor:
        abs_waveform = torch.abs(waveform)
        threshold = torch.quantile(abs_waveform, self.percentile / 100, interpolation="linear")
        return waveform / threshold * peak if threshold > 0 else waveform


class PerChannelNormalizationStrategy(NormalizationStrategy):
    """Normalize each channel independently to its own peak."""

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
    r"""Apply a simple reverb effect using a feedback delay network.

    The reverb effect is computed as:

    .. math::

        y[n] = (1 - mix) x[n] + mix (x[n] + decay x[n - delay])

    where:
        - x[n] is the input signal,
        - y[n] is the output signal,
        - delay is the number of samples for the delay,
        - decay is the feedback decay factor,
        - mix is the wet/dry mix parameter.

    Attributes
    ----------
    delay : int
        Delay in samples for the feedback comb filter. Default is 4410 (100ms at 44.1kHz).
    decay : float
        Feedback decay factor. Must be between 0 and 1. Default is 0.5.
    mix : float
        Wet/dry mix. 0 = dry, 1 = wet. Default is 0.5.

    Examples
    --------
    >>> import torchfx as fx
    >>> wave = fx.Wave.from_file("path_to_audio.wav")
    >>> reverb = fx.effect.Reverb(delay=4410, decay=0.5, mix=0.3)
    >>> reverberated = wave | reverb
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
    delay_samples : int, optional
        Delay time in samples. If provided, this is used directly.
        Default is None (requires bpm and delay_time).
    bpm : float, optional
        Beats per minute for BPM-synced delay. Required if delay_samples is None.
    delay_time : str, optional
        Musical time division for BPM-synced delay. Options:
        - '1/4': Quarter note
        - '1/8': Eighth note
        - '1/16': Sixteenth note
        - '1/8d': Dotted eighth note
        - '1/4d': Dotted quarter note
        - '1/8t': Eighth note triplet
        Default is '1/8'.
    sample_rate : int, optional
        Sample rate in Hz. Required if using BPM-synced delay. Must be positive.
        Default is 44100.
    feedback : float, optional
        Feedback amount (0-0.95). Controls amplitude of taps 2 and beyond.
        First tap always has amplitude 1.0. Higher values create more prominent echoes.
        Default is 0.3.
    mix : float, optional
        Wet/dry mix. 0 = dry (original signal only), 1 = wet (delayed echoes only).
        Default is 0.2.
    taps : int, optional
        Number of delay taps (echoes). Each tap is delayed by delay_samples * tap_number.
        Default is 3.
    stereo_mode : str, optional
        Stereo processing mode. Options:
        - 'mono': Same delay on all channels
        - 'stereo': Same delay on all channels (same as mono for now)
        - 'pingpong': Alternates delay between left and right channels
        Default is 'mono'.

    Examples
    --------
    >>> import torchfx as fx
    >>> import torch
    >>> 
    >>> # BPM-synced delay (1/8 note at 128 BPM)
    >>> waveform = torch.randn(2, 44100)  # (channels, samples)
    >>> delay = fx.effect.Delay(bpm=128, delay_time='1/8', sample_rate=44100, feedback=0.3, mix=0.2)
    >>> delayed = delay(waveform)
    >>> 
    >>> # Direct delay in samples
    >>> delay = fx.effect.Delay(delay_samples=2205, feedback=0.4, mix=0.3)
    >>> delayed = delay(waveform)
    >>> 
    >>> # Ping-pong delay
    >>> delay = fx.effect.Delay(bpm=128, delay_time='1/4', sample_rate=44100, 
    ...                         feedback=0.5, mix=0.4, stereo_mode='pingpong')
    >>> delayed = delay(waveform)
    """

    def __init__(
        self,
        delay_samples: int | None = None,
        bpm: float | None = None,
        delay_time: str = "1/8",
        sample_rate: int = 44100,
        feedback: float = 0.3,
        mix: float = 0.2,
        taps: int = 3,
        stereo_mode: str = "mono",
    ) -> None:
        super().__init__()

        # Calculate delay_samples if not provided
        if delay_samples is None:
            if bpm is None:
                raise ValueError("Either delay_samples or bpm must be provided.")
            if bpm <= 0:
                raise ValueError("BPM must be positive.")
            if sample_rate <= 0:
                raise ValueError("Sample rate must be positive.")
            delay_samples = self._calculate_delay_samples(bpm, delay_time, sample_rate)

        # Validate parameters
        if delay_samples <= 0:
            raise ValueError("Delay must be positive.")
        if not (0 <= feedback <= 0.95):
            raise ValueError("Feedback must be between 0 and 0.95.")
        if not (0 <= mix <= 1):
            raise ValueError("Mix must be between 0 and 1.")
        if taps < 1:
            raise ValueError("Taps must be at least 1.")
        if stereo_mode not in ["mono", "stereo", "pingpong"]:
            raise ValueError("stereo_mode must be 'mono', 'stereo', or 'pingpong'.")

        self.delay_samples = delay_samples
        self.feedback = feedback
        self.mix = mix
        self.taps = taps
        self.stereo_mode = stereo_mode

    @staticmethod
    def _calculate_delay_samples(bpm: float, delay_time: str, sample_rate: int) -> int:
        """Calculate delay time in samples from BPM and musical division."""
        # Seconds per beat
        beat_duration = 60.0 / bpm

        # Parse delay time
        if delay_time == "1/4":
            delay_sec = beat_duration
        elif delay_time == "1/8":
            delay_sec = beat_duration / 2
        elif delay_time == "1/16":
            delay_sec = beat_duration / 4
        elif delay_time == "1/8d":  # Dotted eighth
            delay_sec = beat_duration / 2 * 1.5
        elif delay_time == "1/4d":  # Dotted quarter
            delay_sec = beat_duration * 1.5
        elif delay_time == "1/8t":  # Eighth triplet
            delay_sec = beat_duration / 3
        else:
            delay_sec = beat_duration / 2  # Default to 1/8

        return int(delay_sec * sample_rate)

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
        # Calculate required output length to accommodate all delayed taps
        original_length = waveform.size(-1)
        max_delay_samples = self.delay_samples * self.taps
        required_length = original_length + max_delay_samples

        # If waveform is shorter than first delay, extend it for processing
        if original_length <= self.delay_samples:
            waveform = self._extend_waveform(waveform, required_length)

        # Process based on stereo mode
        if self.stereo_mode == "pingpong" and waveform.ndim >= 2 and waveform.size(-2) == 2:
            delayed = self._apply_pingpong_delay(waveform)
        else:
            delayed = self._apply_mono_delay(waveform)

        # Extend original waveform to match delayed length for mixing
        waveform = self._extend_waveform(waveform, delayed.size(-1))

        # Wet/dry mix
        output = (1 - self.mix) * waveform + self.mix * delayed
        return output

    def _apply_mono_delay(self, waveform: Tensor) -> Tensor:
        """Apply delay with multiple taps and feedback.
        
        Output length is extended to accommodate all delayed taps.
        """
        # Calculate required output length
        original_length = waveform.size(-1)
        max_delay_samples = self.delay_samples * self.taps
        output_length = original_length + max_delay_samples
        
        # waveform shape: (..., time) or (channels, time)
        if waveform.ndim == 1:
            # Single channel: (time,)
            delayed = torch.zeros(output_length, dtype=waveform.dtype, device=waveform.device)
            for tap in range(1, self.taps + 1):
                tap_delay = self.delay_samples * tap
                # First tap always has amplitude 1.0, subsequent taps use feedback
                if tap == 1:
                    feedback_amt = 1.0
                else:
                    feedback_amt = self.feedback ** (tap - 1)
                # Copy original signal starting at tap_delay
                copy_length = min(original_length, output_length - tap_delay)
                if copy_length > 0:
                    delayed[tap_delay:tap_delay + copy_length] += waveform[:copy_length] * feedback_amt
            return delayed

        elif waveform.ndim == 2:
            # Multi-channel: (channels, time)
            delayed = torch.zeros(waveform.size(0), output_length, dtype=waveform.dtype, device=waveform.device)
            for ch in range(waveform.size(0)):
                for tap in range(1, self.taps + 1):
                    tap_delay = self.delay_samples * tap
                    # First tap always has amplitude 1.0, subsequent taps use feedback
                    if tap == 1:
                        feedback_amt = 1.0
                    else:
                        feedback_amt = self.feedback ** (tap - 1)
                    # Copy original signal starting at tap_delay
                    copy_length = min(original_length, output_length - tap_delay)
                    if copy_length > 0:
                        delayed[ch, tap_delay:tap_delay + copy_length] += waveform[ch, :copy_length] * feedback_amt
            return delayed

        else:
            # Higher dimensions: (..., time)
            # Flatten to (channels, time) for processing
            original_shape = list(waveform.shape)
            flattened = waveform.view(-1, waveform.size(-1))
            processed = self._apply_mono_delay(flattened)
            # Reshape with extended time dimension
            new_shape = original_shape[:-1] + [processed.size(-1)]
            return processed.view(new_shape)

    def _apply_pingpong_delay(self, waveform: Tensor) -> Tensor:
        """Apply ping-pong delay (alternates between channels).
        
        Output length is extended to accommodate all delayed taps.
        """
        if waveform.ndim < 2 or waveform.size(-2) != 2:
            # Not stereo, fall back to mono
            return self._apply_mono_delay(waveform)

        # Calculate required output length
        original_length = waveform.size(-1)
        max_delay_samples = self.delay_samples * self.taps
        output_length = original_length + max_delay_samples

        # waveform: (2, time) or (..., 2, time)
        if waveform.ndim == 2:
            # Simple case: (2, time)
            delayed = torch.zeros(2, output_length, dtype=waveform.dtype, device=waveform.device)
            for tap in range(1, self.taps + 1):
                tap_delay = self.delay_samples * tap
                # First tap always has amplitude 1.0, subsequent taps use feedback
                if tap == 1:
                    feedback_amt = 1.0
                else:
                    feedback_amt = self.feedback ** (tap - 1)

                # Copy length for this tap
                copy_length = min(original_length, output_length - tap_delay)
                if copy_length > 0:
                    # Odd taps: left delays to right, even taps: right delays to left
                    if tap % 2 == 1:
                        # Left -> Right
                        delayed[1, tap_delay:tap_delay + copy_length] += waveform[0, :copy_length] * feedback_amt
                    else:
                        # Right -> Left
                        delayed[0, tap_delay:tap_delay + copy_length] += waveform[1, :copy_length] * feedback_amt
            return delayed

        else:
            # Higher dimensions: (..., 2, time)
            original_shape = list(waveform.shape)
            original_shape[-1] = output_length
            delayed = torch.zeros(original_shape, dtype=waveform.dtype, device=waveform.device)
            for tap in range(1, self.taps + 1):
                tap_delay = self.delay_samples * tap
                # First tap always has amplitude 1.0, subsequent taps use feedback
                if tap == 1:
                    feedback_amt = 1.0
                else:
                    feedback_amt = self.feedback ** (tap - 1)

                # Copy length for this tap
                copy_length = min(original_length, output_length - tap_delay)
                if copy_length > 0:
                    if tap % 2 == 1:
                        # Left -> Right
                        delayed[..., 1, tap_delay:tap_delay + copy_length] += waveform[..., 0, :copy_length] * feedback_amt
                    else:
                        # Right -> Left
                        delayed[..., 0, tap_delay:tap_delay + copy_length] += waveform[..., 1, :copy_length] * feedback_amt

            return delayed
