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
    def __init__(self, *args, **kwargs) -> None:  # type: ignore
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

    Delay strategies define how the delay effect is applied to the audio signal,
    allowing for different stereo behaviors and custom processing logic.

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
            Delay time in samples.
        taps : int
            Number of delay taps (echoes).
        feedback : float
            Feedback amount (0-0.95).

        Returns
        -------
        Tensor
            Delayed audio with extended length to accommodate all taps.

        """
        pass


class MonoDelayStrategy(DelayStrategy):
    """Apply same delay to all channels with multiple taps and feedback."""

    def apply_delay(
        self, waveform: Tensor, delay_samples: int, taps: int, feedback: float
    ) -> Tensor:
        """Apply mono delay with multiple taps and feedback.

        Output length is extended to accommodate all delayed taps.

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
    """Apply ping-pong delay alternating between left and right channels."""

    def apply_delay(
        self, waveform: Tensor, delay_samples: int, taps: int, feedback: float
    ) -> Tensor:
        """Apply ping-pong delay (alternates between channels).

        Output length is extended to accommodate all delayed taps.

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
