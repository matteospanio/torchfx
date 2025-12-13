"""Module to define custom types and dataclasses for the reverse detection module."""

import re
import typing as tp
from dataclasses import dataclass

import torch
from annotated_types import Ge, Le

Decibel = tp.Annotated[float, Le(0)]
Millisecond = tp.Annotated[float, Ge(0)]
Second = tp.Annotated[float, Ge(0)]
BitRate = tp.Literal[16, 24, 32]

# Type of spectrogram to compute.
# It can be either "mel", "linear", or "log".
SpecScale = tp.Literal["mel", "lin", "log"]

# Type of filter to apply.
# It can be either a string ("low" or "high") or a tuple of two floats.
FilterType = tp.Literal["low", "high"]

# Type of filter order.
# It can be either a string ("linear" or "db") or an integer.
FilterOrderScale = tp.Literal["db", "linear"]

# Type of device to use for computation.
# It can be either a string ("cpu" or "cuda") or a torch.device object.
# The string "cuda" will use the first available CUDA device.
Device = tp.Literal["cpu", "cuda"] | torch.device

# Type of window functions.
# See `scipy.signal.get_window` for more information.
WindowType = tp.Literal[
    "hann",
    "hamming",
    "blackman",
    "kaiser",
    "boxcar",
    "bartlett",
    "flattop",
    "parzen",
    "bohman",
    "nuttall",
    "barthann",
]


@dataclass(frozen=True)
class MusicalTime:
    """Represents a musical time duration as a fraction of a bar (measure).

    Musical time divisions (e.g., 1/4, 1/8) represent fractions of a bar.
    This is how synced delays work in music production: a 1/4 delay is one quarter
    of a bar, regardless of time signature. What changes across time signatures
    is the bar duration itself.

    Attributes
    ----------
    numerator : int
        The numerator of the time fraction (e.g., 1 in "1/4").
    denominator : int
        The denominator of the time fraction (e.g., 4 in "1/4").
    modifier : str
        Optional modifier: "" (none), "d" (dotted), "t" (triplet). Default is "".

    """

    numerator: int
    denominator: int
    modifier: str = ""

    _modifier_values = {
        "": 1.0,
        "d": 1.5,
        "t": 1 / 3,
    }

    def fraction(self) -> float:
        """Compute the fraction of a bar, considering the modifier.

        Returns
        -------
        float
            The duration as a fraction of a bar (measure).
            For example, 1/4 = 0.25 bars, 1/8 = 0.125 bars, 3/16 = 0.1875 bars.

        Raises
        ------
        ValueError
            If the modifier is invalid.

        """
        base = self.numerator / self.denominator
        modifier_coeff = self._modifier_values.get(self.modifier)
        if modifier_coeff is None:
            raise ValueError(f"Invalid time duration modifier: {self.modifier}")

        return base * modifier_coeff

    def duration_seconds(self, bpm: float, beats_per_bar: int = 4) -> Second:
        """Compute the duration in seconds of the musical time.

        Musical time divisions (1/4, 1/8, etc.) represent fractions of a bar (measure).
        This way a 1/4 delay is one quarter of a bar, regardless of time signature.

        Parameters
        ----------
        bpm : float
            Beats per minute. Must be positive.
        beats_per_bar : int, optional
            Number of beats in one bar. Defaults to 4 (for 4/4 time).

        Returns
        -------
        Second
            Duration in seconds.

        Raises
        ------
        ValueError
            If BPM is not positive.

        """
        assert bpm > 0, "BPM must be positive"
        beat_duration = 60.0 / bpm
        bar_duration = beat_duration * beats_per_bar
        return self.fraction() * bar_duration

    @classmethod
    def from_string(cls, s: str) -> "MusicalTime":
        """Create a MusicalTime instance from a string representation.

        Parameters
        ----------
        s : str
            Musical time string (e.g., "1/4", "1/8d", "1/8t").
            Format: numerator/denominator[modifier]
            where modifier is optional and can be "d" (dotted) or "t" (triplet).

        Returns
        -------
        MusicalTime
            A new MusicalTime instance.

        Raises
        ------
        ValueError
            If the string format is invalid.

        """
        m = re.match(r"(\d+)/(\d+)([dt]?)$", s)
        if not m:
            raise ValueError(f"Invalid musical time string: {s}")
        numerator = int(m.group(1))
        denominator = int(m.group(2))
        modifier = m.group(3)
        return cls(numerator, denominator, modifier)
