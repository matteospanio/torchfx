"""Module to define custom types and dataclasses for the reverse detection module."""

import typing as tp

import torch
from annotated_types import Ge, Le

Decibel = tp.Annotated[float, Le(0)]
Millisecond = tp.Annotated[int, Ge(0)]
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
