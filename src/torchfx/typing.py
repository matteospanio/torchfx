"""Module to define custom types and dataclasses for the reverse detection module."""

import typing as tp

from annotated_types import Ge, Le
import torch

Decibel = tp.Annotated[float, Le(0)]
Millisecond = tp.Annotated[int, Ge(0)]
Second = tp.Annotated[float, Ge(0)]
BitRate = tp.Literal[16, 24, 32]

SpecScale = tp.Literal["mel", "lin", "log"]
FilterType = tp.Literal["low", "high"]
FilterOrderScale = tp.Literal["db", "linear"]
Device = tp.Literal["cpu", "cuda"] | torch.device
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
