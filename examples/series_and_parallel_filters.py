#!/usr/bin/env python3
# This example demonstrates how to combine different filters in series and parallel using torchfx.
# The target circuit is:
#                                                  |-> Highpass Butterworth (2000 Hz, order 4) - |
# Input -> Lowpass Butterworth (100 Hz, order 2) - |                                             | (+) -> Volume (0.5) -> Output
#                                                  |-> Highpass Chebyshev1  (2000 Hz, order 2) - |

import torchfx as fx
import torch
import torchaudio.transforms as T
import torchaudio

signal = fx.Wave.from_file("sample_input.wav")
signal = signal.to("cuda" if torch.cuda.is_available() else "cpu")

result = (signal
    | fx.filter.LoButterworth(100, order=2)
    | fx.filter.HiButterworth(2000, order=2) + fx.filter.HiChebyshev1(2000, order=2)
    | T.Vol(0.5)
)

torchaudio.save("examples/out.wav", result.ys.cpu(), signal.fs)