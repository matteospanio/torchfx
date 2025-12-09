#!/usr/bin/env python3
# This example demonstrates how to use delay effect

import torchfx as fx
import torch
import torchaudio

signal = fx.Wave.from_file("examples/sample_input.wav")
signal = signal.to("cuda" if torch.cuda.is_available() else "cpu")

result = signal | fx.effect.Delay(bpm=100, delay_time="1/4", taps=10, mix=0.5)

torchaudio.save("examples/out.wav", result.ys.cpu(), signal.fs)
