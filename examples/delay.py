#!/usr/bin/env python3
# This example demonstrates how to use delay effect

import torchfx as fx
import torch

signal = fx.Wave.from_file("examples/sample_input.wav")
signal = signal.to("cuda" if torch.cuda.is_available() else "cpu")

result = signal | fx.effect.Delay(bpm=100, delay_time="1/4", taps=10, mix=0.5)

result.save("examples/out.wav")
