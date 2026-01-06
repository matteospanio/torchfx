import torch
from torchfx import Wave
from torchfx.filter import LoButterworth, ParametricEQ

# Load audio
wave = Wave.from_file("sample_input.wav")

# Create filters
lowpass = LoButterworth(cutoff=2000, order=6, fs=wave.fs)
eq = ParametricEQ(frequency=1000, q=2.0, gain=3.0, fs=wave.fs)

# Sequential processing
processed = wave | lowpass | eq

# Parallel processing (filter combination)
stereo_enhancer = lowpass + eq
enhanced = wave | stereo_enhancer

# Save result
processed.save("output.wav")
