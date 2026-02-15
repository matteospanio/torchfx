"""Real-time audio processing example.

Demonstrates live audio processing through a microphone with
GPU-accelerated effects using TorchFX.

Requirements:
    pip install sounddevice

Usage:
    python realtime_processing.py
"""

from torchfx.effect import Gain, Reverb
from torchfx.filter import HiButterworth, LoButterworth, BiquadHPF, BiquadLPF
from torchfx.realtime import RealtimeProcessor, SoundDeviceBackend, StreamConfig

# Configure duplex stream (mic in -> speakers out)
config = StreamConfig(
    sample_rate=44100,
    buffer_size=512,
    channels_in=1,
    channels_out=1,
    latency="low",
)

# Build effect chain
effects = [
    BiquadHPF(220, 0.707),         # Remove low-frequency rumble
    BiquadLPF(8000, 0.707),       # Tame harsh highs
    Gain(1.5),                 # Boost signal
    Reverb(),     # Add ambience
]

# Create and run processor using context manager
# The processor automatically starts on enter and stops on exit.
print(f"Starting real-time processing at {config.sample_rate}Hz")
print(f"Buffer size: {config.buffer_size} samples")
print(f"Estimated latency: {config.latency_ms:.1f}ms")
print()

with RealtimeProcessor(
    effects=effects,
    backend=SoundDeviceBackend(),
    config=config,
) as processor:
    print("Processing audio... Press Enter to stop.")
    try:
        input()
    except KeyboardInterrupt:
        pass

print("Stopped.")
