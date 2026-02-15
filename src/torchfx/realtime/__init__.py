"""Real-time audio processing for TorchFX.

This module provides real-time audio I/O, ring buffering, and stream
processing capabilities for the TorchFX library.

The ``sounddevice`` backend is optional and only required for real-time
audio I/O. The ``StreamProcessor`` and ``TensorRingBuffer`` work without
any optional dependencies.

Classes
-------
AudioBackend
    Abstract base class for audio I/O backends.
SoundDeviceBackend
    PortAudio backend via sounddevice (optional dependency).
StreamConfig
    Configuration for audio streams.
StreamDirection
    Audio stream direction enum.
StreamState
    Audio stream lifecycle state enum.
TensorRingBuffer
    Lock-free SPSC ring buffer for tensor data.
RealtimeProcessor
    Real-time audio processor connecting backend to effect chain.
StreamProcessor
    Chunk-based file processor for large audio files.

Examples
--------
Real-time processing (requires sounddevice):

>>> from torchfx.realtime import RealtimeProcessor, SoundDeviceBackend, StreamConfig
>>> config = StreamConfig(sample_rate=48000, buffer_size=512,
...                       channels_in=2, channels_out=2)
>>> # processor = RealtimeProcessor(
>>> #     effects=[fx.Gain(0.5)],
>>> #     backend=SoundDeviceBackend(),
>>> #     config=config,
>>> # )

File stream processing (no optional dependencies):

>>> from torchfx.realtime import StreamProcessor
>>> # processor = StreamProcessor(effects=[fx.Gain(0.5)])
>>> # processor.process_file("large_input.wav", "output.wav")

"""

from torchfx.realtime.backend import (
    AudioBackend,
    AudioCallback,
    StreamConfig,
    StreamDirection,
    StreamState,
)
from torchfx.realtime.exceptions import (
    BackendNotAvailableError,
    BufferOverrunError,
    BufferUnderrunError,
    RealtimeError,
    StreamError,
)
from torchfx.realtime.processor import RealtimeProcessor
from torchfx.realtime.ring_buffer import TensorRingBuffer
from torchfx.realtime.stream import StreamProcessor


def __getattr__(name: str) -> type:
    """Lazy import for optional backends.

    This allows ``from torchfx.realtime import SoundDeviceBackend``
    without requiring sounddevice to be installed unless actually used.

    """
    if name == "SoundDeviceBackend":
        from torchfx.realtime.sounddevice_backend import SoundDeviceBackend

        return SoundDeviceBackend
    raise AttributeError(f"module 'torchfx.realtime' has no attribute {name!r}")


__all__ = [
    # Core
    "AudioBackend",
    "AudioCallback",
    "StreamConfig",
    "StreamDirection",
    "StreamState",
    "TensorRingBuffer",
    "RealtimeProcessor",
    "StreamProcessor",
    # Optional backends
    "SoundDeviceBackend",
    # Exceptions
    "RealtimeError",
    "BackendNotAvailableError",
    "StreamError",
    "BufferOverrunError",
    "BufferUnderrunError",
]
