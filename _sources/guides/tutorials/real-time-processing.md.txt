# Real-time Audio Processing

Discover how to implement real-time audio processing with TorchFX.

## Overview

This tutorial covers:
- Setting up real-time audio I/O with the SoundDevice backend
- Building effect chains for live processing
- Thread-safe parameter updates during playback
- Stream processing for large files

## Prerequisites

- Completion of [Getting Started](../getting-started/getting_started.md)
- Understanding of audio streaming concepts
- Install the optional `sounddevice` dependency:

```bash
pip install sounddevice
```

## Audio Backend Setup

TorchFX uses an abstract audio backend system. The primary backend uses
[sounddevice](https://python-sounddevice.readthedocs.io/) (PortAudio) for
cross-platform audio I/O.

### Stream Configuration

Configure your audio stream with `StreamConfig`:

```python
from torchfx.realtime import StreamConfig, StreamDirection

# Output-only stream (e.g., playback)
playback_config = StreamConfig(
    sample_rate=48000,
    buffer_size=512,
    channels_out=2,
)
print(playback_config.direction)  # StreamDirection.OUTPUT
print(f"Latency: {playback_config.latency_ms:.1f}ms")  # ~10.7ms

# Duplex stream (e.g., guitar processing)
duplex_config = StreamConfig(
    sample_rate=48000,
    buffer_size=256,
    channels_in=1,
    channels_out=1,
    latency="low",
)
print(duplex_config.direction)  # StreamDirection.DUPLEX
print(f"Latency: {duplex_config.latency_ms:.1f}ms")  # ~5.3ms
```

### Listing Audio Devices

```python
from torchfx.realtime import SoundDeviceBackend

backend = SoundDeviceBackend()
for device in backend.get_devices():
    print(f"{device['index']}: {device['name']} "
          f"(in={device['max_input_channels']}, out={device['max_output_channels']})")
```

## Real-Time Processing

### Basic Passthrough with Effects

The `RealtimeProcessor` connects an audio backend to an effect chain.
Use it as a context manager for automatic start/stop:

```python
from torchfx.realtime import RealtimeProcessor, SoundDeviceBackend, StreamConfig
from torchfx.effect import Gain, Reverb
from torchfx.filter.iir import LoButterworth, HiButterworth

config = StreamConfig(
    sample_rate=48000,
    buffer_size=512,
    channels_in=1,
    channels_out=1,
)

with RealtimeProcessor(
    effects=[
        HiButterworth(80),         # Remove low-frequency rumble
        LoButterworth(8000),       # Tame harsh highs
        Gain(1.5),                 # Boost signal
        Reverb(room_size=0.3),     # Add ambience
    ],
    backend=SoundDeviceBackend(),
    config=config,
) as processor:
    print(f"Processing at {processor.latency_ms:.1f}ms latency")
    # Audio flows: mic -> effects -> speakers
    input("Press Enter to stop...")
# Processor automatically stops when the block exits
```

The context manager ensures the stream is properly stopped even if an
exception occurs. You can also use `start()`/`stop()` manually if needed.

### Thread-Safe Parameter Updates

You can update effect parameters while processing is running:

```python
import time

with RealtimeProcessor(
    effects=[LoButterworth(500), Gain(1.0)],
    backend=SoundDeviceBackend(),
    config=config,
) as processor:
    # Sweep a low-pass filter cutoff from 500Hz to 5000Hz
    for cutoff in range(500, 5001, 100):
        processor.set_parameter("0.cutoff", cutoff)
        time.sleep(0.05)
```

The parameter format is `"<effect_index>.<attribute_name>"`. Parameters
are applied at buffer boundaries for thread safety.

### Automatic Sample Rate Propagation

Filters with `fs=None` automatically receive the sample rate from the
`StreamConfig`, just like with the `Wave` pipe operator:

```python
# No need to specify fs — it comes from config.sample_rate
lpf = LoButterworth(cutoff=2000)  # fs=None
processor = RealtimeProcessor(
    effects=[lpf],
    backend=SoundDeviceBackend(),
    config=StreamConfig(sample_rate=48000, ...),
)
print(lpf.fs)  # 48000
```

## Stream Processing for Large Files

For files too large to load into memory, use `StreamProcessor` as a
context manager:

```python
from torchfx.realtime import StreamProcessor
from torchfx.effect import Gain, Normalize

with StreamProcessor(
    effects=[Gain(0.8), Normalize(peak=0.95)],
    chunk_size=65536,  # 64K samples per chunk
) as processor:
    processor.process_file("large_podcast.wav", "normalized_podcast.wav")
```

### Generator API

For custom pipelines, use the generator:

```python
with StreamProcessor(effects=[Gain(0.5)], chunk_size=65536) as processor:
    for chunk in processor.process_chunks("large_file.wav"):
        # Each chunk is a tensor of shape (channels, chunk_size)
        print(f"Chunk: {chunk.shape}, max={chunk.abs().max():.3f}")
```

### GPU Acceleration

Process chunks on GPU for faster throughput:

```python
with StreamProcessor(
    effects=[Gain(0.5)],
    chunk_size=131072,  # Larger chunks benefit from GPU
    device="cuda",
) as processor:
    processor.process_file("input.wav", "output.wav")
```

## Performance Considerations

- **Buffer size**: Smaller buffers = lower latency but higher CPU load. 256-512 samples at 48kHz is typical.
- **GPU vs CPU**: For real-time processing with small buffers, CPU is often faster due to transfer overhead. Use GPU for stream processing with large chunks.
- **Effect chain length**: Each effect adds processing time. Keep chains short for real-time use.

## Next Steps

- [Filters Design](filters-design.md)
- [Effects Design](effects-design.md)
- [Custom Effects](custom-effects.md)
