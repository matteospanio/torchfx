# Wave - Digital Audio Representation

The {class}`~torchfx.Wave` class is the foundation of TorchFX's audio processing system. It wraps PyTorch {class}`tensors <torch.Tensor>` with audio-specific metadata and methods, making it easy to work with digital audio signals.

## What is a Wave?

In digital audio processing, a signal is represented as a discrete sequence of samples. The {class}`~torchfx.Wave` class encapsulates:

- **Audio Data** ({term}`tensor`): A 2D PyTorch tensor with shape `(channels, samples)`
- **Sample Rate** ({term}`sampling frequency` or `fs`): Number of samples per second (e.g., 44100 Hz)
- **Metadata**: Optional information about the audio (encoding, bit depth, etc.)
- **Device**: Where the audio data lives (CPU or CUDA GPU)

```{mermaid}
classDiagram
    class Wave {
        +Tensor ys
        +int fs
        +dict metadata
        -Device __device
        +from_file(path) Wave$
        +save(path) None
        +to(device) Wave
        +transform(func) Wave
        +__or__(fx) Wave
        +channels() int
        +duration(unit) float
        +get_channel(index) Wave
        +merge(waves) Wave$
    }

    class Tensor {
        <<PyTorch>>
        Shape: (channels, samples)
    }

    Wave --> Tensor : wraps

    note for Wave "Container for digital audio\nwith sample rate and metadata"
    note for Tensor "2D tensor:\n- Dim 0: channels\n- Dim 1: time (samples)"
```

## Creating Waves

### From Audio Files

The most common way to create a {class}`~torchfx.Wave` is to load it from an audio file:

```python
import torchfx as fx

# Load from file (sample rate inferred from file)
wave = fx.Wave.from_file("audio.wav")

print(f"Sample rate: {wave.fs} Hz")
print(f"Channels: {wave.channels()}")
print(f"Duration: {wave.duration('sec')} seconds")
print(f"Samples: {len(wave)}")
print(f"Metadata: {wave.metadata}")
```

**Supported formats**: WAV, FLAC, MP3, OGG (depends on torchaudio backend)

### From NumPy/PyTorch Arrays

You can create a {class}`~torchfx.Wave` from existing array data:

```python
import torch
import numpy as np
import torchfx as fx

# From PyTorch tensor
stereo_data = torch.randn(2, 44100)  # 2 channels, 1 second at 44.1kHz
wave = fx.Wave(stereo_data, fs=44100)

# From NumPy array
mono_data = np.random.randn(1, 22050)  # 1 channel, 0.5 seconds at 44.1kHz
wave = fx.Wave(mono_data, fs=44100)
```

**Shape requirement**: Audio data must be 2D with shape `(channels, samples)`. For mono audio, use shape `(1, samples)`.

### Synthetic Signals

Generate test signals for development and testing:

```python
import torch
import torchfx as fx

# Generate a sine wave
fs = 44100
duration = 1.0  # seconds
frequency = 440  # Hz (A4 note)

t = torch.linspace(0, duration, int(fs * duration))
sine = torch.sin(2 * torch.pi * frequency * t).unsqueeze(0)  # Add channel dimension
wave = fx.Wave(sine, fs=fs)
```

## Working with Waves

### Device Management

{class}`~torchfx.Wave` objects can be moved between CPU and GPU for accelerated processing:

```python
import torchfx as fx

wave = fx.Wave.from_file("audio.wav")

# Move to GPU
wave.to("cuda")

# Or use the device property
wave.device = "cuda"

# Check current device
print(wave.device)  # "cuda"

# Move back to CPU
wave.to("cpu")
```

This is useful for batch processing large datasets or real-time effects that benefit from GPU acceleration.

```{seealso}
[PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html) - Understanding device management in PyTorch
```

### Channel Operations

Work with individual channels or multi-channel audio:

```python
import torchfx as fx

wave = fx.Wave.from_file("stereo.wav")

# Get number of channels
num_channels = wave.channels()  # 2

# Extract a specific channel (returns new Wave)
left = wave.get_channel(0)
right = wave.get_channel(1)

# Process individual channels
processed_left = left | SomeEffect()
```

### Duration and Length

Get audio duration in different units:

```python
import torchfx as fx

wave = fx.Wave.from_file("audio.wav")

# Duration in seconds
duration_sec = wave.duration("sec")  # 3.5

# Duration in milliseconds
duration_ms = wave.duration("ms")  # 3500.0

# Length in samples
num_samples = len(wave)  # 154350 (at 44.1kHz)
```

### Merging Waves

Combine multiple waves into one:

```python
import torchfx as fx

wave1 = fx.Wave.from_file("audio1.wav")
wave2 = fx.Wave.from_file("audio2.wav")

# Mix waves (sum channels)
mixed = fx.Wave.merge([wave1, wave2], split_channels=False)

# Keep channels separate (stack as new channels)
stacked = fx.Wave.merge([wave1, wave2], split_channels=True)
```

**Note**: All waves must have the same {term}`sample rate`. Shorter waves are zero-padded when mixing.

## Transformations

### Functional Transformations

Apply any function to the underlying tensor:

```python
import torch
import torchfx as fx

wave = fx.Wave.from_file("audio.wav")

# Apply FFT
spectrum = wave.transform(torch.fft.fft)

# Apply custom function
def amplify(x):
    return x * 2.0

louder = wave.transform(amplify)

# Chain transformations
processed = wave.transform(torch.fft.fft).transform(torch.abs)
```

The {meth}`~torchfx.Wave.transform` method returns a new {class}`~torchfx.Wave` with the same sample rate and metadata.

### Pipeline Operator

The recommended way to apply effects is using the pipeline operator (`|`):

```python
import torchfx as fx
from torchfx.filter import iir

wave = fx.Wave.from_file("audio.wav")

# Apply single effect
filtered = wave | iir.LoButterworth(cutoff=1000)

# Chain multiple effects
processed = wave | iir.HiButterworth(100) | fx.effect.Reverb() | fx.effect.Normalize()
```

The pipeline operator:
- Automatically configures effects with the wave's sample rate
- Returns a new {class}`~torchfx.Wave` (immutable operations)
- Supports any {class}`~torchfx.FX` subclass or {class}`torch.nn.Module`

```{seealso}
{doc}`pipeline-operator` - Deep dive into the pipeline operator
```

## Saving Waves

### Basic Saving

Save processed audio back to disk:

```python
import torchfx as fx

wave = fx.Wave.from_file("input.wav")
processed = wave | SomeEffect()

# Save with default settings (format inferred from extension)
processed.save("output.wav")
```

### Advanced Saving Options

Control encoding and bit depth:

```python
import torchfx as fx

wave = fx.Wave.from_file("audio.wav")

# Save as 24-bit FLAC
wave.save("output.flac", encoding="PCM_S", bits_per_sample=24)

# Save as 32-bit float WAV
wave.save("output.wav", encoding="PCM_F", bits_per_sample=32)

# Force specific format (overrides extension)
wave.save("output.ogg", format="wav")
```

**Encoding options**:
- `PCM_S`: Signed integer PCM (most common)
- `PCM_U`: Unsigned integer PCM
- `PCM_F`: Floating-point PCM (32-bit or 64-bit)
- `ULAW`, `ALAW`: Compressed formats

**Bit depth options**: 8, 16, 24, 32, 64 (depending on encoding)

```{note}
The {meth}`~torchfx.Wave.save` method automatically creates parent directories and moves data to CPU before saving.
```

## Implementation Details

### Internal Representation

```{mermaid}
sequenceDiagram
    participant User
    participant Wave
    participant Tensor
    participant Device

    User->>Wave: from_file("audio.wav")
    Wave->>Tensor: Load audio data
    Tensor-->>Wave: shape (channels, samples)
    Wave->>Device: Set device="cpu"
    Wave-->>User: Return Wave object

    User->>Wave: wave.to("cuda")
    Wave->>Tensor: tensor.to("cuda")
    Wave->>Device: Update __device="cuda"
    Wave-->>User: Return self

    User->>Wave: wave | Effect()
    Wave->>Wave: __or__(effect)
    Wave->>Wave: __update_config(effect)
    Wave->>Wave: transform(effect.forward)
    Wave-->>User: Return new Wave
```

### Sample Rate Configuration

When using the pipeline operator, {class}`~torchfx.Wave` automatically configures effects:

1. Checks if the effect is an {class}`~torchfx.FX` instance
2. Sets the effect's `fs` attribute if it's `None`
3. For filters, calls `compute_coefficients()` if not yet computed
4. Applies the effect's `forward()` method to the audio tensor

This ensures effects always have the correct sample rate without manual configuration.

```python
import torchfx as fx
from torchfx.filter import iir

wave = fx.Wave.from_file("audio.wav")  # fs = 44100

# Filter automatically configured with fs=44100
filtered = wave | iir.LoButterworth(cutoff=1000)
```

## Mathematical Representation

A discrete-time signal in TorchFX is represented as:

$$
\mathbf{x}[n] = [x_1[n], x_2[n], \ldots, x_C[n]]^T
$$

where:
- $\mathbf{x}[n]$ is the multi-channel signal at sample index $n$
- $x_c[n]$ is the signal for channel $c$ at sample $n$
- $C$ is the number of channels
- $n \in [0, N-1]$ where $N$ is the total number of samples

The {term}`sample rate` $f_s$ determines the relationship between sample index and time:

$$
t = \frac{n}{f_s}
$$

## Best Practices

### Memory Management

```python
import torchfx as fx

# ✅ GOOD: Reuse wave object with pipeline
wave = fx.Wave.from_file("audio.wav")
processed = wave | Effect1() | Effect2() | Effect3()

# ❌ BAD: Creating multiple intermediate wave objects
wave = fx.Wave.from_file("audio.wav")
wave1 = wave | Effect1()
wave2 = wave1 | Effect2()
wave3 = wave2 | Effect3()
```

### Batch Processing

```python
import torchfx as fx
from pathlib import Path

# Process multiple files
input_dir = Path("audio_files")
output_dir = Path("processed")

effect_chain = Effect1() | Effect2()

for audio_file in input_dir.glob("*.wav"):
    wave = fx.Wave.from_file(audio_file)
    processed = wave | effect_chain
    processed.save(output_dir / audio_file.name)
```

### GPU Acceleration

```python
import torchfx as fx
import torch

# Check GPU availability
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

wave = fx.Wave.from_file("audio.wav").to(device)
processed = wave | HeavyEffect()
processed.to("cpu").save("output.wav")  # Move back to CPU for saving
```

## Common Pitfalls

### Incorrect Tensor Shape

```python
import torch
import torchfx as fx

# ❌ WRONG: 1D tensor
mono = torch.randn(44100)
wave = fx.Wave(mono, fs=44100)  # Error!

# ✅ CORRECT: 2D tensor with channel dimension
mono = torch.randn(1, 44100)
wave = fx.Wave(mono, fs=44100)
```

### Sample Rate Mismatch

```python
import torchfx as fx

wave1 = fx.Wave.from_file("audio_44k.wav")  # fs = 44100
wave2 = fx.Wave.from_file("audio_48k.wav")  # fs = 48000

# ❌ WRONG: Cannot merge waves with different sample rates
mixed = fx.Wave.merge([wave1, wave2])  # ValueError!

# ✅ CORRECT: Resample first (using external library)
import torchaudio.transforms as T

resampler = T.Resample(orig_freq=48000, new_freq=44100)
wave2_resampled = fx.Wave(resampler(wave2.ys), fs=44100)
mixed = fx.Wave.merge([wave1, wave2_resampled])
```

## Related Concepts

- {doc}`fx` - Effects that process Wave objects
- {doc}`pipeline-operator` - Chaining effects with `|`
- {doc}`type-system` - Time units and musical notation
- {doc}`/guides/tutorials/ml-batch-processing` - Processing waves in ML pipelines

## External Resources

- [Digital Signal on Wikipedia](https://en.wikipedia.org/wiki/Digital_signal) - Understanding discrete-time signals
- [Sample Rate on Wikipedia](https://en.wikipedia.org/wiki/Sampling_(signal_processing)) - How sampling works
- [PyTorch Tensor Documentation](https://pytorch.org/docs/stable/tensors.html) - Working with tensors
- [torchaudio I/O Documentation](https://pytorch.org/audio/stable/io.html) - Loading and saving audio

## References

```{bibliography}
:filter: docname in docnames
:style: alpha
```
