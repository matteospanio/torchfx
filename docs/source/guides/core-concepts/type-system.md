# Type System - Musical Time and Audio Units

TorchFX provides a rich type system designed specifically for audio processing. This system enables you to express time in musical terms (BPM, note divisions) rather than just samples or seconds, making effects more intuitive for musical applications.

## Overview

The type system includes:

- **Musical Time**: BPM-synchronized time divisions (quarter notes, eighth notes, triplets, dotted notes)
- **Time Units**: Seconds, milliseconds, samples
- **Audio Units**: Decibels, bit rates
- **Device Types**: CPU and CUDA device specifications
- **Window Types**: Window functions for spectral analysis

```{mermaid}
graph TB
    subgraph "TorchFX Type System"
        Musical["Musical Time<br/>MusicalTime class"]
        Time["Time Units<br/>Second, Millisecond"]
        Audio["Audio Units<br/>Decibel, BitRate"]
        Device["Device Types<br/>CPU, CUDA"]
        Window["Window Types<br/>Hann, Hamming, etc."]
    end

    subgraph "Applications"
        Delay["BPM-synced Delay"]
        Filter["Filters with cutoff"]
        Save["Audio file saving"]
        GPU["GPU acceleration"]
    end

    Musical --> Delay
    Time --> Filter
    Audio --> Save
    Device --> GPU
    Window --> Spectral["Spectral analysis"]

    style Musical fill:#e1f5ff
    style Time fill:#fff5e1
    style Audio fill:#ffe1e1
    style Device fill:#e8f5e1
    style Window fill:#f5e1ff
```

## Musical Time

The {class}`~torchfx.typing.MusicalTime` class represents musical time divisions as fractions of a bar (measure).

### Basic Concept

In music production, delays and rhythmic effects are often synchronized to the tempo (BPM). Instead of specifying delay times in milliseconds, you specify them as note divisions:

- `1/4`: Quarter note (one beat in 4/4 time)
- `1/8`: Eighth note (half a beat)
- `1/16`: Sixteenth note (quarter of a beat)
- `1/2`: Half note (two beats)

### Creating Musical Times

#### From Strings

```python
from torchfx.typing import MusicalTime

# Basic note divisions
quarter = MusicalTime.from_string("1/4")
eighth = MusicalTime.from_string("1/8")
sixteenth = MusicalTime.from_string("1/16")

# Dotted notes (1.5x duration)
dotted_quarter = MusicalTime.from_string("1/4d")
dotted_eighth = MusicalTime.from_string("1/8d")

# Triplets (2/3x duration)
eighth_triplet = MusicalTime.from_string("1/8t")
quarter_triplet = MusicalTime.from_string("1/4t")
```

#### Direct Construction

```python
from torchfx.typing import MusicalTime

# Create directly
quarter = MusicalTime(numerator=1, denominator=4)
dotted_eighth = MusicalTime(numerator=1, denominator=8, modifier="d")
eighth_triplet = MusicalTime(numerator=1, denominator=8, modifier="t")
```

### Converting to Time

Convert musical time to seconds based on BPM:

```python
from torchfx.typing import MusicalTime

quarter = MusicalTime.from_string("1/4")

# At 120 BPM in 4/4 time
duration = quarter.duration_seconds(bpm=120, beats_per_bar=4)
print(f"Duration: {duration} seconds")  # 0.5 seconds

# At 140 BPM
duration = quarter.duration_seconds(bpm=140)
print(f"Duration: {duration} seconds")  # ~0.428 seconds
```

### Using with Effects

The {class}`~torchfx.effect.Delay` effect uses {class}`~torchfx.typing.MusicalTime` for BPM-synced delays:

```python
import torchfx as fx

wave = fx.Wave.from_file("audio.wav")

# Delay synced to 128 BPM, eighth note
delay = fx.effect.Delay(bpm=128, delay_time="1/8", feedback=0.4, mix=0.3)
delayed = wave | delay

# Dotted quarter note delay (classic rock delay)
delay = fx.effect.Delay(bpm=120, delay_time="1/4d", feedback=0.5, mix=0.25)
delayed = wave | delay

# Triplet delay
delay = fx.effect.Delay(bpm=140, delay_time="1/8t", feedback=0.3, mix=0.2)
delayed = wave | delay
```

### Mathematical Representation

The duration of a musical time division is calculated as:

$$
t = \frac{n}{d} \times m \times \frac{60}{BPM} \times \text{beats\_per\_bar}
$$

where:
- $n$ is the numerator (e.g., 1 in "1/4")
- $d$ is the denominator (e.g., 4 in "1/4")
- $m$ is the modifier coefficient:
  - $m = 1.0$ for normal notes
  - $m = 1.5$ for dotted notes (`d`)
  - $m = \frac{1}{3}$ for triplets (`t`)
- $BPM$ is beats per minute
- $\text{beats\_per\_bar}$ is the number of beats in a bar (default 4 for 4/4 time)

### Note Division Reference

Common note divisions and their durations at 120 BPM in 4/4 time:

| Division | Name | Modifier | Duration (seconds) | Duration (ms) |
|----------|------|----------|-------------------|---------------|
| `1/1` | Whole note | - | 2.000 | 2000 |
| `1/2` | Half note | - | 1.000 | 1000 |
| `1/4` | Quarter note | - | 0.500 | 500 |
| `1/4d` | Dotted quarter | Dotted | 0.750 | 750 |
| `1/8` | Eighth note | - | 0.250 | 250 |
| `1/8d` | Dotted eighth | Dotted | 0.375 | 375 |
| `1/8t` | Eighth triplet | Triplet | 0.167 | 167 |
| `1/16` | Sixteenth note | - | 0.125 | 125 |
| `1/16d` | Dotted sixteenth | Dotted | 0.188 | 188 |

```{seealso}
[Note Value on Wikipedia](https://en.wikipedia.org/wiki/Note_value) - Understanding musical time divisions
```

## Time Units

### Second and Millisecond

Type-annotated aliases for time values:

```python
from torchfx.typing import Second, Millisecond

# These are annotated types ensuring non-negative values
duration_sec: Second = 1.5  # 1.5 seconds
duration_ms: Millisecond = 1500.0  # 1500 milliseconds
```

Used in {class}`~torchfx.Wave` duration methods:

```python
import torchfx as fx

wave = fx.Wave.from_file("audio.wav")

# Get duration in seconds
duration_sec = wave.duration("sec")  # Returns Second type

# Get duration in milliseconds
duration_ms = wave.duration("ms")  # Returns Millisecond type
```

### Sample-Based Time

Convert between samples and time units:

```python
import torchfx as fx

wave = fx.Wave.from_file("audio.wav")
fs = wave.fs  # Sample rate

# Samples to seconds
num_samples = 44100
duration_sec = num_samples / fs  # 1.0 second at 44100 Hz

# Seconds to samples
duration_sec = 0.5
num_samples = int(duration_sec * fs)  # 22050 samples at 44100 Hz

# Milliseconds to samples
duration_ms = 100.0  # 100 ms
num_samples = int((duration_ms / 1000.0) * fs)  # 4410 samples at 44100 Hz
```

## Audio Units

### Decibels

Type-annotated for decibel values (≤ 0):

```python
from torchfx.typing import Decibel

# Decibel values must be non-positive (≤ 0)
gain_db: Decibel = -6.0  # -6 dB attenuation
reference_db: Decibel = 0.0  # 0 dB (unity gain)
```

Used in effects like {class}`~torchfx.effect.Gain`:

```python
import torchfx as fx

wave = fx.Wave.from_file("audio.wav")

# Gain in decibels
gained = wave | fx.effect.Gain(gain=-3.0, gain_type="db")

# Convert between amplitude and dB
import math

amplitude = 0.5
db = 20 * math.log10(amplitude)  # -6.02 dB

db_value = -6.0
amplitude = 10 ** (db_value / 20)  # 0.501
```

### Bit Rate

Audio bit depth specification:

```python
from torchfx.typing import BitRate

# Valid bit rates: 8, 16, 24, 32
bit_depth: BitRate = 24
```

Used when saving audio files:

```python
import torchfx as fx

wave = fx.Wave.from_file("audio.wav")

# Save as 24-bit audio
wave.save("output.wav", bits_per_sample=24, encoding="PCM_S")

# Save as 16-bit audio (CD quality)
wave.save("output.wav", bits_per_sample=16, encoding="PCM_S")
```

## Device Types

### CPU and CUDA

Specify where computations should run:

```python
from torchfx.typing import Device
import torch

# Type can be string literal or torch.device
device1: Device = "cpu"
device2: Device = "cuda"
device3: Device = torch.device("cuda:0")  # Specific GPU
```

Used in {class}`~torchfx.Wave` for device management:

```python
import torchfx as fx
import torch

wave = fx.Wave.from_file("audio.wav")

# Move to GPU
wave.to("cuda")

# Check device
print(wave.device)  # "cuda"

# Create wave directly on GPU
if torch.cuda.is_available():
    tensor = torch.randn(2, 44100, device="cuda")
    wave = fx.Wave(tensor, fs=44100, device="cuda")
```

```{seealso}
[PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html) - Device management in PyTorch
```

## Window Types

### Window Functions

For spectral analysis and filter design:

```python
from torchfx.typing import WindowType

# Available window types
valid_windows: list[WindowType] = [
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
```

Used in spectral analysis and filter design:

```python
import torch
from torchfx.typing import WindowType

# Create window for FFT
window_type: WindowType = "hann"
window_length = 2048

# Using torch directly
window = torch.hann_window(window_length)

# Or using scipy.signal.get_window (if needed)
from scipy import signal
window = signal.get_window(window_type, window_length)
```

### Window Characteristics

Different windows have different characteristics:

| Window | Main Lobe Width | Side Lobe Level | Use Case |
|--------|----------------|-----------------|----------|
| `boxcar` | Narrow | High (-13 dB) | Fast transitions, high resolution |
| `hann` | Medium | Medium (-31 dB) | General purpose, balanced |
| `hamming` | Medium | Low (-43 dB) | Low side lobes, smooth |
| `blackman` | Wide | Very Low (-58 dB) | Minimal leakage, wide transitions |
| `kaiser` | Adjustable | Adjustable | Configurable trade-offs |
| `flattop` | Very Wide | Very Low | Amplitude accuracy |

```{seealso}
[Window Function on Wikipedia](https://en.wikipedia.org/wiki/Window_function) - Detailed window function analysis
```

## Filter-Specific Types

### Filter Type

Specify filter mode:

```python
from torchfx.typing import FilterType

# Valid filter types
mode1: FilterType = "low"   # Low-pass
mode2: FilterType = "high"  # High-pass
```

### Filter Order Scale

For displaying filter order:

```python
from torchfx.typing import FilterOrderScale

# Display order in dB/octave or linear units
scale1: FilterOrderScale = "db"       # dB/octave (e.g., -40 dB/oct)
scale2: FilterOrderScale = "linear"   # Linear order (e.g., order=4)
```

## Spectral Types

### Spectrogram Scale

Specify frequency scale for spectrograms:

```python
from torchfx.typing import SpecScale

# Valid spectrogram scales
scale1: SpecScale = "mel"    # Mel scale (perceptual)
scale2: SpecScale = "lin"    # Linear scale
scale3: SpecScale = "log"    # Logarithmic scale
```

## Type Annotations in Custom Effects

Use TorchFX types for better type safety:

```python
from torchfx.effect import FX
from torchfx.typing import Second, Millisecond, Device, Decibel
from torch import Tensor
import torch

class CustomEffect(FX):
    """Custom effect with type-annotated parameters."""

    def __init__(
        self,
        duration_ms: Millisecond,
        gain_db: Decibel,
        fs: int | None = None,
        device: Device = "cpu",
    ) -> None:
        super().__init__()
        assert duration_ms >= 0, "Duration must be non-negative"
        assert gain_db <= 0, "Gain in dB must be non-positive"

        self.duration_ms = duration_ms
        self.gain_db = gain_db
        self.fs = fs
        self.device = device

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        # Convert dB to linear gain
        linear_gain = 10 ** (self.gain_db / 20)

        # Convert ms to samples (if fs is set)
        if self.fs is not None:
            duration_samples = int((self.duration_ms / 1000.0) * self.fs)
            # Use duration_samples for processing

        return x * linear_gain
```

## Best Practices

### Use Musical Time for Rhythmic Effects

```python
# ✅ GOOD: Musical time for delay
delay = fx.effect.Delay(bpm=128, delay_time="1/8", feedback=0.4)

# ❌ LESS GOOD: Hardcoded samples (not tempo-aware)
delay = fx.effect.Delay(delay_samples=5512)  # What BPM is this?
```

### Validate Type Constraints

```python
from torchfx.typing import Decibel

def set_gain(gain_db: Decibel) -> None:
    # Decibel type hints that gain_db should be ≤ 0
    assert gain_db <= 0, "Gain in dB must be non-positive"
    # ... implementation

# This hints at incorrect usage
set_gain(6.0)  # Type checker may warn (positive dB)

# Correct usage
set_gain(-6.0)  # Attenuation
```

### Document Units in Docstrings

```python
class MyEffect(FX):
    """Custom effect with time-based parameter.

    Parameters
    ----------
    delay_ms : Millisecond
        Delay time in milliseconds. Must be non-negative.
    gain_db : Decibel
        Gain in decibels. Must be non-positive (≤ 0).
    fs : int, optional
        Sample rate in Hz. If None, inferred from Wave.
    """
```

## Common Conversions

### BPM to Time

```python
from torchfx.typing import MusicalTime

bpm = 120
time_div = "1/4"  # Quarter note

musical_time = MusicalTime.from_string(time_div)
duration_sec = musical_time.duration_seconds(bpm=bpm)

print(f"At {bpm} BPM, a {time_div} note lasts {duration_sec} seconds")
# At 120 BPM, a 1/4 note lasts 0.5 seconds
```

### Samples to Time

```python
fs = 44100  # Sample rate

# Samples to seconds
samples = 22050
seconds = samples / fs  # 0.5 seconds

# Samples to milliseconds
milliseconds = (samples / fs) * 1000  # 500 ms
```

### Amplitude to Decibels

```python
import math

# Amplitude to dB
amplitude = 0.5
db = 20 * math.log10(amplitude)  # -6.02 dB

# dB to amplitude
db = -3.0
amplitude = 10 ** (db / 20)  # 0.708
```

## Related Concepts

- {doc}`wave` - Uses time types for duration methods
- {doc}`fx` - Effects use musical time and audio units
- {doc}`/guides/tutorials/filters-design` - Filters using window types
- {doc}`/guides/tutorials/bpm-delay` - BPM-synced delay using MusicalTime

## External Resources

- [Musical Note Values on Wikipedia](https://en.wikipedia.org/wiki/Note_value) - Understanding musical time divisions
- [Decibel on Wikipedia](https://en.wikipedia.org/wiki/Decibel) - Understanding dB scale
- [Audio Bit Depth on Wikipedia](https://en.wikipedia.org/wiki/Audio_bit_depth) - Bit rate and audio quality
- [Python Type Hints](https://docs.python.org/3/library/typing.html) - Type annotation system

## References

```{bibliography}
:filter: docname in docnames
:style: alpha
```
