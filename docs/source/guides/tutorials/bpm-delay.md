# BPM-Synced Delay Effects

Learn how to create musically-timed delay effects using TorchFX's BPM synchronization system. This tutorial covers musical time divisions, tempo-synced delays, and rhythmic echo patterns.

## Overview

In music production, delays are often synchronized to the tempo (BPM) rather than specified in milliseconds. This ensures delays stay musically coherent even when the tempo changes. TorchFX's {class}`~torchfx.effect.Delay` effect supports:

- **Musical time notation**: Quarter notes (`1/4`), eighth notes (`1/8`), etc.
- **BPM synchronization**: Automatic calculation from tempo
- **Dotted and triplet notes**: `1/4d`, `1/8t`, etc.
- **Multiple taps**: Create rhythmic echo patterns
- **Delay strategies**: Mono and ping-pong stereo effects

```{mermaid}
graph LR
    Input[User Input] --> MT[Musical Time<br/>e.g., 1/8]
    BPM[BPM<br/>e.g., 120] --> Calc[Calculate<br/>Duration]
    MT --> Calc
    FS[Sample Rate<br/>fs=44100] --> Samples[Convert to<br/>Samples]
    Calc --> Samples
    Samples --> Delay[Apply Delay<br/>with Taps]
    Delay --> Output[Delayed Audio]

    style Input fill:#e1f5ff
    style Output fill:#e1f5ff
    style MT fill:#fff5e1
    style BPM fill:#ffe1e1
    style Calc fill:#e8f5e1
```

## Musical Time Notation

### Basic Time Divisions

Musical time in TorchFX uses the format `n/d[modifier]`:

| Notation | Name | Bar Fraction | At 120 BPM (4/4) |
|----------|------|--------------|------------------|
| `1/1` | Whole note | 1.0 bars | 2.0 seconds |
| `1/2` | Half note | 0.5 bars | 1.0 seconds |
| `1/4` | Quarter note | 0.25 bars | 0.5 seconds |
| `1/8` | Eighth note | 0.125 bars | 0.25 seconds |
| `1/16` | Sixteenth note | 0.0625 bars | 0.125 seconds |
| `3/16` | Three sixteenths | 0.1875 bars | 0.375 seconds |

### Modifiers

Add modifiers for dotted and triplet notes:

| Modifier | Multiplier | Example | Bar Fraction | At 120 BPM (4/4) |
|----------|------------|---------|--------------|------------------|
| ` ` (none) | ×1.0 | `1/4` | 0.25 bars | 0.5 seconds |
| `d` (dotted) | ×1.5 | `1/4d` | 0.375 bars | 0.75 seconds |
| `t` (triplet) | ×1/3 | `1/4t` | 0.0833 bars | 0.167 seconds |

**Dotted notes**: Add half the original duration (1.5×)
**Triplet notes**: Fit three notes in the space of two (1/3×)

```python
from torchfx.typing import MusicalTime

# Parse musical time notation
quarter = MusicalTime.from_string("1/4")
print(quarter.fraction())  # 0.25 bars

dotted_quarter = MusicalTime.from_string("1/4d")
print(dotted_quarter.fraction())  # 0.375 bars

eighth_triplet = MusicalTime.from_string("1/8t")
print(eighth_triplet.fraction())  # 0.0417 bars (1/24)

# Convert to seconds at specific BPM
duration = quarter.duration_seconds(bpm=120, beats_per_bar=4)
print(f"Duration: {duration} seconds")  # 0.5 seconds
```

```{seealso}
{doc}`/guides/core-concepts/type-system` - Complete type system documentation
```

## BPM-to-Samples Conversion

Understanding how musical time converts to sample-accurate delays:

```{mermaid}
graph TB
    Start[Input: BPM=120<br/>delay_time=1/4<br/>fs=44100]

    Parse[Parse Musical Time<br/>MusicalTime.from_string]
    Fraction[Calculate Bar Fraction<br/>fraction = 0.25]
    BeatDur[Calculate Beat Duration<br/>60 / 120 = 0.5s]
    BarDur[Calculate Bar Duration<br/>0.5s × 4 beats = 2.0s]
    NoteDur[Calculate Note Duration<br/>0.25 bars × 2.0s = 0.5s]
    Samples[Convert to Samples<br/>0.5s × 44100 Hz = 22050]

    Start --> Parse
    Parse --> Fraction
    Start --> BeatDur
    BeatDur --> BarDur
    Fraction --> NoteDur
    BarDur --> NoteDur
    NoteDur --> Samples

    style Start fill:#e1f5ff
    style Samples fill:#e1f5ff
```

**Formula**:

$$
\text{samples} = \frac{n}{d} \times m \times \frac{60}{BPM} \times \text{beats\_per\_bar} \times f_s
$$

Where:
- $n/d$ is the note division (e.g., 1/4)
- $m$ is the modifier (1.0, 1.5, or 1/3)
- $BPM$ is beats per minute
- $\text{beats\_per\_bar}$ is typically 4 (for 4/4 time)
- $f_s$ is the sample rate

## Basic Usage

### With Wave Pipeline (Automatic fs)

The recommended way—let {class}`~torchfx.Wave` configure the sample rate:

```python
import torchfx as fx

# Load audio
wave = fx.Wave.from_file("audio.wav")  # fs = 44100

# Create BPM-synced delay (fs auto-configured)
delay = fx.effect.Delay(
    bpm=128,           # 128 beats per minute
    delay_time="1/8",  # Eighth note delay
    feedback=0.3,      # 30% feedback for echoes
    mix=0.2,           # 20% wet/dry mix
    taps=3             # 3 delay taps
)

# Apply using pipeline operator
delayed = wave | delay  # fs automatically set to 44100

delayed.save("delayed.wav")
```

### With Explicit Sample Rate

When working directly with tensors:

```python
import torch
import torchfx as fx

# Create or load waveform
waveform = torch.randn(2, 44100)  # Stereo, 1 second

# Create delay with explicit fs
delay = fx.effect.Delay(
    bpm=128,
    delay_time="1/8",
    fs=44100,  # Explicit sample rate
    feedback=0.3,
    mix=0.2
)

# Apply to tensor
delayed = delay(waveform)
```

### Direct Sample Specification

For non-musical delays or precise control:

```python
import torchfx as fx

# Delay by exact number of samples
delay = fx.effect.Delay(
    delay_samples=2205,  # 50ms at 44.1kHz
    feedback=0.4,
    mix=0.3,
    taps=4
)

# No BPM or fs needed when using delay_samples
delayed = delay(waveform)
```

## Taps and Feedback

### Understanding Taps

**Taps** create multiple echoes of the original signal. Each tap is delayed by `tap_number × delay_time`:

```python
import torchfx as fx

wave = fx.Wave.from_file("audio.wav")

# Create delay with 4 taps
delay = fx.effect.Delay(
    bpm=120,
    delay_time="1/4",  # 0.5s at 120 BPM
    feedback=0.6,
    taps=4
)

delayed = wave | delay

# Results in echoes at:
# Tap 1: 0.5s (amplitude = 1.0)
# Tap 2: 1.0s (amplitude = 0.6^1 = 0.6)
# Tap 3: 1.5s (amplitude = 0.6^2 = 0.36)
# Tap 4: 2.0s (amplitude = 0.6^3 = 0.216)
```

### Feedback Decay Pattern

The first tap always has amplitude 1.0. Subsequent taps decay exponentially:

$$
\text{amplitude}_n = \begin{cases}
1.0 & \text{if } n = 1 \\
\text{feedback}^{n-1} & \text{if } n > 1
\end{cases}
$$

**Example visualization**:

```python
import matplotlib.pyplot as plt
import numpy as np

feedback = 0.6
taps = 6

tap_numbers = np.arange(1, taps + 1)
amplitudes = np.array([1.0] + [feedback**(n-1) for n in range(2, taps + 1)])

plt.stem(tap_numbers, amplitudes)
plt.xlabel("Tap Number")
plt.ylabel("Amplitude")
plt.title(f"Delay Tap Amplitudes (feedback={feedback})")
plt.grid(True)
plt.show()
```

### Output Length Extension

The output is extended to accommodate all delay taps:

```python
original_length = 44100  # 1 second at 44.1kHz
delay_samples = 22050    # 0.5 second delay
taps = 3

output_length = original_length + (delay_samples * taps)
# output_length = 44100 + (22050 * 3) = 110250 samples (~2.5 seconds)
```

## Delay Strategies

### MonoDelayStrategy (Default)

Applies identical delay to all channels:

```python
import torchfx as fx
from torchfx.effect import Delay, MonoDelayStrategy

wave = fx.Wave.from_file("stereo.wav")

# Mono strategy (default)
delay = Delay(
    bpm=120,
    delay_time="1/8",
    feedback=0.4,
    mix=0.3,
    strategy=MonoDelayStrategy()  # Optional, this is default
)

delayed = wave | delay
```

**Behavior**: Each channel receives the same delay pattern independently.

### PingPongDelayStrategy

Creates alternating delays between left and right stereo channels:

```python
import torchfx as fx
from torchfx.effect import Delay, PingPongDelayStrategy

wave = fx.Wave.from_file("stereo.wav")  # Must be stereo

# Ping-pong delay
delay = Delay(
    bpm=120,
    delay_time="1/8",
    feedback=0.5,
    mix=0.4,
    taps=6,
    strategy=PingPongDelayStrategy()
)

delayed = wave | delay
```

**Ping-pong pattern**:
- **Odd taps (1, 3, 5)**: Left → Right
- **Even taps (2, 4, 6)**: Right → Left

```{mermaid}
sequenceDiagram
    participant L as Left Channel
    participant R as Right Channel

    Note over L,R: Original
    rect rgb(200, 220, 255)
        L->>L: Original left
        R->>R: Original right
    end

    Note over L,R: Tap 1 (amp=1.0)
    rect rgb(255, 220, 200)
        L->>R: L→R
    end

    Note over L,R: Tap 2 (amp=feedback)
    rect rgb(200, 255, 220)
        R->>L: R→L
    end

    Note over L,R: Tap 3 (amp=feedback²)
    rect rgb(255, 220, 200)
        L->>R: L→R
    end

    Note over L,R: Result: Bouncing effect
```

**Fallback**: If input is not stereo, automatically uses {class}`~torchfx.effect.MonoDelayStrategy`.

## Musical Applications

### Common Genre Patterns

| Genre | Typical BPM | Common Delays | Use Case |
|-------|-------------|---------------|----------|
| House | 120-130 | `1/8`, `1/16` | Rhythmic vocal delays |
| Techno | 125-135 | `1/16`, `1/32` | Fast percussion echoes |
| Dubstep | 140 | `1/4d`, `1/8t` | Syncopated delays |
| Hip-Hop | 80-110 | `1/4`, `1/8` | Vocal doubling |
| Ambient | 60-90 | `1/2`, `1/4d` | Long atmospheric delays |

### Dubstep Delay Example

Classic dubstep delay using dotted eighth notes:

```python
import torchfx as fx
from torchfx.effect import Delay, PingPongDelayStrategy

wave = fx.Wave.from_file("synth.wav")

# Dotted eighth creates syncopated rhythm
delay = Delay(
    bpm=140,
    delay_time="1/8d",  # Dotted eighth = 0.1875 bars
    feedback=0.5,
    mix=0.4,
    taps=3,
    strategy=PingPongDelayStrategy()
)

delayed = wave | delay
delayed.save("synth_delayed.wav")
```

**Why dotted eighth?** At 140 BPM, a dotted eighth note creates a syncopated rhythm that's iconic in dubstep.

### Vocal Doubling

Subtle delay for thickening vocals:

```python
import torchfx as fx

wave = fx.Wave.from_file("vocal.wav")

# Short delay for doubling effect
doubler = fx.effect.Delay(
    bpm=100,
    delay_time="1/16",  # Very short delay
    feedback=0.0,       # No feedback (single tap)
    mix=0.3,            # Subtle blend
    taps=1              # Just one echo
)

doubled = wave | doubler
```

### Ambient Atmosphere

Long delays with high feedback:

```python
import torchfx as fx

wave = fx.Wave.from_file("pad.wav")

# Long atmospheric delay
ambient_delay = fx.effect.Delay(
    bpm=70,
    delay_time="1/2",   # Half note = long delay
    feedback=0.7,       # High feedback for many echoes
    mix=0.5,            # Prominent effect
    taps=8              # Many echoes
)

atmospheric = wave | ambient_delay
```

## Complex Processing Chains

### Combining with Filters

Create frequency-dependent delays:

```python
import torchfx as fx
from torchfx.filter import iir

wave = fx.Wave.from_file("audio.wav")

# Pre-filter the delay
processed = (
    wave
    | iir.HiButterworth(cutoff=100, order=2)  # Remove rumble
    | fx.effect.Delay(
        bpm=128,
        delay_time="1/4",
        feedback=0.5,
        mix=0.3
    )
    | iir.LoButterworth(cutoff=8000, order=4)  # Soften delays
)
```

### Multi-Effect Chain

Complete production-ready chain:

```python
import torchfx as fx
from torchfx.filter import iir

wave = fx.Wave.from_file("vocal.wav")

# Professional vocal processing
processed = (
    wave
    # EQ
    | iir.HiButterworth(cutoff=80, order=2)
    | iir.PeakingEQ(freq=3000, gain_db=3, q=1.0)

    # Dynamics
    | fx.effect.Compressor(threshold=0.5, ratio=4.0)

    # Delay (quarter note at 120 BPM)
    | fx.effect.Delay(
        bpm=120,
        delay_time="1/4",
        feedback=0.4,
        mix=0.25,
        taps=3
    )

    # Reverb
    | fx.effect.Reverb(delay=4410, decay=0.5, mix=0.2)

    # Final normalization
    | fx.effect.Normalize(peak=0.95)
)

processed.save("vocal_processed.wav")
```

## Advanced Techniques

### Tempo Automation

Apply different delays for different sections:

```python
import torchfx as fx

wave = fx.Wave.from_file("full_track.wav")

# Process intro (slow tempo)
intro_delay = fx.effect.Delay(bpm=90, delay_time="1/4", feedback=0.6, mix=0.4)

# Process verse (faster tempo)
verse_delay = fx.effect.Delay(bpm=120, delay_time="1/8", feedback=0.4, mix=0.3)

# Split audio and apply different delays
# (Manual splitting required)
intro = wave.ys[:, :44100*8]  # First 8 seconds
verse = wave.ys[:, 44100*8:]  # Rest

intro_processed = intro_delay(intro)
verse_processed = verse_delay(verse)
```

### Creative Feedback Patterns

Use extreme feedback for creative effects:

```python
import torchfx as fx

wave = fx.Wave.from_file("drum_loop.wav")

# High feedback creates infinite delays (carefully!)
infinite_delay = fx.effect.Delay(
    bpm=128,
    delay_time="1/16",
    feedback=0.95,  # Very high feedback (max allowed)
    mix=0.3,
    taps=16         # Many taps
)

# Result: Long, dense echo tail
infinite = wave | infinite_delay
```

⚠️ **Warning**: Feedback > 0.95 can cause numerical instability and is clamped.

### Layered Delays

Stack multiple delays for complex patterns:

```python
import torchfx as fx

wave = fx.Wave.from_file("audio.wav")

# Quarter note and dotted eighth
delay1 = fx.effect.Delay(bpm=128, delay_time="1/4", feedback=0.4, mix=0.2)
delay2 = fx.effect.Delay(bpm=128, delay_time="1/8d", feedback=0.3, mix=0.15)

# Apply both
layered = wave | delay1 | delay2
```

## Lazy vs Eager Calculation

The {class}`~torchfx.effect.Delay` effect supports two calculation modes:

### Eager Calculation

When `fs` is provided at initialization:

```python
delay = fx.effect.Delay(
    bpm=120,
    delay_time="1/4",
    fs=44100,  # fs provided: calculate immediately
    feedback=0.4
)

# Delay samples already calculated: 22050
print(delay.delay_samples)  # 22050
```

### Lazy Calculation

When `fs` is `None` (deferred until `forward()`):

```python
delay = fx.effect.Delay(
    bpm=120,
    delay_time="1/4",
    # fs=None: defer calculation
    feedback=0.4
)

print(delay._needs_calculation)  # True

# Wave sets fs automatically
wave = fx.Wave.from_file("audio.wav")
delayed = wave | delay

print(delay._needs_calculation)  # False (calculated during forward)
print(delay.delay_samples)  # 22050 (at fs=44100)
```

**Use lazy calculation** when using the {term}`pipeline operator` with {class}`~torchfx.Wave`.

## Error Handling

### Common Errors

```python
import torchfx as fx

# ❌ ERROR: BPM required without delay_samples
delay = fx.effect.Delay(delay_time="1/4")
# AssertionError: BPM must be provided

# ❌ ERROR: fs required for lazy calculation
delay = fx.effect.Delay(bpm=120, delay_time="1/4")
waveform = torch.randn(2, 44100)
delayed = delay(waveform)
# AssertionError: Sample rate (fs) is required

# ❌ ERROR: Invalid musical time format
delay = fx.effect.Delay(bpm=120, delay_time="invalid", fs=44100)
# ValueError: Invalid musical time string

# ❌ ERROR: Feedback out of range
delay = fx.effect.Delay(delay_samples=1000, feedback=1.2)
# AssertionError: Feedback must be between 0 and 0.95
```

### Validation Checklist

Before applying delay, ensure:
- ✅ `bpm` is provided (if using musical time)
- ✅ `fs` is set (either explicitly or via Wave)
- ✅ `delay_time` matches pattern `n/d[d|t]`
- ✅ `feedback` is in [0, 0.95]
- ✅ `mix` is in [0, 1]
- ✅ `taps` >= 1

## Best Practices

### Choose Appropriate Mix Levels

```python
# ✅ GOOD: Subtle delay (20-30% mix)
subtle = fx.effect.Delay(bpm=120, delay_time="1/8", mix=0.25)

# ✅ GOOD: Prominent delay (40-60% mix)
prominent = fx.effect.Delay(bpm=120, delay_time="1/4", mix=0.5)

# ⚠️  USE CAREFULLY: Heavy delay (70-100% mix)
heavy = fx.effect.Delay(bpm=120, delay_time="1/2", mix=0.8)
```

### Match Delay to Tempo

```python
# ✅ GOOD: Delay matches song tempo
song_bpm = 128
delay = fx.effect.Delay(bpm=song_bpm, delay_time="1/8")

# ❌ BAD: Hardcoded delay time (not tempo-aware)
delay = fx.effect.Delay(delay_samples=10000)  # What tempo is this?
```

### Use Feedback Wisely

```python
# ✅ GOOD: Moderate feedback (0.3-0.6)
musical = fx.effect.Delay(bpm=120, delay_time="1/4", feedback=0.4)

# ⚠️  CAREFUL: High feedback (0.7-0.95)
dense = fx.effect.Delay(bpm=120, delay_time="1/16", feedback=0.9)
# Can create very dense, potentially muddy delays
```

## Related Concepts

- {doc}`/guides/core-concepts/type-system` - Musical time and type system
- {doc}`custom-effects` - Creating custom effects
- {doc}`multi-channel-effects` - Multi-channel processing
- {doc}`/guides/core-concepts/pipeline-operator` - Pipeline operator usage

## External Resources

- [Note Values on Wikipedia](https://en.wikipedia.org/wiki/Note_value) - Understanding musical note durations
- [Delay Effect on Wikipedia](https://en.wikipedia.org/wiki/Delay_(audio_effect)) - Delay effect theory
- [Tempo on Wikipedia](https://en.wikipedia.org/wiki/Tempo) - Understanding BPM and tempo

## References

```{bibliography}
:filter: docname in docnames
:style: alpha
```
