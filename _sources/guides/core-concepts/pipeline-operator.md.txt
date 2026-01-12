# Pipeline Operator - Functional Composition

The pipeline operator (`|`) is one of TorchFX's most distinctive features. It provides an intuitive, readable way to chain audio effects together, creating complex processing pipelines with simple, linear code.

## What is the Pipeline Operator?

The pipeline operator is Python's bitwise OR operator (`|`) repurposed for functional composition. In TorchFX, it means "apply this effect to the audio":

```python
import torchfx as fx

wave = fx.Wave.from_file("audio.wav")

# Read this as: "wave, apply lowpass filter"
filtered = wave | fx.filter.iir.LoButterworth(1000)
```

This approach is inspired by Unix pipes and functional programming languages, making audio processing code read like a natural description of the signal flow.

```{mermaid}
graph LR
    A[Input Wave] -->|"|"| B[Effect 1]
    B -->|"|"| C[Effect 2]
    C -->|"|"| D[Effect 3]
    D --> E[Output Wave]

    style A fill:#e1f5ff
    style E fill:#e1f5ff
    style B fill:#fff5e1
    style C fill:#fff5e1
    style D fill:#fff5e1
```

## Basic Usage

### Single Effect

Apply one effect to a wave:

```python
import torchfx as fx
from torchfx.filter import iir

wave = fx.Wave.from_file("audio.wav")

# Apply a single effect
filtered = wave | iir.LoButterworth(cutoff=1000, order=4)
```

### Chaining Effects

Chain multiple effects sequentially:

```python
import torchfx as fx
from torchfx.filter import iir

wave = fx.Wave.from_file("audio.wav")

# Chain multiple effects: read left to right
processed = wave | iir.HiButterworth(100) | iir.LoButterworth(5000) | fx.effect.Normalize()
```

This is equivalent to:

```python
step1 = wave | iir.HiButterworth(100)
step2 = step1 | iir.LoButterworth(5000)
processed = step2 | fx.effect.Normalize()
```

But much more concise and readable.

### Multi-line Pipelines

For complex pipelines, use parentheses for multi-line formatting:

```python
import torchfx as fx
from torchfx.filter import iir

wave = fx.Wave.from_file("audio.wav")

processed = (
    wave
    | iir.HiButterworth(cutoff=80, order=2)      # Remove subsonic rumble
    | iir.LoButterworth(cutoff=12000, order=4)   # Remove high-frequency noise
    | fx.effect.Normalize(peak=0.9)               # Normalize to -0.9 dB
    | fx.effect.Reverb(delay=4410, decay=0.3)    # Add subtle reverb
)

processed.save("output.wav")
```

## How It Works

### The `__or__` Method

The pipeline operator is implemented via Python's `__or__` magic method in the {class}`~torchfx.Wave` class:

```python
# Simplified implementation from torchfx/wave.py
class Wave:
    def __or__(self, f: nn.Module) -> "Wave":
        """Apply a module to the wave through the pipeline operator."""
        # Validate input
        if not isinstance(f, nn.Module):
            raise TypeError(f"Expected nn.Module, but got {type(f).__name__}")

        # Auto-configure FX effects
        if isinstance(f, FX):
            self.__update_config(f)

        # Apply effect and return new Wave
        return self.transform(f.forward)
```

### Automatic Configuration

When you use the pipeline operator, TorchFX automatically:

1. **Validates the effect**: Ensures it's a {class}`torch.nn.Module`
2. **Sets the sample rate**: Configures the effect's `fs` attribute from the wave
3. **Computes coefficients**: For filters, calls `compute_coefficients()` if needed
4. **Applies the effect**: Calls the effect's `forward()` method on the audio tensor
5. **Returns a new Wave**: Preserves immutability

```{mermaid}
sequenceDiagram
    participant User
    participant Wave
    participant Effect
    participant Tensor

    User->>Wave: wave | effect
    Wave->>Wave: __or__(effect)
    Wave->>Effect: Check if FX instance
    Wave->>Effect: Set effect.fs = wave.fs
    Wave->>Effect: Call compute_coefficients() (if filter)
    Wave->>Effect: Call effect.forward(wave.ys)
    Effect->>Tensor: Process audio tensor
    Tensor-->>Effect: Processed tensor
    Effect-->>Wave: Return processed tensor
    Wave->>Wave: Create new Wave(processed, fs)
    Wave-->>User: Return new Wave
```

### Immutability

The pipeline operator returns a **new** {class}`~torchfx.Wave` object, leaving the original unchanged:

```python
import torchfx as fx

original = fx.Wave.from_file("audio.wav")
filtered = original | fx.filter.iir.LoButterworth(1000)

# Original is unchanged
assert len(original) == len(original.ys)

# Filtered is a new object
assert filtered is not original
```

This functional programming pattern:
- Prevents accidental mutations
- Makes debugging easier
- Enables safe parallel processing
- Supports method chaining

## Advanced Patterns

### Combining with PyTorch Modules

The pipeline operator works with any {class}`torch.nn.Module`, not just {class}`~torchfx.FX` effects:

```python
import torch
import torchfx as fx

wave = fx.Wave.from_file("audio.wav")

# Use PyTorch Sequential
effect_chain = torch.nn.Sequential(
    fx.filter.iir.HiButterworth(100),
    fx.filter.iir.LoButterworth(5000),
    fx.effect.Normalize()
)

processed = wave | effect_chain
```

### Parallel Filter Combination

Use the `+` operator to combine filters in parallel:

```python
import torchfx as fx
from torchfx.filter import iir

wave = fx.Wave.from_file("audio.wav")

# Create bandpass filter: parallel high-pass + low-pass
bandpass = iir.HiButterworth(200) + iir.LoButterworth(2000)

# Apply combined filter
filtered = wave | bandpass
```

The `+` operator creates a {class}`~torchfx.filter.ParallelFilterCombination` that:
- Applies both filters to the input
- Sums the outputs
- Returns the combined result

```{mermaid}
graph TB
    Input[Input Signal] --> Split{Split}
    Split --> HP[High-Pass<br/>200 Hz]
    Split --> LP[Low-Pass<br/>2000 Hz]
    HP --> Sum((+))
    LP --> Sum
    Sum --> Output[Output Signal]

    style Input fill:#e1f5ff
    style Output fill:#e1f5ff
    style HP fill:#fff5e1
    style LP fill:#fff5e1
```

### Reusable Effect Chains

Define effect chains once and reuse them:

```python
import torchfx as fx
from torchfx.filter import iir
import torch.nn as nn

# Define a reusable mastering chain
mastering_chain = nn.Sequential(
    iir.HiButterworth(cutoff=30, order=2),     # Remove DC offset and rumble
    iir.LoButterworth(cutoff=18000, order=4),  # Remove ultrasonic content
    fx.effect.Normalize(peak=0.95),            # Normalize
)

# Apply to multiple files
for audio_file in ["track1.wav", "track2.wav", "track3.wav"]:
    wave = fx.Wave.from_file(audio_file)
    mastered = wave | mastering_chain
    mastered.save(f"mastered_{audio_file}")
```

### Conditional Processing

Apply effects conditionally:

```python
import torchfx as fx

def process_audio(wave: fx.Wave, apply_reverb: bool = False) -> fx.Wave:
    # Always apply EQ
    processed = wave | fx.filter.iir.LoButterworth(5000)

    # Conditionally apply reverb
    if apply_reverb:
        processed = processed | fx.effect.Reverb(delay=4410, decay=0.4)

    # Always normalize
    processed = processed | fx.effect.Normalize()

    return processed

wave = fx.Wave.from_file("audio.wav")
dry = process_audio(wave, apply_reverb=False)
wet = process_audio(wave, apply_reverb=True)
```

### Per-Channel Processing

Process individual channels separately:

```python
import torchfx as fx

wave = fx.Wave.from_file("stereo.wav")  # 2 channels

# Extract and process left channel
left = wave.get_channel(0) | fx.effect.Delay(delay_samples=4410)

# Extract and process right channel differently
right = wave.get_channel(1) | fx.effect.Delay(delay_samples=2205)

# Merge back to stereo
stereo = fx.Wave.merge([left, right], split_channels=True)
```

## Comparison with Other Approaches

### Pipeline Operator (TorchFX)

```python
import torchfx as fx

wave = fx.Wave.from_file("audio.wav")
processed = wave | Effect1() | Effect2() | Effect3()
```

**Advantages**:
- Reads left to right (natural flow)
- Concise and expressive
- Auto-configures effects
- Type-safe (IDE support)

### Sequential Container (PyTorch)

```python
import torch.nn as nn

effects = nn.Sequential(Effect1(), Effect2(), Effect3())
processed_tensor = effects(audio_tensor)
```

**Advantages**:
- Standard PyTorch pattern
- Works with neural networks
- Explicit container

**Disadvantages**:
- Need to manage sample rate manually
- Returns tensor, not Wave
- More verbose

### Manual Chaining

```python
x = audio_tensor
x = effect1(x)
x = effect2(x)
x = effect3(x)
processed_tensor = x
```

**Advantages**:
- Explicit and clear
- Easy to debug intermediate steps

**Disadvantages**:
- Verbose
- Need to manage variables
- Easy to make mistakes

## Best Practices

### Use Multi-line for Readability

For complex pipelines, format across multiple lines:

```python
# ✅ GOOD: Multi-line with comments
processed = (
    wave
    | iir.HiButterworth(80, order=2)     # Remove rumble
    | iir.LoButterworth(12000, order=4)  # Remove noise
    | fx.effect.Normalize(peak=0.9)      # Normalize
)

# ❌ BAD: Everything on one line
processed = wave | iir.HiButterworth(80, order=2) | iir.LoButterworth(12000, order=4) | fx.effect.Normalize(peak=0.9)
```

### Extract Complex Chains

For reusable or complex chains, use {class}`torch.nn.Sequential`:

```python
import torch.nn as nn
import torchfx as fx

# ✅ GOOD: Named, reusable chain
vocal_chain = nn.Sequential(
    iir.HiButterworth(cutoff=100, order=2),
    iir.PeakingEQ(freq=3000, gain_db=3, q=1.0),  # Presence boost
    fx.effect.Compressor(threshold=0.5, ratio=4.0),
    fx.effect.Normalize(peak=0.95),
)

processed = wave | vocal_chain

# ❌ LESS GOOD: Inline complex chain
processed = wave | iir.HiButterworth(100, 2) | iir.PeakingEQ(3000, 3, 1.0) | fx.effect.Compressor(0.5, 4.0) | fx.effect.Normalize(0.95)
```

### Avoid Deep Nesting

Keep pipelines flat for readability:

```python
# ✅ GOOD: Flat pipeline
processed = wave | Effect1() | Effect2() | Effect3()

# ❌ BAD: Nested pipelines (confusing)
processed = wave | (Effect1() | (Effect2() | Effect3()))
```

### Name Intermediate Results

For debugging or visualization, name intermediate steps:

```python
import torchfx as fx

wave = fx.Wave.from_file("audio.wav")

# Name each step for debugging
highpassed = wave | fx.filter.iir.HiButterworth(100)
lowpassed = highpassed | fx.filter.iir.LoButterworth(5000)
normalized = lowpassed | fx.effect.Normalize()

# Can save intermediate results
highpassed.save("debug_highpass.wav")
lowpassed.save("debug_lowpass.wav")
normalized.save("output.wav")
```

## Common Pitfalls

### Modifying Effects After Application

Effects are configured when applied to the wave. Modifying them afterward has no effect:

```python
import torchfx as fx

wave = fx.Wave.from_file("audio.wav")
effect = fx.effect.Gain(gain=1.0)

processed = wave | effect

# ❌ This doesn't change the processed wave
effect.gain = 2.0

# ✅ Create a new effect and apply it
processed = wave | fx.effect.Gain(gain=2.0)
```

### Forgetting to Assign Result

The pipeline operator returns a **new** Wave; forgetting to assign loses the result:

```python
import torchfx as fx

wave = fx.Wave.from_file("audio.wav")

# ❌ WRONG: Result is lost
wave | fx.effect.Normalize()

# ✅ CORRECT: Assign result
normalized = wave | fx.effect.Normalize()

# ✅ ALSO CORRECT: Reassign to same variable (if you don't need original)
wave = wave | fx.effect.Normalize()
```

### Mixing Sample Rates

All effects in a chain should work with the same sample rate:

```python
import torchfx as fx

wave_44k = fx.Wave.from_file("audio_44100.wav")  # 44100 Hz
wave_48k = fx.Wave.from_file("audio_48000.wav")  # 48000 Hz

# ❌ WRONG: Effects configured for different rates
effect_44k = wave_44k | fx.effect.Delay(bpm=120, delay_time="1/4")
processed_48k = wave_48k | effect_44k  # Effect still has fs=44100!

# ✅ CORRECT: Each wave configures its own effects
processed_44k = wave_44k | fx.effect.Delay(bpm=120, delay_time="1/4")
processed_48k = wave_48k | fx.effect.Delay(bpm=120, delay_time="1/4")
```

## Performance Considerations

### In-Place vs. Immutable

The pipeline operator creates new Wave objects. For memory-constrained environments, consider:

```python
import torchfx as fx

wave = fx.Wave.from_file("audio.wav")

# Memory-friendly: reuse variable name
wave = wave | Effect1()
wave = wave | Effect2()
wave = wave | Effect3()
# Old Wave objects are garbage collected

# Memory-intensive: keep all intermediates
step1 = wave | Effect1()
step2 = step1 | Effect2()
step3 = step2 | Effect3()
# All Wave objects remain in memory
```

### GPU Acceleration

Move waves to GPU before applying effects:

```python
import torchfx as fx
import torch

if torch.cuda.is_available():
    wave = fx.Wave.from_file("audio.wav").to("cuda")

    # Effects run on GPU
    processed = wave | HeavyEffect1() | HeavyEffect2()

    # Move back to CPU for saving
    processed.to("cpu").save("output.wav")
else:
    wave = fx.Wave.from_file("audio.wav")
    processed = wave | HeavyEffect1() | HeavyEffect2()
    processed.save("output.wav")
```

### Batch Processing

Reuse effect instances for batch processing:

```python
import torchfx as fx
from pathlib import Path

# Create effects once
effect_chain = fx.effect.Normalize() | fx.effect.Reverb()

# Reuse for multiple files
for audio_file in Path("audio").glob("*.wav"):
    wave = fx.Wave.from_file(audio_file)
    processed = wave | effect_chain
    processed.save(f"processed/{audio_file.name}")
```

## Mathematical Interpretation

The pipeline operator can be understood as function composition. Given effects $f$, $g$, and $h$:

```python
result = wave | f | g | h
```

Is equivalent to:

$$
\text{result} = h(g(f(\text{wave})))
$$

Or in mathematical notation with the composition operator $\circ$:

$$
\text{result} = (h \circ g \circ f)(\text{wave})
$$

But TorchFX's syntax reads left-to-right (natural order) instead of right-to-left (mathematical order):

$$
\text{wave} \xrightarrow{f} \text{temp}_1 \xrightarrow{g} \text{temp}_2 \xrightarrow{h} \text{result}
$$

## Related Concepts

- {doc}`wave` - The audio container that supports the pipeline operator
- {doc}`fx` - Effects that can be used with the pipeline operator
- {doc}`/guides/tutorials/ml-batch-processing` - Using pipelines in ML workflows
- {doc}`/guides/tutorials/effects-design` - Designing effect chains

## External Resources

- [Unix Pipes on Wikipedia](https://en.wikipedia.org/wiki/Pipeline_(Unix)) - Inspiration for the pipeline pattern
- [Fluent Interface on Wikipedia](https://en.wikipedia.org/wiki/Fluent_interface) - Design pattern for method chaining
- [Python Magic Methods](https://docs.python.org/3/reference/datamodel.html#special-method-names) - Understanding `__or__`

## References

```{bibliography}
:filter: docname in docnames
:style: alpha
```
