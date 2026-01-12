# Series and Parallel Filter Combinations

Learn how to create complex filter networks by combining filters in series (sequential) and parallel (simultaneous) configurations. This tutorial demonstrates TorchFX's intuitive operators for building sophisticated audio processing chains.

## Overview

TorchFX provides two fundamental ways to combine filters:

| Configuration | Operator | Behavior | Use Case |
|---------------|----------|----------|----------|
| **Series** | `\|` (pipe) | Sequential processing | Filter chains, cascaded stages |
| **Parallel** | `+` (addition) | Simultaneous processing with summed outputs | Band-pass filters, multi-band processing |

Both can be combined to create complex signal processing networks with minimal code.

```{mermaid}
graph TB
    subgraph "Series Configuration"
        Input1[Input] --> F1[Filter 1]
        F1 --> F2[Filter 2]
        F2 --> F3[Filter 3]
        F3 --> Output1[Output]
    end

    subgraph "Parallel Configuration"
        Input2[Input] --> Split{Split}
        Split --> FA[Filter A]
        Split --> FB[Filter B]
        Split --> FC[Filter C]
        FA --> Sum((+))
        FB --> Sum
        FC --> Sum
        Sum --> Output2[Output]
    end

    style Input1 fill:#e1f5ff
    style Output1 fill:#e1f5ff
    style Input2 fill:#e1f5ff
    style Output2 fill:#e1f5ff
    style F1 fill:#fff5e1
    style F2 fill:#fff5e1
    style F3 fill:#fff5e1
    style FA fill:#fff5e1
    style FB fill:#fff5e1
    style FC fill:#fff5e1
```

## Series Filter Chains

Series combinations use the {term}`pipeline operator` (`|`) to process audio sequentially through multiple filters.

### Basic Series Chain

```python
import torchfx as fx

# Load audio
wave = fx.Wave.from_file("audio.wav")

# Apply filters in series
processed = (
    wave
    | fx.filter.iir.HiButterworth(cutoff=100, order=2)    # Remove low frequencies
    | fx.filter.iir.LoButterworth(cutoff=5000, order=4)   # Remove high frequencies
    | fx.effect.Normalize(peak=0.9)                       # Normalize
)

processed.save("filtered.wav")
```

**Signal flow**: Input → High-pass (100 Hz) → Low-pass (5000 Hz) → Normalize → Output

### How Series Works

When you use the pipe operator:

1. The {class}`~torchfx.Wave` object flows through each filter sequentially
2. Each filter's `forward()` method processes the audio tensor
3. A new {class}`~torchfx.Wave` is returned at each stage
4. Sample rate (`fs`) is automatically propagated through the pipeline

```{mermaid}
sequenceDiagram
    participant Wave
    participant Filter1
    participant Filter2
    participant Filter3

    Wave->>Filter1: wave | filter1
    Note over Filter1: fs auto-configured
    Filter1->>Filter1: forward(wave.ys)
    Filter1->>Wave: Return new Wave

    Wave->>Filter2: result | filter2
    Note over Filter2: fs auto-configured
    Filter2->>Filter2: forward(result.ys)
    Filter2->>Wave: Return new Wave

    Wave->>Filter3: result | filter3
    Note over Filter3: fs auto-configured
    Filter3->>Filter3: forward(result.ys)
    Filter3->>Wave: Return final Wave
```

### Multi-Stage Processing

Build complex filter chains with multiple stages:

```python
import torchfx as fx
from torchfx.filter import iir

wave = fx.Wave.from_file("vocal.wav")

# Multi-stage vocal processing chain
processed = (
    wave
    | iir.HiButterworth(cutoff=80, order=2)          # Remove rumble
    | iir.PeakingEQ(freq=3000, gain_db=3, q=1.0)     # Presence boost
    | iir.PeakingEQ(freq=200, gain_db=-2, q=0.7)     # Reduce muddiness
    | fx.effect.Compressor(threshold=0.5, ratio=4.0) # Dynamic control
    | fx.effect.Normalize(peak=0.95)                 # Final normalization
)
```

Each stage processes the output of the previous stage, allowing you to build sophisticated processing chains.

```{seealso}
{doc}`/guides/core-concepts/pipeline-operator` - Deep dive into the pipeline operator
```

## Parallel Filter Combinations

Parallel combinations use the addition operator (`+`) to apply multiple filters simultaneously and sum their outputs.

### Basic Parallel Combination

```python
import torchfx as fx
from torchfx.filter import iir

wave = fx.Wave.from_file("audio.wav")

# Create a band-pass filter using parallel high-pass and low-pass
bandpass = iir.HiButterworth(200, order=2) + iir.LoButterworth(2000, order=4)

# Apply the combined filter
filtered = wave | bandpass
```

**Signal flow**: Input → [High-pass (200 Hz) + Low-pass (2000 Hz)] → Output

The output is the sum: `result = highpass(input) + lowpass(input)`

### The ParallelFilterCombination Class

When you use the `+` operator with filters, TorchFX creates a {class}`~torchfx.filter.ParallelFilterCombination` instance:

```python
import torchfx as fx
from torchfx.filter import iir

# These are equivalent
combination1 = iir.HiButterworth(200) + iir.LoButterworth(2000)
combination2 = fx.filter.ParallelFilterCombination(
    iir.HiButterworth(200),
    iir.LoButterworth(2000)
)
```

The {class}`~torchfx.filter.ParallelFilterCombination` class:
- Stores all filters in a `filters` attribute
- Automatically propagates sample rate (`fs`) to all child filters
- Sums outputs: `result = sum(filter(input) for filter in filters)`

```{mermaid}
classDiagram
    class ParallelFilterCombination {
        +Sequence~AbstractFilter~ filters
        +int|None fs
        +compute_coefficients() None
        +forward(x) Tensor
    }

    class AbstractFilter {
        <<abstract>>
        +__add__(other) ParallelFilterCombination
        +compute_coefficients() None
        +forward(x) Tensor
    }

    class HiButterworth {
        +float cutoff
        +int order
        +forward(x) Tensor
    }

    class LoButterworth {
        +float cutoff
        +int order
        +forward(x) Tensor
    }

    AbstractFilter <|-- ParallelFilterCombination
    AbstractFilter <|-- HiButterworth
    AbstractFilter <|-- LoButterworth
    ParallelFilterCombination o-- AbstractFilter : contains

    note for ParallelFilterCombination "Sums outputs of all child filters"
```

### How Parallel Works

The `forward()` method of {class}`~torchfx.filter.ParallelFilterCombination`:

```python
@torch.no_grad()
def forward(self, x: Tensor) -> Tensor:
    # Apply each filter independently
    outputs = [f.forward(x) for f in self.filters]

    # Create result tensor on same device as input
    results = torch.zeros_like(x)

    # Sum all outputs
    for output in outputs:
        results += output

    return results
```

This implementation:
- Processes all filters in parallel (conceptually)
- Uses `torch.zeros_like(x)` to ensure device compatibility
- Accumulates outputs via element-wise addition
- Disables gradients with `@torch.no_grad()` for efficiency

### Multiple Parallel Filters

Combine more than two filters:

```python
import torchfx as fx
from torchfx.filter import iir

wave = fx.Wave.from_file("audio.wav")

# Three-way parallel combination
multiband = (
    iir.LoButterworth(500, order=4) +      # Low band
    iir.BandPass(500, 2000, order=2) +     # Mid band
    iir.HiButterworth(2000, order=4)       # High band
)

processed = wave | multiband
```

## Combining Series and Parallel

The real power comes from combining both patterns in a single pipeline.

### Complete Example

Here's a complete example demonstrating mixed series/parallel processing:

```python
import torchfx as fx
from torchfx.filter import iir
import torch

# Load audio
wave = fx.Wave.from_file("audio.wav")

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
wave = wave.to(device)

# Complex processing chain
processed = (
    wave
    # Stage 1: Remove low-frequency rumble (series)
    | iir.LoButterworth(cutoff=100, order=2)

    # Stage 2: Parallel high-pass filters (parallel)
    | iir.HiButterworth(2000, order=4) + iir.HiChebyshev1(2000, order=2)

    # Stage 3: Reduce level (series)
    | fx.effect.Gain(gain=0.5, gain_type="amplitude")
)

# Save result (move to CPU for I/O)
processed.to("cpu").save("processed.wav")
```

### Signal Flow Visualization

```{mermaid}
graph TB
    Input[Input Signal<br/>from_file] --> Stage1[Stage 1: LoButterworth<br/>fc=100 Hz, order=2]

    Stage1 --> Split{Split}

    subgraph Stage2 [Stage 2: Parallel Combination]
        Split --> HP1[HiButterworth<br/>fc=2000 Hz, order=4]
        Split --> HP2[HiChebyshev1<br/>fc=2000 Hz, order=2]
        HP1 --> Sum((+))
        HP2 --> Sum
    end

    Sum --> Stage3[Stage 3: Gain<br/>amplitude=0.5]

    Stage3 --> Output[Output Signal<br/>save to file]

    style Input fill:#e1f5ff
    style Output fill:#e1f5ff
    style Stage1 fill:#fff5e1
    style HP1 fill:#fff5e1
    style HP2 fill:#fff5e1
    style Stage3 fill:#fff5e1
    style Stage2 fill:#f9f9f9
```

### Operator Precedence

Python's operator precedence affects how series and parallel combine:

```python
# Addition (+) has HIGHER precedence than pipe (|)
result = wave | filter1 + filter2 | filter3

# This is parsed as:
result = wave | (filter1 + filter2) | filter3

# For clarity, use explicit parentheses:
result = wave | (filter1 + filter2) | filter3

# Or break into variables:
parallel_section = filter1 + filter2
result = wave | parallel_section | filter3
```

**Best practice**: Use parentheses or intermediate variables for complex combinations.

## Sample Rate Management

Both series and parallel combinations handle sample rate automatically.

### Automatic Propagation

```python
import torchfx as fx
from torchfx.filter import iir

# Load audio (fs automatically extracted from file)
wave = fx.Wave.from_file("audio.wav")  # fs = 44100

# Filters auto-configured with fs=44100
processed = wave | (
    iir.LoButterworth(100) + iir.HiButterworth(5000)
)
```

When you use the pipeline operator:
1. {class}`~torchfx.Wave` sets each filter's `fs` attribute
2. For {class}`~torchfx.filter.ParallelFilterCombination`, `fs` propagates to all child filters
3. Filters compute coefficients using the correct sample rate

### Manual Configuration

You can also set sample rate explicitly:

```python
from torchfx.filter import iir, ParallelFilterCombination

# Create parallel combination with explicit fs
parallel = ParallelFilterCombination(
    iir.HiButterworth(2000, order=2),
    iir.HiChebyshev1(2000, order=2),
    fs=44100
)

# All child filters now have fs=44100
```

The `fs` property automatically propagates to child filters that have `fs=None`.

## Performance Considerations

### GPU Acceleration

Both series and parallel work seamlessly on GPU:

```python
import torchfx as fx
import torch

wave = fx.Wave.from_file("audio.wav")

# Move to GPU
if torch.cuda.is_available():
    wave = wave.to("cuda")

    # All processing happens on GPU
    processed = wave | (filter1 + filter2) | filter3

    # Move back to CPU for saving
    processed.to("cpu").save("output.wav")
```

The device is automatically propagated because:
- Filters inherit from {class}`torch.nn.Module`
- Filter coefficients are stored as {class}`torch.Tensor`
- The {class}`~torchfx.Wave` tensor carries device information

```{seealso}
{doc}`/guides/advanced/gpu-acceleration` - GPU acceleration guide
```

### Memory Efficiency

For parallel combinations with many filters:

```python
# Current implementation collects all outputs
outputs = [f.forward(x) for f in self.filters]  # List of N tensors

# Then sums them
for output in outputs:
    results += output
```

This holds all filter outputs in memory simultaneously. For very large signals or many filters, consider processing in chunks.

## Mixing with PyTorch Modules

Since filters inherit from {class}`torch.nn.Module`, they work with standard PyTorch containers:

```python
import torch.nn as nn
import torchfx as fx
from torchfx.filter import iir

# Use nn.Sequential
effect_chain = nn.Sequential(
    iir.LoButterworth(100, order=2),
    iir.HiButterworth(5000, order=4),
    fx.effect.Normalize()
)

wave = fx.Wave.from_file("audio.wav")
processed = wave | effect_chain
```

You can also mix TorchFX filters with torchaudio transforms:

```python
import torchaudio.transforms as T
from torchfx.filter import iir

processed = (
    wave
    | iir.LoButterworth(100, order=2)
    | iir.HiButterworth(2000, order=2) + iir.HiChebyshev1(2000, order=2)
    | T.Vol(0.5)  # torchaudio volume transform
    | fx.effect.Reverb()  # torchfx reverb effect
)
```

## Real-World Examples

### Vocal Processing Chain

```python
import torchfx as fx
from torchfx.filter import iir

wave = fx.Wave.from_file("vocal.wav")

# Professional vocal processing
processed = (
    wave
    # Remove rumble and noise
    | iir.HiButterworth(cutoff=80, order=2)
    | iir.LoButterworth(cutoff=12000, order=4)

    # Parallel EQ boosts for presence and air
    | (
        iir.PeakingEQ(freq=3000, gain_db=3, q=1.0) +   # Presence
        iir.PeakingEQ(freq=10000, gain_db=2, q=0.7)    # Air
    )

    # Dynamics and final polish
    | fx.effect.Compressor(threshold=0.5, ratio=4.0)
    | fx.effect.Normalize(peak=0.95)
)
```

### Multi-Band Processing

```python
import torchfx as fx
from torchfx.filter import iir

wave = fx.Wave.from_file("master.wav")

# Three-band processing
low = wave | iir.LoButterworth(200, order=4) | fx.effect.Gain(1.2)
mid = wave | iir.BandPass(200, 2000, order=2) | fx.effect.Compressor(0.6, 3.0)
high = wave | iir.HiButterworth(2000, order=4) | fx.effect.Gain(1.1)

# Recombine bands
multiband = fx.Wave.merge([low, mid, high], split_channels=False)
```

### Creative Parallel Effects

```python
import torchfx as fx
from torchfx.filter import iir

wave = fx.Wave.from_file("guitar.wav")

# Parallel processing for thickness
processed = (
    wave
    # Three parallel delay lines
    | (
        fx.effect.Delay(delay_samples=1000, feedback=0.3) +
        fx.effect.Delay(delay_samples=1500, feedback=0.25) +
        fx.effect.Delay(delay_samples=2000, feedback=0.2)
    )
    # Final processing
    | fx.effect.Normalize(peak=0.9)
)
```

## Best Practices

### Use Multi-Line for Readability

```python
# ✅ GOOD: Clear, readable
processed = (
    wave
    | filter1
    | filter2 + filter3
    | filter4
)

# ❌ BAD: Hard to read
processed = wave | filter1 | filter2 + filter3 | filter4
```

### Name Complex Sections

```python
# ✅ GOOD: Named sections
parallel_hp = iir.HiButterworth(2000, order=4) + iir.HiChebyshev1(2000, order=2)
processed = wave | iir.LoButterworth(100) | parallel_hp | fx.effect.Gain(0.5)

# ❌ LESS GOOD: Inline everything
processed = wave | iir.LoButterworth(100) | iir.HiButterworth(2000, order=4) + iir.HiChebyshev1(2000, order=2) | fx.effect.Gain(0.5)
```

### Use Parentheses for Clarity

```python
# ✅ GOOD: Explicit grouping
result = wave | (lowpass + highpass) | gain

# ⚠️  WORKS: Relies on operator precedence
result = wave | lowpass + highpass | gain
```

### Reuse Filter Combinations

```python
# ✅ GOOD: Reusable filter
mastering_chain = (
    iir.HiButterworth(30, order=2) +
    iir.LoButterworth(18000, order=4)
)

# Apply to multiple files
for file in audio_files:
    wave = fx.Wave.from_file(file)
    processed = wave | mastering_chain | fx.effect.Normalize()
    processed.save(f"mastered_{file}")
```

## Common Pitfalls

### Forgetting Device Management

```python
# ❌ WRONG: Mixing CPU and GPU
wave_gpu = fx.Wave.from_file("audio.wav").to("cuda")
processed = wave_gpu | filter_chain
processed.save("output.wav")  # Error: can't save CUDA tensor

# ✅ CORRECT: Move to CPU before saving
processed.to("cpu").save("output.wav")
```

### Misunderstanding Parallel Summation

```python
# Parallel combination SUMS outputs, not average
parallel = filter1 + filter2

# Output amplitude may be higher than input
# Consider adding gain reduction if needed
parallel_with_gain = (filter1 + filter2) | fx.effect.Gain(0.5)
```

### Sample Rate Mismatch

```python
# ❌ WRONG: Manually created filters without fs
filter1 = iir.LoButterworth(100)
filter1.fs = 44100
filter2 = iir.HiButterworth(5000)
# filter2.fs is None!

parallel = ParallelFilterCombination(filter1, filter2)

# ✅ CORRECT: Let Wave set fs automatically
parallel = iir.LoButterworth(100) + iir.HiButterworth(5000)
wave = fx.Wave.from_file("audio.wav")  # fs auto-configured
processed = wave | parallel
```

## Related Concepts

- {doc}`/guides/core-concepts/pipeline-operator` - Understanding the pipe operator
- {doc}`/guides/core-concepts/fx` - FX base class architecture
- {doc}`/guides/tutorials/custom-filters` - Creating custom filters
- {doc}`/guides/advanced/pytorch-integration` - PyTorch nn.Module integration

## External Resources

- [Digital Filter on Wikipedia](https://en.wikipedia.org/wiki/Digital_filter) - Filter theory and design
- [Series and Parallel Circuits](https://en.wikipedia.org/wiki/Series_and_parallel_circuits) - Electrical circuit analogy
- [PyTorch nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) - Module base class

## References

```{bibliography}
:filter: docname in docnames
:style: alpha
```
