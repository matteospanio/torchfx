# FX - Audio Effect Base Class

The {class}`~torchfx.FX` class is the abstract base class for all audio effects and filters in TorchFX. It defines the interface that all effects must implement and provides the foundation for building custom audio processors.

## What is FX?

{class}`~torchfx.FX` inherits from {class}`torch.nn.Module`, making all TorchFX effects compatible with PyTorch's neural network ecosystem. This design choice enables:

- **Gradient Computation**: Effects can be used in differentiable audio processing pipelines
- **Parameter Management**: Automatic device handling and parameter tracking
- **Modularity**: Effects can be combined with neural networks
- **Composability**: Easy integration with {class}`torch.nn.Sequential` and other PyTorch containers

```{mermaid}
classDiagram
    class Module {
        <<PyTorch>>
        +forward(x)*
        +to(device)
        +parameters()
        +train() / eval()
    }

    class FX {
        <<abstract>>
        +forward(x) Tensor*
        +__init__()*
    }

    class Filter {
        <<abstract>>
        +int fs
        +compute_coefficients()*
    }

    class Effect {
        +forward(x) Tensor
    }

    class IIRFilter {
        +Tensor b
        +Tensor a
        +compute_coefficients()
        +forward(x) Tensor
    }

    class Delay {
        +int delay_samples
        +float feedback
        +forward(x) Tensor
    }

    class Reverb {
        +int delay
        +float decay
        +forward(x) Tensor
    }

    Module <|-- FX
    FX <|-- Filter
    FX <|-- Effect
    Filter <|-- IIRFilter
    Effect <|-- Delay
    Effect <|-- Reverb

    note for Module "PyTorch base class\nprovides device management,\nparameter tracking"
    note for FX "TorchFX base class\ndefines audio effect interface"
```

## The FX Interface

Every effect must implement two key methods:

### 1. Constructor (`__init__`)

Initializes the effect with its parameters:

```python
from torchfx.effect import FX
import torch

class MyEffect(FX):
    def __init__(self, param1: float, param2: int):
        super().__init__()  # Always call parent constructor
        self.param1 = param1
        self.param2 = param2
        # Initialize any learnable parameters
        self.gain = torch.nn.Parameter(torch.tensor([1.0]))
```

**Key points**:
- Always call `super().__init__()` first
- Store effect parameters as attributes
- Use {class}`torch.nn.Parameter` for learnable parameters (if needed)

### 2. Forward Method (`forward`)

Processes the audio signal:

```python
from torch import Tensor
import torch

class MyEffect(FX):
    def __init__(self, gain: float):
        super().__init__()
        self.gain = gain

    @torch.no_grad()  # Disable gradients for efficiency (optional)
    def forward(self, x: Tensor) -> Tensor:
        """Apply effect to audio tensor.

        Parameters
        ----------
        x : Tensor
            Input audio of shape (..., time) or (channels, time)

        Returns
        -------
        Tensor
            Processed audio with the same shape as input
        """
        return x * self.gain
```

**Key points**:
- Input shape: `(channels, samples)` or `(..., samples)`
- Output shape: Should match input shape (unless explicitly extending signal)
- Use `@torch.no_grad()` for efficiency if gradients aren't needed
- Handle multi-channel audio appropriately

## Built-in Effects

TorchFX provides several built-in effects demonstrating different patterns:

### Simple Effects

Effects with straightforward signal processing:

```python
import torchfx as fx

# Gain adjustment
wave = fx.Wave.from_file("audio.wav")
louder = wave | fx.effect.Gain(gain=2.0, gain_type="amplitude")

# Normalization
normalized = wave | fx.effect.Normalize(peak=0.9)
```

### Time-Based Effects

Effects that use delay and feedback:

```python
import torchfx as fx

wave = fx.Wave.from_file("audio.wav")

# Simple reverb
reverb = fx.effect.Reverb(delay=4410, decay=0.5, mix=0.3)
wet = wave | reverb

# BPM-synced delay
delay = fx.effect.Delay(bpm=120, delay_time="1/8", feedback=0.4, mix=0.3)
delayed = wave | delay
```

### Strategy Pattern Effects

Effects using the {term}`Strategy Pattern` for flexible behavior:

```python
import torchfx as fx
from torchfx.effect import (
    Normalize,
    RMSNormalizationStrategy,
    PercentileNormalizationStrategy
)

wave = fx.Wave.from_file("audio.wav")

# Peak normalization (default)
norm1 = wave | Normalize(peak=1.0)

# RMS normalization
norm2 = wave | Normalize(peak=0.5, strategy=RMSNormalizationStrategy())

# Percentile normalization
norm3 = wave | Normalize(peak=0.9, strategy=PercentileNormalizationStrategy(percentile=99))
```

```{seealso}
{cite:t}`gamma1994design` - Design Patterns book covering the Strategy Pattern
```

## Filters vs Effects

TorchFX distinguishes between filters and effects:

### Filters

Inherit from {class}`~torchfx.filter.AbstractFilter` (which inherits from {class}`~torchfx.FX`):

- **Frequency-domain processing**: IIR, FIR filters
- **Require sample rate**: `fs` attribute is mandatory
- **Compute coefficients**: Must implement `compute_coefficients()` method
- **Parallel combination**: Support `+` operator for parallel filter banks

```python
from torchfx.filter import iir
import torchfx as fx

wave = fx.Wave.from_file("audio.wav")

# IIR filters with automatic fs configuration
lowpass = wave | iir.LoButterworth(cutoff=1000, order=4)
highpass = wave | iir.HiButterworth(cutoff=200, order=2)

# Parallel combination (bandpass filter)
bandpass = wave | (iir.HiButterworth(200) + iir.LoButterworth(1000))
```

### Effects

Inherit directly from {class}`~torchfx.FX`:

- **Time-domain processing**: Delay, reverb, dynamics, etc.
- **Optional sample rate**: May or may not need `fs`
- **Direct implementation**: No coefficient computation required
- **Flexible parameters**: Can use any processing strategy

```python
import torchfx as fx

wave = fx.Wave.from_file("audio.wav")

# Effects don't require fs (unless BPM-synced)
gained = wave | fx.effect.Gain(2.0)
normalized = wave | fx.effect.Normalize(peak=0.8)

# BPM-synced effects auto-configure fs from Wave
delayed = wave | fx.effect.Delay(bpm=120, delay_time="1/4")
```

## Creating Custom Effects

### Basic Custom Effect

```python
from torchfx.effect import FX
from torch import Tensor
import torch

class SimpleDistortion(FX):
    """Apply soft clipping distortion."""

    def __init__(self, drive: float = 2.0, mix: float = 0.5):
        """
        Parameters
        ----------
        drive : float
            Amount of distortion (>1.0). Higher values = more distortion.
        mix : float
            Wet/dry mix (0 = dry, 1 = wet).
        """
        super().__init__()
        assert drive >= 1.0, "Drive must be >= 1.0"
        assert 0 <= mix <= 1, "Mix must be in [0, 1]"

        self.drive = drive
        self.mix = mix

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """Apply soft clipping distortion.

        Uses tanh for smooth saturation:
        y[n] = tanh(drive * x[n])
        """
        # Apply distortion
        distorted = torch.tanh(self.drive * x)

        # Wet/dry mix
        output = (1 - self.mix) * x + self.mix * distorted

        return output
```

Usage:

```python
import torchfx as fx

wave = fx.Wave.from_file("guitar.wav")
distorted = wave | SimpleDistortion(drive=3.0, mix=0.7)
distorted.save("guitar_distorted.wav")
```

### Multi-Channel Effect

Handle stereo and multi-channel audio correctly:

```python
from torchfx.effect import FX
from torch import Tensor
import torch

class StereoWidener(FX):
    """Widen stereo image using Mid/Side processing."""

    def __init__(self, width: float = 1.5):
        """
        Parameters
        ----------
        width : float
            Stereo width multiplier (1.0 = no change, >1.0 = wider, <1.0 = narrower).
        """
        super().__init__()
        assert width >= 0, "Width must be non-negative"
        self.width = width

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """Apply stereo widening.

        Converts to Mid/Side, scales Side, converts back to L/R.
        """
        # Only works on stereo audio
        if x.shape[0] != 2:
            return x  # Return unchanged for non-stereo

        left = x[0]
        right = x[1]

        # Convert to Mid/Side
        mid = (left + right) / 2
        side = (left - right) / 2

        # Widen by scaling Side component
        side = side * self.width

        # Convert back to L/R
        new_left = mid + side
        new_right = mid - side

        return torch.stack([new_left, new_right])
```

### Effect with Strategy Pattern

Use the {term}`Strategy Pattern` for flexible behavior:

```python
from torchfx.effect import FX
from torch import Tensor
import torch
import abc

class CompressionStrategy(abc.ABC):
    """Abstract base for compression algorithms."""

    @abc.abstractmethod
    def compress(self, x: Tensor, threshold: float, ratio: float) -> Tensor:
        pass

class HardKneeCompression(CompressionStrategy):
    """Hard-knee compression with sharp threshold."""

    def compress(self, x: Tensor, threshold: float, ratio: float) -> Tensor:
        abs_x = torch.abs(x)
        mask = abs_x > threshold

        # Compress values above threshold
        compressed = torch.where(
            mask,
            threshold + (abs_x - threshold) / ratio,
            abs_x
        )

        # Restore sign
        return torch.sign(x) * compressed

class SoftKneeCompression(CompressionStrategy):
    """Soft-knee compression with gradual transition."""

    def compress(self, x: Tensor, threshold: float, ratio: float) -> Tensor:
        # Implementation of soft-knee compression
        # (simplified for brevity)
        return x  # Placeholder

class Compressor(FX):
    """Dynamic range compressor with configurable strategy."""

    def __init__(
        self,
        threshold: float = 0.5,
        ratio: float = 4.0,
        strategy: CompressionStrategy | None = None
    ):
        super().__init__()
        self.threshold = threshold
        self.ratio = ratio
        self.strategy = strategy or HardKneeCompression()

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        return self.strategy.compress(x, self.threshold, self.ratio)
```

Usage:

```python
import torchfx as fx

wave = fx.Wave.from_file("vocals.wav")

# Hard knee compression (default)
compressed1 = wave | Compressor(threshold=0.5, ratio=4.0)

# Soft knee compression
compressed2 = wave | Compressor(
    threshold=0.5,
    ratio=4.0,
    strategy=SoftKneeCompression()
)
```

## Sample Rate Handling

Many effects need the sample rate to function correctly. TorchFX provides automatic configuration:

### Automatic Configuration

When using the pipeline operator with {class}`~torchfx.Wave`, the sample rate is automatically set:

```python
import torchfx as fx

wave = fx.Wave.from_file("audio.wav")  # fs = 44100

# Effect's fs is automatically set to 44100
delayed = wave | fx.effect.Delay(bpm=120, delay_time="1/8")
```

### Manual Configuration

For standalone use or custom effects:

```python
from torchfx.effect import FX
from torch import Tensor
import torch

class MyTimedEffect(FX):
    """Effect that needs sample rate."""

    def __init__(self, duration_ms: float, fs: int | None = None):
        super().__init__()
        self.fs = fs
        self.duration_ms = duration_ms
        self._duration_samples = None

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        # Lazy calculation when fs becomes available
        if self._duration_samples is None:
            assert self.fs is not None, "Sample rate (fs) must be set"
            self._duration_samples = int(self.duration_ms * self.fs / 1000)

        # Use self._duration_samples for processing
        return x  # Placeholder
```

## Gradient Support

While most audio effects run with `@torch.no_grad()` for efficiency, you can enable gradients for differentiable audio processing:

```python
from torchfx.effect import FX
from torch import Tensor
import torch

class LearnableGain(FX):
    """Gain effect with learnable parameter."""

    def __init__(self, initial_gain: float = 1.0):
        super().__init__()
        # Learnable parameter
        self.gain = torch.nn.Parameter(torch.tensor([initial_gain]))

    def forward(self, x: Tensor) -> Tensor:
        """Forward with gradient support."""
        return x * self.gain

# Usage in a differentiable pipeline
effect = LearnableGain(initial_gain=0.5)
optimizer = torch.optim.Adam(effect.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    output = effect(input_audio)
    loss = some_loss_function(output, target_audio)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

```{seealso}
{cite:t}`engel2020ddsp` - Differentiable Digital Signal Processing
```

## Best Practices

### Parameter Validation

Always validate parameters in the constructor:

```python
class MyEffect(FX):
    def __init__(self, param: float):
        super().__init__()
        assert param > 0, "Parameter must be positive"
        assert param <= 1.0, "Parameter must be <= 1.0"
        self.param = param
```

### Handle Edge Cases

Consider boundary conditions:

```python
@torch.no_grad()
def forward(self, x: Tensor) -> Tensor:
    # Check for empty input
    if x.numel() == 0:
        return x

    # Check for very short signals
    if x.shape[-1] < self.required_length:
        return x  # Or pad, or raise error

    # Process normally
    return processed
```

### Preserve Tensor Properties

Maintain dtype and device:

```python
@torch.no_grad()
def forward(self, x: Tensor) -> Tensor:
    # Create new tensors on same device with same dtype
    buffer = torch.zeros_like(x)

    # Or explicitly specify
    buffer = torch.zeros(
        x.shape,
        dtype=x.dtype,
        device=x.device
    )

    return buffer
```

### Document Mathematical Formulation

Include formulas in docstrings:

```python
class MyEffect(FX):
    r"""Apply custom effect.

    The effect is computed as:

    .. math::
        y[n] = \alpha x[n] + (1-\alpha) x[n-1]

    where:
        - x[n] is the input signal
        - y[n] is the output signal
        - \alpha is the blend factor
    """
```

## Common Patterns

### Wet/Dry Mix

Almost all effects benefit from a mix parameter:

```python
@torch.no_grad()
def forward(self, x: Tensor) -> Tensor:
    # Process signal
    processed = self.process(x)

    # Mix with dry signal
    output = (1 - self.mix) * x + self.mix * processed

    return output
```

### Extend Signal Length

For delay-based effects:

```python
@torch.no_grad()
def forward(self, x: Tensor) -> Tensor:
    original_length = x.shape[-1]
    extended_length = original_length + self.delay_samples

    # Create extended buffer
    output = torch.zeros(
        *x.shape[:-1], extended_length,
        dtype=x.dtype, device=x.device
    )

    # Copy original signal
    output[..., :original_length] = x

    # Add delayed signal
    output[..., self.delay_samples:] += x * self.feedback

    return output
```

### Per-Channel Processing

Use {class}`torch.nn.ModuleList` for per-channel effects:

```python
class PerChannelEffect(FX):
    def __init__(self, num_channels: int):
        super().__init__()
        self.processors = torch.nn.ModuleList([
            ChannelProcessor() for _ in range(num_channels)
        ])

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        # x shape: (channels, samples)
        outputs = []
        for ch in range(x.shape[0]):
            processed = self.processors[ch](x[ch:ch+1])
            outputs.append(processed)

        return torch.cat(outputs, dim=0)
```

## Related Concepts

- {doc}`wave` - The audio container that uses FX effects
- {doc}`pipeline-operator` - How to chain effects together
- {doc}`/guides/tutorials/custom-effects` - Detailed tutorial on creating custom effects
- {doc}`/guides/tutorials/effects-design` - Design patterns for audio effects

## External Resources

- [PyTorch nn.Module Documentation](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) - Understanding the base class
- [Abstract Base Classes in Python](https://docs.python.org/3/library/abc.html) - Using ABC for interfaces
- [DAFX Book](http://www.dafx.de/) - Digital Audio Effects theory and algorithms

## References

```{bibliography}
:filter: docname in docnames
:style: alpha
```
