(tutorial-custom-effects)=
# Creating Custom Effects

This tutorial explains how to create custom audio effects in TorchFX by subclassing the {class}`~torchfx.FX` abstract base class. You'll learn about parameter handling, the strategy pattern for extensibility, and multi-channel processing.

## Prerequisites

Before starting this tutorial, you should be familiar with:

- {doc}`../getting-started/getting_started` - TorchFX basics
- [PyTorch nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) - Understanding PyTorch modules
- Basic [digital signal processing concepts](https://en.wikipedia.org/wiki/Digital_signal_processing)

Recommended reading:

- {term}`Audio Effect` - Glossary definition
- {term}`Strategy Pattern` - Design pattern overview
- {cite:t}`gamma1994design` - Design patterns reference

## Understanding the FX Base Class

All effects in TorchFX must inherit from the {class}`~torchfx.FX` abstract base class, which itself inherits from {class}`torch.nn.Module`. This design ensures effects are compatible with PyTorch's module system and can use standard PyTorch features like {term}`device <Device>` management, gradient computation, and serialization.

### Class Hierarchy

```{mermaid}
classDiagram
    class Module["torch.nn.Module"] {
        +forward()
        +to(device)
        +parameters()
    }

    class FX {
        <<abstract>>
        +__init__()
        +forward(x: Tensor) Tensor
    }

    class Gain {
        +gain: float
        +gain_type: str
        +clamp: bool
        +forward(waveform: Tensor) Tensor
    }

    class Normalize {
        +peak: float
        +strategy: NormalizationStrategy
        +forward(waveform: Tensor) Tensor
    }

    class Reverb {
        +delay: int
        +decay: float
        +mix: float
        +forward(waveform: Tensor) Tensor
    }

    class CustomEffect {
        +custom_param: float
        +fs: int | None
        +forward(waveform: Tensor) Tensor
    }

    Module <|-- FX
    FX <|-- Gain
    FX <|-- Normalize
    FX <|-- Reverb
    FX <|-- CustomEffect
```

**FX Class Inheritance Hierarchy** - All effects inherit from FX, which inherits from PyTorch's Module.

### Required Methods

The {class}`~torchfx.FX` class defines two abstract methods that must be implemented by all subclasses:

| Method | Signature | Purpose |
|--------|-----------|---------|
| `__init__` | `__init__(self, *args, **kwargs) -> None` | Initialize effect parameters and call `super().__init__()` |
| `forward` | `forward(self, x: Tensor) -> Tensor` | Process input {term}`tensor <Tensor>` and return transformed output |

Both methods are marked with the `@abc.abstractmethod` decorator, ensuring that subclasses must implement them.

```{seealso}
- [Abstract Base Classes in Python](https://docs.python.org/3/library/abc.html)
- [PyTorch Module API](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
```

## Creating a Basic Custom Effect

The minimal structure for a custom effect requires:

1. Subclassing {class}`~torchfx.FX`
2. Calling `super().__init__()` in the constructor
3. Implementing the `forward` method with tensor input/output
4. Optionally decorating `forward` with `@torch.no_grad()` for inference-only effects

### Effect Lifecycle

```{mermaid}
sequenceDiagram
    participant User
    participant CustomEffect
    participant FX
    participant Module["torch.nn.Module"]

    User->>CustomEffect: __init__(param1, param2)
    CustomEffect->>FX: super().__init__()
    FX->>Module: super().__init__()
    Note over CustomEffect: Store parameters as attributes

    User->>CustomEffect: forward(waveform)
    Note over CustomEffect: Validate input shape/type
    Note over CustomEffect: Apply transformation
    CustomEffect->>User: Return transformed tensor
```

**Effect Initialization and Forward Pass Lifecycle** - Shows the call sequence when creating and using a custom effect.

### Structure Template

A basic custom effect follows this structure:

```python
from torchfx import FX
import torch
from torch import Tensor
from typing import override

class CustomEffect(FX):
    """A simple custom effect that scales the input signal.

    This effect demonstrates the minimal requirements for creating
    a custom audio effect in TorchFX.

    Parameters
    ----------
    param1 : float
        Scaling factor, must be positive
    param2 : int, optional
        Processing window size, default is 100

    Examples
    --------
    >>> from torchfx import Wave
    >>> effect = CustomEffect(param1=0.5, param2=200)
    >>> wave = Wave.from_file("audio.wav")
    >>> result = wave | effect
    """

    def __init__(self, param1: float, param2: int = 100) -> None:
        super().__init__()

        # Validate parameters
        assert param1 > 0, "param1 must be positive"
        assert param2 > 0, "param2 must be positive"

        # Store as instance attributes
        self.param1 = param1
        self.param2 = param2

    @override
    @torch.no_grad()
    def forward(self, waveform: Tensor) -> Tensor:
        """Apply the effect to the input waveform.

        Parameters
        ----------
        waveform : Tensor
            Input audio tensor with shape (..., time) or (channels, time)

        Returns
        -------
        Tensor
            Processed audio tensor with same shape as input
        """
        # Process waveform tensor
        processed = waveform * self.param1
        return processed
```

```{note}
The `@torch.no_grad()` decorator disables gradient computation for inference-only effects. Omit this decorator if your effect will be used in training pipelines that require backpropagation. See [PyTorch autograd](https://pytorch.org/docs/stable/notes/autograd.html) for more details.
```

```{tip}
Use NumPy-style docstrings as shown above. This format integrates well with Sphinx autodoc and provides clear, structured documentation. See the [NumPy docstring guide](https://numpydoc.readthedocs.io/en/latest/format.html).
```

## Parameter Configuration and Validation

Custom effects should validate parameters in the constructor and store them as instance attributes. TorchFX uses Python assertions for parameter validation, following a fail-fast principle.

### Common Validation Patterns

| Validation Type | Pattern | Example Use Case |
|----------------|---------|------------------|
| Positive values | `assert value > 0` | {term}`Sample rate <Sample Rate>`, frequencies, durations |
| Range bounds (exclusive) | `assert 0 < value < 1` | Probability values, normalized parameters |
| Range bounds (inclusive) | `assert 0 <= value <= 1` | Mix levels, feedback amounts |
| Non-empty collections | `assert len(value) > 0` | Filter coefficient arrays |
| Type checking | `assert isinstance(x, Type)` | Strategy pattern validation |

### Sample Rate Handling

Effects that depend on the {term}`sampling frequency <Sample Rate>` (like time-based effects) should accept an optional `fs` parameter. This parameter can be automatically configured when the effect is used with the {term}`Wave <Wave>` {term}`pipeline operator <Pipeline Operator>`.

```python
class TimeBasedEffect(FX):
    """An effect that processes audio based on time duration.

    The sample rate (fs) can be provided at initialization or
    automatically configured when used in a Wave pipeline.

    Parameters
    ----------
    duration_ms : float
        Effect duration in milliseconds
    fs : int or None, optional
        Sample rate in Hz. If None, will be auto-configured by Wave
    """

    def __init__(self, duration_ms: float, fs: int | None = None) -> None:
        super().__init__()

        assert duration_ms > 0, "Duration must be positive"

        self.duration_ms = duration_ms
        self.fs = fs  # Can be set by Wave.__update_config

        if fs is not None:
            assert fs > 0, "Sample rate must be positive"
            self.samples = int((duration_ms / 1000) * fs)
            self._needs_calculation = False
        else:
            self.samples = None
            self._needs_calculation = True

    @override
    def forward(self, waveform: Tensor) -> Tensor:
        if self.fs is None:
            raise ValueError("Sample rate (fs) must be set before processing")

        # Calculate samples if needed (lazy initialization)
        if self._needs_calculation:
            self.samples = int((self.duration_ms / 1000) * self.fs)
            self._needs_calculation = False

        # Use self.samples for processing
        return waveform
```

When `fs` is `None` at initialization, the {class}`~torchfx.Wave` class automatically sets it when the effect is used in a {term}`pipeline <Pipeline>`. This allows for flexible effect creation:

```python
from torchfx import Wave

# Option 1: Explicit sample rate
effect = TimeBasedEffect(duration_ms=100, fs=44100)

# Option 2: Auto-configured (recommended)
effect = TimeBasedEffect(duration_ms=100)  # fs is None
wave = Wave.from_file("audio.wav")  # fs = 44100
result = wave | effect  # fs automatically set to 44100
```

```{seealso}
- [Lazy initialization pattern](https://en.wikipedia.org/wiki/Lazy_initialization)
```

## Strategy Pattern for Extensibility

The [strategy pattern](https://en.wikipedia.org/wiki/Strategy_pattern) allows effects to support multiple processing algorithms while maintaining a clean interface {cite:p}`gamma1994design`. TorchFX uses this pattern extensively in the {class}`~torchfx.Normalize` and {class}`~torchfx.Delay` effects.

### Pattern Architecture

```{mermaid}
classDiagram
    class Normalize {
        +peak: float
        +strategy: NormalizationStrategy
        +forward(waveform: Tensor) Tensor
    }

    class NormalizationStrategy {
        <<abstract>>
        +__call__(waveform: Tensor, peak: float) Tensor
    }

    class PeakNormalizationStrategy {
        +__call__(waveform: Tensor, peak: float) Tensor
    }

    class RMSNormalizationStrategy {
        +__call__(waveform: Tensor, peak: float) Tensor
    }

    class PercentileNormalizationStrategy {
        +percentile: float
        +__call__(waveform: Tensor, peak: float) Tensor
    }

    class PerChannelNormalizationStrategy {
        +__call__(waveform: Tensor, peak: float) Tensor
    }

    class CustomNormalizationStrategy {
        +func: Callable
        +__call__(waveform: Tensor, peak: float) Tensor
    }

    Normalize --> NormalizationStrategy
    NormalizationStrategy <|-- PeakNormalizationStrategy
    NormalizationStrategy <|-- RMSNormalizationStrategy
    NormalizationStrategy <|-- PercentileNormalizationStrategy
    NormalizationStrategy <|-- PerChannelNormalizationStrategy
    NormalizationStrategy <|-- CustomNormalizationStrategy
```

**Normalization Strategy Pattern Structure** - Effect delegates processing to interchangeable strategy objects.

### Implementing a Strategy-Based Effect

A strategy-based effect consists of three components:

1. **Abstract strategy base class** with a `__call__` method
2. **Concrete strategy implementations**
3. **Effect class** that delegates processing to the strategy

#### Step 1: Define the Abstract Strategy

```python
import abc
from torch import Tensor

class ProcessingStrategy(abc.ABC):
    """Abstract base class for processing strategies.

    Strategies implement different algorithms for processing audio,
    allowing the same effect to support multiple behaviors.
    """

    @abc.abstractmethod
    def __call__(self, waveform: Tensor, param: float) -> Tensor:
        """Apply the processing strategy.

        Parameters
        ----------
        waveform : Tensor
            Input audio tensor
        param : float
            Strategy-specific parameter

        Returns
        -------
        Tensor
            Processed audio tensor
        """
        pass
```

#### Step 2: Implement Concrete Strategies

```python
class LinearStrategy(ProcessingStrategy):
    """Linear scaling strategy."""

    def __call__(self, waveform: Tensor, param: float) -> Tensor:
        return waveform * param

class ExponentialStrategy(ProcessingStrategy):
    """Exponential scaling strategy."""

    def __call__(self, waveform: Tensor, param: float) -> Tensor:
        return torch.sign(waveform) * torch.abs(waveform) ** param

class SoftClipStrategy(ProcessingStrategy):
    """Soft clipping strategy using tanh."""

    def __call__(self, waveform: Tensor, param: float) -> Tensor:
        return torch.tanh(waveform * param)
```

#### Step 3: Create the Effect Class

```python
class StrategyBasedEffect(FX):
    """An effect that uses the strategy pattern for processing.

    This effect demonstrates how to implement flexible, extensible
    effects by delegating processing to strategy objects.

    Parameters
    ----------
    param : float
        Processing parameter passed to the strategy
    strategy : ProcessingStrategy or Callable, optional
        Processing strategy. Can be a ProcessingStrategy instance
        or a callable function. If None, uses LinearStrategy.

    Examples
    --------
    >>> # Using built-in strategy
    >>> effect = StrategyBasedEffect(param=0.5, strategy=LinearStrategy())
    >>>
    >>> # Using custom callable
    >>> def custom_process(waveform, param):
    ...     return waveform * param * 2
    >>> effect = StrategyBasedEffect(param=0.5, strategy=custom_process)
    """

    def __init__(
        self,
        param: float,
        strategy: ProcessingStrategy | Callable | None = None
    ) -> None:
        super().__init__()

        self.param = param

        # Support custom callable functions
        if callable(strategy) and not isinstance(strategy, ProcessingStrategy):
            # Wrap callable in a strategy object
            strategy = CustomStrategy(strategy)

        # Use default strategy if none provided
        self.strategy = strategy or LinearStrategy()

        # Validate strategy type
        if not isinstance(self.strategy, ProcessingStrategy):
            raise TypeError("Strategy must be ProcessingStrategy or callable")

    @override
    def forward(self, waveform: Tensor) -> Tensor:
        """Process waveform using the configured strategy."""
        return self.strategy(waveform, self.param)

class CustomStrategy(ProcessingStrategy):
    """Wrapper for custom callable strategies."""

    def __init__(self, func: Callable):
        self.func = func

    def __call__(self, waveform: Tensor, param: float) -> Tensor:
        return self.func(waveform, param)
```

### Custom Strategy Example

Users can provide custom strategies as callable functions or strategy class instances:

```python
# As a callable function (recommended for simple cases)
def median_normalize(waveform: Tensor, peak: float) -> Tensor:
    """Custom normalization using median instead of peak."""
    median = torch.median(torch.abs(waveform))
    return waveform / median * peak if median > 0 else waveform

# Use with effect
effect = StrategyBasedEffect(param=0.8, strategy=median_normalize)

# As a strategy class (recommended for complex logic)
class AdaptiveStrategy(ProcessingStrategy):
    """Strategy that adapts based on signal characteristics."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def __call__(self, waveform: Tensor, param: float) -> Tensor:
        # Adaptive processing based on signal level
        level = torch.mean(torch.abs(waveform))
        if level > self.threshold:
            return waveform * param * 0.5  # Reduce gain for loud signals
        else:
            return waveform * param  # Normal gain for quiet signals

# Use with effect
effect = StrategyBasedEffect(param=1.0, strategy=AdaptiveStrategy(threshold=0.6))
```

```{tip}
The strategy pattern is particularly useful when:

- You need multiple algorithms for the same operation
- Algorithms may be added or changed frequently
- You want users to provide custom processing logic
- Different algorithms have different configuration needs
```

## Multi-Channel Effects

Multi-channel effects process each audio channel independently or with channel-specific processing chains. TorchFX provides the {class}`torch.nn.ModuleList` pattern for implementing per-channel processing.

### Multi-Channel Architecture

```{mermaid}
graph TB
    subgraph ComplexEffect["ComplexEffect (FX)"]
        Init["__init__(num_channels, fs)"]
        Forward["forward(x: Tensor)"]
        ModuleList["ch: nn.ModuleList"]

        subgraph Channel1["ch[0]: Channel 1 Chain"]
            HiButter1["HiButterworth(1000, fs)"]
            LoButter1["LoButterworth(2000, fs)"]
            HiButter1 --> LoButter1
        end

        subgraph Channel2["ch[1]: Channel 2 Chain"]
            HiButter2["HiButterworth(2000, fs)"]
            LoButter2["LoButterworth(4000, fs)"]
            Vol["Vol(0.5)"]
            HiButter2 --> LoButter2 --> Vol
        end

        Init --> ModuleList
        ModuleList --> Channel1
        ModuleList --> Channel2
        Forward --> Channel1
        Forward --> Channel2
    end

    Input["Input Tensor<br/>(channels, time)"] --> Forward
    Forward --> Output["Output Tensor<br/>(channels, time)"]
```

**Multi-Channel Effect Architecture Using nn.ModuleList** - Each channel can have its own processing chain.

### Implementation Pattern

```python
from torch import nn
from torchfx import FX
from torchfx.filter import HiButterworth, LoButterworth
from torchfx import Vol

class MultiChannelEffect(FX):
    """A multi-channel effect with per-channel processing chains.

    This effect demonstrates how to create effects that process
    each channel differently, useful for stereo enhancement,
    frequency splitting, and crossover designs.

    Parameters
    ----------
    num_channels : int
        Number of audio channels to process
    fs : int or None, optional
        Sample rate in Hz

    Examples
    --------
    >>> effect = MultiChannelEffect(num_channels=2, fs=44100)
    >>> stereo_wave = Wave.from_file("stereo.wav")  # (2, time)
    >>> result = stereo_wave | effect
    """

    ch: nn.ModuleList
    fs: int | None

    def __init__(self, num_channels: int, fs: int | None = None) -> None:
        super().__init__()

        assert num_channels > 0, "Number of channels must be positive"

        self.num_channels = num_channels
        self.fs = fs

        # Create per-channel processing chains
        self.ch = nn.ModuleList([
            self.create_channel_chain(i)
            for i in range(num_channels)
        ])

    def create_channel_chain(self, channel_idx: int) -> nn.Module:
        """Create processing chain for a specific channel.

        Parameters
        ----------
        channel_idx : int
            Zero-based channel index

        Returns
        -------
        nn.Module
            Processing module or chain for this channel
        """
        if self.fs is None:
            raise ValueError("fs must be set to create filters")

        # Example: Different processing per channel
        if channel_idx == 0:
            # Channel 0: Bandpass 1000-2000 Hz
            return nn.Sequential(
                HiButterworth(cutoff=1000, order=4, fs=self.fs),
                LoButterworth(cutoff=2000, order=4, fs=self.fs)
            )
        else:
            # Channel 1: Bandpass 2000-4000 Hz with volume reduction
            return nn.Sequential(
                HiButterworth(cutoff=2000, order=4, fs=self.fs),
                LoButterworth(cutoff=4000, order=4, fs=self.fs),
                Vol(volume=0.5)
            )

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Process multi-channel audio.

        Parameters
        ----------
        x : Tensor
            Input audio with shape (channels, time)

        Returns
        -------
        Tensor
            Processed audio with shape (channels, time)
        """
        if self.fs is None:
            raise ValueError("Sampling frequency (fs) must be set")

        # Process each channel with its own chain
        for i in range(self.num_channels):
            x[i] = self.ch[i](x[i])

        return x
```

This pattern enables complex routing and processing scenarios:

- **Frequency splitting (crossovers)** - Send different frequency bands to different channels
- **Stereo widening effects** - Apply different processing to L/R channels
- **Mid-side processing** - Process mid and side components separately
- **Per-channel dynamics** - Apply different compression/limiting per channel

```{note}
For processing that affects all channels equally, you don't need {class}`torch.nn.ModuleList`. Simply process the entire tensor at once:

```python
def forward(self, x: Tensor) -> Tensor:
    # Process all channels identically
    return x * self.gain
```
```

```{seealso}
- [PyTorch ModuleList](https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html)
- {term}`Multi-channel Audio` in the glossary
```

## Integration with Wave Pipeline

Custom effects integrate seamlessly with the {class}`~torchfx.Wave` {term}`pipeline operator <Pipeline Operator>` (`|`) and inherit automatic configuration capabilities.

### Automatic Configuration Flow

```{mermaid}
sequenceDiagram
    participant User
    participant Wave
    participant CustomEffect
    participant Tensor

    User->>Wave: Wave.from_file("audio.wav")
    Note over Wave: fs = 44100<br/>ys = Tensor(2, 44100)

    User->>CustomEffect: CustomEffect(params, fs=None)
    Note over CustomEffect: fs attribute is None<br/>Will be auto-configured

    User->>Wave: wave | custom_effect
    Wave->>Wave: __or__(custom_effect)
    Wave->>Wave: __update_config(custom_effect)
    Note over Wave,CustomEffect: If custom_effect has 'fs' attribute<br/>and it's None, set it to wave.fs

    Wave->>CustomEffect: forward(wave.ys)
    Note over CustomEffect: fs is now 44100<br/>Can compute time-based parameters

    CustomEffect->>Tensor: Process audio
    Tensor->>Wave: Return new Wave
    Wave->>User: Result Wave
```

**Automatic Configuration Flow in Pipeline** - Wave automatically configures effects when used in pipelines.

### Auto-Configuration Mechanism

When a custom effect has an `fs` attribute set to `None`, the {class}`~torchfx.Wave` class automatically configures it when used in a pipeline:

```python
from torchfx import Wave

# Create effect without sample rate
effect = TimeBasedCustomEffect(duration_ms=100)  # fs is None

# Load audio file
wave = Wave.from_file("audio.wav")  # fs is set from file

# Pipeline operator automatically sets effect.fs = wave.fs
result = wave | effect  # effect.fs is now 44100 (or whatever wave.fs is)
```

This mechanism allows effects to be created once and reused with audio at different sample rates:

```python
# Create effect once
reverb = MyReverb(decay=0.5, room_size=0.8)  # fs=None

# Use with different sample rates
wave_44k = Wave.from_file("audio_44100.wav")
wave_48k = Wave.from_file("audio_48000.wav")

result_44k = wave_44k | reverb  # reverb.fs temporarily 44100
result_48k = wave_48k | reverb  # reverb.fs temporarily 48000
```

### Device Handling

Custom effects automatically inherit {term}`device <Device>` management from {class}`torch.nn.Module`. Effects can be moved to {term}`GPU` using `.to()`:

```python
# Create effect
custom_effect = CustomEffect(param=0.5)

# Move to GPU
custom_effect.to("cuda")

# Or use in pipeline - Wave handles device
wave = Wave.from_file("audio.wav").to("cuda")
result = wave | custom_effect  # Effect processes on GPU
```

TorchFX automatically propagates device placement through pipelines, so you typically only need to set the device on the {class}`~torchfx.Wave` object.

```python
from torchfx import Wave
from torchfx.filter import LoButterworth
import torch

# Check CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create pipeline
wave = Wave.from_file("audio.wav").to(device)
lowpass = LoButterworth(cutoff=5000, order=4, fs=wave.fs)
custom = CustomEffect(param=0.8)

# All processing happens on device
result = wave | lowpass | custom
```

```{seealso}
- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- {term}`CUDA` in the glossary
```

## Complete Example: Multi-Tap Delay

This example combines all concepts covered in this tutorial: strategy pattern, sample rate handling, parameter validation, and device compatibility.

### Implementation

```python
import abc
import torch
from torch import Tensor
from typing import override
from torchfx import FX

# Step 1: Define strategy interface
class DecayStrategy(abc.ABC):
    """Abstract strategy for calculating tap amplitudes."""

    @abc.abstractmethod
    def calculate_amplitude(self, tap_index: int, base_feedback: float) -> float:
        """Calculate amplitude for a delay tap.

        Parameters
        ----------
        tap_index : int
            Tap number (1-indexed)
        base_feedback : float
            Base feedback amount in range [0, 1]

        Returns
        -------
        float
            Amplitude multiplier for this tap
        """
        pass

# Step 2: Implement concrete strategies
class ExponentialDecayStrategy(DecayStrategy):
    """Exponential decay - each tap is feedback^tap_index."""

    def calculate_amplitude(self, tap_index: int, base_feedback: float) -> float:
        return base_feedback ** tap_index

class LinearDecayStrategy(DecayStrategy):
    """Linear decay - each tap decreases by constant amount."""

    def calculate_amplitude(self, tap_index: int, base_feedback: float) -> float:
        return max(0.0, 1.0 - (tap_index * (1.0 - base_feedback)))

class FibonacciDecayStrategy(DecayStrategy):
    """Fibonacci-based decay for interesting rhythmic patterns."""

    def __init__(self):
        self.fib_cache = {0: 0, 1: 1}

    def _fibonacci(self, n: int) -> int:
        if n not in self.fib_cache:
            self.fib_cache[n] = self._fibonacci(n-1) + self._fibonacci(n-2)
        return self.fib_cache[n]

    def calculate_amplitude(self, tap_index: int, base_feedback: float) -> float:
        fib_sum = sum(self._fibonacci(i) for i in range(1, tap_index + 2))
        fib_val = self._fibonacci(tap_index)
        return (fib_val / fib_sum) * base_feedback if fib_sum > 0 else 0.0

# Step 3: Implement the effect
class MultiTapDelay(FX):
    """Multi-tap delay effect with configurable decay strategies.

    This effect creates multiple delayed copies of the input signal,
    each with decreasing amplitude according to the selected strategy.
    Useful for creating echo effects, rhythmic patterns, and spatial depth.

    The delay can be synchronized to musical time using the BPM parameter,
    or specified in absolute milliseconds.

    Parameters
    ----------
    delay_ms : float
        Delay time in milliseconds for each tap
    taps : int, optional
        Number of delay taps (repetitions), default is 3
    feedback : float, optional
        Base feedback amount in range [0, 1], default is 0.5
    fs : int or None, optional
        Sample rate in Hz, auto-configured if None
    strategy : DecayStrategy or None, optional
        Strategy for calculating tap amplitudes.
        If None, uses ExponentialDecayStrategy

    Attributes
    ----------
    delay_samples : int or None
        Delay time in samples, calculated from delay_ms and fs

    Examples
    --------
    >>> # Basic usage with exponential decay
    >>> from torchfx import Wave
    >>> delay = MultiTapDelay(delay_ms=100, taps=4, feedback=0.6)
    >>> wave = Wave.from_file("audio.wav")
    >>> result = wave | delay
    >>>
    >>> # With custom linear decay strategy
    >>> delay = MultiTapDelay(
    ...     delay_ms=100,
    ...     taps=4,
    ...     feedback=0.6,
    ...     strategy=LinearDecayStrategy()
    ... )
    >>> result = wave | delay
    >>>
    >>> # With Fibonacci decay for rhythmic interest
    >>> delay = MultiTapDelay(
    ...     delay_ms=150,
    ...     taps=6,
    ...     feedback=0.7,
    ...     strategy=FibonacciDecayStrategy()
    ... )
    >>> result = wave | delay

    See Also
    --------
    torchfx.Delay : Built-in delay effect with mono/ping-pong strategies

    Notes
    -----
    The output length is increased by ``delay_ms * taps`` to accommodate
    all delay taps. The effect supports both mono and multi-channel audio.

    For tempo-synchronized delays, consider using the BPM-to-milliseconds
    conversion: ``delay_ms = (60000 / bpm) * beat_division``

    References
    ----------
    .. [1] Zölzer, U. (2011). DAFX: Digital Audio Effects (2nd ed.).
           John Wiley & Sons. Chapter on Delay Effects.
    """

    def __init__(
        self,
        delay_ms: float,
        taps: int = 3,
        feedback: float = 0.5,
        fs: int | None = None,
        strategy: DecayStrategy | None = None
    ) -> None:
        super().__init__()

        # Parameter validation
        assert delay_ms > 0, "Delay must be positive"
        assert taps >= 1, "At least one tap required"
        assert 0 <= feedback <= 1, "Feedback must be in [0, 1]"

        self.delay_ms = delay_ms
        self.taps = taps
        self.feedback = feedback
        self.fs = fs
        self.strategy = strategy or ExponentialDecayStrategy()

        # Calculate delay samples if fs is available
        if fs is not None:
            assert fs > 0, "Sample rate must be positive"
            self.delay_samples = int((delay_ms / 1000) * fs)
        else:
            self.delay_samples = None

    @override
    @torch.no_grad()
    def forward(self, waveform: Tensor) -> Tensor:
        """Apply multi-tap delay to the waveform.

        Parameters
        ----------
        waveform : Tensor
            Input audio tensor with shape (..., time) or (channels, time)

        Returns
        -------
        Tensor
            Delayed audio with extended length to accommodate all taps

        Raises
        ------
        ValueError
            If sample rate (fs) has not been set
        """
        if self.fs is None:
            raise ValueError("Sample rate (fs) must be set before processing")

        # Lazy calculation of delay samples
        if self.delay_samples is None:
            self.delay_samples = int((self.delay_ms / 1000) * self.fs)

        # Calculate output length
        max_delay = self.delay_samples * self.taps
        output_length = waveform.size(-1) + max_delay

        # Create output tensor with appropriate shape and device
        if waveform.ndim == 1:
            output = torch.zeros(
                output_length,
                dtype=waveform.dtype,
                device=waveform.device
            )
        else:
            output = torch.zeros(
                *waveform.shape[:-1], output_length,
                dtype=waveform.dtype,
                device=waveform.device
            )

        # Add original signal (tap 0)
        output[..., :waveform.size(-1)] = waveform

        # Add delayed taps with strategy-based decay
        for tap in range(1, self.taps + 1):
            tap_delay = self.delay_samples * tap
            amplitude = self.strategy.calculate_amplitude(tap, self.feedback)

            if waveform.ndim == 1:
                output[tap_delay:tap_delay + waveform.size(-1)] += (
                    waveform * amplitude
                )
            else:
                output[..., tap_delay:tap_delay + waveform.size(-1)] += (
                    waveform * amplitude
                )

        return output
```

### Usage Examples

```python
from torchfx import Wave

# Example 1: Basic multi-tap delay with exponential decay
delay = MultiTapDelay(delay_ms=100, taps=4, feedback=0.6, fs=44100)
wave = Wave.from_file("vocals.wav")
result = wave | delay
result.save("vocals_delayed.wav")

# Example 2: Auto-configured sample rate
delay = MultiTapDelay(delay_ms=150, taps=3, feedback=0.5)
wave = Wave.from_file("drums.wav")  # fs auto-detected
result = wave | delay  # fs automatically configured

# Example 3: Linear decay for more uniform echoes
delay = MultiTapDelay(
    delay_ms=100,
    taps=5,
    feedback=0.7,
    strategy=LinearDecayStrategy()
)
result = wave | delay

# Example 4: Fibonacci decay for rhythmic interest
delay = MultiTapDelay(
    delay_ms=200,
    taps=6,
    feedback=0.8,
    strategy=FibonacciDecayStrategy()
)
result = wave | delay

# Example 5: Tempo-synchronized delay (quarter note at 120 BPM)
bpm = 120
quarter_note_ms = (60000 / bpm)  # 500ms
delay = MultiTapDelay(
    delay_ms=quarter_note_ms,
    taps=4,
    feedback=0.5
)
result = wave | delay

# Example 6: GPU processing
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
delay = MultiTapDelay(delay_ms=100, taps=4).to(device)
wave = Wave.from_file("audio.wav").to(device)
result = wave | delay  # Processes on GPU
```

### Key Features Demonstrated

This example demonstrates:

1. **Strategy Pattern** - `DecayStrategy` abstract class with multiple implementations
2. **Sample Rate Handling** - Optional `fs` parameter with lazy calculation
3. **Parameter Validation** - Comprehensive assertions for all parameters
4. **Tensor Shape Handling** - Support for both 1D and multi-dimensional tensors
5. **Device Compatibility** - Uses tensor device and dtype from input
6. **Documentation** - Complete NumPy-style docstrings with examples
7. **Musical Timing** - BPM-to-milliseconds conversion for tempo sync

```{tip}
This multi-tap delay can be extended further:

- Add stereo spread by delaying L/R channels differently
- Implement filtering on delayed taps (e.g., lowpass for darker echoes)
- Add modulation to delay time for chorus-like effects
- Combine with reverb for rich spatial effects
```

### Best Practices

**Documentation**
- Use NumPy-style docstrings with parameter descriptions
- Include usage examples in docstrings
- Document expected tensor shapes and dimensions
- Add `See Also` sections linking related functionality

**Parameter Validation**
- Validate all parameters in `__init__` with clear error messages
- Use assertions for preconditions
- Document valid parameter ranges in docstrings

**Device Handling**
- Use `waveform.device` and `waveform.dtype` when creating new tensors
- Don't hardcode device or dtype
- Test on both CPU and CUDA if GPU support is important

**Tensor Shapes**
- Support both 1D (mono) and multi-dimensional (multi-channel) tensors
- Use `...` indexing for flexibility: `output[..., :length]`
- Document expected input/output shapes clearly

**Gradient Computation**
- Use `@torch.no_grad()` for inference-only effects
- Omit decorator if effect should support backpropagation
- Document gradient behavior in docstring

**Strategy Pattern**
- Use when multiple algorithms are possible
- Provide sensible default strategy
- Allow callable functions as strategies for convenience
- Document available strategies and their behavior

Custom effects automatically integrate with:

- {class}`~torchfx.Wave` {term}`pipeline operator <Pipeline Operator>` (`|`)
- PyTorch device management (`.to(device)`)
- Automatic {term}`sample rate <Sample Rate>` configuration
- Neural network training (if gradients are enabled)

## Next Steps

Now that you understand custom effects, explore:

- [Custom Filters](custom-filters.md) - Create custom filter designs
- [Series and Parallel Filters](series-parallel-filters.md) - Compose effects and filters in pipelines
- [ML Batch Processing](ml-batch-processing.md) - Integrate with neural networks

### External Resources

- [Digital Audio Effects (DAFX)](https://www.dafx.de/) - Research and resources
- [Julius O. Smith's Books](https://ccrma.stanford.edu/~jos/) - Free online DSP books
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) - Official PyTorch docs
- [Audio Signal Processing](https://en.wikipedia.org/wiki/Audio_signal_processing) - Wikipedia overview

## References

```{bibliography}
:filter: docname in docnames
:style: alpha
```
