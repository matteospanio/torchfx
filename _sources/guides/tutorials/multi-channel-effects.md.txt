# Multi-Channel Audio Processing

Learn how to create audio effects that process multiple channels independently or with channel interaction. This tutorial covers tensor shape conventions, per-channel processing patterns, and stereo-specific effects.

## Overview

Multi-channel audio is ubiquitous in modern audio production—from simple stereo (2 channels) to surround sound (5.1, 7.1) and beyond. TorchFX provides flexible patterns for handling multi-channel audio:

- **Independent processing**: Apply different effects to each channel
- **Broadcast processing**: Apply the same effect to all channels
- **Interactive processing**: Channels affect each other (e.g., ping-pong delay)
- **Channel-aware strategies**: Normalization, delay, and custom algorithms

```{mermaid}
graph TB
    subgraph "Multi-Channel Processing Patterns"
        Input[Multi-Channel<br/>Input]

        subgraph Independent[Independent Processing]
            Ch1[Channel 1:<br/>Filter A]
            Ch2[Channel 2:<br/>Filter B]
            ChN[Channel N:<br/>Filter C]
        end

        subgraph Broadcast[Broadcast Processing]
            Same[Same Filter<br/>All Channels]
        end

        subgraph Interactive[Interactive Processing]
            PingPong[Ping-Pong<br/>L→R, R→L]
        end

        Input --> Independent
        Input --> Broadcast
        Input --> Interactive
    end

    style Input fill:#e1f5ff
    style Independent fill:#fff5e1
    style Broadcast fill:#e8f5e1
    style Interactive fill:#f5e1ff
```

## Tensor Shape Conventions

TorchFX follows PyTorch audio conventions:

| Shape | Description | Use Case |
|-------|-------------|----------|
| `(time,)` | Mono audio | Single microphone recording |
| `(channels, time)` | Multi-channel | Stereo, surround sound |
| `(batch, channels, time)` | Batched multi-channel | ML training batches |
| `(..., time)` | Arbitrary dimensions | General tensor processing |

**Key principle**: The **last dimension is always time**, earlier dimensions are channels/batches.

```python
import torch
import torchfx as fx

# Mono audio: (time,)
mono = torch.randn(44100)  # 1 second at 44.1kHz
wave_mono = fx.Wave(mono.unsqueeze(0), fs=44100)  # Add channel dimension → (1, 44100)

# Stereo audio: (channels, time)
stereo = torch.randn(2, 44100)  # 2 channels, 1 second
wave_stereo = fx.Wave(stereo, fs=44100)

# 5.1 surround: (channels, time)
surround = torch.randn(6, 44100)  # 6 channels
wave_surround = fx.Wave(surround, fs=44100)

# Batched stereo: (batch, channels, time)
batch = torch.randn(8, 2, 44100)  # 8 samples of stereo audio
```

```{seealso}
{doc}`/guides/core-concepts/wave` - Wave class and tensor handling
```

## Built-in Multi-Channel Strategies

### Per-Channel Normalization

The {class}`~torchfx.effect.PerChannelNormalizationStrategy` normalizes each channel independently to its own peak:

```python
import torchfx as fx
from torchfx.effect import Normalize, PerChannelNormalizationStrategy

# Load stereo audio
wave = fx.Wave.from_file("stereo.wav")  # (2, time)

# Standard normalization (uses global peak across all channels)
global_norm = wave | Normalize(peak=1.0)

# Per-channel normalization (each channel normalized to its own peak)
strategy = PerChannelNormalizationStrategy()
perchannel_norm = wave | Normalize(peak=1.0, strategy=strategy)
```

**Behavior comparison**:

```python
# Example with imbalanced channels
left_loud = torch.randn(44100) * 0.8    # Peak ~0.8
right_quiet = torch.randn(44100) * 0.3  # Peak ~0.3
stereo = torch.stack([left_loud, right_quiet])

wave = fx.Wave(stereo, fs=44100)

# Global normalization: both scaled by same factor (based on loudest channel)
global_norm = wave | Normalize(peak=1.0)
# Result: left ~1.0, right ~0.375

# Per-channel normalization: each scaled independently
perchannel_norm = wave | Normalize(peak=1.0, strategy=PerChannelNormalizationStrategy())
# Result: left ~1.0, right ~1.0
```

### Delay Strategies

The {class}`~torchfx.effect.Delay` effect supports two multi-channel strategies:

#### MonoDelayStrategy (Default)

Applies the same delay to all channels independently:

```python
import torchfx as fx
from torchfx.effect import Delay, MonoDelayStrategy

wave = fx.Wave.from_file("stereo.wav")

# Mono strategy: identical delay on both channels
delay = Delay(
    bpm=120,
    delay_time="1/4",
    feedback=0.4,
    mix=0.3,
    strategy=MonoDelayStrategy()  # Default, can be omitted
)

delayed = wave | delay
```

#### PingPongDelayStrategy

Creates alternating delays between left and right stereo channels:

```python
import torchfx as fx
from torchfx.effect import Delay, PingPongDelayStrategy

wave = fx.Wave.from_file("stereo.wav")  # Must be stereo (2 channels)

# Ping-pong delay: alternates between L→R and R→L
delay = Delay(
    bpm=120,
    delay_time="1/8",
    feedback=0.5,
    mix=0.4,
    strategy=PingPongDelayStrategy()
)

delayed = wave | delay
```

**Ping-pong pattern**:
- **Tap 1**: Left channel → delays to → Right channel
- **Tap 2**: Right channel → delays to → Left channel
- **Tap 3**: Left channel → delays to → Right channel
- And so on...

```{mermaid}
sequenceDiagram
    participant L as Left Channel
    participant R as Right Channel

    Note over L,R: Original Signal
    L->>L: Original left
    R->>R: Original right

    Note over L,R: Tap 1 (100% amplitude)
    L->>R: Left delays to Right

    Note over L,R: Tap 2 (feedback^1)
    R->>L: Right delays to Left

    Note over L,R: Tap 3 (feedback^2)
    L->>R: Left delays to Right

    Note over L,R: Result: Ping-pong pattern
```

**Fallback**: If audio is not stereo, {class}`~torchfx.effect.PingPongDelayStrategy` automatically falls back to {class}`~torchfx.effect.MonoDelayStrategy`.

## Creating Per-Channel Effects

### Pattern 1: Independent Channel Processing

Use {class}`torch.nn.ModuleList` to store per-channel processing chains:

```python
import torch
from torch import Tensor, nn
from torchfx import FX, Wave
from torchfx.filter import iir
import torchaudio.transforms as T

class StereoProcessor(FX):
    """Apply different processing to left and right channels."""

    def __init__(self, fs: int | None = None):
        super().__init__()
        self.fs = fs

        # Store per-channel chains in ModuleList
        self.channels = nn.ModuleList([
            self.left_channel(),
            self.right_channel(),
        ])

    def left_channel(self) -> nn.Sequential:
        """Processing chain for left channel."""
        return nn.Sequential(
            iir.HiButterworth(cutoff=100, order=2, fs=self.fs),  # Remove rumble
            iir.LoButterworth(cutoff=8000, order=4, fs=self.fs),  # Remove high freq
        )

    def right_channel(self) -> nn.Sequential:
        """Processing chain for right channel."""
        return nn.Sequential(
            iir.HiButterworth(cutoff=150, order=2, fs=self.fs),   # Different HPF
            iir.LoButterworth(cutoff=10000, order=4, fs=self.fs),  # Different LPF
            T.Vol(0.9),  # Slight volume reduction
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply per-channel processing."""
        if self.fs is None:
            raise ValueError("Sample rate must be set")

        # Process each channel with its own chain
        for i in range(len(self.channels)):
            x[i] = self.channels[i](x[i])

        return x

# Usage
wave = Wave.from_file("stereo.wav")
processor = StereoProcessor(fs=wave.fs)
processed = wave | processor
processed.save("processed_stereo.wav")
```

**Key points**:
- Use {class}`torch.nn.ModuleList` to register submodules
- Pass `fs` to filters that need sample rate
- Process each channel independently in `forward()`
- Each channel can have completely different processing

```{mermaid}
graph TB
    Input[Stereo Input<br/>2, time]

    subgraph Processor[StereoProcessor]
        subgraph Left[Left Channel Chain]
            LH[HiButterworth<br/>100 Hz]
            LL[LoButterworth<br/>8000 Hz]
        end

        subgraph Right[Right Channel Chain]
            RH[HiButterworth<br/>150 Hz]
            RL[LoButterworth<br/>10000 Hz]
            RV[Vol 0.9]
        end
    end

    Output[Stereo Output<br/>2, time]

    Input -->|x[0]| Left
    Input -->|x[1]| Right

    LH --> LL
    RH --> RL --> RV

    Left -->|processed[0]| Output
    Right -->|processed[1]| Output

    style Input fill:#e1f5ff
    style Output fill:#e1f5ff
    style Left fill:#fff5e1
    style Right fill:#ffe1e1
```

### Pattern 2: Dynamic Channel Count

Handle any number of channels dynamically:

```python
from torch import Tensor, nn
from torchfx import FX

class FlexibleMultiChannel(FX):
    """Effect that adapts to any number of channels."""

    def __init__(self, fs: int | None = None):
        super().__init__()
        self.fs = fs
        self.channels = None  # Created dynamically

    def _create_channels(self, num_channels: int):
        """Create processing chains for given number of channels."""
        from torchfx.filter import iir

        self.channels = nn.ModuleList([
            nn.Sequential(
                iir.HiButterworth(cutoff=100 * (i + 1), order=2, fs=self.fs),
                iir.LoButterworth(cutoff=1000 * (i + 1), order=2, fs=self.fs),
            )
            for i in range(num_channels)
        ])

    def forward(self, x: Tensor) -> Tensor:
        num_channels = x.shape[0] if x.ndim >= 2 else 1

        # Create channels on first forward pass
        if self.channels is None:
            self._create_channels(num_channels)

        # Process each channel
        if x.ndim >= 2:
            for i in range(num_channels):
                x[i] = self.channels[i](x[i])
        else:
            x = self.channels[0](x)

        return x
```

### Pattern 3: Complete Example - ComplexEffect

Here's a complete, production-ready example adapted from the TorchFX examples:

```python
import torch
from torch import Tensor, nn
import torchaudio.transforms as T

from torchfx import FX, Wave
from torchfx.filter import iir

class ComplexEffect(FX):
    """Multi-channel effect with different processing per channel.

    Channel 1: Bandpass 1000-2000 Hz
    Channel 2: Bandpass 2000-4000 Hz with volume reduction

    Parameters
    ----------
    num_channels : int
        Number of channels to process
    fs : int, optional
        Sample rate in Hz

    Examples
    --------
    >>> wave = Wave.from_file("stereo.wav")
    >>> fx = ComplexEffect(num_channels=2, fs=wave.fs)
    >>> processed = wave | fx
    """

    def __init__(self, num_channels: int, fs: int | None = None):
        super().__init__()
        self.num_channels = num_channels
        self.fs = fs

        # Per-channel processing chains
        self.ch = nn.ModuleList([
            self.channel1(),
            self.channel2(),
        ])

    def channel1(self) -> nn.Sequential:
        """Processing chain for channel 1."""
        return nn.Sequential(
            iir.HiButterworth(1000, fs=self.fs),  # High-pass at 1000 Hz
            iir.LoButterworth(2000, fs=self.fs),  # Low-pass at 2000 Hz
        )

    def channel2(self) -> nn.Sequential:
        """Processing chain for channel 2."""
        return nn.Sequential(
            iir.HiButterworth(2000, fs=self.fs),  # High-pass at 2000 Hz
            iir.LoButterworth(4000, fs=self.fs),  # Low-pass at 4000 Hz
            T.Vol(0.5),  # Reduce volume by 50%
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply per-channel processing."""
        if self.fs is None:
            raise ValueError("Sampling frequency (fs) must be set")

        # Process each channel independently
        for i in range(self.num_channels):
            x[i] = self.ch[i](x[i])

        return x

# Complete usage example
if __name__ == "__main__":
    # Load stereo audio
    wave = Wave.from_file("input.wav")

    # Create and apply effect
    fx = ComplexEffect(num_channels=2, fs=wave.fs)
    result = wave | fx

    # Save result
    result.save("output.wav")
```

## Dimension-Agnostic Processing

For effects that should work with any tensor shape, detect and handle dimensions:

```python
from torch import Tensor
from torchfx import FX

class DimensionAgnosticEffect(FX):
    """Effect that handles 1D, 2D, and 3D+ tensors."""

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 1:
            # Mono: (time,)
            return self._process_mono(x)

        elif x.ndim == 2:
            # Multi-channel: (channels, time)
            return self._process_multi_channel(x)

        elif x.ndim == 3:
            # Batched: (batch, channels, time)
            return self._process_batched(x)

        else:
            # Higher dimensions: flatten, process, reshape
            original_shape = x.shape
            flattened = x.view(-1, x.size(-1))  # Flatten to (N, time)
            processed = self._process_multi_channel(flattened)
            return processed.view(original_shape)

    def _process_mono(self, x: Tensor) -> Tensor:
        # Process single channel
        return x * 0.5  # Example: reduce volume

    def _process_multi_channel(self, x: Tensor) -> Tensor:
        # Process each channel
        for i in range(x.shape[0]):
            x[i] = self._process_mono(x[i])
        return x

    def _process_batched(self, x: Tensor) -> Tensor:
        # Process batched data
        for b in range(x.shape[0]):
            x[b] = self._process_multi_channel(x[b])
        return x
```

## Channel Interaction Patterns

### Cross-Channel Effects

For effects where channels affect each other:

```python
from torch import Tensor
import torch
from torchfx import FX

class StereoWidener(FX):
    """Widen stereo image using Mid/Side processing."""

    def __init__(self, width: float = 1.5):
        """
        Parameters
        ----------
        width : float
            Stereo width multiplier (1.0 = no change, >1.0 = wider, <1.0 = narrower)
        """
        super().__init__()
        assert width >= 0, "Width must be non-negative"
        self.width = width

    def forward(self, x: Tensor) -> Tensor:
        """Apply stereo widening."""
        # Only works on stereo (2-channel) audio
        if x.ndim < 2 or x.shape[0] != 2:
            return x  # Return unchanged for non-stereo

        left = x[0]
        right = x[1]

        # Convert to Mid/Side
        mid = (left + right) / 2
        side = (left - right) / 2

        # Widen by scaling Side component
        side_widened = side * self.width

        # Convert back to L/R
        new_left = mid + side_widened
        new_right = mid - side_widened

        return torch.stack([new_left, new_right])

# Usage
wave = Wave.from_file("stereo.wav")
widener = StereoWidener(width=1.5)
wider = wave | widener
```

### Channel-Aware Validation

Validate expected channel configuration:

```python
def forward(self, x: Tensor) -> Tensor:
    # Require at least 2D tensor
    if x.ndim < 2:
        raise ValueError("Input must be at least 2D (channels, time)")

    # Require stereo
    if x.shape[-2] != 2:
        raise ValueError(f"Expected stereo (2 channels), got {x.shape[-2]}")

    # Process stereo audio
    # ...
```

## Integration with PyTorch

### Using with DataLoader

Multi-channel effects work seamlessly in PyTorch data pipelines:

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torchfx import Wave

class AudioDataset(Dataset):
    """Dataset with multi-channel audio augmentation."""

    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load audio
        wave = Wave.from_file(self.file_paths[idx])

        # Apply multi-channel transform
        if self.transform:
            wave = wave | self.transform

        return wave.ys, wave.fs

# Create dataset with multi-channel effect
dataset = AudioDataset(
    file_paths=["audio1.wav", "audio2.wav", "audio3.wav"],
    transform=ComplexEffect(num_channels=2, fs=44100)
)

# Use with DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for batch_audio, batch_fs in dataloader:
    # batch_audio shape: (batch, channels, time)
    print(f"Batch shape: {batch_audio.shape}")
```

### GPU Acceleration

Multi-channel effects automatically support GPU:

```python
import torch
from torchfx import Wave

wave = Wave.from_file("stereo.wav")

# Move to GPU
if torch.cuda.is_available():
    wave = wave.to("cuda")

    # Effect runs on GPU
    fx = ComplexEffect(num_channels=2, fs=wave.fs)
    processed = wave | fx

    # Move back to CPU for saving
    processed.to("cpu").save("output.wav")
```

```{seealso}
{doc}`/guides/advanced/gpu-acceleration` - GPU acceleration guide
```

## Best Practices

### Use ModuleList for Channel Chains

```python
# ✅ GOOD: Proper module registration
self.channels = nn.ModuleList([
    self.create_chain(0),
    self.create_chain(1),
])

# ❌ BAD: Regular list won't register modules
self.channels = [
    self.create_chain(0),
    self.create_chain(1),
]
```

### Handle Variable Channel Counts

```python
# ✅ GOOD: Flexible channel handling
def forward(self, x: Tensor) -> Tensor:
    num_channels = x.shape[0] if x.ndim >= 2 else 1

    if self.channels is None or len(self.channels) != num_channels:
        self._create_channels(num_channels)

    # Process channels
    # ...

# ❌ BAD: Hardcoded channel count
def forward(self, x: Tensor) -> Tensor:
    x[0] = self.process_left(x[0])
    x[1] = self.process_right(x[1])
    # Fails for mono or surround
```

### Preserve Tensor Properties

```python
# ✅ GOOD: Preserve device and dtype
output = torch.zeros_like(input_tensor)

# ❌ BAD: May create tensor on wrong device
output = torch.zeros(input_tensor.shape)
```

### Validate Input Shape

```python
# ✅ GOOD: Clear error messages
if x.ndim < 2:
    raise ValueError(
        f"Expected at least 2D tensor (channels, time), got shape {x.shape}"
    )

if x.shape[0] != self.expected_channels:
    raise ValueError(
        f"Expected {self.expected_channels} channels, got {x.shape[0]}"
    )
```

## Common Pitfalls

### In-Place Modifications

```python
# ❌ WRONG: Modifying input in-place can cause issues
def forward(self, x: Tensor) -> Tensor:
    for i in range(x.shape[0]):
        x[i] = self.process(x[i])  # In-place modification
    return x

# ✅ CORRECT: Create output tensor
def forward(self, x: Tensor) -> Tensor:
    output = torch.zeros_like(x)
    for i in range(x.shape[0]):
        output[i] = self.process(x[i])
    return output
```

### Broadcasting Errors

```python
# ❌ WRONG: Shape mismatch
max_val = torch.max(torch.abs(x), dim=1).values  # Shape: (channels,)
normalized = x / max_val * peak  # Error: can't broadcast (channels,) to (channels, time)

# ✅ CORRECT: Use keepdim=True
max_val = torch.max(torch.abs(x), dim=1, keepdim=True).values  # Shape: (channels, 1)
normalized = x / max_val * peak  # Works: broadcasts correctly
```

### Forgetting Sample Rate

```python
# ❌ WRONG: No fs validation
def forward(self, x: Tensor) -> Tensor:
    # self.fs might be None!
    return self.filter(x)

# ✅ CORRECT: Validate fs
def forward(self, x: Tensor) -> Tensor:
    if self.fs is None:
        raise ValueError("Sample rate must be set before processing")
    return self.filter(x)
```

## Related Concepts

- {doc}`/guides/core-concepts/wave` - Wave class and tensor handling
- {doc}`custom-effects` - Creating custom effects
- {doc}`/guides/advanced/pytorch-integration` - PyTorch integration patterns
- {doc}`/guides/advanced/gpu-acceleration` - GPU acceleration

## External Resources

- [PyTorch Audio Documentation](https://pytorch.org/audio/stable/index.html) - torchaudio tensor conventions
- [Multi-Channel Audio on Wikipedia](https://en.wikipedia.org/wiki/Surround_sound) - Multi-channel audio formats
- [Mid/Side Processing](https://en.wikipedia.org/wiki/Stereophonic_sound#M/S_technique:_mid/side_stereophony) - Stereo imaging technique

## References

```{bibliography}
:filter: docname in docnames
:style: alpha
```
