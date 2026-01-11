(multi-channel)=
# Multi-Channel Processing

Learn how to process multi-channel audio in TorchFX, from simple stereo to complex surround sound configurations. This guide covers tensor shape conventions, per-channel vs. cross-channel processing strategies, and production-ready patterns for building multi-channel effects.

## Prerequisites

Before starting this guide, you should be familiar with:

- {doc}`../core-concepts/wave` - Wave class and tensor handling
- {doc}`../core-concepts/fx` - FX base class for effects
- {doc}`../tutorials/custom-effects` - Creating custom effects
- Basic PyTorch {class}`torch.nn.Module` and {class}`torch.nn.ModuleList` usage

## Overview

Multi-channel audio is everywhere in modern audio production—from stereo music to 5.1/7.1 surround sound in film and immersive spatial audio formats. TorchFX provides flexible patterns for handling multi-channel audio with three main processing approaches:

| Processing Type | Description | Example Use Case |
|----------------|-------------|------------------|
| **Broadcast** | Same effect applied to all channels | Global EQ, normalization |
| **Per-channel** | Different processing for each channel | Independent channel EQ, stereo mastering |
| **Cross-channel** | Channels interact with each other | Ping-pong delay, stereo widening |

```{mermaid}
graph TB
    subgraph "Multi-Channel Processing Strategies"
        Input[Multi-Channel Input<br/>channels, time]

        subgraph Broadcast[Broadcast Processing]
            BC[Same filter<br/>applied to all channels]
        end

        subgraph PerChannel[Per-Channel Processing]
            PC1[Channel 1: Filter A]
            PC2[Channel 2: Filter B]
            PCN[Channel N: Filter C]
        end

        subgraph CrossChannel[Cross-Channel Processing]
            CC[Channels interact<br/>e.g., L→R, R→L]
        end

        Input --> Broadcast
        Input --> PerChannel
        Input --> CrossChannel

        Broadcast --> Output1[Output: Same processing]
        PerChannel --> Output2[Output: Different per channel]
        CrossChannel --> Output3[Output: Channel interaction]
    end

    style Input fill:#e1f5ff
    style Broadcast fill:#e8f5e1
    style PerChannel fill:#fff5e1
    style CrossChannel fill:#f5e1ff
```

## Tensor Shape Conventions

TorchFX follows standard PyTorch audio conventions where **the last dimension represents time** and earlier dimensions represent channels and/or batches.

### Standard Audio Shapes

| Shape | Description | Example Use Case |
|-------|-------------|------------------|
| `(time,)` | Mono audio | Single microphone recording |
| `(channels, time)` | Multi-channel audio | Stereo (2), 5.1 surround (6), 7.1 surround (8) |
| `(batch, channels, time)` | Batched multi-channel | Neural network training batches |
| `(..., time)` | Arbitrary leading dimensions | Generic tensor processing |

All effects inheriting from {class}`~torchfx.FX` accept tensors with these shapes. By default, effects broadcast operations across all dimensions except time, unless they implement channel-specific logic.

```{important}
The **last dimension is always time**. This is critical for proper tensor handling in TorchFX and PyTorch audio libraries.
```

### Shape Convention Examples

```python
import torch
import torchfx as fx

# Mono audio: (time,)
mono = torch.randn(44100)  # 1 second at 44.1kHz
print(f"Mono shape: {mono.shape}")  # (44100,)

# Stereo audio: (channels, time)
stereo = torch.randn(2, 44100)  # 2 channels, 1 second
print(f"Stereo shape: {stereo.shape}")  # (2, 44100)

# 5.1 surround: (channels, time)
# Order: Front L, Front R, Center, LFE, Rear L, Rear R
surround_51 = torch.randn(6, 44100)
print(f"5.1 shape: {surround_51.shape}")  # (6, 44100)

# 7.1 surround: (channels, time)
surround_71 = torch.randn(8, 44100)
print(f"7.1 shape: {surround_71.shape}")  # (8, 44100)

# Batched stereo for ML: (batch, channels, time)
batch = torch.randn(32, 2, 44100)  # 32 stereo samples
print(f"Batched shape: {batch.shape}")  # (32, 2, 44100)
```

### Channel Processing Flow

```{mermaid}
graph LR
    subgraph "Input Tensor Shapes"
        Mono["(time,)<br/>Mono"]
        Stereo["(2, time)<br/>Stereo"]
        Multi["(N, time)<br/>N channels"]
        Batch["(B, N, time)<br/>Batched"]
    end

    subgraph "FX Base Class"
        FX["FX.forward(x: Tensor)<br/>All shapes supported"]
    end

    subgraph "Processing Strategies"
        Broadcast["Broadcast<br/>Same to all"]
        PerCh["Per-Channel<br/>Independent"]
        Cross["Cross-Channel<br/>Interactive"]
    end

    Mono --> FX
    Stereo --> FX
    Multi --> FX
    Batch --> FX

    FX --> Broadcast
    FX --> PerCh
    FX --> Cross
```

## Per-Channel Processing Patterns

Per-channel processing applies different effects to each channel independently. This is the most common pattern for multi-channel effects.

### Pattern 1: Fixed Channel Count with ModuleList

Use {class}`torch.nn.ModuleList` to store separate processing chains for each channel:

```python
import torch
from torch import Tensor, nn
import torchaudio.transforms as T
from torchfx import FX, Wave
from torchfx.filter import iir

class StereoProcessor(FX):
    """Apply different processing to left and right channels.

    Left channel: Remove rumble, gentle high-pass
    Right channel: Different EQ curve with volume reduction

    Parameters
    ----------
    fs : int, optional
        Sample rate in Hz. Can be set via Wave pipeline.

    Examples
    --------
    >>> wave = Wave.from_file("stereo.wav")
    >>> processor = StereoProcessor(fs=wave.fs)
    >>> processed = wave | processor
    """

    def __init__(self, fs: int | None = None):
        super().__init__()
        self.fs = fs

        # Store per-channel processing chains
        self.channels = nn.ModuleList([
            self.left_channel(),
            self.right_channel(),
        ])

    def left_channel(self) -> nn.Sequential:
        """Processing chain for left channel."""
        return nn.Sequential(
            iir.HiButterworth(cutoff=100, order=2, fs=self.fs),   # Remove rumble
            iir.LoButterworth(cutoff=8000, order=4, fs=self.fs),  # Gentle rolloff
        )

    def right_channel(self) -> nn.Sequential:
        """Processing chain for right channel."""
        return nn.Sequential(
            iir.HiButterworth(cutoff=150, order=2, fs=self.fs),    # Higher HPF
            iir.LoButterworth(cutoff=10000, order=4, fs=self.fs),  # Different LPF
            T.Vol(0.9),  # Slight volume reduction
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply per-channel processing.

        Parameters
        ----------
        x : Tensor
            Input audio with shape (2, time) for stereo

        Returns
        -------
        Tensor
            Processed audio with same shape as input
        """
        if self.fs is None:
            raise ValueError("Sample rate (fs) must be set before processing")

        # Process each channel independently
        for i in range(len(self.channels)):
            x[i] = self.channels[i](x[i])

        return x

# Usage
wave = Wave.from_file("stereo_music.wav")
processor = StereoProcessor(fs=wave.fs)
processed = wave | processor
processed.save("stereo_processed.wav")
```

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

**Key implementation details**:

1. Use {class}`torch.nn.ModuleList` to properly register submodules
2. Accept `fs` parameter and pass it to filters requiring sample rate
3. Validate `fs` is set before processing
4. Process each channel independently in `forward()`

```{seealso}
{doc}`../tutorials/custom-effects` - General patterns for creating custom effects
```

### Pattern 2: Dynamic Channel Count

Handle any number of channels dynamically by creating processing chains on the fly:

```python
from torch import Tensor, nn
from torchfx import FX, Wave
from torchfx.filter import iir

class FlexibleMultiChannel(FX):
    """Effect that adapts to any number of channels.

    Creates independent processing chains for each channel,
    with frequency ranges scaled based on channel index.

    Parameters
    ----------
    fs : int, optional
        Sample rate in Hz

    Examples
    --------
    >>> # Works with stereo
    >>> stereo_wave = Wave.from_file("stereo.wav")
    >>> fx = FlexibleMultiChannel(fs=stereo_wave.fs)
    >>> result = stereo_wave | fx

    >>> # Also works with 5.1 surround
    >>> surround_wave = Wave.from_file("surround_51.wav")
    >>> result = surround_wave | fx
    """

    def __init__(self, fs: int | None = None):
        super().__init__()
        self.fs = fs
        self.channels = None  # Created dynamically on first forward pass

    def _create_channels(self, num_channels: int):
        """Create processing chains for given number of channels.

        Each channel gets a bandpass filter with different frequency range.
        """
        self.channels = nn.ModuleList([
            nn.Sequential(
                iir.HiButterworth(cutoff=100 * (i + 1), order=2, fs=self.fs),
                iir.LoButterworth(cutoff=1000 * (i + 1), order=2, fs=self.fs),
            )
            for i in range(num_channels)
        ])

    def forward(self, x: Tensor) -> Tensor:
        """Process audio with dynamic channel adaptation."""
        if self.fs is None:
            raise ValueError("Sample rate (fs) must be set")

        # Determine number of channels
        num_channels = x.shape[0] if x.ndim >= 2 else 1

        # Create channels on first forward pass
        if self.channels is None:
            self._create_channels(num_channels)

        # Process each channel
        if x.ndim >= 2:
            for i in range(num_channels):
                x[i] = self.channels[i](x[i])
        else:
            # Handle mono input
            x = self.channels[0](x)

        return x

# Usage with different channel counts
# Stereo
stereo = Wave.from_file("stereo.wav")
fx_stereo = FlexibleMultiChannel(fs=stereo.fs)
result_stereo = stereo | fx_stereo

# 5.1 Surround
surround = Wave.from_file("surround_51.wav")
fx_surround = FlexibleMultiChannel(fs=surround.fs)
result_surround = surround | fx_surround
```

### Pattern 3: Complete Production Example

Here's a complete, production-ready multi-channel effect based on the TorchFX examples:

```python
import torch
from torch import Tensor, nn
import torchaudio.transforms as T

from torchfx import FX, Wave
from torchfx.filter import HiButterworth, LoButterworth

class ComplexEffect(FX):
    """Multi-channel effect with different processing per channel.

    This effect demonstrates a complete production pattern for
    multi-channel processing with independent channel chains.

    Channel 1: Bandpass 1000-2000 Hz (mid-range focus)
    Channel 2: Bandpass 2000-4000 Hz with 50% volume (presence range)

    Parameters
    ----------
    num_channels : int
        Number of channels to process (typically 2 for stereo)
    fs : int, optional
        Sample rate in Hz. Can be set automatically via Wave pipeline.

    Examples
    --------
    >>> # Basic usage with Wave pipeline
    >>> wave = Wave.from_file("stereo.wav")
    >>> fx = ComplexEffect(num_channels=2, fs=wave.fs)
    >>> processed = wave | fx
    >>> processed.save("output.wav")

    >>> # With GPU acceleration
    >>> wave = Wave.from_file("stereo.wav").to("cuda")
    >>> fx = ComplexEffect(num_channels=2, fs=wave.fs).to("cuda")
    >>> processed = wave | fx
    >>> processed.to("cpu").save("output.wav")
    """

    def __init__(self, num_channels: int, fs: int | None = None):
        super().__init__()
        self.num_channels = num_channels
        self.fs = fs

        # Per-channel processing chains stored in ModuleList
        self.ch = nn.ModuleList([
            self.channel1(),
            self.channel2(),
        ])

    def channel1(self) -> nn.Sequential:
        """Processing chain for channel 1.

        Creates a bandpass filter focusing on mid-range frequencies.
        """
        return nn.Sequential(
            HiButterworth(1000, fs=self.fs),  # High-pass at 1 kHz
            LoButterworth(2000, fs=self.fs),  # Low-pass at 2 kHz
        )

    def channel2(self) -> nn.Sequential:
        """Processing chain for channel 2.

        Creates a bandpass filter focusing on presence range
        with volume reduction.
        """
        return nn.Sequential(
            HiButterworth(2000, fs=self.fs),  # High-pass at 2 kHz
            LoButterworth(4000, fs=self.fs),  # Low-pass at 4 kHz
            T.Vol(0.5),  # Reduce volume by 50%
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply per-channel processing.

        Parameters
        ----------
        x : Tensor
            Input audio with shape (num_channels, time)

        Returns
        -------
        Tensor
            Processed audio with same shape as input

        Raises
        ------
        ValueError
            If sample rate (fs) has not been set
        """
        if self.fs is None:
            raise ValueError("Sampling frequency (fs) must be set")

        # Process each channel independently
        for i in range(self.num_channels):
            x[i] = self.ch[i](x[i])

        return x

# Complete usage example
if __name__ == "__main__":
    # Load stereo audio
    wave = Wave.from_file("input_stereo.wav")
    print(f"Loaded: {wave.ys.shape}, fs={wave.fs}")

    # Create and apply effect
    fx = ComplexEffect(num_channels=2, fs=wave.fs)
    result = wave | fx

    # Save result
    result.save("output_processed.wav")
    print("Processing complete!")
```

```{tip}
Use the {class}`torch.nn.ModuleList` pattern even if all channels have the same processing. This keeps your code flexible and makes it easy to customize individual channels later.
```

## Cross-Channel Processing Patterns

Cross-channel processing enables channels to interact, creating spatial effects and channel-aware processing.

### Built-in Strategy: Ping-Pong Delay

The {class}`~torchfx.effect.PingPongDelayStrategy` creates alternating delays between stereo channels:

```python
import torchfx as fx
from torchfx.effect import Delay, PingPongDelayStrategy

# Load stereo audio
wave = fx.Wave.from_file("stereo.wav")

# Create ping-pong delay effect
delay = Delay(
    bpm=120,
    delay_time="1/8",      # 8th note delay
    feedback=0.5,          # 50% feedback
    mix=0.4,               # 40% wet signal
    strategy=PingPongDelayStrategy()
)

# Apply effect
result = wave | delay
result.save("pingpong_delayed.wav")
```

**Ping-pong delay pattern**:

```{mermaid}
sequenceDiagram
    participant L as Left Channel
    participant R as Right Channel

    Note over L,R: Original Signal
    L->>L: Original left audio
    R->>R: Original right audio

    Note over L,R: Tap 1 (100% amplitude)
    L->>R: Left delays into Right

    Note over L,R: Tap 2 (feedback^1)
    R->>L: Right delays into Left

    Note over L,R: Tap 3 (feedback^2)
    L->>R: Left delays into Right

    Note over L,R: Tap 4 (feedback^3)
    R->>L: Right delays into Left

    Note over L,R: Result: Ping-pong stereo pattern
```

```{note}
If the input is not stereo (not exactly 2 channels), {class}`~torchfx.effect.PingPongDelayStrategy` automatically falls back to {class}`~torchfx.effect.MonoDelayStrategy`.
```

### Custom Cross-Channel Effect: Stereo Widener

Create a custom cross-channel effect using Mid/Side processing:

```python
from torch import Tensor
import torch
from torchfx import FX, Wave

class StereoWidener(FX):
    """Widen stereo image using Mid/Side processing.

    Converts stereo L/R to Mid/Side, scales the Side component,
    then converts back to L/R.

    Parameters
    ----------
    width : float
        Stereo width multiplier:
        - 1.0 = no change (original stereo)
        - >1.0 = wider stereo image
        - <1.0 = narrower stereo image
        - 0.0 = pure mono (no stereo)

    Examples
    --------
    >>> # Widen stereo image by 50%
    >>> wave = Wave.from_file("stereo.wav")
    >>> widener = StereoWidener(width=1.5)
    >>> wider = wave | widener

    >>> # Narrow to 50% stereo width
    >>> narrower = wave | StereoWidener(width=0.5)

    >>> # Convert to mono
    >>> mono = wave | StereoWidener(width=0.0)
    """

    def __init__(self, width: float = 1.5):
        super().__init__()
        if width < 0:
            raise ValueError("Width must be non-negative")
        self.width = width

    def forward(self, x: Tensor) -> Tensor:
        """Apply stereo widening.

        Only processes stereo (2-channel) audio.
        Non-stereo audio is returned unchanged.
        """
        # Only works on stereo audio
        if x.ndim < 2 or x.shape[0] != 2:
            return x  # Return unchanged for non-stereo

        # Extract left and right channels
        left = x[0]
        right = x[1]

        # Convert to Mid/Side
        mid = (left + right) / 2    # Sum of L+R (mono content)
        side = (left - right) / 2   # Difference L-R (stereo width)

        # Widen by scaling Side component
        side_widened = side * self.width

        # Convert back to L/R
        new_left = mid + side_widened
        new_right = mid - side_widened

        return torch.stack([new_left, new_right])

# Usage examples
wave = Wave.from_file("stereo_mix.wav")

# Widen stereo image
widener = StereoWidener(width=1.5)
wider = wave | widener
wider.save("widened_mix.wav")

# Narrow stereo image
narrower = StereoWidener(width=0.5)
narrow = wave | narrower
narrow.save("narrowed_mix.wav")
```

```{mermaid}
graph TB
    Input[Stereo Input<br/>Left, Right]

    subgraph "Mid/Side Conversion"
        Mid["Mid = (L + R) / 2<br/>(Mono content)"]
        Side["Side = (L - R) / 2<br/>(Stereo width)"]
    end

    subgraph "Stereo Widening"
        Scale["Side × width<br/>(Adjust stereo width)"]
    end

    subgraph "L/R Conversion"
        NewL["New Left = Mid + Side×width"]
        NewR["New Right = Mid - Side×width"]
    end

    Output[Stereo Output<br/>Wider/Narrower]

    Input --> Mid
    Input --> Side
    Side --> Scale
    Mid --> NewL
    Mid --> NewR
    Scale --> NewL
    Scale --> NewR
    NewL --> Output
    NewR --> Output

    style Input fill:#e1f5ff
    style Output fill:#e1f5ff
    style Scale fill:#fff5e1
```

## Built-in Multi-Channel Strategies

TorchFX provides several built-in strategies for multi-channel processing.

### Per-Channel Normalization

The {class}`~torchfx.effect.PerChannelNormalizationStrategy` normalizes each channel independently to its own peak value:

```python
import torch
import torchfx as fx
from torchfx.effect import Normalize, PerChannelNormalizationStrategy

# Create stereo with imbalanced channels
left_loud = torch.randn(44100) * 0.8    # Peak ~0.8
right_quiet = torch.randn(44100) * 0.3  # Peak ~0.3
stereo = torch.stack([left_loud, right_quiet])

wave = fx.Wave(stereo, fs=44100)

# Global normalization: both channels scaled by same factor
global_norm = wave | Normalize(peak=1.0)
# Result: left ~1.0, right ~0.375 (preserves balance)

# Per-channel normalization: each channel scaled independently
strategy = PerChannelNormalizationStrategy()
perchannel_norm = wave | Normalize(peak=1.0, strategy=strategy)
# Result: left ~1.0, right ~1.0 (changes balance)
```

**Comparison**:

```{mermaid}
graph TB
    Input["Stereo Input<br/>Left peak: 0.8<br/>Right peak: 0.3"]

    subgraph Global[Global Normalization]
        GMax["Find global max: 0.8"]
        GScale["Scale both by: 1.0/0.8 = 1.25"]
        GResult["Left: 1.0, Right: 0.375"]
    end

    subgraph PerChannel[Per-Channel Normalization]
        PCMax1["Left max: 0.8"]
        PCMax2["Right max: 0.3"]
        PCScale["Scale left: 1.0/0.8<br/>Scale right: 1.0/0.3"]
        PCResult["Left: 1.0, Right: 1.0"]
    end

    Input --> Global
    Input --> PerChannel

    GMax --> GScale --> GResult
    PCMax1 --> PCScale
    PCMax2 --> PCScale
    PCScale --> PCResult

    style Input fill:#e1f5ff
    style Global fill:#fff5e1
    style PerChannel fill:#e8f5e1
```

**Implementation details**:
- For 2D tensors `(channels, time)`: computes max per channel along `dim=1` with `keepdim=True`
- For 3D tensors `(batch, channels, time)`: computes max per channel along `dim=2` with `keepdim=True`
- Uses `keepdim=True` to maintain broadcasting compatibility

### Mono Delay Strategy

{class}`~torchfx.effect.MonoDelayStrategy` applies identical delay processing to all channels independently:

```python
import torchfx as fx
from torchfx.effect import Delay, MonoDelayStrategy

wave = fx.Wave.from_file("stereo.wav")

# Mono delay: same delay pattern on both channels
delay = Delay(
    bpm=120,
    delay_time="1/4",
    feedback=0.4,
    mix=0.3,
    strategy=MonoDelayStrategy()  # Default, can be omitted
)

delayed = wave | delay
```

## Dimension-Agnostic Processing

For effects that should work with any tensor shape, implement dimension detection and handling:

```python
from torch import Tensor
from torchfx import FX

class DimensionAgnosticEffect(FX):
    """Effect that handles 1D, 2D, and 3D+ tensors.

    Automatically detects tensor dimensionality and routes
    to appropriate processing method.

    Examples
    --------
    >>> # Works with mono
    >>> mono = torch.randn(44100)
    >>> fx = DimensionAgnosticEffect()
    >>> result = fx(mono)

    >>> # Works with stereo
    >>> stereo = torch.randn(2, 44100)
    >>> result = fx(stereo)

    >>> # Works with batched multi-channel
    >>> batch = torch.randn(8, 2, 44100)
    >>> result = fx(batch)
    """

    def forward(self, x: Tensor) -> Tensor:
        """Process tensor with automatic dimension detection."""
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
        """Process single channel."""
        # Example: reduce volume by 50%
        return x * 0.5

    def _process_multi_channel(self, x: Tensor) -> Tensor:
        """Process each channel independently."""
        for i in range(x.shape[0]):
            x[i] = self._process_mono(x[i])
        return x

    def _process_batched(self, x: Tensor) -> Tensor:
        """Process batched data."""
        for b in range(x.shape[0]):
            x[b] = self._process_multi_channel(x[b])
        return x
```

```{mermaid}
graph TD
    Forward["forward(waveform)"]

    CheckDim{"waveform.ndim?"}

    Mono["ndim == 1<br/>(time,)<br/>Process as mono"]
    MultiCh["ndim == 2<br/>(channels, time)<br/>Process per channel"]
    Batched["ndim == 3<br/>(batch, channels, time)<br/>Process with batch dim"]
    Higher["ndim > 3<br/>(..., time)<br/>Flatten → process → reshape"]

    Forward --> CheckDim
    CheckDim -->|"1"| Mono
    CheckDim -->|"2"| MultiCh
    CheckDim -->|"3"| Batched
    CheckDim -->|">3"| Higher

    style CheckDim fill:#fff5e1
    style Mono fill:#e1f5ff
    style MultiCh fill:#e8f5e1
    style Batched fill:#f5e1ff
    style Higher fill:#ffe1e1
```

## Surround Sound Configurations

TorchFX supports standard surround sound channel configurations.

### 5.1 Surround Sound

5.1 surround has 6 channels with standard ordering:

```python
import torch
from torchfx import Wave

# 5.1 channel order: FL, FR, FC, LFE, BL, BR
# FL = Front Left, FR = Front Right, FC = Front Center
# LFE = Low Frequency Effects (subwoofer)
# BL = Back Left, BR = Back Right

surround_51 = torch.randn(6, 44100)  # 6 channels, 1 second
wave_51 = Wave(surround_51, fs=44100)

# Access individual channels
front_left = surround_51[0]
front_right = surround_51[1]
center = surround_51[2]
lfe = surround_51[3]
back_left = surround_51[4]
back_right = surround_51[5]
```

### 7.1 Surround Sound

7.1 surround has 8 channels:

```python
# 7.1 channel order: FL, FR, FC, LFE, BL, BR, SL, SR
# SL = Side Left, SR = Side Right

surround_71 = torch.randn(8, 44100)  # 8 channels
wave_71 = Wave(surround_71, fs=44100)
```

### Custom Surround Effect

Process surround sound with channel-specific effects:

```python
from torch import Tensor, nn
from torchfx import FX, Wave
from torchfx.filter import iir
import torchaudio.transforms as T

class SurroundProcessor(FX):
    """Process 5.1 surround sound with channel-specific effects.

    - Front channels: Full range processing
    - Center: Voice-optimized (bandpass)
    - LFE: Low-pass only (subwoofer)
    - Rear channels: Ambient processing

    Parameters
    ----------
    fs : int
        Sample rate in Hz
    """

    def __init__(self, fs: int):
        super().__init__()
        self.fs = fs

        # Channel order: FL, FR, FC, LFE, BL, BR
        self.channels = nn.ModuleList([
            self.front_lr(),    # 0: Front Left
            self.front_lr(),    # 1: Front Right
            self.center(),      # 2: Front Center
            self.lfe(),         # 3: LFE (subwoofer)
            self.rear(),        # 4: Back Left
            self.rear(),        # 5: Back Right
        ])

    def front_lr(self) -> nn.Sequential:
        """Full-range processing for front L/R."""
        return nn.Sequential(
            iir.HiButterworth(cutoff=80, order=2, fs=self.fs),
            iir.LoButterworth(cutoff=18000, order=4, fs=self.fs),
        )

    def center(self) -> nn.Sequential:
        """Voice-optimized for center channel."""
        return nn.Sequential(
            iir.HiButterworth(cutoff=200, order=2, fs=self.fs),   # Remove rumble
            iir.LoButterworth(cutoff=8000, order=4, fs=self.fs),  # Voice range
        )

    def lfe(self) -> nn.Sequential:
        """Subwoofer channel (low-pass only)."""
        return nn.Sequential(
            iir.LoButterworth(cutoff=120, order=8, fs=self.fs),   # Sharp LPF
        )

    def rear(self) -> nn.Sequential:
        """Ambient processing for rear channels."""
        return nn.Sequential(
            iir.HiButterworth(cutoff=100, order=2, fs=self.fs),
            iir.LoButterworth(cutoff=12000, order=4, fs=self.fs),
            T.Vol(0.8),  # Slightly quieter for ambience
        )

    def forward(self, x: Tensor) -> Tensor:
        """Process 5.1 surround audio."""
        if x.shape[0] != 6:
            raise ValueError(f"Expected 6 channels for 5.1, got {x.shape[0]}")

        for i in range(6):
            x[i] = self.channels[i](x[i])

        return x

# Usage
surround_wave = Wave.from_file("movie_51.wav")
processor = SurroundProcessor(fs=surround_wave.fs)
processed = surround_wave | processor
processed.save("processed_51.wav")
```

## Integration with Wave Pipeline

The {class}`~torchfx.Wave` class automatically configures effects with the `fs` (sample rate) parameter when using the pipeline operator (`|`).

### Automatic Sample Rate Configuration

```python
# Create effect without fs
fx = ComplexEffect(num_channels=2, fs=None)

# Wave automatically sets fs via __update_config
wave = Wave.from_file("stereo.wav")
result = wave | fx  # fx.fs is automatically set to wave.fs
```

The {class}`~torchfx.Wave` class calls `__update_config` on effects that have an `fs` attribute, automatically setting the sample rate before processing.

```{seealso}
{doc}`../core-concepts/wave` - Wave class architecture and automatic configuration
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

Using {class}`torch.nn.ModuleList` ensures:
- Proper parameter registration for GPU transfer
- Correct gradient tracking for trainable parameters
- Integration with PyTorch's module system

### Validate Sample Rate

```python
# ✅ GOOD: Clear error messages
def forward(self, x: Tensor) -> Tensor:
    if self.fs is None:
        raise ValueError("Sample rate (fs) must be set before processing")
    # Process audio...

# ❌ BAD: No validation, may fail later
def forward(self, x: Tensor) -> Tensor:
    # self.fs might be None!
    return self.filter(x)
```

### Handle Variable Channel Counts

```python
# ✅ GOOD: Flexible channel handling
def forward(self, x: Tensor) -> Tensor:
    num_channels = x.shape[0] if x.ndim >= 2 else 1

    if self.channels is None or len(self.channels) != num_channels:
        self._create_channels(num_channels)

    # Process channels...

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

### Use keepdim for Broadcasting

```python
# ✅ CORRECT: keepdim=True allows broadcasting
max_per_channel = torch.max(torch.abs(waveform), dim=1, keepdim=True).values
normalized = waveform / max_per_channel * peak

# ❌ WRONG: Shape mismatch
max_per_channel = torch.max(torch.abs(waveform), dim=1).values
normalized = waveform / max_per_channel * peak  # Broadcasting error!
```

## Common Pitfalls

### Pitfall 1: In-Place Modifications

```python
# ❌ WRONG: In-place modification can cause issues
def forward(self, x: Tensor) -> Tensor:
    for i in range(x.shape[0]):
        x[i] = self.process(x[i])  # Modifies input
    return x

# ✅ CORRECT: Create output tensor
def forward(self, x: Tensor) -> Tensor:
    output = torch.zeros_like(x)
    for i in range(x.shape[0]):
        output[i] = self.process(x[i])
    return output
```

### Pitfall 2: Incorrect Channel Validation

```python
# ❌ WRONG: Doesn't handle batched input
if x.shape[0] != 2:
    raise ValueError("Expected stereo")

# ✅ CORRECT: Check channel dimension properly
if x.ndim < 2 or x.shape[-2] != 2:
    raise ValueError(f"Expected stereo, got shape {x.shape}")
```

### Pitfall 3: Forgetting Output Length

```python
# ❌ WRONG: May truncate delay tails
def forward(self, x: Tensor) -> Tensor:
    output = torch.zeros_like(x)  # Same length as input
    # Process with delay... output might be too short!

# ✅ CORRECT: Account for extended output
def forward(self, x: Tensor) -> Tensor:
    output_length = x.size(-1) + self.delay_samples
    output = torch.zeros(x.size(0), output_length, ...)
    # Process with proper length
```

## PyTorch Integration

Multi-channel effects work seamlessly with PyTorch's data pipeline.

### Using with DataLoader

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
    file_paths=["song1.wav", "song2.wav", "song3.wav"],
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
    fx = ComplexEffect(num_channels=2, fs=wave.fs).to("cuda")
    processed = wave | fx

    # Move back to CPU for saving
    processed.to("cpu").save("output.wav")
```

```{seealso}
{doc}`gpu-acceleration` - Complete guide to GPU acceleration in TorchFX
```

## Summary

Multi-channel processing in TorchFX follows these key principles:

| Principle | Implementation |
|-----------|----------------|
| **Tensor conventions** | Last dimension is time, earlier dimensions are channels/batches |
| **Default behavior** | Broadcast operations unless channel-specific logic is implemented |
| **Strategy pattern** | Use strategy classes for pluggable channel behavior |
| **ModuleList** | Use {class}`torch.nn.ModuleList` for per-channel processing chains |
| **Dimension handling** | Detect and handle 1D, 2D, 3D+ tensors appropriately |
| **Cross-channel effects** | Explicitly access channels via indexing for interaction patterns |
| **Sample rate** | Store `fs` attribute for automatic configuration via {class}`~torchfx.Wave` pipeline |

**Architecture Overview**:

```{mermaid}
graph TB
    subgraph "Multi-Channel Architecture"
        Input[Input Audio<br/>channels, time]

        subgraph FX[FX Base Class]
            Init[__init__<br/>Set fs, num_channels]
            Chains[nn.ModuleList<br/>Per-channel chains]
            Forward[forward<br/>Process channels]
        end

        subgraph Strategies
            Broadcast[Broadcast Strategy<br/>Same to all]
            PerChannel[Per-Channel Strategy<br/>Independent]
            CrossChannel[Cross-Channel Strategy<br/>Interactive]
        end

        Output[Output Audio<br/>channels, time]
    end

    Input --> FX
    Init --> Chains
    Chains --> Forward
    Forward --> Strategies
    Strategies --> Output

    style Input fill:#e1f5ff
    style Output fill:#e1f5ff
    style FX fill:#fff5e1
    style Strategies fill:#e8f5e1
```

## Related Concepts

- {doc}`../core-concepts/wave` - Wave class and tensor handling
- {doc}`../core-concepts/fx` - FX base class for effects
- {doc}`../tutorials/custom-effects` - Creating custom effects
- {doc}`gpu-acceleration` - GPU acceleration for multi-channel audio
- {doc}`../tutorials/ml-batch-processing` - Batch processing for ML workflows

## External Resources

- [PyTorch Audio Documentation](https://pytorch.org/audio/stable/index.html) - torchaudio tensor conventions
- [Surround Sound on Wikipedia](https://en.wikipedia.org/wiki/Surround_sound) - Multi-channel audio formats
- [Mid/Side Processing](https://en.wikipedia.org/wiki/Stereophonic_sound#M/S_technique:_mid/side_stereophony) - Stereo imaging technique
- [ITU-R BS.775](https://www.itu.int/rec/R-REC-BS.775/) - Multi-channel audio standard

## References

```{bibliography}
:filter: docname in docnames
:style: alpha
```
