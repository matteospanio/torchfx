(pytorch-integration)=
# PyTorch Integration

Learn how TorchFX seamlessly integrates with PyTorch's neural network ecosystem. This guide covers using TorchFX modules with {class}`torch.nn.Sequential`, creating custom modules, gradient computation, and mixing with torchaudio transforms.

## Prerequisites

Before starting this tutorial, you should be familiar with:

- {doc}`../core-concepts/wave` - Wave class fundamentals
- {doc}`../core-concepts/pipeline-operator` - Pipeline operator basics
- {doc}`../core-concepts/fx` - FX base class architecture
- [PyTorch nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) - Module fundamentals
- Basic PyTorch concepts (tensors, device management, forward passes)

## Overview

All TorchFX audio effects and filters are implemented as subclasses of {class}`torch.nn.Module`, making them fully compatible with PyTorch's neural network ecosystem. This design enables:

- **Seamless composition** with other PyTorch modules
- **Integration** into neural network training pipelines
- **Automatic gradient computation** for differentiable operations
- **Device management** through PyTorch's standard API
- **Compatibility** with PyTorch utilities like {class}`torch.nn.Sequential` and {class}`torch.nn.ModuleList`

```{tip}
TorchFX modules behave exactly like standard PyTorch modules, so you can use them anywhere you'd use a PyTorch layer or transform.
```

## Core Integration Architecture

TorchFX's integration with PyTorch is built on inheritance from {class}`torch.nn.Module`. Understanding this architecture helps you leverage the full power of both libraries.

```{mermaid}
graph TB
    subgraph "PyTorch Framework"
        Module["torch.nn.Module<br/>Base class for all neural network modules"]
    end

    subgraph "torchfx Core"
        FX["FX (Abstract Base)<br/>Inherits from torch.nn.Module<br/>src/torchfx/effect.py"]

        subgraph "Effects"
            Gain["Gain"]
            Normalize["Normalize"]
            Reverb["Reverb"]
            Delay["Delay"]
        end

        subgraph "Filters"
            AbstractFilter["AbstractFilter"]
            IIR["IIR Filters<br/>(Butterworth, Chebyshev)"]
            FIR["FIR Filters<br/>(DesignableFIR)"]
        end

        Wave["Wave<br/>Audio data container<br/>Implements pipe operator"]
    end

    subgraph "PyTorch Containers"
        Sequential["nn.Sequential<br/>Sequential composition"]
        ModuleList["nn.ModuleList<br/>Module container"]
        Custom["Custom nn.Module<br/>User-defined classes"]
    end

    Module -->|"inherits"| FX
    FX -->|"base for"| Gain
    FX -->|"base for"| Normalize
    FX -->|"base for"| Reverb
    FX -->|"base for"| Delay
    FX -->|"base for"| AbstractFilter
    AbstractFilter -->|"base for"| IIR
    AbstractFilter -->|"base for"| FIR

    Wave -->|"pipe operator accepts"| Module
    Wave -->|"works with"| Sequential
    Wave -->|"works with"| ModuleList
    Wave -->|"works with"| Custom

    Custom -->|"can contain"| FX
    Sequential -->|"can contain"| FX
    ModuleList -->|"can contain"| FX

    style Module fill:#e1f5ff
    style FX fill:#fff5e1
    style Wave fill:#e1ffe1
```

**TorchFX Module Hierarchy** - All TorchFX effects and filters inherit from {class}`torch.nn.Module`, enabling seamless integration with the PyTorch ecosystem.

## Module Inheritance Benefits

Because TorchFX modules inherit from {class}`torch.nn.Module`, they automatically gain all PyTorch module capabilities:

| Feature | Benefit | Example |
|---------|---------|---------|
| **Parameter Registration** | Filter coefficients tracked by PyTorch | `list(module.parameters())` |
| **Device Management** | `.to(device)` moves all tensors | `filter.to("cuda")` |
| **State Dict** | Serialization and deserialization | `torch.save(module.state_dict(), "model.pt")` |
| **Training Mode** | `.train()` and `.eval()` support | `module.eval()` |
| **Nested Modules** | Automatic recursive operations | `module.to("cuda")` moves all children |
| **Hooks** | Register forward/backward hooks | `module.register_forward_hook(hook_fn)` |

```{seealso}
{doc}`gpu-acceleration` - Using device management for GPU acceleration
```

## Wave Pipe Operator with nn.Module

The {class}`~torchfx.Wave` class implements the {term}`pipeline operator` (`|`) to accept **any** {class}`torch.nn.Module`, not just TorchFX-specific effects and filters. This design choice enables integration with the entire PyTorch ecosystem.

### Implementation Details

The pipe operator performs the following steps:

```{mermaid}
sequenceDiagram
    participant User
    participant Wave
    participant Module as "nn.Module"
    participant FX as "FX Instance?"

    User->>Wave: wave | module
    Wave->>Wave: __or__(module)

    Wave->>Module: Check isinstance(module, nn.Module)

    alt Not an nn.Module
        Wave-->>User: TypeError
    end

    Wave->>FX: Check isinstance(module, FX)

    alt Is FX instance
        Wave->>Module: Set module.fs = wave.fs
        Wave->>Module: Call compute_coefficients()

        alt Is Sequential/ModuleList
            Wave->>Module: Recursively configure FX children
        end
    end

    Wave->>Module: Call module.forward(wave.ys)
    Module-->>Wave: Return processed tensor
    Wave->>Wave: Create new Wave(result, fs)
    Wave-->>User: Return new Wave
```

**Pipeline Operator Flow** - Shows how the pipe operator processes any {class}`torch.nn.Module`.

### Type Validation and Configuration

| Step | Action | Code Reference |
|------|--------|----------------|
| 1. Type validation | Checks if right operand is {class}`torch.nn.Module` | `src/torchfx/wave.py:163-164` |
| 2. FX configuration | Updates `fs` and computes coefficients for FX instances | `src/torchfx/wave.py:166-172` |
| 3. Sequential handling | Recursively configures FX instances in Sequential/ModuleList | `src/torchfx/wave.py:169-172` |
| 4. Forward pass | Applies module's `forward` method to audio tensor | `src/torchfx/wave.py:174` |

### Usage Patterns

The pipe operator works with any module that implements a `forward` method accepting and returning tensors:

```python
import torch
import torch.nn as nn
import torchfx as fx

wave = fx.Wave.from_file("audio.wav")

# Works with TorchFX filters
filtered = wave | fx.filter.LoButterworth(1000)

# Works with torch.nn.Sequential
chained = wave | nn.Sequential(
    fx.filter.HiButterworth(100, order=2),
    fx.filter.LoButterworth(5000, order=4)
)

# Works with any custom nn.Module
class CustomGain(nn.Module):
    def __init__(self, gain: float):
        super().__init__()
        self.gain = gain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gain

custom = wave | CustomGain(gain=0.5)
```

```{note}
The module's `forward` method should accept a tensor of shape `(channels, time)` and return a tensor of the same or compatible shape.
```

## Using nn.Sequential

TorchFX filters and effects can be composed using {class}`torch.nn.Sequential`, providing an alternative to the pipeline operator for creating processing chains.

### Sequential Composition Pattern

{class}`torch.nn.Sequential` creates a container that applies modules in order:

```{mermaid}
graph LR
    Input["Input Tensor<br/>(C, T)"]

    Sequential["nn.Sequential"]

    subgraph "Sequential Container"
        F1["HiChebyshev1(20)"]
        F2["HiChebyshev1(60)"]
        F3["HiChebyshev1(65)"]
        F4["LoButterworth(5000)"]
        F5["LoButterworth(4900)"]
        F6["LoButterworth(4850)"]
    end

    Output["Output Tensor<br/>(C, T)"]

    Input --> Sequential
    Sequential --> F1
    F1 --> F2
    F2 --> F3
    F3 --> F4
    F4 --> F5
    F5 --> F6
    F6 --> Output

    style Input fill:#e1f5ff
    style Output fill:#e1f5ff
    style F1 fill:#fff5e1
    style F2 fill:#fff5e1
    style F3 fill:#fff5e1
    style F4 fill:#fff5e1
    style F5 fill:#fff5e1
    style F6 fill:#fff5e1
```

**Sequential Processing Flow** - Audio flows through each module in the container sequentially.

### Three Equivalent Approaches

TorchFX provides three equivalent ways to chain filters, each with different trade-offs:

```python
import torch.nn as nn
import torchfx as fx
from torchfx.filter import HiChebyshev1, LoButterworth

wave = fx.Wave.from_file("audio.wav")

# Approach 1: Custom Module
class FilterChain(nn.Module):
    def __init__(self, fs):
        super().__init__()
        self.f1 = HiChebyshev1(20, fs=fs)
        self.f2 = HiChebyshev1(60, fs=fs)
        self.f3 = HiChebyshev1(65, fs=fs)
        self.f4 = LoButterworth(5000, fs=fs)
        self.f5 = LoButterworth(4900, fs=fs)
        self.f6 = LoButterworth(4850, fs=fs)

    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        x = self.f5(x)
        x = self.f6(x)
        return x

custom_chain = FilterChain(wave.fs)
result1 = wave | custom_chain

# Approach 2: nn.Sequential
seq_chain = nn.Sequential(
    HiChebyshev1(20, fs=wave.fs),
    HiChebyshev1(60, fs=wave.fs),
    HiChebyshev1(65, fs=wave.fs),
    LoButterworth(5000, fs=wave.fs),
    LoButterworth(4900, fs=wave.fs),
    LoButterworth(4850, fs=wave.fs),
)
result2 = wave | seq_chain

# Approach 3: Pipe Operator
result3 = (
    wave
    | HiChebyshev1(20)
    | HiChebyshev1(60)
    | HiChebyshev1(65)
    | LoButterworth(5000)
    | LoButterworth(4900)
    | LoButterworth(4850)
)
```

### Comparison of Approaches

| Approach | Pros | Cons | Best For |
|----------|------|------|----------|
| **Custom Module** | Named attributes, can add logic, best for reuse | More verbose | Reusable components, complex logic |
| **nn.Sequential** | Standard PyTorch pattern, works with PyTorch tools | Must specify `fs` manually | PyTorch integration, model composition |
| **Pipe Operator** | Most concise, auto `fs` configuration | Less familiar to PyTorch users | Quick prototyping, scripts |

```{tip}
Use the pipe operator for exploration and scripts. Use {class}`torch.nn.Sequential` or custom modules when building larger systems or when you need to integrate with PyTorch training pipelines.
```

## Creating Custom Neural Network Modules

TorchFX filters and effects can be embedded in custom {class}`torch.nn.Module` classes to create reusable processing blocks with custom logic.

### Pattern 1: Audio Preprocessing Module

Create a preprocessing module for machine learning pipelines:

```python
import torch
import torch.nn as nn
import torchfx as fx

class AudioPreprocessor(nn.Module):
    """Preprocessing module for audio classification."""

    def __init__(self, sample_rate: int = 44100):
        super().__init__()
        # Filtering layers
        self.rumble_filter = fx.filter.HiButterworth(cutoff=80, order=2, fs=sample_rate)
        self.noise_filter = fx.filter.LoButterworth(cutoff=12000, order=4, fs=sample_rate)

        # Normalization
        self.normalizer = fx.effect.Normalize(peak=0.9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Audio tensor of shape (batch, channels, time)

        Returns
        -------
        Tensor
            Preprocessed audio of shape (batch, channels, time)
        """
        # Process each sample in batch
        batch_size = x.size(0)
        results = []

        for i in range(batch_size):
            # Extract single sample
            sample = x[i]  # (channels, time)

            # Apply filters
            sample = self.rumble_filter(sample)
            sample = self.noise_filter(sample)
            sample = self.normalizer(sample)

            results.append(sample)

        return torch.stack(results)

# Usage
preprocessor = AudioPreprocessor(sample_rate=44100)
preprocessor.to("cuda")  # Move to GPU

# In training loop
audio_batch = torch.randn(32, 2, 44100).to("cuda")  # (batch, channels, time)
preprocessed = preprocessor(audio_batch)
```

### Pattern 2: Multi-Stage Effects Chain

Create a reusable effects chain module:

```python
import torch.nn as nn
import torchfx as fx

class VocalProcessor(nn.Module):
    """Professional vocal processing chain."""

    def __init__(self, sample_rate: int = 44100):
        super().__init__()
        # Stage 1: Cleanup
        self.rumble_removal = fx.filter.HiButterworth(cutoff=80, order=2, fs=sample_rate)
        self.air_filter = fx.filter.LoButterworth(cutoff=15000, order=4, fs=sample_rate)

        # Stage 2: Tonal shaping (using Sequential)
        self.eq_chain = nn.Sequential(
            fx.filter.PeakingEQ(freq=200, gain_db=-2, q=0.7, fs=sample_rate),    # Reduce mud
            fx.filter.PeakingEQ(freq=3000, gain_db=3, q=1.0, fs=sample_rate),    # Presence
            fx.filter.PeakingEQ(freq=10000, gain_db=2, q=0.7, fs=sample_rate),   # Brightness
        )

        # Stage 3: Dynamics and final polish
        self.compressor = fx.effect.Compressor(threshold=0.5, ratio=4.0)
        self.limiter = fx.effect.Normalize(peak=0.95)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stage 1
        x = self.rumble_removal(x)
        x = self.air_filter(x)

        # Stage 2
        x = self.eq_chain(x)

        # Stage 3
        x = self.compressor(x)
        x = self.limiter(x)

        return x

# Usage with Wave
wave = fx.Wave.from_file("vocal.wav")
processor = VocalProcessor(wave.fs)

processed = wave | processor
processed.save("processed_vocal.wav")
```

### Pattern 3: Parameterized Filter Bank

Create a module with learnable or configurable parameters:

```python
import torch
import torch.nn as nn
import torchfx as fx

class MultiFrequencyFilter(nn.Module):
    """Parallel filter bank with multiple cutoff frequencies."""

    def __init__(self, cutoff_freqs: list[float], sample_rate: int = 44100):
        super().__init__()
        self.filters = nn.ModuleList([
            fx.filter.LoButterworth(cutoff=freq, order=4, fs=sample_rate)
            for freq in cutoff_freqs
        ])

        # Learnable weights for each filter (optional)
        self.weights = nn.Parameter(torch.ones(len(cutoff_freqs)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply all filters and combine with learned weights
        outputs = []
        for i, filter in enumerate(self.filters):
            filtered = filter(x)
            weighted = filtered * self.weights[i]
            outputs.append(weighted)

        # Sum all weighted outputs
        return sum(outputs)

# Usage
filter_bank = MultiFrequencyFilter(
    cutoff_freqs=[500, 1000, 2000, 4000],
    sample_rate=44100
)

wave = fx.Wave.from_file("audio.wav")
result = wave | filter_bank
```

### Key Module Characteristics

| Property | Behavior |
|----------|----------|
| **Module registration** | Filters assigned as attributes are automatically registered by PyTorch |
| **Parameter tracking** | Filter coefficients become part of the module's parameters |
| **Device management** | Calling `.to(device)` on the parent module moves all child filters |
| **State dict** | Filter states are included in `state_dict()` for serialization |
| **Nested modules** | Modules can contain other modules arbitrarily deep |

```{seealso}
[PyTorch Module Documentation](https://pytorch.org/docs/stable/notes/modules.html) - Official guide to building custom modules
```

## Gradient Computation and Differentiability

TorchFX operations maintain gradient flow where applicable, enabling their use in differentiable audio processing and neural network training.

### Differentiability Status

```{mermaid}
graph TD
    Operations["TorchFX Operations"]

    subgraph "Differentiable Operations"
        Filters["Filter Forward Pass<br/>IIR and FIR convolutions<br/>Gradients flow through"]
        Gain["Gain Effect<br/>Amplitude scaling<br/>Gradients flow through"]
        Transform["Wave.transform<br/>Functional operations<br/>Conditional on function"]
    end

    subgraph "Non-Differentiable Operations"
        FileIO["File I/O<br/>from_file, save<br/>No gradients"]
        CoeffComp["Coefficient Computation<br/>compute_coefficients()<br/>Uses SciPy, no gradients"]
        Design["Filter Design<br/>Cutoff/order to coefficients<br/>No gradients"]
    end

    Operations --> Filters
    Operations --> Gain
    Operations --> Transform
    Operations --> FileIO
    Operations --> CoeffComp
    Operations --> Design

    Filters -->|"backward()"| Backprop["Backpropagation"]
    Gain -->|"backward()"| Backprop
    Transform -->|"if func differentiable"| Backprop

    style Filters fill:#e1ffe1
    style Gain fill:#e1ffe1
    style Transform fill:#e1ffe1
    style FileIO fill:#ffe1e1
    style CoeffComp fill:#ffe1e1
    style Design fill:#ffe1e1
```

**Differentiability Map** - Shows which TorchFX operations support gradient computation.

### Gradient Flow Through Processing Chain

When using TorchFX modules in a training pipeline:

1. **Forward Pass**: Audio tensors flow through filter convolutions and effects
2. **Backward Pass**: Gradients flow back through differentiable operations
3. **Parameter Updates**: Upstream parameters receive gradients; filter coefficients remain fixed

```{mermaid}
sequenceDiagram
    participant Upstream as "Upstream Layer<br/>(learnable)"
    participant Filter as "TorchFX Filter<br/>(fixed coefficients)"
    participant Loss as "Loss Function"

    Note over Upstream,Loss: Forward Pass
    Upstream->>Filter: x (requires_grad=True)
    Filter->>Filter: Apply filter convolution
    Filter->>Loss: filtered_x (grad_fn=<...>)
    Loss->>Loss: Compute loss

    Note over Upstream,Loss: Backward Pass
    Loss->>Loss: loss.backward()
    Loss->>Filter: Gradients for filtered_x
    Filter->>Filter: Compute gradients wrt input
    Filter->>Upstream: Gradients for x
    Upstream->>Upstream: Update parameters

    Note over Filter: Filter coefficients<br/>do NOT receive gradients
```

**Gradient Flow in Training** - Gradients flow through TorchFX filters but don't update filter coefficients.

### Important Notes on Gradients

```{warning}
**Filter coefficients are NOT learnable parameters.** Filter coefficients are computed from design parameters (cutoff frequency, order) using non-differentiable SciPy functions. The coefficients themselves do not receive gradients during backpropagation.

If you need learnable filtering, consider using learnable FIR filters where the filter taps are {class}`torch.nn.Parameter` objects.
```

### Example: Differentiable Audio Augmentation

```python
import torch
import torch.nn as nn
import torchfx as fx

class AudioClassifier(nn.Module):
    """Example classifier with TorchFX augmentation."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Fixed augmentation filters
        self.augment = nn.Sequential(
            fx.filter.HiButterworth(cutoff=80, order=2, fs=44100),
            fx.filter.LoButterworth(cutoff=12000, order=4, fs=44100),
        )

        # Learnable classification layers
        self.conv1 = nn.Conv1d(2, 64, kernel_size=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time)

        # Augmentation (gradients flow through)
        batch_results = []
        for i in range(x.size(0)):
            augmented = self.augment(x[i])
            batch_results.append(augmented)
        x = torch.stack(batch_results)

        # Classification (learnable)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = x.mean(dim=-1)  # Global average pooling
        x = self.fc(x)

        return x

# Training example
model = AudioClassifier(num_classes=10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Forward pass
audio_batch = torch.randn(32, 2, 44100, requires_grad=True)
labels = torch.randint(0, 10, (32,))

# Compute loss
outputs = model(audio_batch)
loss = nn.CrossEntropyLoss()(outputs, labels)

# Backward pass (gradients flow through filters to conv layers)
loss.backward()

# Update ONLY the learnable parameters (conv1, conv2, fc)
optimizer.step()

# Filter coefficients remain unchanged
```

```{tip}
TorchFX filters work great as **fixed augmentation layers** in neural network pipelines, providing consistent audio preprocessing that gradients can flow through.
```

## Mixing with torchaudio.transforms

TorchFX modules can be mixed with [torchaudio](https://pytorch.org/audio/) transforms in the same processing pipeline, leveraging the best of both libraries.

### Integration Pattern

```python
import torch
import torchfx as fx
import torchaudio.transforms as T

wave = fx.Wave.from_file("audio.wav")

# Mix TorchFX filters with torchaudio transforms
processed = (
    wave
    | fx.filter.LoButterworth(100, order=2)                    # TorchFX filter
    | fx.filter.HiButterworth(2000, order=2)                   # TorchFX filter
    | T.Vol(gain=0.5)                                          # torchaudio volume
    | fx.effect.Normalize(peak=0.9)                            # TorchFX effect
)

processed.save("mixed_processing.wav")
```

### Compatible torchaudio Transforms

| Transform | Use Case | Integration |
|-----------|----------|-------------|
| `T.Vol` | Volume adjustment | Direct pipe operator |
| `T.Resample` | Sample rate conversion | Direct pipe operator |
| `T.Fade` | Fade in/out | Direct pipe operator |
| `T.FrequencyMasking` | Spectrogram augmentation | Requires spectrogram conversion |
| `T.TimeMasking` | Spectrogram augmentation | Requires spectrogram conversion |
| `T.MelScale` | Mel-frequency processing | Requires spectrogram conversion |

### Example: Complete Audio Pipeline

```python
import torch
import torch.nn as nn
import torchfx as fx
import torchaudio.transforms as T

class AudioPipeline(nn.Module):
    """Complete audio processing pipeline mixing TorchFX and torchaudio."""

    def __init__(self, sample_rate: int = 44100):
        super().__init__()
        # TorchFX preprocessing
        self.lowpass = fx.filter.LoButterworth(cutoff=100, order=2, fs=sample_rate)

        # torchaudio transforms
        self.resample = T.Resample(orig_freq=sample_rate, new_freq=16000)
        self.volume = T.Vol(gain=0.8, gain_type="amplitude")

        # TorchFX final processing
        self.normalize = fx.effect.Normalize(peak=0.95)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lowpass(x)
        x = self.resample(x)
        x = self.volume(x)
        x = self.normalize(x)
        return x

# Usage
pipeline = AudioPipeline(sample_rate=44100)
wave = fx.Wave.from_file("audio.wav")

# Apply pipeline
result = wave | pipeline
```

### Module Compatibility Requirements

Any {class}`torch.nn.Module` is compatible with TorchFX's pipe operator if it:

1. **Implements** a `forward` method
2. **Accepts** a tensor as input with shape `(channels, time)` for audio
3. **Returns** a tensor of the same or compatible shape
4. **Operates** on the same device as the input tensor

```{note}
Some torchaudio transforms expect mono audio or specific channel configurations. Check the transform documentation and adjust channel count if needed using {meth}`~torchfx.Wave.to_mono()` or other methods.
```

## Complete Working Examples

### Example 1: Audio Classification with TorchFX Augmentation

```python
import torch
import torch.nn as nn
import torchfx as fx
from torch.utils.data import Dataset, DataLoader

class AudioDataset(Dataset):
    """Simple audio dataset."""

    def __init__(self, file_paths: list[str], labels: list[int]):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        wave = fx.Wave.from_file(self.file_paths[idx])
        # Ensure fixed length (e.g., 3 seconds at 44100 Hz)
        target_length = 3 * 44100
        if wave.ys.size(-1) > target_length:
            wave.ys = wave.ys[..., :target_length]
        else:
            # Pad if too short
            padding = target_length - wave.ys.size(-1)
            wave.ys = torch.nn.functional.pad(wave.ys, (0, padding))

        return wave.ys, self.labels[idx]

class AudioClassifierWithAugmentation(nn.Module):
    """Classifier with TorchFX augmentation."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Data augmentation (fixed filters)
        self.augmentation = nn.Sequential(
            fx.filter.HiButterworth(cutoff=80, order=2, fs=44100),
            fx.filter.LoButterworth(cutoff=15000, order=4, fs=44100),
            fx.effect.Normalize(peak=0.9),
        )

        # Feature extraction
        self.conv1 = nn.Conv1d(2, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Classification
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time)

        # Apply augmentation to each sample
        batch_size = x.size(0)
        augmented = []
        for i in range(batch_size):
            aug = self.augmentation(x[i])
            augmented.append(aug)
        x = torch.stack(augmented)

        # Feature extraction
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)

        # Classification
        x = self.fc(x)
        return x

# Training setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create dataset (example)
file_paths = ["audio1.wav", "audio2.wav", "audio3.wav"]
labels = [0, 1, 2]
dataset = AudioDataset(file_paths, labels)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Initialize model
model = AudioClassifierWithAugmentation(num_classes=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop (one epoch)
model.train()
for batch_audio, batch_labels in dataloader:
    batch_audio = batch_audio.to(device)
    batch_labels = batch_labels.to(device)

    # Forward pass
    outputs = model(batch_audio)
    loss = criterion(outputs, batch_labels)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Loss: {loss.item():.4f}")
```

### Example 2: Custom Effect Module with State

```python
import torch
import torch.nn as nn
import torchfx as fx

class StatefulReverb(nn.Module):
    """Custom reverb with learnable parameters."""

    def __init__(self, max_delay: int = 44100):
        super().__init__()
        # Learnable delay time (in samples)
        self.delay_time = nn.Parameter(torch.tensor(22050.0))

        # Learnable decay factor
        self.decay = nn.Parameter(torch.tensor(0.5))

        self.max_delay = max_delay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Clamp delay to valid range
        delay = torch.clamp(self.delay_time, 0, self.max_delay).long()

        # Simple delay-based reverb
        output = x.clone()
        if delay > 0:
            # Add delayed signal with decay
            padded = torch.nn.functional.pad(x, (delay, 0))
            delayed = padded[..., :-delay] if delay > 0 else padded
            output = x + delayed * self.decay

        return output

class ProcessorWithLearnableReverb(nn.Module):
    """Combines fixed filters with learnable reverb."""

    def __init__(self, sample_rate: int = 44100):
        super().__init__()
        # Fixed preprocessing
        self.preprocess = nn.Sequential(
            fx.filter.HiButterworth(cutoff=80, order=2, fs=sample_rate),
            fx.filter.LoButterworth(cutoff=12000, order=4, fs=sample_rate),
        )

        # Learnable reverb
        self.reverb = StatefulReverb(max_delay=sample_rate)

        # Fixed normalization
        self.normalize = fx.effect.Normalize(peak=0.9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fixed filtering
        x = self.preprocess(x)

        # Learnable reverb
        x = self.reverb(x)

        # Fixed normalization
        x = self.normalize(x)

        return x

# Usage
processor = ProcessorWithLearnableReverb(sample_rate=44100)
wave = fx.Wave.from_file("audio.wav")

# Process (can be used in training loop)
processed = wave | processor

# Access learnable parameters
print(f"Delay time: {processor.reverb.delay_time.item():.0f} samples")
print(f"Decay factor: {processor.reverb.decay.item():.3f}")
```

### Example 3: Mixing TorchFX with torchaudio in Sequential

```python
import torch.nn as nn
import torchfx as fx
import torchaudio.transforms as T

class HybridAudioProcessor(nn.Module):
    """Processor combining TorchFX and torchaudio transforms."""

    def __init__(self, sample_rate: int = 44100, target_sr: int = 16000):
        super().__init__()
        self.pipeline = nn.Sequential(
            # TorchFX: Remove low-frequency rumble
            fx.filter.HiButterworth(cutoff=80, order=2, fs=sample_rate),

            # TorchFX: Remove high-frequency noise
            fx.filter.LoButterworth(cutoff=15000, order=4, fs=sample_rate),

            # torchaudio: Resample to lower rate
            T.Resample(orig_freq=sample_rate, new_freq=target_sr),

            # torchaudio: Adjust volume
            T.Vol(gain=0.8, gain_type="amplitude"),

            # TorchFX: Final normalization
            fx.effect.Normalize(peak=0.95),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pipeline(x)

# Usage
processor = HybridAudioProcessor(sample_rate=44100, target_sr=16000)
wave = fx.Wave.from_file("audio.wav")

# Process
result = wave | processor

# Note: Output has different sample rate!
# You need to create a new Wave with the correct fs
result_wave = fx.Wave(result.ys if isinstance(result, fx.Wave) else processor(wave.ys), fs=16000)
result_wave.save("processed.wav")
```

## Architecture Diagrams

### TorchFX in a Training Pipeline

```{mermaid}
graph TB
    subgraph "Data Loading"
        Files["Audio Files"] --> Loader["DataLoader<br/>(PyTorch)"]
    end

    subgraph "Model Architecture"
        Loader --> Aug["Augmentation Layer<br/>(TorchFX filters - fixed)"]
        Aug --> Conv1["Conv1D Layer<br/>(learnable)"]
        Conv1 --> Conv2["Conv1D Layer<br/>(learnable)"]
        Conv2 --> Pool["Pooling Layer"]
        Pool --> FC["Fully Connected<br/>(learnable)"]
    end

    subgraph "Training Loop"
        FC --> Loss["Loss Function"]
        Loss --> Backward["Backward Pass"]
        Backward --> Optimizer["Optimizer<br/>(updates Conv + FC)"]
    end

    Optimizer -.->|"gradients flow<br/>but don't update"| Aug
    Optimizer -->|"updates parameters"| Conv1
    Optimizer -->|"updates parameters"| Conv2
    Optimizer -->|"updates parameters"| FC

    style Aug fill:#fff5e1
    style Conv1 fill:#e1ffe1
    style Conv2 fill:#e1ffe1
    style FC fill:#e1ffe1
    style Loss fill:#ffe1e1
```

**Training Pipeline Architecture** - TorchFX filters serve as fixed augmentation layers while gradients flow through to update learnable layers.

### Module Composition Patterns

```{mermaid}
graph TB
    subgraph "Pattern 1: Sequential Chain"
        S1["nn.Sequential"] --> SF1["TorchFX Filter 1"]
        SF1 --> SF2["TorchFX Filter 2"]
        SF2 --> SF3["TorchFX Effect"]
    end

    subgraph "Pattern 2: Custom Module"
        C1["Custom Module"] --> CM1["def __init__"]
        CM1 --> CF1["self.filter1 = ..."]
        CM1 --> CF2["self.filter2 = ..."]
        C1 --> CM2["def forward"]
        CM2 --> CL1["x = self.filter1(x)"]
        CL1 --> CL2["x = self.filter2(x)"]
    end

    subgraph "Pattern 3: Hybrid Module"
        H1["Hybrid Module"] --> HT1["TorchFX Filters"]
        H1 --> HT2["torchaudio Transforms"]
        H1 --> HT3["Custom Logic"]
        HT1 --> HF["forward()"]
        HT2 --> HF
        HT3 --> HF
    end

    style S1 fill:#e1f5ff
    style C1 fill:#e1f5ff
    style H1 fill:#e1f5ff
```

**Module Composition Patterns** - Three common ways to structure TorchFX modules in larger systems.

## Best Practices

### Use nn.Sequential for Simple Chains

```python
# ✅ GOOD: Clear, standard PyTorch pattern
preprocessing = nn.Sequential(
    fx.filter.HiButterworth(cutoff=80, order=2, fs=44100),
    fx.filter.LoButterworth(cutoff=12000, order=4, fs=44100),
    fx.effect.Normalize(peak=0.9),
)

# ❌ LESS GOOD: Custom module for simple chain
class Preprocessing(nn.Module):
    def __init__(self):
        super().__init__()
        self.f1 = fx.filter.HiButterworth(cutoff=80, order=2, fs=44100)
        self.f2 = fx.filter.LoButterworth(cutoff=12000, order=4, fs=44100)
        self.norm = fx.effect.Normalize(peak=0.9)

    def forward(self, x):
        return self.norm(self.f2(self.f1(x)))
```

### Set Sample Rate Explicitly for Reusable Modules

```python
# ✅ GOOD: Sample rate specified at module creation
class ReusableFilter(nn.Module):
    def __init__(self, sample_rate: int = 44100):
        super().__init__()
        self.filter = fx.filter.LoButterworth(cutoff=1000, order=4, fs=sample_rate)

# ❌ BAD: Relying on pipe operator to set fs
class ReusableFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = fx.filter.LoButterworth(cutoff=1000, order=4)  # fs=None!
```

### Move Entire Modules to Device Together

```python
# ✅ GOOD: Move entire module at once
processor = VocalProcessor(sample_rate=44100)
processor.to("cuda")  # Moves all child modules

# ❌ BAD: Moving individual components
processor = VocalProcessor(sample_rate=44100)
processor.filter1.to("cuda")
processor.filter2.to("cuda")
processor.effect1.to("cuda")  # Easy to miss one!
```

### Use ModuleList for Dynamic Filter Collections

```python
# ✅ GOOD: ModuleList for dynamic collections
class MultiFilterBank(nn.Module):
    def __init__(self, cutoffs: list[float], fs: int = 44100):
        super().__init__()
        self.filters = nn.ModuleList([
            fx.filter.LoButterworth(cutoff=f, order=4, fs=fs)
            for f in cutoffs
        ])

    def forward(self, x):
        return sum(f(x) for f in self.filters)

# ❌ BAD: Regular list (filters won't be registered!)
class MultiFilterBank(nn.Module):
    def __init__(self, cutoffs: list[float], fs: int = 44100):
        super().__init__()
        self.filters = [  # ⚠️ Regular list!
            fx.filter.LoButterworth(cutoff=f, order=4, fs=fs)
            for f in cutoffs
        ]
```

## Common Pitfalls

### Pitfall 1: Forgetting to Process Batches Correctly

```python
# ❌ WRONG: TorchFX filters expect (channels, time), not (batch, channels, time)
class BadProcessor(nn.Module):
    def forward(self, x):
        # x is (batch, channels, time)
        return self.filter(x)  # Error! Filter expects (channels, time)

# ✅ CORRECT: Process each sample in batch
class GoodProcessor(nn.Module):
    def forward(self, x):
        # x is (batch, channels, time)
        results = [self.filter(x[i]) for i in range(x.size(0))]
        return torch.stack(results)
```

### Pitfall 2: Mixing Sample Rates

```python
# ❌ WRONG: Sample rate mismatch
audio_44k = fx.Wave.from_file("audio_44100.wav")  # 44100 Hz
filter_16k = fx.filter.LoButterworth(cutoff=1000, fs=16000)  # Wrong fs!
result = audio_44k | filter_16k  # Incorrect filtering!

# ✅ CORRECT: Match sample rates
audio_44k = fx.Wave.from_file("audio_44100.wav")
filter_44k = fx.filter.LoButterworth(cutoff=1000, fs=44100)
result = audio_44k | filter_44k
```

### Pitfall 3: Expecting Filter Coefficients to Update

```python
# ❌ WRONG: Expecting filter coefficients to learn
model = nn.Sequential(
    fx.filter.LoButterworth(cutoff=1000, fs=44100),
    SomeLearnableLayer(),
)
# Training will NOT update the filter's cutoff frequency or coefficients!

# ✅ CORRECT: Use fixed filters or create learnable FIR filters
# Option 1: Use as fixed preprocessing
preprocessing = fx.filter.LoButterworth(cutoff=1000, fs=44100)

# Option 2: Create learnable FIR filter
class LearnableFIR(nn.Module):
    def __init__(self, num_taps: int):
        super().__init__()
        self.taps = nn.Parameter(torch.randn(num_taps))

    def forward(self, x):
        return torch.nn.functional.conv1d(
            x.unsqueeze(0),
            self.taps.unsqueeze(0).unsqueeze(0),
            padding=len(self.taps)//2
        ).squeeze(0)
```

## Related Concepts

- {doc}`gpu-acceleration` - Using TorchFX with CUDA
- {doc}`../tutorials/series-parallel-filters` - Combining filters in complex networks
- {doc}`../core-concepts/fx` - Understanding the FX base class
- {doc}`../tutorials/custom-effects` - Creating custom effects as nn.Modules

## External Resources

- [PyTorch nn.Module Tutorial](https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_module.html) - Building custom modules
- [torchaudio Documentation](https://pytorch.org/audio/stable/index.html) - Audio processing in PyTorch
- [PyTorch Device Management](https://pytorch.org/docs/stable/notes/cuda.html) - Working with GPUs
- [Automatic Differentiation](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) - Understanding gradients in PyTorch

## Summary

Key takeaways for PyTorch integration:

1. **Module Inheritance**: All TorchFX effects and filters inherit from {class}`torch.nn.Module`, enabling full PyTorch compatibility
2. **Pipe Operator**: The `|` operator accepts any {class}`torch.nn.Module`, not just TorchFX components
3. **Sequential Composition**: Use {class}`torch.nn.Sequential` for standard PyTorch-style chains
4. **Custom Modules**: Embed TorchFX filters in custom modules for reusable processing blocks
5. **Gradient Flow**: Gradients flow through TorchFX operations, but filter coefficients are fixed (non-learnable)
6. **Library Mixing**: Seamlessly combine TorchFX with torchaudio transforms and custom modules

TorchFX's deep integration with PyTorch makes it a natural fit for machine learning pipelines, data augmentation, and differentiable audio processing workflows.
