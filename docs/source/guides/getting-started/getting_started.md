# Getting Started

This guide will help you get started with **TorchFX**, a GPU-accelerated audio DSP library built on top of PyTorch. You'll learn the fundamental workflow from loading audio files to creating complex processing pipelines.

## What You'll Learn

This guide demonstrates the fundamental TorchFX workflow:

1. Installing the library
2. Loading audio with {meth}`Wave.from_file() <torchfx.Wave.from_file>`
3. Chaining filters and effects using the {term}`pipeline operator` `|`
4. Managing device placement (CPU/GPU)
5. Combining filters in series and parallel
6. Saving processed audio

For detailed installation options including platform-specific PyTorch configuration, see {doc}`installation`. For in-depth explanations of the core concepts, see {doc}`/guides/core-concepts/index`.

## Installation

Install TorchFX using pip:

```bash
pip install torchfx
```

This command installs TorchFX along with its dependencies: PyTorch, torchaudio, NumPy, SciPy, and soundfile. For advanced installation options, dependency management with `uv`, or platform-specific PyTorch builds (CPU vs CUDA), see {doc}`installation`.

```{seealso}
{doc}`installation` - Complete installation guide with GPU setup and development options
```

## Basic Concepts

TorchFX uses an object-oriented interface where audio signals are wrapped in a {class}`~torchfx.Wave` object that holds both the audio samples and the sampling rate.

You can build audio processing pipelines by chaining operations using the **pipe operator (`|`)**, thanks to Python operator overloading.

### Key Components

- **{class}`~torchfx.Wave` class**: Wraps audio data ({attr}`ys <torchfx.Wave.ys>`) and sample rate ({attr}`fs <torchfx.Wave.fs>`)
- **{class}`~torchfx.FX` base class**: Foundation for all audio effects and filters
- **Pipeline operator (`|`)**: Chains processing modules together
- **PyTorch integration**: All modules inherit from {class}`torch.nn.Module`

## Your First Audio Processing Pipeline

The following example demonstrates the core TorchFX workflow:

```python
import torch
import torchfx as fx

# Load audio file
wave = fx.Wave.from_file("path_to_audio.wav")

# Apply processing pipeline
filtered_wave = (
    wave
    | fx.filter.LoButterworth(8000)
    | fx.filter.HiShelving(2000)
    | fx.effect.Reverb()
)

# Access the processed audio tensor
output_tensor = filtered_wave.ys
```

This example creates a processing chain that:

1. Loads an audio file into a {class}`~torchfx.Wave` object
2. Applies a low-pass Butterworth filter at 8000 Hz
3. Applies a high-shelving filter at 2000 Hz
4. Applies a reverb effect
5. Returns a new {class}`~torchfx.Wave` object containing the processed audio

## Wave Object and Pipeline Processing

The following diagram illustrates how the {class}`~torchfx.Wave` class and pipeline operator work together:

```{mermaid}
graph LR
    AudioFile["Audio File<br/>(WAV/MP3/etc)"]
    WaveFromFile["Wave.from_file()"]
    WaveObj1["Wave object<br/>ys: Tensor<br/>fs: int"]

    Filter1["fx.filter.LoButterworth<br/>torch.nn.Module"]
    WaveObj2["Wave object<br/>(filtered)"]

    Filter2["fx.filter.HiShelving<br/>torch.nn.Module"]
    WaveObj3["Wave object<br/>(filtered + shaped)"]

    Effect["fx.effect.Reverb<br/>torch.nn.Module"]
    WaveObjFinal["Wave object<br/>(final output)"]

    AudioFile -->|"load"| WaveFromFile
    WaveFromFile --> WaveObj1
    WaveObj1 -->|"| operator"| Filter1
    Filter1 -->|"returns"| WaveObj2
    WaveObj2 -->|"| operator"| Filter2
    Filter2 -->|"returns"| WaveObj3
    WaveObj3 -->|"| operator"| Effect
    Effect -->|"returns"| WaveObjFinal
```

Each {class}`~torchfx.Wave` object encapsulates:

- {attr}`ys <torchfx.Wave.ys>`: A PyTorch {class}`~torch.Tensor` containing audio samples (shape: `[channels, samples]`)
- {attr}`fs <torchfx.Wave.fs>`: An integer representing the {term}`sampling frequency` in Hz

The pipe operator `|` is overloaded on the {class}`~torchfx.Wave` class to enable functional chaining. Each filter or effect in the chain receives a {class}`~torchfx.Wave`, processes its {attr}`ys <torchfx.Wave.ys>` tensor, and returns a new {class}`~torchfx.Wave` object.

## Working with the Wave Class

To begin, import the library and load a waveform from file:

```python
import torchfx as fx

# Load an audio file
wave = fx.Wave.from_file("path_to_audio.wav")

# Access the raw audio data and sampling rate
print(wave.ys.shape)   # e.g., torch.Size([2, 44100])
print(wave.fs)         # e.g., 44100
print(f"Duration: {wave.duration('sec')} seconds")
print(f"Channels: {wave.channels()}")
```

The {class}`~torchfx.Wave` object automatically handles stereo or multichannel data and ensures that filters retain sample rate context.

```{seealso}
{doc}`/guides/core-concepts/wave` - Complete guide to the Wave class with detailed examples
```

## Complete Example with File I/O

Here's a complete example that loads a file, processes it, and saves the output:

```python
import torch
import torchfx as fx
import torchaudio

# Load audio
signal = fx.Wave.from_file("input.wav")

# Optional: Move to GPU for acceleration
if torch.cuda.is_available():
    signal = signal.to("cuda")

# Apply processing pipeline
result = (
    signal
    | fx.filter.LoButterworth(100, order=2)
    | fx.filter.HiButterworth(2000, order=2)
    | fx.effect.Gain(db=-6)
)

# Save output (move back to CPU for I/O)
torchaudio.save("output.wav", result.ys.cpu(), result.fs)
```

This example demonstrates the complete workflow including:

- Loading audio from disk
- GPU acceleration for processing
- Applying multiple filters in series
- Saving the result back to disk

## Processing Flow with Device Management

The following sequence diagram shows the complete processing flow including device management:

```{mermaid}
sequenceDiagram
    participant User
    participant WaveFromFile as "Wave.from_file()"
    participant WaveObj as "Wave object"
    participant GPU as "GPU Device"
    participant Filter as "Filter/Effect Module"
    participant TorchAudio as "torchaudio.save()"

    User->>WaveFromFile: "load audio file"
    WaveFromFile->>WaveObj: "create Wave(ys, fs)"
    Note over WaveObj: "ys is on CPU by default"

    User->>WaveObj: "wave.to('cuda')"
    WaveObj->>GPU: "move tensor to GPU"
    GPU-->>WaveObj: "return GPU Wave"

    User->>Filter: "wave | filter"
    Filter->>Filter: "process ys tensor on GPU"
    Filter-->>WaveObj: "return new Wave on GPU"

    User->>WaveObj: "result.ys.cpu()"
    WaveObj->>GPU: "move tensor to CPU"
    GPU-->>WaveObj: "return CPU tensor"

    User->>TorchAudio: "save(path, ys, fs)"
    TorchAudio-->>User: "file written"
```

### Key Points

1. {meth}`Wave.from_file() <torchfx.Wave.from_file>` loads audio onto CPU by default
2. {meth}`wave.to("cuda") <torchfx.Wave.to>` moves the {class}`~torchfx.Wave` and its {attr}`ys <torchfx.Wave.ys>` tensor to GPU
3. Filters and effects process tensors on whatever device they reside
4. {meth}`ys.cpu() <torch.Tensor.cpu>` moves the tensor back to CPU for file I/O
5. {func}`torchaudio.save` requires CPU tensors for writing to disk

```{seealso}
{doc}`/guides/advanced/gpu-acceleration` - GPU acceleration best practices and performance optimization
```

## Applying Built-in Filters

TorchFX provides a collection of IIR and FIR filters under the {mod}`torchfx.filter` module. All filters are implemented as subclasses of {class}`torch.nn.Module`.

Here's an example of chaining filters with the pipe operator:

```python
from torchfx import filter as fx_filter

# Apply a low-pass Butterworth filter at 8 kHz and a high-shelving filter at 2 kHz
filtered = (
    fx.Wave.from_file("example.wav")
    | fx_filter.LoButterworth(8000)
    | fx_filter.HiShelving(2000)
)

# Save the processed signal
filtered.save("filtered_output.wav")
```

You can also build pipelines using {class}`torch.nn.Sequential` or define custom modules as in PyTorch.

## Parallel Filter Combination

TorchFX supports combining filters in parallel using the `+` operator:

```python
result = (
    signal
    | fx.filter.LoButterworth(100, order=2)
    | fx.filter.HiButterworth(2000, order=4) + fx.filter.HiChebyshev1(2000, order=2)
)
```

This creates a parallel combination where the signal is split, processed by both filters independently, and then summed. The `+` operator creates a {class}`~torchfx.filter.ParallelFilterCombination` object that handles this routing automatically.

### Series and Parallel Filter Topology

```{mermaid}
graph TB
    Input["Input Wave"]
    LoFilter["fx.filter.LoButterworth<br/>cutoff=100, order=2"]

    Split["Split"]
    HiButterworth["fx.filter.HiButterworth<br/>cutoff=2000, order=4"]
    HiChebyshev["fx.filter.HiChebyshev1<br/>cutoff=2000, order=2"]
    Sum["Sum (ParallelFilterCombination)"]

    Output["Output Wave"]

    Input -->|"| operator (series)"| LoFilter
    LoFilter --> Split
    Split -->|"+ operator (parallel)"| HiButterworth
    Split -->|"+ operator (parallel)"| HiChebyshev
    HiButterworth --> Sum
    HiChebyshev --> Sum
    Sum --> Output
```

For more details on parallel filter combinations, see {doc}`/guides/tutorials/series-parallel-filters`.

```{seealso}
{doc}`/guides/tutorials/series-parallel-filters` - Complete tutorial on series and parallel filter combinations
```

## Complete Series and Parallel Example

Here's a complete example demonstrating mixed series/parallel processing:

```python
import torchfx as fx
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
    | fx.filter.LoButterworth(cutoff=100, order=2)

    # Stage 2: Parallel high-pass filters (parallel)
    | fx.filter.HiButterworth(2000, order=4) + fx.filter.HiChebyshev1(2000, order=2)

    # Stage 3: Reduce level (series)
    | fx.effect.Gain(gain=0.5, gain_type="amplitude")
)

# Save result (move to CPU for I/O)
processed.to("cpu").save("processed.wav")
```

## Creating Your Own Effect

To create your own audio effect, subclass the {class}`~torchfx.FX` class (a utility base class derived from {class}`torch.nn.Module`):

```python
from torchfx.core import FX

class Invert(FX):
    def forward(self, wave):
        return wave.new(-wave.ys)
```

This custom `Invert` effect simply negates the audio signal. You can now use it like any other TorchFX module:

```python
inverted = wave | Invert()

# Listen or save the output
inverted.save("inverted.wav")
```

The {class}`~torchfx.FX` base class ensures that your custom effect works seamlessly with the {class}`~torchfx.Wave` class and supports the pipe operator.

```{seealso}
- {doc}`/guides/core-concepts/fx` - Understanding the FX base class architecture
- {doc}`/guides/tutorials/custom-effects` - Complete tutorial on creating custom effects
```

## Available Filters and Effects

The following table lists commonly used filters and effects available in TorchFX:

| Category | Class Name | Description |
|----------|-----------|-------------|
| **IIR Filters** | {class}`~torchfx.filter.LoButterworth` | Low-pass Butterworth filter |
| | {class}`~torchfx.filter.HiButterworth` | High-pass Butterworth filter |
| | {class}`~torchfx.filter.BandButterworth` | Band-pass Butterworth filter |
| | {class}`~torchfx.filter.LoChebyshev1` | Low-pass Chebyshev Type I filter |
| | {class}`~torchfx.filter.HiChebyshev1` | High-pass Chebyshev Type I filter |
| | {class}`~torchfx.filter.LoShelving` | Low-frequency shelving filter |
| | {class}`~torchfx.filter.HiShelving` | High-frequency shelving filter |
| | {class}`~torchfx.filter.Peaking` | Peaking/notch filter |
| **FIR Filters** | {class}`~torchfx.filter.FIR` | Finite impulse response filter |
| | {class}`~torchfx.filter.DesignableFIR` | FIR with automatic coefficient design |
| **Effects** | {class}`~torchfx.effect.Gain` | Amplitude/dB gain control |
| | {class}`~torchfx.effect.Normalize` | Signal normalization |
| | {class}`~torchfx.effect.Reverb` | Reverb effect |
| | {class}`~torchfx.effect.Delay` | BPM-synced delay effect |

For complete API documentation, see the API Reference.

## Best Practices

### Use Multi-Line Pipelines for Readability

```python
# ✅ GOOD: Clear, readable pipeline
filtered = (
    wave
    | fx.filter.LoButterworth(8000)
    | fx.filter.HiShelving(2000)
    | fx.effect.Reverb()
)

# ❌ BAD: Hard to read single line
filtered = wave | fx.filter.LoButterworth(8000) | fx.filter.HiShelving(2000) | fx.effect.Reverb()
```

### Device Management

```python
# ✅ GOOD: Explicit device management
wave = fx.Wave.from_file("audio.wav")
if torch.cuda.is_available():
    wave = wave.to("cuda")

processed = wave | effect_chain

# Move back to CPU for saving
processed.to("cpu").save("output.wav")
```

### Reuse Filter Chains

```python
# ✅ GOOD: Define reusable processing chains
mastering_chain = (
    fx.filter.HiButterworth(30, order=2)
    | fx.filter.LoButterworth(18000, order=4)
    | fx.effect.Normalize()
)

# Apply to multiple files
for audio_file in audio_files:
    wave = fx.Wave.from_file(audio_file)
    processed = wave | mastering_chain
    processed.save(f"mastered_{audio_file}")
```

## Common Pitfalls

### Forgetting to Move to CPU Before Saving

```python
# ❌ WRONG: Trying to save CUDA tensor
wave_gpu = fx.Wave.from_file("audio.wav").to("cuda")
processed = wave_gpu | effect_chain
processed.save("output.wav")  # Error: can't save CUDA tensor

# ✅ CORRECT: Move to CPU before saving
processed.to("cpu").save("output.wav")
```

### Incorrect Tensor Shape

```python
import torch

# ❌ WRONG: 1D tensor
mono = torch.randn(44100)
wave = fx.Wave(mono, fs=44100)  # Error!

# ✅ CORRECT: 2D tensor with channel dimension
mono = torch.randn(1, 44100)
wave = fx.Wave(mono, fs=44100)
```

## Next Steps

After completing this quick start guide, explore these topics to deepen your understanding:

### Core Concepts

- {doc}`/guides/core-concepts/wave` - Learn about the Wave class in detail
- {doc}`/guides/core-concepts/fx` - Understand the FX base class and architecture
- {doc}`/guides/core-concepts/pipeline-operator` - Deep dive into the pipeline operator
- {doc}`/guides/core-concepts/type-system` - Time units and musical notation system

### Tutorials

- {doc}`/guides/tutorials/series-parallel-filters` - Master series and parallel filter combinations
- {doc}`/guides/tutorials/custom-filters` - Create your own custom filters
- {doc}`/guides/tutorials/custom-effects` - Design custom audio effects
- {doc}`/guides/tutorials/ml-batch-processing` - Process audio in ML pipelines

### Advanced Topics

- {doc}`/guides/advanced/gpu-acceleration` - GPU acceleration and performance optimization
- {doc}`/guides/advanced/pytorch-integration` - Deep PyTorch integration patterns
- {doc}`/guides/advanced/multi-channel` - Working with multi-channel audio
- {doc}`/guides/advanced/performance` - Performance tuning and optimization

### Examples

See the [examples directory](https://github.com/matteospanio/torchfx/tree/master/examples) for more complete examples including:

- Real-time audio processing
- BPM-synchronized effects
- Multi-band processing
- Vocal processing chains
- And more!

## External Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) - PyTorch framework documentation
- [torchaudio Documentation](https://pytorch.org/audio/stable/index.html) - Audio I/O and transformations
- [Digital Signal Processing on Wikipedia](https://en.wikipedia.org/wiki/Digital_signal_processing) - DSP fundamentals

## References

```{bibliography}
:filter: docname in docnames
:style: alpha
```
