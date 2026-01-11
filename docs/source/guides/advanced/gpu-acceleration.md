(gpu-acceleration)=
# GPU Acceleration

Learn how to leverage CUDA-enabled GPUs for accelerated audio processing in TorchFX. This tutorial covers device management, performance optimization, and best practices for moving audio processing workflows to the GPU.

## Prerequisites

Before starting this tutorial, you should be familiar with:

- {doc}`../core-concepts/wave` - Wave class fundamentals
- {doc}`../core-concepts/pipeline-operator` - Pipeline operator basics
- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html) - PyTorch device management
- Basic understanding of GPU computing concepts

## Overview

TorchFX leverages PyTorch's device management system to enable GPU acceleration for audio processing. All audio data ({class}`~torchfx.Wave` objects) and filter coefficients can be seamlessly moved between CPU and GPU memory using standard PyTorch device APIs.

### When to Use GPU Acceleration

GPU acceleration provides significant performance benefits in specific scenarios:

| Scenario | GPU Advantage | Reason |
|----------|---------------|---------|
| Long audio files (>60 seconds) | **High** | Amortizes data transfer overhead |
| Multi-channel audio (≥4 channels) | **High** | Parallel processing across channels |
| Complex filter chains (≥3 filters) | **Medium-High** | Accumulated compute savings |
| Short audio (<5 seconds) | **Low** | Data transfer overhead dominates |
| Single channel, simple processing | **Low-Medium** | Insufficient parallelism |

```{tip}
For batch processing of many audio files, GPU acceleration can provide substantial speedups even for shorter files, as the overhead is amortized across the entire batch.
```

## Device Management Architecture

TorchFX uses PyTorch's device management system for both {class}`~torchfx.Wave` objects and filter modules.

```{mermaid}
graph TB
    subgraph CPU["CPU Memory Space"]
        WaveCPU["Wave Object<br/>ys: Tensor (CPU)<br/>fs: int<br/>device: 'cpu'"]
        FilterCPU["Filter Modules<br/>coefficients on CPU"]
    end

    subgraph GPU["GPU Memory Space (CUDA)"]
        WaveGPU["Wave Object<br/>ys: Tensor (CUDA)<br/>fs: int<br/>device: 'cuda'"]
        FilterGPU["Filter Modules<br/>coefficients on CUDA"]
    end

    subgraph API["Device Management API"]
        ToMethod["Wave.to(device)"]
        DeviceProp["Wave.device property"]
        ModuleTo["nn.Module.to(device)"]
    end

    WaveCPU -->|"wave.to('cuda')"| ToMethod
    ToMethod -->|"moves ys tensor"| WaveGPU

    WaveGPU -->|"wave.to('cpu')"| ToMethod
    ToMethod -->|"moves ys tensor"| WaveCPU

    DeviceProp -->|"setter calls to()"| ToMethod

    FilterCPU -->|"filter.to('cuda')"| ModuleTo
    ModuleTo -->|"moves parameters"| FilterGPU

    FilterGPU -->|"filter.to('cpu')"| ModuleTo
    ModuleTo -->|"moves parameters"| FilterCPU

    style WaveCPU fill:#e1f5ff
    style WaveGPU fill:#e1ffe1
    style FilterCPU fill:#fff5e1
    style FilterGPU fill:#fff5e1
```

**Device Transfer Architecture** - Wave objects and filters can be moved between CPU and GPU memory using standard PyTorch APIs.

## Moving Wave Objects to GPU

The {class}`~torchfx.Wave` class provides two methods for device management: the `to()` method and the `device` property setter.

### The `to()` Method

The primary method for moving a {class}`~torchfx.Wave` object between devices is {meth}`~torchfx.Wave.to`, which returns the modified {class}`~torchfx.Wave` object for method chaining:

```python
import torchfx as fx

# Load audio file (defaults to CPU)
wave = fx.Wave.from_file("audio.wav")
print(wave.device)  # 'cpu'

# Move to GPU
wave.to("cuda")
print(wave.device)  # 'cuda'

# Move back to CPU
wave.to("cpu")
print(wave.device)  # 'cpu'
```

The `to()` method performs two operations:
1. Updates the internal `__device` field to track the current device
2. Moves the underlying `ys` tensor using PyTorch's `Tensor.to(device)` method

```{seealso}
{meth}`torchfx.Wave.to` - API documentation for the `to()` method
```

### The `device` Property

The {attr}`~torchfx.Wave.device` property provides both getter and setter functionality:

```python
import torchfx as fx

wave = fx.Wave.from_file("audio.wav")

# Reading current device
current_device = wave.device  # Returns "cpu" or "cuda"
print(f"Wave is on: {current_device}")

# Setting device via property (equivalent to wave.to("cuda"))
wave.device = "cuda"
print(f"Wave moved to: {wave.device}")
```

The property setter internally calls `to()`, so both approaches are equivalent. Use whichever is more readable in your code.

### Method Chaining

The `to()` method returns `self`, enabling method chaining with other {class}`~torchfx.Wave` operations:

```python
import torchfx as fx

# Method chaining with device transfer
result = (
    fx.Wave.from_file("audio.wav")
    .to("cuda")  # Move to GPU
    | fx.filter.LoButterworth(cutoff=1000, order=4)
    | fx.effect.Normalize(peak=0.9)
)

# Save result (automatically on same device as input)
result.to("cpu").save("output.wav")
```

## Filter and Effect Device Management

All filters and effects in TorchFX inherit from {class}`torch.nn.Module`, enabling standard PyTorch device management for their parameters and buffers.

### Moving Filters to GPU

Filters store their coefficients as PyTorch tensors or buffers. To enable GPU-accelerated filtering, move these coefficients to the GPU:

```python
import torchfx as fx

# Create and configure filter
lowpass = fx.filter.LoButterworth(cutoff=1000, order=4, fs=44100)
lowpass.compute_coefficients()  # Compute coefficients on CPU

# Move filter to GPU
lowpass.to("cuda")

# Now the filter is ready for GPU processing
```

### Moving Filter Chains to GPU

When using {class}`torch.nn.Sequential` or other PyTorch containers, all modules in the chain are moved together:

```python
import torch.nn as nn
import torchfx as fx

# Create filter chain
filter_chain = nn.Sequential(
    fx.filter.HiButterworth(cutoff=100, order=2),
    fx.filter.LoButterworth(cutoff=5000, order=4),
    fx.effect.Normalize(peak=0.9)
)

# Move entire chain to GPU
filter_chain.to("cuda")  # All filters and effects now on CUDA
```

The `to()` method propagates through all child modules, ensuring consistent device placement.

## Device Coordination in Processing Pipelines

When using the {term}`pipeline operator` (`|`), device compatibility is the user's responsibility. Both the {class}`~torchfx.Wave` object and the filter/effect must be on the same device.

```{mermaid}
sequenceDiagram
    participant User
    participant Wave as "Wave Object"
    participant Filter as "Filter Module"
    participant GPU as "CUDA Device"

    User->>Wave: Wave.from_file("audio.wav")
    Note over Wave: ys on CPU<br/>device = "cpu"

    User->>Wave: wave.to("cuda")
    Wave->>GPU: Transfer ys tensor
    Note over Wave: ys on CUDA<br/>device = "cuda"

    User->>Filter: filter.to("cuda")
    Filter->>GPU: Transfer coefficients
    Note over Filter: coefficients on CUDA

    User->>Wave: wave | filter
    Note over Wave,Filter: Both on same device ✓
    Wave->>Filter: forward(ys)
    Filter->>GPU: Execute convolution on GPU
    GPU-->>Filter: Result tensor (CUDA)
    Filter-->>Wave: Return new Wave (CUDA)
    Note over Wave: New Wave object<br/>ys on CUDA
```

**Pipeline Processing Flow with GPU** - Shows the sequence of device transfers and processing operations.

### Device Compatibility Rules

The pipeline operator validates device compatibility at runtime:

| Wave Device | Filter/Effect Device | Result |
|-------------|----------------------|---------|
| `"cuda"` | `"cuda"` | ✅ Processing on GPU |
| `"cpu"` | `"cpu"` | ✅ Processing on CPU |
| `"cuda"` | `"cpu"` | ❌ Runtime error |
| `"cpu"` | `"cuda"` | ❌ Runtime error |

```{warning}
Device mismatches will raise a runtime error from PyTorch. Always ensure the {class}`~torchfx.Wave` object and all filters/effects in the pipeline are on the same device.
```

### Automatic Device Propagation Pattern

While TorchFX doesn't automatically move filters to match the Wave's device, you can establish a consistent pattern:

```python
import torch
import torchfx as fx

# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load and move Wave to device
wave = fx.Wave.from_file("audio.wav").to(device)

# Create filters (they start on CPU by default)
lowpass = fx.filter.LoButterworth(cutoff=1000, order=4)
highpass = fx.filter.HiButterworth(cutoff=100, order=2)

# Move filters to match Wave's device
lowpass.to(device)
highpass.to(device)

# Now processing works on the selected device
result = wave | lowpass | highpass
```

```{tip}
The tensor returned by the filter's `forward()` method maintains the same device as the input tensor, so all intermediate {class}`~torchfx.Wave` objects in a pipeline chain stay on the same device.
```

## Performance Considerations

GPU acceleration provides the greatest benefits when data transfer overhead is amortized by significant computation.

### Data Transfer Overhead

Moving data between CPU and GPU incurs overhead from PCIe bus transfers:

| Operation | Cost | Notes |
|-----------|------|-------|
| `Wave.to("cuda")` | O(n) where n = sample count | Transfer audio data to GPU |
| `nn.Module.to("cuda")` | O(p) where p = parameter count | Transfer filter coefficients |
| `Tensor.cpu()` | O(n) where n = tensor size | Transfer results back to CPU |

**Optimization Strategy**: Minimize device transfers by:

1. Loading and moving to GPU **once** at the start
2. Performing **all processing** on GPU
3. Moving back to CPU **only** for final I/O operations

### Benchmarking Example

The following example demonstrates proper device management for performance:

```python
import torch
import torchfx as fx
from torchfx.filter import DesignableFIR
import torch.nn as nn
import timeit

# Configuration
SAMPLE_RATE = 44100
DURATION = 60  # seconds
NUM_CHANNELS = 4

# Create test audio
signal = torch.randn(NUM_CHANNELS, int(SAMPLE_RATE * DURATION))
wave = fx.Wave(signal, SAMPLE_RATE)

# Create filter chain
filter_chain = nn.Sequential(
    DesignableFIR(num_taps=101, cutoff=1000, fs=SAMPLE_RATE),
    DesignableFIR(num_taps=102, cutoff=5000, fs=SAMPLE_RATE),
    DesignableFIR(num_taps=103, cutoff=1500, fs=SAMPLE_RATE),
)

# Compute coefficients before moving to GPU
for f in filter_chain:
    f.compute_coefficients()

# Benchmark GPU processing
wave.to("cuda")
filter_chain.to("cuda")
gpu_time = timeit.timeit(lambda: wave | filter_chain, number=10)

# Benchmark CPU processing
wave.to("cpu")
filter_chain.to("cpu")
cpu_time = timeit.timeit(lambda: wave | filter_chain, number=10)

print(f"GPU time: {gpu_time/10:.4f}s")
print(f"CPU time: {cpu_time/10:.4f}s")
print(f"Speedup: {cpu_time/gpu_time:.2f}x")
```

### When GPU Provides Maximum Benefit

Based on empirical benchmarking, GPU acceleration is most beneficial when:

**Audio Duration**
- Files longer than 60 seconds see significant speedups
- Transfer overhead is amortized over longer computation time

**Number of Channels**
- 4+ channels leverage GPU's parallel processing capabilities
- Single-channel audio sees modest gains

**Filter Complexity**
- FIR filters with >100 taps benefit significantly
- IIR filter chains (3+ cascaded stages) show good speedups
- Parallel filter combinations ({doc}`series-parallel-filters`) see excellent performance

**Batch Processing**
- Processing multiple files in a batch maximizes GPU utilization
- Transfer overhead amortized across entire batch

```{seealso}
{doc}`performance` - Comprehensive performance benchmarks and optimization guidelines
```

### Memory Considerations

GPU memory is typically more limited than system RAM:

| Constraint | Typical Limit | Mitigation Strategy |
|------------|---------------|---------------------|
| GPU VRAM capacity | 4-24 GB (consumer GPUs) | Process audio in chunks |
| Audio file size | Limited by VRAM | Stream processing for very long files |
| Filter coefficient storage | Usually negligible | Pre-compute coefficients before transfer |
| Batch size | Limited by VRAM | Reduce batch size if OOM errors occur |

For very long audio files (e.g., >1 hour), consider chunked processing:

```python
import torch
import torchfx as fx

def process_in_chunks(wave, filter_chain, chunk_duration=60):
    """Process audio in chunks to manage GPU memory."""
    chunk_samples = int(chunk_duration * wave.fs)
    num_chunks = (wave.ys.size(-1) + chunk_samples - 1) // chunk_samples

    results = []
    for i in range(num_chunks):
        start = i * chunk_samples
        end = min((i + 1) * chunk_samples, wave.ys.size(-1))

        # Extract chunk
        chunk = fx.Wave(wave.ys[..., start:end], wave.fs)
        chunk.to("cuda")

        # Process chunk
        processed_chunk = chunk | filter_chain

        # Move back to CPU and store
        results.append(processed_chunk.ys.cpu())

    # Concatenate results
    return fx.Wave(torch.cat(results, dim=-1), wave.fs)

# Usage
wave = fx.Wave.from_file("very_long_audio.wav")
filter_chain = nn.Sequential(
    fx.filter.LoButterworth(cutoff=1000, order=4),
    fx.filter.HiButterworth(cutoff=100, order=2),
).to("cuda")

result = process_in_chunks(wave, filter_chain, chunk_duration=60)
result.save("processed.wav")
```

## Best Practices

### Conditional Device Selection

Production code should handle systems without CUDA support gracefully:

```python
import torch
import torchfx as fx

# Conditional device selection
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load and move to selected device
wave = fx.Wave.from_file("audio.wav").to(device)

# Create and move filters
filter_chain = torch.nn.Sequential(
    fx.filter.LoButterworth(cutoff=1000, order=4),
    fx.filter.HiButterworth(cutoff=100, order=2),
).to(device)

# Process on appropriate device
result = wave | filter_chain
```

This pattern:
- Checks for CUDA availability at runtime
- Falls back to CPU if CUDA is unavailable
- Enables cross-platform compatibility

```{tip}
For multi-GPU systems, you can specify a specific GPU using `"cuda:0"`, `"cuda:1"`, etc. Use {func}`torch.cuda.device_count()` to check available GPUs.
```

### CPU Transfer for I/O Operations

File I/O operations require CPU tensors. Always move tensors to CPU before saving:

```python
import torchfx as fx
import torchaudio

# Process on GPU
wave = fx.Wave.from_file("input.wav").to("cuda")
result = wave | filter_chain  # Processing on GPU

# Option 1: Use ys.cpu() for saving
torchaudio.save("output.wav", result.ys.cpu(), result.fs)

# Option 2: Move entire Wave to CPU
result.to("cpu").save("output.wav")
```

The `Tensor.cpu()` method creates a copy on CPU without modifying the original GPU tensor, while `Wave.to("cpu")` moves the Wave object's internal state.

### Complete Processing Pipeline Pattern

Here's a complete example demonstrating best practices for GPU-accelerated audio processing:

```python
import torch
import torch.nn as nn
import torchfx as fx
import torchaudio

def process_audio_gpu(input_path, output_path):
    """Process audio with GPU acceleration and proper device management."""

    # Step 1: Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Step 2: Load audio
    wave = fx.Wave.from_file(input_path)
    print(f"Loaded audio: {wave.ys.shape}, fs={wave.fs}")

    # Step 3: Create processing chain
    processing_chain = nn.Sequential(
        # Pre-processing: remove rumble and noise
        fx.filter.HiButterworth(cutoff=80, order=2),
        fx.filter.LoButterworth(cutoff=15000, order=4),

        # Main processing: EQ and dynamics
        fx.effect.Normalize(peak=0.8),
    )

    # Step 4: Move to selected device
    wave = wave.to(device)
    processing_chain = processing_chain.to(device)

    # Step 5: Process audio (all on same device)
    result = wave | processing_chain
    print(f"Processing completed on {device}")

    # Step 6: Save result (move to CPU if needed)
    if result.device == "cuda":
        result = result.to("cpu")

    result.save(output_path)
    print(f"Saved to: {output_path}")

# Usage
process_audio_gpu("input.wav", "output.wav")
```

### Processing Pipeline Visualization

```{mermaid}
graph TD
    Start([Start]) --> CheckGPU{torch.cuda<br/>.is_available?}

    CheckGPU -->|Yes| SetCUDA["device = 'cuda'"]
    CheckGPU -->|No| SetCPU["device = 'cpu'"]

    SetCUDA --> Load[Load Audio<br/>Wave.from_file]
    SetCPU --> Load

    Load --> CreateChain[Create Processing Chain<br/>nn.Sequential]

    CreateChain --> MoveData["Move to Device<br/>wave.to(device)<br/>chain.to(device)"]

    MoveData --> Process[Process Audio<br/>wave | chain]

    Process --> CheckDevice{result.device<br/>== 'cuda'?}

    CheckDevice -->|Yes| MoveCPU["Move to CPU<br/>result.to('cpu')"]
    CheckDevice -->|No| Save

    MoveCPU --> Save[Save to File<br/>result.save]

    Save --> End([End])

    style Start fill:#e1f5ff
    style End fill:#e1f5ff
    style Process fill:#e1ffe1
    style CheckGPU fill:#fff5e1
    style CheckDevice fill:#fff5e1
```

**Complete GPU Processing Workflow** - Shows the full lifecycle from device selection to final output.

### Reusable Device Management Wrapper

For production code, consider creating a wrapper class:

```python
import torch
import torchfx as fx
from pathlib import Path

class GPUAudioProcessor:
    """Wrapper for GPU-accelerated audio processing."""

    def __init__(self, processing_chain, device=None):
        """Initialize processor with a processing chain.

        Parameters
        ----------
        processing_chain : nn.Module
            PyTorch module for audio processing
        device : str or None
            Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.processing_chain = processing_chain.to(device)
        print(f"Initialized on device: {device}")

    def process_file(self, input_path, output_path):
        """Process a single audio file.

        Parameters
        ----------
        input_path : str or Path
            Path to input audio file
        output_path : str or Path
            Path to save processed audio
        """
        # Load and move to device
        wave = fx.Wave.from_file(input_path).to(self.device)

        # Process
        result = wave | self.processing_chain

        # Save (automatically moves to CPU)
        result.to("cpu").save(output_path)

    def process_batch(self, input_files, output_dir):
        """Process multiple audio files.

        Parameters
        ----------
        input_files : list of str or Path
            List of input audio files
        output_dir : str or Path
            Directory to save processed files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for input_path in input_files:
            input_path = Path(input_path)
            output_path = output_dir / f"processed_{input_path.name}"

            print(f"Processing: {input_path.name}")
            self.process_file(input_path, output_path)

# Usage
import torch.nn as nn

# Create processing chain
chain = nn.Sequential(
    fx.filter.HiButterworth(cutoff=80, order=2),
    fx.filter.LoButterworth(cutoff=12000, order=4),
    fx.effect.Normalize(peak=0.9),
)

# Create processor (auto-detects GPU)
processor = GPUAudioProcessor(chain)

# Process single file
processor.process_file("input.wav", "output.wav")

# Process batch
files = ["song1.wav", "song2.wav", "song3.wav"]
processor.process_batch(files, "processed/")
```

## Working Examples

### Example 1: Basic GPU Processing

```python
import torch
import torchfx as fx

# Check GPU availability
if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU available, using CPU")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load audio and move to GPU
wave = fx.Wave.from_file("audio.wav").to(device)

# Create and move filter to GPU
lowpass = fx.filter.LoButterworth(cutoff=1000, order=4).to(device)

# Process on GPU
result = wave | lowpass

# Save (move to CPU first)
result.to("cpu").save("filtered.wav")
```

### Example 2: Multi-Stage Pipeline

```python
import torch
import torch.nn as nn
import torchfx as fx

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load audio
wave = fx.Wave.from_file("vocal.wav").to(device)

# Create complex processing chain
processing = nn.Sequential(
    # Stage 1: Remove rumble
    fx.filter.HiButterworth(cutoff=80, order=2),

    # Stage 2: Parallel filters for thickness
    fx.filter.HiButterworth(cutoff=2000, order=4) +
    fx.filter.HiChebyshev1(cutoff=2000, order=2),

    # Stage 3: Normalize
    fx.effect.Normalize(peak=0.9),
).to(device)

# Process
result = wave | processing

# Save
result.to("cpu").save("processed_vocal.wav")
```

### Example 3: Batch Processing with Progress

```python
import torch
import torchfx as fx
from pathlib import Path
from tqdm import tqdm

def batch_process_gpu(input_files, output_dir, filter_chain):
    """Process multiple audio files on GPU with progress bar."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    filter_chain = filter_chain.to(device)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_path in tqdm(input_files, desc="Processing"):
        # Load and process
        wave = fx.Wave.from_file(input_path).to(device)
        result = wave | filter_chain

        # Save
        output_path = output_dir / Path(input_path).name
        result.to("cpu").save(output_path)

# Usage
files = list(Path("audio_dataset").glob("*.wav"))
chain = fx.filter.LoButterworth(cutoff=1000, order=4)
batch_process_gpu(files, "processed_dataset", chain)
```

### Example 4: Memory-Efficient Chunked Processing

```python
import torch
import torchfx as fx

def process_long_audio(input_path, output_path, filter_chain, chunk_seconds=30):
    """Process very long audio files in chunks to manage GPU memory."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    filter_chain = filter_chain.to(device)

    # Load entire audio on CPU
    wave = fx.Wave.from_file(input_path)
    chunk_samples = int(chunk_seconds * wave.fs)

    processed_chunks = []
    num_chunks = (wave.ys.size(-1) + chunk_samples - 1) // chunk_samples

    print(f"Processing {num_chunks} chunks on {device}")

    for i in range(num_chunks):
        start = i * chunk_samples
        end = min((i + 1) * chunk_samples, wave.ys.size(-1))

        # Extract, process, and move back to CPU
        chunk = fx.Wave(wave.ys[..., start:end], wave.fs)
        chunk.to(device)

        processed = chunk | filter_chain
        processed_chunks.append(processed.ys.cpu())

        # Clear GPU cache periodically
        if device == "cuda":
            torch.cuda.empty_cache()

    # Combine chunks and save
    result = fx.Wave(torch.cat(processed_chunks, dim=-1), wave.fs)
    result.save(output_path)
    print(f"Saved to {output_path}")

# Usage
chain = fx.filter.LoButterworth(cutoff=1000, order=4)
process_long_audio("long_recording.wav", "processed.wav", chain, chunk_seconds=30)
```

## Common Pitfalls and Solutions

### Pitfall 1: Device Mismatch Errors

**Problem**: RuntimeError when Wave and filter are on different devices

```python
# ❌ WRONG: Device mismatch
wave = fx.Wave.from_file("audio.wav")  # CPU
filter = fx.filter.LoButterworth(cutoff=1000).to("cuda")  # GPU
result = wave | filter  # RuntimeError!
```

**Solution**: Ensure both are on the same device

```python
# ✅ CORRECT: Both on same device
device = "cuda" if torch.cuda.is_available() else "cpu"
wave = fx.Wave.from_file("audio.wav").to(device)
filter = fx.filter.LoButterworth(cutoff=1000).to(device)
result = wave | filter  # Works!
```

### Pitfall 2: Forgetting to Move Back to CPU for I/O

**Problem**: Error when trying to save GPU tensors

```python
# ❌ WRONG: Trying to save GPU tensor
wave = fx.Wave.from_file("audio.wav").to("cuda")
result = wave | filter_chain
result.save("output.wav")  # May fail depending on backend
```

**Solution**: Always move to CPU before saving

```python
# ✅ CORRECT: Move to CPU before saving
wave = fx.Wave.from_file("audio.wav").to("cuda")
result = wave | filter_chain
result.to("cpu").save("output.wav")  # Works!

# Or use ys.cpu() directly with torchaudio
import torchaudio
torchaudio.save("output.wav", result.ys.cpu(), result.fs)
```

### Pitfall 3: Inefficient Repeated Transfers

**Problem**: Moving data back and forth unnecessarily

```python
# ❌ WRONG: Inefficient transfers
wave = fx.Wave.from_file("audio.wav").to("cuda")
result1 = wave.to("cpu") | filter1  # CPU
result2 = result1.to("cuda") | filter2  # GPU
result3 = result2.to("cpu") | filter3  # CPU
```

**Solution**: Do all processing on one device

```python
# ✅ CORRECT: Single device for entire pipeline
device = "cuda" if torch.cuda.is_available() else "cpu"
wave = fx.Wave.from_file("audio.wav").to(device)
filter1.to(device)
filter2.to(device)
filter3.to(device)

result = wave | filter1 | filter2 | filter3  # All on same device
```

### Pitfall 4: Out of Memory on GPU

**Problem**: CUDA out of memory error with large audio files

```python
# ❌ WRONG: Loading entire 2-hour file on GPU
wave = fx.Wave.from_file("2_hour_recording.wav").to("cuda")  # OOM!
```

**Solution**: Use chunked processing (see Example 4 above) or reduce batch size

```python
# ✅ CORRECT: Process in chunks
process_long_audio("2_hour_recording.wav", "output.wav", filter_chain, chunk_seconds=30)
```

## Related Concepts

- {doc}`../core-concepts/wave` - Wave class architecture and methods
- {doc}`series-parallel-filters` - Combining filters in complex chains
- {doc}`performance` - Performance benchmarks and optimization
- {doc}`pytorch-integration` - Integration with PyTorch ecosystem

## External Resources

- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html) - Official PyTorch CUDA documentation
- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) - CUDA programming fundamentals
- [PyTorch Device Management](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device) - Device attribute documentation
- [torchaudio GPU Tutorial](https://pytorch.org/audio/stable/tutorials/device_avsr.html) - GPU acceleration in torchaudio

## Summary

Key takeaways for GPU acceleration in TorchFX:

1. **Device Management**: Use `Wave.to(device)` and `Module.to(device)` for consistent device placement
2. **Compatibility**: Ensure Wave objects and filters are on the same device
3. **Performance**: GPU acceleration is most beneficial for long audio, multi-channel files, and complex filter chains
4. **I/O Operations**: Always move tensors to CPU before saving to disk
5. **Best Practices**: Use conditional device selection and minimize data transfers

GPU acceleration can provide significant speedups for audio processing workflows when used correctly. Follow the patterns and best practices in this tutorial to leverage CUDA-enabled GPUs effectively in your TorchFX pipelines.
