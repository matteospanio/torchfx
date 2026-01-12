(performance)=
# Performance Optimization and Benchmarking

Learn how to measure, optimize, and maximize the performance of your TorchFX audio processing pipelines. This comprehensive guide covers benchmarking methodologies, GPU vs CPU performance comparisons, filter type trade-offs, and best practices for building high-throughput audio processing systems.

## Prerequisites

Before starting this guide, you should be familiar with:

- {doc}`../core-concepts/wave` - Wave class fundamentals
- {doc}`../core-concepts/pipeline-operator` - Pipeline operator basics
- {doc}`gpu-acceleration` - GPU device management
- {doc}`../filters/iir-filters` - IIR filter characteristics
- {doc}`../filters/fir-filters` - FIR filter characteristics

## Overview

Performance optimization in TorchFX involves understanding three key trade-offs:

| Dimension | Trade-off | Optimization Strategy |
|-----------|-----------|----------------------|
| **Execution Backend** | GPU vs CPU vs SciPy | Choose based on duration, channels, batch size |
| **Filter Type** | FIR vs IIR | Balance computational cost vs phase response |
| **API Pattern** | Classes vs Sequential vs Pipe | Select based on ergonomics vs performance needs |

```{mermaid}
graph TB
    subgraph "Performance Optimization Dimensions"
        AudioTask["Audio Processing Task"]

        subgraph Backend["Backend Selection"]
            GPU["GPU (CUDA)<br/>High parallelism"]
            CPU["CPU (PyTorch)<br/>Moderate performance"]
            SciPy["SciPy<br/>Baseline reference"]
        end

        subgraph FilterType["Filter Type"]
            FIR["FIR Filters<br/>Linear phase<br/>Higher compute"]
            IIR["IIR Filters<br/>Non-linear phase<br/>Lower compute"]
        end

        subgraph API["API Pattern"]
            Pipe["Pipeline Operator<br/>Auto sample rate"]
            Sequential["nn.Sequential<br/>Standard PyTorch"]
            Custom["Custom Module<br/>Maximum control"]
        end

        AudioTask --> Backend
        AudioTask --> FilterType
        AudioTask --> API

        Backend --> Optimize["Optimized Pipeline"]
        FilterType --> Optimize
        API --> Optimize
    end

    style AudioTask fill:#e1f5ff
    style Optimize fill:#e1ffe1
    style GPU fill:#fff5e1
    style FIR fill:#f5e1ff
    style Pipe fill:#ffe1e1
```

**Performance Optimization Framework** - Three key dimensions for optimizing TorchFX pipelines.

```{seealso}
For detailed GPU acceleration patterns, see {doc}`gpu-acceleration`. For comprehensive benchmarking infrastructure, see the benchmark suite in the `benchmark/` directory.
```

## Benchmark Methodology

TorchFX includes a comprehensive benchmarking suite that evaluates performance across different dimensions. The suite consists of three benchmark scripts, each targeting a specific aspect of performance.

### Benchmark Suite Architecture

```{mermaid}
graph TB
    subgraph "TorchFX Benchmark Suite"
        API["api_bench.py<br/>API Pattern Comparison"]
        FIR["fir_bench.py<br/>FIR Filter Performance"]
        IIR["iir_bench.py<br/>IIR Filter Performance"]
    end

    subgraph "Common Test Parameters"
        SR["Sample Rate: 44.1 kHz"]
        REP["Repetitions: 50"]
        Signal["Signal: Random noise<br/>Float32, normalized"]
    end

    subgraph "Variable Parameters"
        Duration["Duration: 1s - 10 min"]
        Channels["Channels: 1, 2, 4, 8, 12"]
        Backends["Backends: GPU, CPU, SciPy"]
    end

    subgraph "Output"
        CSV["CSV Results<br/>.out files"]
    end

    API --> SR
    FIR --> SR
    IIR --> SR

    API --> REP
    FIR --> REP
    IIR --> REP

    FIR --> Duration
    FIR --> Channels
    IIR --> Duration
    IIR --> Channels

    API --> Backends
    FIR --> Backends
    IIR --> Backends

    API --> CSV
    FIR --> CSV
    IIR --> CSV

    style API fill:#e1f5ff
    style FIR fill:#e8f5e1
    style IIR fill:#fff5e1
```

**Benchmark Suite Organization** - Three complementary benchmarks measuring different performance aspects.

### Test Signal Generation

All benchmarks use consistent signal generation to ensure comparable results:

```python
import numpy as np

def create_audio(sample_rate, duration, num_channels):
    """Generate multi-channel random noise for benchmarking.

    Parameters
    ----------
    sample_rate : int
        Sample rate in Hz (typically 44100)
    duration : float
        Duration in seconds
    num_channels : int
        Number of audio channels

    Returns
    -------
    signal : np.ndarray
        Shape (num_channels, num_samples), float32, normalized to [-1, 1]
    """
    signal = np.random.randn(num_channels, int(sample_rate * duration))
    signal = signal.astype(np.float32)
    # Normalize each channel independently
    signal /= np.max(np.abs(signal), axis=1, keepdims=True)
    return signal
```

**Key Characteristics**:
- **Distribution**: Gaussian random noise (`np.random.randn`)
- **Data Type**: Float32 for GPU compatibility
- **Normalization**: Per-channel normalization to [-1, 1] range
- **Deterministic**: Same random seed produces consistent results (when seeded)

```{tip}
Using random noise ensures benchmarks test worst-case performance without special structure or patterns that could be optimized by the hardware.
```

### Timing Methodology

All benchmarks use Python's {func}`timeit.timeit` for accurate timing measurements:

```python
import timeit

# Standard pattern across all benchmarks
REP = 50  # Number of repetitions

# Time the operation
elapsed = timeit.timeit(
    lambda: process_audio(wave, filter_chain),
    number=REP
)

# Report average time per iteration
avg_time = elapsed / REP
print(f"Average time: {avg_time:.6f}s")
```

**Timing Best Practices**:

1. **Multiple Repetitions**: 50 repetitions minimize variance and warm-up effects
2. **Lambda Wrapper**: Captures all setup costs in the closure
3. **Average Reporting**: Reports per-iteration time for easy comparison
4. **Excluded Overhead**: Setup (loading, coefficient computation) excluded from timing

```{important}
Timing only measures the core processing operation. Setup costs like loading audio files, computing filter coefficients, and device transfers are performed **before** timing begins.
```

## API Performance Comparison

The `api_bench.py` benchmark compares different API patterns for applying the same filter chain to audio. This helps users understand the performance and ergonomic trade-offs of different coding styles.

### Test Configuration

**Filter Chain**: 6 cascaded IIR filters
- 3 × High-pass Chebyshev Type I filters (20 Hz, 60 Hz, 65 Hz)
- 3 × Low-pass Butterworth filters (5000 Hz, 4900 Hz, 4850 Hz)

**Test Signal**: 8-channel audio, 2 minutes duration, 44.1 kHz sample rate

```{mermaid}
graph LR
    Input["Input Audio<br/>8 channels, 2 min"] --> F1["HiChebyshev1<br/>20 Hz"]
    F1 --> F2["HiChebyshev1<br/>60 Hz"]
    F2 --> F3["HiChebyshev1<br/>65 Hz"]
    F3 --> F4["LoButterworth<br/>5000 Hz"]
    F4 --> F5["LoButterworth<br/>4900 Hz"]
    F5 --> F6["LoButterworth<br/>4850 Hz"]
    F6 --> Output["Filtered Output<br/>8 channels, 2 min"]

    style Input fill:#e1f5ff
    style Output fill:#e1ffe1
```

**API Benchmark Filter Chain** - Six cascaded IIR filters applied to 8-channel audio.

### API Pattern 1: Custom nn.Module Class

A traditional PyTorch approach using a custom {class}`torch.nn.Module`:

```python
from torch import nn
from torchfx.filter import HiChebyshev1, LoButterworth

class FilterChain(nn.Module):
    """Custom module for filter chain."""

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

# Usage
fchain = FilterChain(signal.fs)
result = fchain(signal.ys)
```

**Characteristics**:
- **Sample Rate**: Must be passed explicitly to `__init__`
- **Reusability**: Can be instantiated once and reused
- **Flexibility**: Full control over forward pass logic
- **Overhead**: Standard PyTorch `nn.Module` call overhead

### API Pattern 2: nn.Sequential

Using PyTorch's built-in {class}`torch.nn.Sequential` container:

```python
from torch.nn import Sequential
from torchfx.filter import HiChebyshev1, LoButterworth

# Create filter chain inline
fchain = Sequential(
    HiChebyshev1(20, fs=signal.fs),
    HiChebyshev1(60, fs=signal.fs),
    HiChebyshev1(65, fs=signal.fs),
    LoButterworth(5000, fs=signal.fs),
    LoButterworth(4900, fs=signal.fs),
    LoButterworth(4850, fs=signal.fs),
)

# Apply to audio tensor
result = fchain(signal.ys)
```

**Characteristics**:
- **Sample Rate**: Must be passed explicitly to each filter
- **Simplicity**: No custom class needed
- **Performance**: Identical to custom `nn.Module` (same underlying mechanism)
- **Flexibility**: Can add/remove filters easily

```{note}
`nn.Sequential` and custom `nn.Module` classes have **identical performance** characteristics. The choice between them is purely about code organization and readability.
```

### API Pattern 3: Pipeline Operator (Pipe)

TorchFX's idiomatic {term}`pipeline operator` pattern:

```python
from torchfx import Wave
from torchfx.filter import HiChebyshev1, LoButterworth

# Apply filters using pipe operator
result = (
    signal
    | HiChebyshev1(20)
    | HiChebyshev1(60)
    | HiChebyshev1(65)
    | LoButterworth(5000)
    | LoButterworth(4900)
    | LoButterworth(4850)
)
```

**Characteristics**:
- **Sample Rate**: **Automatically configured** from {class}`~torchfx.Wave` object
- **Ergonomics**: Most readable and concise
- **Safety**: Eliminates sample rate mismatch errors
- **Performance**: Minimal overhead for automatic configuration

```{tip}
The pipeline operator is the **recommended pattern** for TorchFX. It provides the best balance of readability, safety (automatic sample rate configuration), and performance.
```

### API Pattern 4: SciPy Baseline

Pure NumPy/SciPy implementation for baseline comparison:

```python
from scipy.signal import butter, cheby1, lfilter

# Pre-compute filter coefficients
b1, a1 = cheby1(2, 0.5, 20, btype='high', fs=SAMPLE_RATE)
b2, a2 = cheby1(2, 0.5, 60, btype='high', fs=SAMPLE_RATE)
b3, a3 = cheby1(2, 0.5, 65, btype='high', fs=SAMPLE_RATE)
b4, a4 = butter(2, 5000, btype='low', fs=SAMPLE_RATE)
b5, a5 = butter(2, 4900, btype='low', fs=SAMPLE_RATE)
b6, a6 = butter(2, 4850, btype='low', fs=SAMPLE_RATE)

# Apply filters sequentially
filtered = lfilter(b1, a1, signal)
filtered = lfilter(b2, a2, filtered)
filtered = lfilter(b3, a3, filtered)
filtered = lfilter(b4, a4, filtered)
filtered = lfilter(b5, a5, filtered)
filtered = lfilter(b6, a6, filtered)
```

**Characteristics**:
- **Performance**: Optimized C implementation
- **GPU**: No GPU acceleration available
- **Integration**: Requires NumPy arrays (no PyTorch tensors)
- **Baseline**: Reference for CPU performance comparison

### API Performance Summary

| API Pattern | Sample Rate Config | Ergonomics | Performance | Use Case |
|-------------|-------------------|------------|-------------|----------|
| Custom `nn.Module` | Manual (`fs=`) | Good | Fast | Complex custom logic |
| `nn.Sequential` | Manual (`fs=`) | Very Good | Fast | Standard PyTorch integration |
| **Pipeline Operator** | **Automatic** | **Excellent** | **Fast** | **Recommended for TorchFX** |
| SciPy `lfilter` | Manual (`fs=`) | Fair | Fast (CPU only) | Baseline comparison |

**Key Insight**: The pipeline operator provides automatic sample rate configuration with negligible performance overhead, making it the most ergonomic choice without sacrificing speed.

```{seealso}
{doc}`../core-concepts/pipeline-operator` - Detailed documentation on the pipeline operator pattern
```

## FIR Filter Performance

The `fir_bench.py` benchmark evaluates FIR filter performance across varying signal durations, channel counts, and execution backends (GPU, CPU, SciPy).

### FIR Test Configuration

**Filter Chain**: 5 cascaded FIR filters with varying tap counts
- DesignableFIR: 101 taps, 1000 Hz cutoff
- DesignableFIR: 102 taps, 5000 Hz cutoff
- DesignableFIR: 103 taps, 1500 Hz cutoff
- DesignableFIR: 104 taps, 1800 Hz cutoff
- DesignableFIR: 105 taps, 1850 Hz cutoff

**Test Parameters**:
- **Durations**: 5s, 60s, 180s, 300s, 600s (10 minutes)
- **Channels**: 1, 2, 4, 8, 12
- **Backends**: GPU (CUDA), CPU (PyTorch), SciPy (NumPy)
- **Repetitions**: 50 per configuration

```{mermaid}
graph TB
    subgraph "FIR Benchmark Test Matrix"
        Input["Input Signal<br/>Variable duration & channels"]

        subgraph Filters["FIR Filter Chain"]
            F1["FIR 101 taps<br/>1000 Hz"]
            F2["FIR 102 taps<br/>5000 Hz"]
            F3["FIR 103 taps<br/>1500 Hz"]
            F4["FIR 104 taps<br/>1800 Hz"]
            F5["FIR 105 taps<br/>1850 Hz"]
        end

        subgraph Durations["Duration Sweep"]
            D1["5 seconds"]
            D2["60 seconds"]
            D3["180 seconds"]
            D4["300 seconds"]
            D5["600 seconds"]
        end

        subgraph Channels["Channel Sweep"]
            C1["1 channel"]
            C2["2 channels"]
            C3["4 channels"]
            C4["8 channels"]
            C5["12 channels"]
        end

        Input --> F1
        F1 --> F2
        F2 --> F3
        F3 --> F4
        F4 --> F5

        Input --> Durations
        Input --> Channels
    end

    style Input fill:#e1f5ff
    style Filters fill:#fff5e1
```

**FIR Benchmark Configuration** - Tests performance across duration and channel count dimensions.

### FIR Coefficient Pre-Computation

FIR filters require coefficient computation before filtering. The benchmark explicitly pre-computes coefficients to exclude design time from performance measurements:

```python
import torch.nn as nn
from torchfx import Wave
from torchfx.filter import DesignableFIR

SAMPLE_RATE = 44100

# Create filter chain
fchain = nn.Sequential(
    DesignableFIR(num_taps=101, cutoff=1000, fs=SAMPLE_RATE),
    DesignableFIR(num_taps=102, cutoff=5000, fs=SAMPLE_RATE),
    DesignableFIR(num_taps=103, cutoff=1500, fs=SAMPLE_RATE),
    DesignableFIR(num_taps=104, cutoff=1800, fs=SAMPLE_RATE),
    DesignableFIR(num_taps=105, cutoff=1850, fs=SAMPLE_RATE),
)

# Pre-compute coefficients before timing
for f in fchain:
    f.compute_coefficients()

# Now ready for benchmarking (coefficient design excluded)
```

**Why Pre-Compute?**
1. **Separation of Concerns**: Design time vs filtering time are separate
2. **Realistic Use Case**: Coefficients are typically computed once and reused
3. **Fair Comparison**: SciPy baseline also pre-computes coefficients

```{important}
In production code, call {meth}`compute_coefficients()` once during initialization, then reuse the filter for processing multiple audio files.
```

### FIR Device Transfer Pattern

The benchmark demonstrates proper device management for GPU acceleration:

```python
import timeit

# GPU benchmarking
wave.to("cuda")
fchain.to("cuda")
gpu_time = timeit.timeit(lambda: wave | fchain, number=REP)

# CPU benchmarking (transfer back)
wave.to("cpu")
fchain.to("cpu")
cpu_time = timeit.timeit(lambda: wave | fchain, number=REP)

# Calculate speedup
speedup = cpu_time / gpu_time
print(f"GPU speedup: {speedup:.2f}x")
```

**Device Transfer Rules**:
1. Move both `Wave` and filter chain to the same device
2. Pre-compute coefficients **before** moving to GPU
3. Time only the filtering operation (exclude transfers)
4. Move back to CPU for result saving

```{seealso}
{doc}`gpu-acceleration` - Comprehensive guide to GPU device management
```

### FIR Performance Characteristics

FIR filters have distinct performance characteristics compared to IIR filters:

| Characteristic | FIR Filters | Reason |
|---------------|-------------|---------|
| **Computational Cost** | Higher | Convolution with many taps |
| **GPU Advantage** | Excellent | High parallelism in convolution |
| **Memory Footprint** | Larger | Must store all tap coefficients |
| **Scaling with Taps** | Linear O(N) | More taps = more multiply-accumulate operations |
| **Scaling with Channels** | Excellent | Independent per-channel convolution |

**When FIR Filters Excel**:
- **Long signals** (>60s): Amortizes setup overhead
- **Many channels** (≥4): Parallel convolution across channels
- **GPU available**: Convolution is highly parallel
- **Linear phase required**: Only FIR can provide linear phase

**When to Avoid FIR**:
- **Real-time processing**: IIR filters have lower latency
- **Limited memory**: FIR coefficients consume more memory
- **CPU-only, short signals**: IIR may be faster

```{tip}
For steep frequency responses, FIR filters require many taps (100+). Consider IIR filters if phase linearity is not critical.
```

### SciPy FIR Baseline Implementation

The benchmark includes a SciPy baseline for CPU performance comparison:

```python
from scipy.signal import firwin, lfilter

# Design FIR coefficients using scipy
b1 = firwin(101, 1000, fs=SAMPLE_RATE)
b2 = firwin(102, 5000, fs=SAMPLE_RATE)
b3 = firwin(103, 1500, fs=SAMPLE_RATE)
b4 = firwin(104, 1800, fs=SAMPLE_RATE)
b5 = firwin(105, 1850, fs=SAMPLE_RATE)

# Apply filters sequentially
a = [1]  # FIR filters have a = [1]
filtered = lfilter(b1, a, signal)
filtered = lfilter(b2, a, filtered)
filtered = lfilter(b3, a, filtered)
filtered = lfilter(b4, a, filtered)
filtered = lfilter(b5, a, filtered)
```

**SciPy Characteristics**:
- **CPU-only**: No GPU acceleration
- **Optimized**: Uses NumPy's optimized C/Fortran backend
- **Reference**: Establishes baseline CPU performance
- **Compatibility**: Requires NumPy arrays (not PyTorch tensors)

## IIR Filter Performance

The `iir_bench.py` benchmark evaluates IIR (Infinite Impulse Response) filter performance using the same test matrix as FIR benchmarks but with recursive filters.

### IIR Test Configuration

**Filter Chain**: 4 cascaded IIR filters
- HiButterworth: 1000 Hz cutoff, order 2
- LoButterworth: 5000 Hz cutoff, order 2
- HiChebyshev1: 1500 Hz cutoff, order 2
- LoChebyshev1: 1800 Hz cutoff, order 2

**Test Parameters**:
- **Durations**: 1s, 5s, 180s, 300s, 600s (10 minutes)
- **Channels**: 1, 2, 4, 8, 12
- **Backends**: GPU (CUDA), CPU (PyTorch), SciPy (NumPy)
- **Repetitions**: 50 per configuration

```{mermaid}
graph TB
    subgraph "IIR Benchmark Architecture"
        Input["Input Signal<br/>Variable duration & channels"]

        subgraph Chain["IIR Filter Chain"]
            F1["HiButterworth<br/>1000 Hz, order 2"]
            F2["LoButterworth<br/>5000 Hz, order 2"]
            F3["HiChebyshev1<br/>1500 Hz, order 2"]
            F4["LoChebyshev1<br/>1800 Hz, order 2"]
        end

        subgraph Setup["IIR-Specific Setup"]
            Compute["compute_coefficients()<br/>Design b, a coefficients"]
            MoveCoeff["move_coeff('cuda'/'cpu')<br/>Transfer to device"]
        end

        subgraph Backends["Execution Backends"]
            GPU["GPU: fchain(wave.ys)"]
            CPU["CPU: fchain(wave.ys)"]
            SciPy["SciPy: lfilter(b, a, signal)"]
        end

        Input --> F1
        F1 --> F2
        F2 --> F3
        F3 --> F4

        F1 --> Compute
        Compute --> MoveCoeff
        MoveCoeff --> GPU
        MoveCoeff --> CPU

        F1 --> SciPy
    end

    style Input fill:#e1f5ff
    style Chain fill:#fff5e1
    style Setup fill:#e8f5e1
```

**IIR Benchmark Structure** - Shows coefficient management and execution backends.

### IIR Coefficient Management

Unlike FIR filters, IIR filters have both numerator (`b`) and denominator (`a`) coefficients that must be explicitly managed:

```python
import torch.nn as nn
from torchfx import Wave
from torchfx.filter import HiButterworth, LoButterworth

SAMPLE_RATE = 44100

# Create IIR filter chain
fchain = nn.Sequential(
    HiButterworth(cutoff=1000, order=2, fs=SAMPLE_RATE),
    LoButterworth(cutoff=5000, order=2, fs=SAMPLE_RATE),
)

# Move wave and module to GPU
wave.to("cuda")
fchain.to("cuda")

# IIR-specific: compute and move coefficients
for f in fchain:
    f.compute_coefficients()  # Design b, a coefficients
    f.move_coeff("cuda")       # Move coefficients to GPU

# Now ready for GPU processing
result = fchain(wave.ys)
```

**Two-Step Device Transfer**:
1. **Module transfer**: `fchain.to("cuda")` moves module parameters
2. **Coefficient transfer**: `f.move_coeff("cuda")` moves filter coefficients

```{warning}
For IIR filters, you must **both** move the module to the device **and** call {meth}`move_coeff()`. Forgetting the second step will cause runtime errors.
```

### IIR vs FIR Performance Trade-offs

IIR and FIR filters have fundamentally different performance characteristics:

| Aspect | IIR Filters | FIR Filters |
|--------|-------------|-------------|
| **Computational Cost** | Lower (fewer operations per sample) | Higher (convolution with many taps) |
| **Memory Footprint** | Small (few coefficients: b, a) | Large (many tap coefficients) |
| **GPU Advantage** | Moderate (less parallelism) | High (highly parallel convolution) |
| **Phase Response** | Non-linear | Can be linear (symmetric taps) |
| **Stability** | Can be unstable if poorly designed | Always stable |
| **Filter Order** | Achieves sharp cutoff with low order | Requires many taps for sharp cutoff |

**Performance Comparison Example**:

```python
# IIR: Order 8 Butterworth (16 coefficients total)
iir_filter = LoButterworth(cutoff=1000, order=8, fs=44100)
# Coefficients: b (9 values) + a (9 values) = 18 total

# Equivalent FIR: ~150+ taps for similar frequency response
fir_filter = DesignableFIR(num_taps=151, cutoff=1000, fs=44100)
# Coefficients: 151 tap values

# IIR is ~8x more memory-efficient and faster on CPU
# FIR has better GPU parallelism and linear phase
```

**Choosing Between IIR and FIR**:

```{mermaid}
flowchart TD
    Start["Choose Filter Type"]

    LinearPhase{"Linear phase<br/>required?"}
    Stability{"Stability<br/>critical?"}
    GPU{"GPU<br/>available?"}
    Memory{"Memory<br/>constrained?"}

    UseFIR["Use FIR Filters<br/>✓ Linear phase<br/>✓ Always stable<br/>✓ GPU-friendly"]
    UseIIR["Use IIR Filters<br/>✓ Low memory<br/>✓ Low latency<br/>✓ Efficient CPU"]

    Start --> LinearPhase
    LinearPhase -->|Yes| UseFIR
    LinearPhase -->|No| Stability
    Stability -->|Critical| UseFIR
    Stability -->|Not critical| GPU
    GPU -->|Yes, long signals| UseFIR
    GPU -->|No or short signals| Memory
    Memory -->|Yes| UseIIR
    Memory -->|No| UseFIR

    style UseFIR fill:#e1ffe1
    style UseIIR fill:#e1f5ff
```

**Filter Type Selection Decision Tree** - Choose based on phase, stability, and resource constraints.

### IIR SciPy Baseline

The IIR benchmark includes SciPy baseline for comparison:

```python
from scipy.signal import butter, cheby1, lfilter

# Design IIR coefficients
b1, a1 = butter(2, 1000, btype='high', fs=SAMPLE_RATE)
b2, a2 = butter(2, 5000, btype='low', fs=SAMPLE_RATE)
b3, a3 = cheby1(2, 0.5, 1500, btype='high', fs=SAMPLE_RATE)
b4, a4 = cheby1(2, 0.5, 1800, btype='low', fs=SAMPLE_RATE)

# Apply filters sequentially
filtered = lfilter(b1, a1, signal)
filtered = lfilter(b2, a2, filtered)
filtered = lfilter(b3, a3, filtered)
filtered = lfilter(b4, a4, filtered)
```

**SciPy IIR Performance**:
- **CPU-optimized**: Highly optimized C implementation
- **No GPU**: SciPy doesn't support CUDA
- **Baseline**: Reference for CPU performance
- **Filter Design**: Uses standard signal processing algorithms

## Performance Optimization Guidelines

Based on the benchmarking results, follow these guidelines to optimize your TorchFX pipelines.

### When to Use GPU Acceleration

GPU acceleration provides the greatest benefit under specific conditions:

```{mermaid}
flowchart TD
    Start["Audio Processing Task"]

    CheckDuration{"Signal duration<br/>> 60 seconds?"}
    CheckChannels{"Channels ≥ 4?"}
    CheckBatch{"Batch processing<br/>multiple files?"}
    CheckFIR{"Using FIR filters<br/>with >100 taps?"}
    CheckRealtime{"Real-time<br/>low-latency requirement?"}

    UseGPU["✓ Use GPU<br/>wave.to('cuda')<br/>fchain.to('cuda')"]
    UseCPU["✓ Use CPU<br/>Default or wave.to('cpu')"]

    Start --> CheckDuration
    CheckDuration -->|Yes| UseGPU
    CheckDuration -->|No| CheckChannels
    CheckChannels -->|Yes| UseGPU
    CheckChannels -->|No| CheckBatch
    CheckBatch -->|Yes| UseGPU
    CheckBatch -->|No| CheckFIR
    CheckFIR -->|Yes| UseGPU
    CheckFIR -->|No| CheckRealtime
    CheckRealtime -->|Yes| UseCPU
    CheckRealtime -->|No| UseGPU

    style UseGPU fill:#e1ffe1
    style UseCPU fill:#e1f5ff
```

**GPU Decision Tree** - Follow this flowchart to determine optimal execution backend.

**GPU Performance Sweet Spot**:

| Factor | Threshold | Reasoning |
|--------|-----------|-----------|
| **Duration** | > 60 seconds | Amortizes data transfer overhead |
| **Channels** | ≥ 4 channels | Exploits parallel processing |
| **Batch Size** | > 5 files | Transfer overhead amortized across batch |
| **FIR Taps** | > 100 taps | Convolution highly parallelizable |
| **IIR Chain** | ≥ 3 filters | Accumulated compute benefits |

**CPU Preferred Cases**:
- **Real-time processing**: More predictable latency
- **Short signals** (<30s): Transfer overhead dominates
- **Single channel**: Insufficient parallelism
- **IIR filters only**: Less GPU benefit than FIR

```{tip}
When in doubt, benchmark your specific workload. Use the patterns from the benchmark suite as templates.
```

### Filter Chain Optimization

Optimize filter chains by pre-computing coefficients and reusing filters:

```python
import torch.nn as nn
from torchfx import Wave
from torchfx.filter import DesignableFIR, HiButterworth

SAMPLE_RATE = 44100

# Create filter chain
fchain = nn.Sequential(
    DesignableFIR(num_taps=101, cutoff=1000, fs=SAMPLE_RATE),
    HiButterworth(cutoff=500, order=2, fs=SAMPLE_RATE),
)

# Pre-compute coefficients once during initialization
for f in fchain:
    f.compute_coefficients()

# For IIR filters, also move coefficients to device
device = "cuda" if torch.cuda.is_available() else "cpu"
fchain.to(device)

for f in fchain:
    if hasattr(f, 'move_coeff'):
        f.move_coeff(device)

# Process multiple files without re-computing coefficients
audio_files = ["song1.wav", "song2.wav", "song3.wav"]

for audio_file in audio_files:
    wave = Wave.from_file(audio_file).to(device)
    result = wave | fchain  # Uses cached coefficients
    result.to("cpu").save(f"processed_{audio_file}")
```

**Optimization Benefits**:
1. **Coefficient caching**: Compute once, reuse for all files
2. **Device pinning**: Keep filters on GPU across iterations
3. **Batch amortization**: Setup cost amortized over multiple files

### Device Placement Strategy

Minimize device transfers by keeping processing on a single device:

```python
import torch
import torchfx as fx

# Strategy 1: Single device throughout (RECOMMENDED)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load and move to device once
wave = fx.Wave.from_file("audio.wav").to(device)

# Create filter chain on same device
fchain = torch.nn.Sequential(
    fx.filter.HiButterworth(cutoff=80, order=2),
    fx.filter.LoButterworth(cutoff=12000, order=4),
).to(device)

# All operations on same device
result = wave | fchain

# Move to CPU only for final I/O
result.to("cpu").save("output.wav")
```

**Avoid Inefficient Transfers** (Anti-pattern):

```python
# ❌ WRONG: Unnecessary device transfers
wave = fx.Wave.from_file("audio.wav").to("cuda")
result1 = wave.to("cpu") | cpu_filter  # Transfer 1
result2 = result1.to("cuda") | gpu_filter  # Transfer 2
result3 = result2.to("cpu") | cpu_filter2  # Transfer 3

# ✅ CORRECT: Single device
device = "cuda" if torch.cuda.is_available() else "cpu"
wave = fx.Wave.from_file("audio.wav").to(device)
cpu_filter.to(device)
gpu_filter.to(device)
cpu_filter2.to(device)
result = wave | cpu_filter | gpu_filter | cpu_filter2
```

**Device Transfer Costs**:
- **CPU → GPU**: O(n) where n = number of samples
- **GPU → CPU**: O(n) where n = number of samples
- **Impact**: Can dominate for short signals

```{seealso}
{doc}`gpu-acceleration` - Comprehensive device management patterns
```

### Memory Management Best Practices

Optimize memory usage for large-scale processing:

| Optimization | Implementation | Impact |
|-------------|----------------|---------|
| **In-place operations** | Use effects that modify tensors in-place where possible | Reduces memory allocations |
| **Chunked processing** | Process long audio in chunks | Prevents GPU OOM errors |
| **Coefficient caching** | Pre-compute and reuse filter coefficients | Eliminates redundant computation |
| **Device pinning** | Keep frequently-used filters on device | Reduces transfer overhead |
| **Batch size tuning** | Adjust batch size to fit GPU memory | Maximizes throughput |

**Memory-Efficient Chunked Processing Example**:

```python
import torch
import torchfx as fx

def process_long_audio_chunked(wave, fchain, chunk_duration=60):
    """Process very long audio in chunks to manage GPU memory.

    Parameters
    ----------
    wave : Wave
        Input audio (can be on CPU or GPU)
    fchain : nn.Module
        Filter chain (must be on same device as intended chunks)
    chunk_duration : float
        Chunk duration in seconds

    Returns
    -------
    Wave
        Processed audio
    """
    chunk_samples = int(chunk_duration * wave.fs)
    num_chunks = (wave.ys.size(-1) + chunk_samples - 1) // chunk_samples

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fchain.to(device)

    results = []
    for i in range(num_chunks):
        start = i * chunk_samples
        end = min((i + 1) * chunk_samples, wave.ys.size(-1))

        # Extract chunk and move to GPU
        chunk = fx.Wave(wave.ys[..., start:end], wave.fs).to(device)

        # Process chunk on GPU
        processed_chunk = chunk | fchain

        # Move back to CPU and store
        results.append(processed_chunk.ys.cpu())

        # Clear GPU cache
        if device == "cuda":
            torch.cuda.empty_cache()

    # Concatenate results
    return fx.Wave(torch.cat(results, dim=-1), wave.fs)

# Usage
wave = fx.Wave.from_file("10_hour_recording.wav")
fchain = fx.filter.LoButterworth(cutoff=1000, order=4)
result = process_long_audio_chunked(wave, fchain, chunk_duration=60)
result.save("processed.wav")
```

**Chunked Processing Benefits**:
- Processes arbitrarily long audio without OOM errors
- Keeps GPU utilization high
- Balances memory usage with throughput

## Benchmarking Your Own Pipelines

Use these patterns to benchmark your custom TorchFX pipelines.

### Basic Benchmarking Template

```python
import timeit
import torch
import torchfx as fx
import numpy as np

# Configuration
SAMPLE_RATE = 44100
DURATION = 60  # seconds
NUM_CHANNELS = 4
REP = 50  # repetitions for timing

# Generate test signal
signal = np.random.randn(NUM_CHANNELS, int(SAMPLE_RATE * DURATION))
signal = signal.astype(np.float32)
signal /= np.max(np.abs(signal), axis=1, keepdims=True)
wave = fx.Wave(signal, SAMPLE_RATE)

# Create your processing pipeline
pipeline = torch.nn.Sequential(
    fx.filter.HiButterworth(cutoff=100, order=2),
    fx.filter.LoButterworth(cutoff=10000, order=4),
    fx.effect.Normalize(peak=0.9),
)

# Pre-compute coefficients
for module in pipeline:
    if hasattr(module, 'compute_coefficients'):
        module.compute_coefficients()

# Benchmark CPU
wave.to("cpu")
pipeline.to("cpu")
cpu_time = timeit.timeit(lambda: wave | pipeline, number=REP)

# Benchmark GPU (if available)
if torch.cuda.is_available():
    wave.to("cuda")
    pipeline.to("cuda")

    # Move IIR coefficients if needed
    for module in pipeline:
        if hasattr(module, 'move_coeff'):
            module.move_coeff("cuda")

    gpu_time = timeit.timeit(lambda: wave | pipeline, number=REP)

    print(f"CPU time: {cpu_time/REP:.6f}s")
    print(f"GPU time: {gpu_time/REP:.6f}s")
    print(f"Speedup: {cpu_time/gpu_time:.2f}x")
else:
    print(f"CPU time: {cpu_time/REP:.6f}s")
    print("GPU not available")
```

### Multi-Configuration Benchmark

Test performance across multiple configurations:

```python
import timeit
import torch
import torchfx as fx
import numpy as np
import pandas as pd

SAMPLE_RATE = 44100
REP = 50

# Test configurations
durations = [5, 30, 60, 120, 300]  # seconds
channel_counts = [1, 2, 4, 8]

# Create filter chain
pipeline = torch.nn.Sequential(
    fx.filter.LoButterworth(cutoff=1000, order=4),
    fx.filter.HiButterworth(cutoff=100, order=2),
)

# Pre-compute coefficients
for module in pipeline:
    if hasattr(module, 'compute_coefficients'):
        module.compute_coefficients()

# Benchmark grid
results = []

for duration in durations:
    for channels in channel_counts:
        # Generate test signal
        signal = np.random.randn(channels, int(SAMPLE_RATE * duration))
        signal = signal.astype(np.float32)
        signal /= np.max(np.abs(signal), axis=1, keepdims=True)
        wave = fx.Wave(signal, SAMPLE_RATE)

        # CPU benchmark
        wave.to("cpu")
        pipeline.to("cpu")
        cpu_time = timeit.timeit(lambda: wave | pipeline, number=REP) / REP

        # GPU benchmark
        if torch.cuda.is_available():
            wave.to("cuda")
            pipeline.to("cuda")
            gpu_time = timeit.timeit(lambda: wave | pipeline, number=REP) / REP
            speedup = cpu_time / gpu_time
        else:
            gpu_time = None
            speedup = None

        results.append({
            'duration': duration,
            'channels': channels,
            'cpu_time': cpu_time,
            'gpu_time': gpu_time,
            'speedup': speedup
        })

# Convert to DataFrame for analysis
df = pd.DataFrame(results)
print(df.to_string(index=False))

# Save to CSV
df.to_csv("benchmark_results.csv", index=False)
```

### Profiling with PyTorch Profiler

For detailed performance analysis, use PyTorch's built-in profiler:

```python
import torch
import torchfx as fx

# Create pipeline
wave = fx.Wave.from_file("audio.wav").to("cuda")
pipeline = torch.nn.Sequential(
    fx.filter.LoButterworth(cutoff=1000, order=4),
    fx.filter.HiButterworth(cutoff=100, order=2),
).to("cuda")

# Profile the pipeline
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    result = wave | pipeline

# Print profiling results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Export for visualization
prof.export_chrome_trace("trace.json")
# Open trace.json in chrome://tracing for detailed visualization
```

```{seealso}
[PyTorch Profiler Documentation](https://pytorch.org/docs/stable/profiler.html) - Official guide to PyTorch profiling tools
```

## Complete Benchmarking Examples

These complete examples demonstrate how to run comprehensive benchmarks for your specific use cases.

### Example 1: API Pattern Comparison

Compare different API patterns for your filter chain:

```python
import timeit
import numpy as np
from torch import nn
from torchfx import Wave
from torchfx.filter import HiChebyshev1, LoButterworth

SAMPLE_RATE = 44100
DURATION = 120  # 2 minutes
NUM_CHANNELS = 8
REP = 50

# Generate test signal
signal_data = np.random.randn(NUM_CHANNELS, int(SAMPLE_RATE * DURATION))
signal_data = signal_data.astype(np.float32)
signal_data /= np.max(np.abs(signal_data), axis=1, keepdims=True)
wave = Wave(signal_data, SAMPLE_RATE)

# Pattern 1: Custom nn.Module class
class FilterChain(nn.Module):
    def __init__(self, fs):
        super().__init__()
        self.f1 = HiChebyshev1(20, fs=fs)
        self.f2 = LoButterworth(5000, fs=fs)

    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        return x

def test_class():
    fchain = FilterChain(wave.fs)
    return fchain(wave.ys)

# Pattern 2: nn.Sequential
def test_sequential():
    fchain = nn.Sequential(
        HiChebyshev1(20, fs=wave.fs),
        LoButterworth(5000, fs=wave.fs),
    )
    return fchain(wave.ys)

# Pattern 3: Pipe operator
def test_pipe():
    return wave | HiChebyshev1(20) | LoButterworth(5000)

# Benchmark each pattern
class_time = timeit.timeit(test_class, number=REP)
seq_time = timeit.timeit(test_sequential, number=REP)
pipe_time = timeit.timeit(test_pipe, number=REP)

print(f"Custom class: {class_time/REP:.6f}s")
print(f"nn.Sequential: {seq_time/REP:.6f}s")
print(f"Pipe operator: {pipe_time/REP:.6f}s")
```

### Example 2: FIR Filter Performance Analysis

Comprehensive FIR filter benchmarking across durations and channel counts:

```python
import timeit
import numpy as np
import torch.nn as nn
import pandas as pd
from torchfx import Wave
from torchfx.filter import DesignableFIR

SAMPLE_RATE = 44100
REP = 50

# Test matrix
durations = [5, 60, 180, 300, 600]
channel_counts = [1, 2, 4, 8, 12]

results = []

for duration in durations:
    for channels in channel_counts:
        # Generate test signal
        signal = np.random.randn(channels, int(SAMPLE_RATE * duration))
        signal = signal.astype(np.float32)
        signal /= np.max(np.abs(signal), axis=1, keepdims=True)
        wave = Wave(signal, SAMPLE_RATE)

        # Create FIR filter chain
        fchain = nn.Sequential(
            DesignableFIR(num_taps=101, cutoff=1000, fs=SAMPLE_RATE),
            DesignableFIR(num_taps=102, cutoff=5000, fs=SAMPLE_RATE),
            DesignableFIR(num_taps=103, cutoff=1500, fs=SAMPLE_RATE),
        )

        # Pre-compute coefficients
        for f in fchain:
            f.compute_coefficients()

        # GPU benchmark
        wave.to("cuda")
        fchain.to("cuda")
        gpu_time = timeit.timeit(lambda: wave | fchain, number=REP) / REP

        # CPU benchmark
        wave.to("cpu")
        fchain.to("cpu")
        cpu_time = timeit.timeit(lambda: wave | fchain, number=REP) / REP

        results.append({
            'duration_sec': duration,
            'channels': channels,
            'gpu_time_sec': gpu_time,
            'cpu_time_sec': cpu_time,
            'speedup': cpu_time / gpu_time
        })

        print(f"Duration: {duration}s, Channels: {channels}, "
              f"GPU: {gpu_time:.6f}s, CPU: {cpu_time:.6f}s, "
              f"Speedup: {cpu_time/gpu_time:.2f}x")

# Save results
df = pd.DataFrame(results)
df.to_csv("fir_benchmark.csv", index=False)
print("\nResults saved to fir_benchmark.csv")
```

### Example 3: IIR Filter Performance Analysis

Complete IIR filter benchmarking with coefficient management:

```python
import timeit
import numpy as np
import torch.nn as nn
import pandas as pd
from torchfx import Wave
from torchfx.filter import HiButterworth, LoButterworth, HiChebyshev1, LoChebyshev1

SAMPLE_RATE = 44100
REP = 50

# Test matrix
durations = [1, 5, 180, 300, 600]
channel_counts = [1, 2, 4, 8, 12]

results = []

for duration in durations:
    for channels in channel_counts:
        # Generate test signal
        signal = np.random.randn(channels, int(SAMPLE_RATE * duration))
        signal = signal.astype(np.float32)
        signal /= np.max(np.abs(signal), axis=1, keepdims=True)
        wave = Wave(signal, SAMPLE_RATE)

        # Create IIR filter chain
        fchain = nn.Sequential(
            HiButterworth(cutoff=1000, order=2, fs=SAMPLE_RATE),
            LoButterworth(cutoff=5000, order=2, fs=SAMPLE_RATE),
            HiChebyshev1(cutoff=1500, order=2, fs=SAMPLE_RATE),
            LoChebyshev1(cutoff=1800, order=2, fs=SAMPLE_RATE),
        )

        # GPU benchmark
        wave.to("cuda")
        fchain.to("cuda")

        # Compute and move coefficients
        for f in fchain:
            f.compute_coefficients()
            f.move_coeff("cuda")

        gpu_time = timeit.timeit(
            lambda: fchain(wave.ys),
            number=REP
        ) / REP

        # CPU benchmark
        wave.to("cpu")
        fchain.to("cpu")

        for f in fchain:
            f.move_coeff("cpu")

        cpu_time = timeit.timeit(
            lambda: fchain(wave.ys),
            number=REP
        ) / REP

        results.append({
            'duration_sec': duration,
            'channels': channels,
            'gpu_time_sec': gpu_time,
            'cpu_time_sec': cpu_time,
            'speedup': cpu_time / gpu_time
        })

        print(f"Duration: {duration}s, Channels: {channels}, "
              f"GPU: {gpu_time:.6f}s, CPU: {cpu_time:.6f}s, "
              f"Speedup: {cpu_time/gpu_time:.2f}x")

# Save results
df = pd.DataFrame(results)
df.to_csv("iir_benchmark.csv", index=False)
print("\nResults saved to iir_benchmark.csv")
```

## Summary

Key takeaways for optimizing TorchFX performance:

1. **GPU Acceleration**: Use GPU for long signals (>60s), multi-channel audio (≥4 channels), and batch processing
2. **Filter Choice**: FIR filters excel on GPU with parallel convolution; IIR filters are more CPU-efficient
3. **API Pattern**: Pipeline operator provides best ergonomics with automatic sample rate configuration and minimal overhead
4. **Coefficient Caching**: Pre-compute filter coefficients once and reuse for multiple files
5. **Device Management**: Minimize transfers by keeping all processing on one device
6. **Memory**: Use chunked processing for very long audio files to prevent OOM errors
7. **Benchmarking**: Use the provided templates to measure performance of your specific pipelines

GPU acceleration can provide 5-20x speedups for appropriate workloads. Follow the decision trees and best practices in this guide to maximize throughput in your audio processing pipelines.

## Related Guides

- {doc}`gpu-acceleration` - Comprehensive GPU device management guide
- {doc}`../filters/iir-filters` - IIR filter design and usage
- {doc}`../filters/fir-filters` - FIR filter design and usage
- {doc}`pytorch-integration` - Integration with PyTorch ecosystem
- {doc}`multi-channel` - Multi-channel processing patterns

## External Resources

- [PyTorch Profiler](https://pytorch.org/docs/stable/profiler.html) - Profiling PyTorch code
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) - NVIDIA optimization guide
- [SciPy Signal Processing](https://docs.scipy.org/doc/scipy/reference/signal.html) - SciPy signal processing reference
- [PyTorch Performance Tuning](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html) - PyTorch optimization guide
