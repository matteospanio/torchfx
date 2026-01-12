# Benchmarking

Comprehensive guide to the TorchFX benchmarking suite for measuring and comparing performance of audio processing operations.

## Overview

The TorchFX benchmarking suite evaluates performance across three key dimensions:

1. **API patterns**: Comparing different usage patterns (FilterChain, Sequential, pipe operator)
2. **FIR filter performance**: GPU vs CPU vs SciPy implementations
3. **IIR filter performance**: GPU vs CPU vs SciPy implementations

All benchmarks compare TorchFX implementations against SciPy baselines to validate performance characteristics and identify optimization opportunities.

```{seealso}
{doc}`/guides/developer/testing` - Testing infrastructure
{doc}`/guides/advanced/performance` - Performance optimization guide
{doc}`/guides/advanced/gpu-acceleration` - GPU acceleration usage
```

## Benchmark Suite Structure

The benchmarking suite consists of three independent scripts:

| Script | Purpose | Comparisons | Output File |
|--------|---------|-------------|-------------|
| `api_bench.py` | API pattern comparison | FilterChain, Sequential, pipe operator, SciPy | `api_bench.out` |
| `fir_bench.py` | FIR filter performance | GPU vs CPU vs SciPy across varying durations and channels | `fir.out` |
| `iir_bench.py` | IIR filter performance | GPU vs CPU vs SciPy across varying durations and channels | `iir.out` |

All benchmarks use Python's `timeit` module for precise timing measurements and output results in CSV format for analysis and visualization.

## Benchmark Architecture

```{mermaid}
graph TB
    subgraph "Benchmark Scripts"
        API["api_bench.py<br/>API pattern comparison"]
        FIR["fir_bench.py<br/>FIR filter performance"]
        IIR["iir_bench.py<br/>IIR filter performance"]
    end

    subgraph "Test Signal Generation"
        CreateAudio["create_audio()<br/>np.random.randn()<br/>Normalized to [-1, 1]"]
    end

    subgraph "Implementations Under Test"
        TorchFX_GPU["torchfx on CUDA<br/>Wave.to('cuda')<br/>filter.to('cuda')"]
        TorchFX_CPU["torchfx on CPU<br/>Wave.to('cpu')<br/>filter.to('cpu')"]
        SciPy_Baseline["SciPy baseline<br/>scipy.signal.lfilter()"]
    end

    subgraph "Timing Infrastructure"
        TimeitModule["timeit.timeit()<br/>REP=50 repetitions"]
    end

    subgraph "Output"
        CSV["CSV files:<br/>api_bench.out<br/>fir.out<br/>iir.out"]
        Visualization["draw3.py<br/>Generates PNG plots"]
    end

    API --> CreateAudio
    FIR --> CreateAudio
    IIR --> CreateAudio

    CreateAudio --> TorchFX_GPU
    CreateAudio --> TorchFX_CPU
    CreateAudio --> SciPy_Baseline

    TorchFX_GPU --> TimeitModule
    TorchFX_CPU --> TimeitModule
    SciPy_Baseline --> TimeitModule

    TimeitModule --> CSV
    CSV --> Visualization
```

## Common Infrastructure

All benchmark scripts share common infrastructure for test signal generation and timing measurement.

### Test Signal Generation

Each benchmark uses the `create_audio()` function to generate synthetic test signals:

```python
def create_audio(duration, num_channels):
    """Create random audio signal for testing.

    Parameters
    ----------
    duration : int
        Duration in seconds
    num_channels : int
        Number of audio channels

    Returns
    -------
    np.ndarray
        Audio signal with shape (num_channels, samples)
    """
    samples = int(duration * SAMPLE_RATE)
    audio = np.random.randn(num_channels, samples)
    return audio / np.max(np.abs(audio))  # Normalize to [-1, 1]
```

**Normalization**: Signals are normalized to the range [-1, 1] to simulate realistic audio levels.

### Timing Methodology

All benchmarks use Python's `timeit.timeit()` function with consistent parameters:

```python
REP = 50  # Number of repetitions

# Measure execution time
time = timeit.timeit(lambda: function_under_test(), number=REP)
average_time = time / REP
```

**Why 50 repetitions?**
- Provides stable averages by reducing variance
- Balances accuracy with total benchmark runtime
- Minimizes impact of system noise and cache effects

### Standard Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `SAMPLE_RATE` | 44100 Hz | Standard CD-quality sampling rate |
| `REP` | 50 | Number of timing repetitions for averaging |
| `DURATION` | Varies | Audio duration in seconds (benchmark-specific) |
| `NUM_CHANNELS` | Varies | Number of audio channels (benchmark-specific) |

## API Benchmark

The API benchmark (`benchmark/api_bench.py`) compares different approaches to chaining filters, evaluating both ergonomics and performance.

### Tested Implementations

```{mermaid}
graph LR
    Signal["Wave object<br/>8 channels<br/>120 seconds"]

    subgraph "Four API Patterns"
        Method1["FilterChain class<br/>nn.Module subclass<br/>explicit forward()"]
        Method2["Sequential<br/>torch.nn.Sequential<br/>functional composition"]
        Method3["Pipe operator<br/>wave | filter1 | filter2"]
        Method4["SciPy baseline<br/>scipy.signal.lfilter()"]
    end

    Output["Filtered signal"]

    Signal --> Method1
    Signal --> Method2
    Signal --> Method3
    Signal --> Method4

    Method1 --> Output
    Method2 --> Output
    Method3 --> Output
    Method4 --> Output
```

### Implementation Patterns

#### FilterChain Class Pattern

Traditional PyTorch module composition with explicit `forward()` method:

```python
class FilterChain(nn.Module):
    """Custom filter chain implementation."""
    def __init__(self, filters):
        super().__init__()
        self.filters = nn.ModuleList(filters)

    def forward(self, x):
        for f in self.filters:
            x = f(x)
        return x

# Usage
chain = FilterChain([filter1, filter2, filter3])
output = chain(wave.ys)
```

**Characteristics**:
- Explicit control over execution
- Standard PyTorch pattern
- Requires boilerplate code

#### Sequential Pattern

PyTorch's built-in sequential container:

```python
from torch import nn

# Create sequential chain
chain = nn.Sequential(filter1, filter2, filter3)

# Apply to audio
output = chain(wave.ys)
```

**Characteristics**:
- Built-in PyTorch functionality
- Minimal boilerplate
- Standard functional composition

#### Pipe Operator Pattern

TorchFX's idiomatic API with automatic configuration:

```python
# Chain filters using pipe operator
output = wave | filter1 | filter2 | filter3
```

**Characteristics**:
- Most ergonomic syntax
- Automatic sample rate configuration
- Pythonic and readable

#### SciPy Baseline

Pure NumPy/SciPy implementation for comparison:

```python
import scipy.signal as signal

# Design filter coefficients
b1, a1 = signal.butter(N=order, Wn=cutoff, btype='high', fs=fs)

# Apply filter
output = signal.lfilter(b1, a1, audio)
```

**Characteristics**:
- CPU-only implementation
- No PyTorch overhead
- Industry-standard baseline

### Filter Configuration

All patterns apply the same six filters in series:

| Filter | Type | Cutoff Frequency | Purpose |
|--------|------|------------------|---------|
| HiChebyshev1 | High-pass | 20 Hz | Remove subsonic content |
| HiChebyshev1 | High-pass | 60 Hz | Remove hum |
| HiChebyshev1 | High-pass | 65 Hz | Additional hum removal |
| LoButterworth | Low-pass | 5000 Hz | Anti-aliasing |
| LoButterworth | Low-pass | 4900 Hz | Transition band shaping |
| LoButterworth | Low-pass | 4850 Hz | Final rolloff |

### Test Parameters

- **Duration**: 120 seconds (2 minutes)
- **Channels**: 8
- **Sample Rate**: 44100 Hz
- **Repetitions**: 50

### Output Format

CSV with the following structure:

```csv
filter_chain,sequential,pipe,scipy
<class_time>,<seq_time>,<pipe_time>,<scipy_time>
```

Each time value represents average execution time in seconds.

### Running API Benchmark

```bash
python benchmark/api_bench.py
```

**Expected output**:

```
API Benchmark
Duration: 120s, Channels: 8, Sample Rate: 44100Hz
Repetitions: 50

FilterChain: 1.234 seconds
Sequential:  1.235 seconds
Pipe:        1.236 seconds
SciPy:       1.450 seconds

Results saved to api_bench.out
```

## FIR Filter Benchmark

The FIR filter benchmark (`benchmark/fir_bench.py`) evaluates FIR filter performance across different audio durations and channel counts.

### Test Matrix

The benchmark tests across two dimensions:

| Dimension | Values |
|-----------|--------|
| **Durations** | 5, 60, 180, 300, 600 seconds |
| **Channels** | 1, 2, 4, 8, 12 |

**Total test cases**: 5 durations × 5 channel counts = 25 data points

```{mermaid}
graph TB
    subgraph "Test Variables"
        Durations["Durations (seconds)<br/>5, 60, 180, 300, 600"]
        Channels["Channels<br/>1, 2, 4, 8, 12"]
    end

    subgraph "Filter Chain"
        F1["DesignableFIR<br/>101 taps, 1000 Hz"]
        F2["DesignableFIR<br/>102 taps, 5000 Hz"]
        F3["DesignableFIR<br/>103 taps, 1500 Hz"]
        F4["DesignableFIR<br/>104 taps, 1800 Hz"]
        F5["DesignableFIR<br/>105 taps, 1850 Hz"]
    end

    subgraph "Implementations"
        GPU["GPU Implementation<br/>wave.to('cuda')<br/>fchain.to('cuda')"]
        CPU["CPU Implementation<br/>wave.to('cpu')<br/>fchain.to('cpu')"]
        SciPy["SciPy Implementation<br/>scipy.signal.firwin()<br/>scipy.signal.lfilter()"]
    end

    Durations --> F1
    Channels --> F1

    F1 --> F2
    F2 --> F3
    F3 --> F4
    F4 --> F5

    F5 --> GPU
    F5 --> CPU
    F5 --> SciPy
```

### Filter Configuration

The benchmark applies five `DesignableFIR` filters in series:

```python
# Create filter chain
fchain = nn.Sequential(
    DesignableFIR(numtaps=101, cutoff=1000, fs=44100),
    DesignableFIR(numtaps=102, cutoff=5000, fs=44100),
    DesignableFIR(numtaps=103, cutoff=1500, fs=44100),
    DesignableFIR(numtaps=104, cutoff=1800, fs=44100),
    DesignableFIR(numtaps=105, cutoff=1850, fs=44100),
)

# Pre-compute coefficients (excluded from timing)
for f in fchain:
    f.compute_coefficients()
```

**Important**: Filter coefficients are pre-computed before timing to measure only filtering performance, not coefficient design.

### Implementation Functions

#### GPU FIR Function

```python
def gpu_fir(wave):
    """Apply FIR filter chain on GPU."""
    return (wave | fchain).ys
```

Moves audio to GPU and applies filter chain using pipe operator.

#### CPU FIR Function

```python
def cpu_fir(wave):
    """Apply FIR filter chain on CPU."""
    return (wave | fchain).ys
```

Applies filter chain on CPU.

#### SciPy FIR Function

```python
def scipy_fir(audio):
    """Apply FIR filters using SciPy."""
    for f in fchain:
        b = f.coefficients.cpu().numpy()
        audio = signal.lfilter(b, [1.0], audio)
    return audio
```

Uses `scipy.signal.lfilter()` for baseline comparison.

### Test Execution Flow

For each combination of duration and channel count:

1. Generate test signal with `create_audio(duration, channels)`
2. Create `Wave` object from signal
3. Build filter chain with `nn.Sequential`
4. Pre-compute all filter coefficients
5. Move to GPU, time GPU execution
6. Move to CPU, time CPU execution
7. Design SciPy coefficients, time SciPy execution

### Output Format

CSV with the following structure:

```csv
time,channels,gpu,cpu,scipy
5,1,0.012,0.015,0.018
5,2,0.013,0.016,0.020
...
600,12,1.234,1.567,1.890
```

### Running FIR Benchmark

```bash
python benchmark/fir_bench.py
```

**Expected output**:

```
FIR Filter Benchmark
Sample Rate: 44100Hz
Repetitions: 50

Testing: 5s, 1 channel...
  GPU:   0.012s
  CPU:   0.015s
  SciPy: 0.018s

Testing: 5s, 2 channels...
  GPU:   0.013s
  CPU:   0.016s
  SciPy: 0.020s

...

Results saved to fir.out
```

## IIR Filter Benchmark

The IIR filter benchmark (`benchmark/iir_bench.py`) evaluates IIR filter performance with similar methodology to the FIR benchmark.

### Test Matrix

| Dimension | Values |
|-----------|--------|
| **Durations** | 1, 5, 180, 300, 600 seconds |
| **Channels** | 1, 2, 4, 8, 12 |

**Total test cases**: 5 durations × 5 channel counts = 25 data points

### Filter Configuration

The benchmark applies four IIR filters in series:

```python
fchain = nn.Sequential(
    HiButterworth(cutoff=1000, order=2, fs=44100),
    LoButterworth(cutoff=5000, order=2, fs=44100),
    HiChebyshev1(cutoff=1500, order=2, ripple=0.5, fs=44100),
    LoChebyshev1(cutoff=1800, order=2, ripple=0.5, fs=44100),
)
```

| Filter | Type | Cutoff | Order | Purpose |
|--------|------|--------|-------|---------|
| `HiButterworth` | High-pass | 1000 Hz | 2 | Remove low frequencies |
| `LoButterworth` | Low-pass | 5000 Hz | 2 | Remove high frequencies |
| `HiChebyshev1` | High-pass | 1500 Hz | 2 | Additional high-pass |
| `LoChebyshev1` | Low-pass | 1800 Hz | 2 | Additional low-pass |

### Implementation Details

```{mermaid}
graph TB
    subgraph "GPU Execution Path"
        GPU_Wave["Wave.to('cuda')"]
        GPU_Chain["fchain.to('cuda')"]
        GPU_Coeff["f.compute_coefficients()<br/>f.move_coeff('cuda')<br/>for each filter"]
        GPU_Execute["fchain(wave.ys)"]

        GPU_Wave --> GPU_Chain
        GPU_Chain --> GPU_Coeff
        GPU_Coeff --> GPU_Execute
    end

    subgraph "CPU Execution Path"
        CPU_Wave["Wave.to('cpu')"]
        CPU_Chain["fchain.to('cpu')"]
        CPU_Coeff["f.move_coeff('cpu')<br/>for each filter"]
        CPU_Execute["fchain(wave.ys)"]

        CPU_Wave --> CPU_Chain
        CPU_Chain --> CPU_Coeff
        CPU_Coeff --> CPU_Execute
    end

    subgraph "SciPy Execution Path"
        SciPy_Design["butter() / cheby1()<br/>Design filter coefficients"]
        SciPy_Filter["lfilter()<br/>Apply filters"]

        SciPy_Design --> SciPy_Filter
    end
```

#### GPU Filter Function

```python
def gpu_iir(wave):
    """Apply IIR filter chain on GPU."""
    # CRITICAL: Move both module and coefficients to GPU
    for f in fchain:
        f.compute_coefficients()
        f.move_coeff("cuda")
    return (wave | fchain).ys
```

**Important**: IIR filters require explicit coefficient movement to GPU.

#### CPU Filter Function

```python
def cpu_iir(wave):
    """Apply IIR filter chain on CPU."""
    # Move coefficients back to CPU
    for f in fchain:
        f.move_coeff("cpu")
    return (wave | fchain).ys
```

#### SciPy Filter Function

```python
def scipy_iir(audio):
    """Apply IIR filters using SciPy."""
    # Design Butterworth coefficients
    b1, a1 = signal.butter(N=2, Wn=1000, btype='high', fs=44100)
    # ... design other filters ...

    # Apply filters sequentially
    audio = signal.lfilter(b1, a1, audio)
    # ... apply other filters ...
    return audio
```

### Output Format

CSV with the following structure:

```csv
time,channels,gpu,cpu,scipy
1,1,0.005,0.008,0.010
1,2,0.006,0.009,0.012
...
600,12,0.987,1.234,1.567
```

### Running IIR Benchmark

```bash
python benchmark/iir_bench.py
```

**Expected output**:

```
IIR Filter Benchmark
Sample Rate: 44100Hz
Repetitions: 50

Testing: 1s, 1 channel...
  GPU:   0.005s
  CPU:   0.008s
  SciPy: 0.010s

Testing: 1s, 2 channels...
  GPU:   0.006s
  CPU:   0.009s
  SciPy: 0.012s

...

Results saved to iir.out
```

## Interpreting Results

### Performance Metrics

All timing values are reported in **seconds**, representing average execution time over 50 repetitions. **Lower values indicate better performance**.

### Expected Performance Characteristics

| Scenario | Expected Behavior |
|----------|-------------------|
| **Short audio, few channels** | CPU may outperform GPU due to transfer overhead |
| **Long audio, many channels** | GPU should significantly outperform CPU |
| **Simple operations** | SciPy may be competitive with CPU implementation |
| **Complex filter chains** | TorchFX benefits from vectorization and batching |

### API Benchmark Interpretation

The API benchmark compares ergonomics and performance:

- **FilterChain**: Traditional PyTorch pattern with explicit control
- **Sequential**: Standard PyTorch composition with automatic forwarding
- **Pipe operator**: Most ergonomic with automatic configuration
- **SciPy**: CPU-only baseline

**Expected results**:
- Performance differences between FilterChain, Sequential, and Pipe should be minimal (same underlying operations)
- Pipe operator provides automatic sampling rate configuration
- SciPy may be slower due to lack of GPU acceleration

### FIR/IIR Benchmark Interpretation

These benchmarks generate multi-dimensional data for analysis:

1. **Duration scaling**: How performance scales with audio length
   - Linear scaling expected for both CPU and GPU
   - GPU overhead amortized over longer durations

2. **Channel scaling**: How performance scales with channel count
   - GPU should show better scaling for many channels
   - CPU performance degrades more with channel count

3. **GPU vs CPU**: When GPU acceleration provides benefits
   - Crossover point varies by filter complexity
   - Generally favorable for >2 channels and >60s duration

4. **TorchFX vs SciPy**: Overhead of PyTorch abstraction
   - TorchFX CPU should be competitive with SciPy
   - GPU should outperform SciPy for suitable workloads

## Visualization

The `draw3.py` script generates PNG plots from CSV output files:

```bash
python benchmark/draw3.py
```

**Generated plots**:
- `api_bench.png`: Bar chart comparing API patterns
- `fir_bench.png`: Performance curves across durations/channels
- `iir_bench.png`: Performance curves across durations/channels

## Running All Benchmarks

### Prerequisites

Ensure development environment is set up:

```bash
# Sync dependencies
uv sync

# Verify CUDA availability (for GPU benchmarks)
python -c "import torch; print(torch.cuda.is_available())"
```

### Execution Script

Run all benchmarks sequentially:

```bash
# Run individual benchmarks
python benchmark/api_bench.py
python benchmark/fir_bench.py
python benchmark/iir_bench.py

# Generate visualizations
python benchmark/draw3.py
```

### GPU Configuration

To disable GPU benchmarks, comment out CUDA calls:

```python
# In benchmark script
# wave.to("cuda")  # Comment to disable GPU
```

## Benchmark Maintenance

### Adding New Benchmarks

To add a new benchmark:

1. Create new Python file in `benchmark/` directory
2. Implement `create_audio()` for test signal generation
3. Use `timeit.timeit()` with `REP=50` for timing
4. Compare against SciPy baseline when applicable
5. Output results in CSV format
6. Update this documentation

**Template**:

```python
import timeit
import numpy as np

SAMPLE_RATE = 44100
REP = 50

def create_audio(duration, num_channels):
    samples = int(duration * SAMPLE_RATE)
    audio = np.random.randn(num_channels, samples)
    return audio / np.max(np.abs(audio))

def benchmark():
    # Setup
    audio = create_audio(duration=60, num_channels=2)

    # Time execution
    def run():
        # Code to benchmark
        pass

    time = timeit.timeit(run, number=REP)
    avg_time = time / REP

    print(f"Average time: {avg_time:.4f}s")

if __name__ == "__main__":
    benchmark()
```

### Modifying Test Parameters

Common parameters to adjust:

```python
# Sample rate (default: 44100 Hz)
SAMPLE_RATE = 48000  # Change to 48kHz

# Repetitions (default: 50)
REP = 100  # Increase for more stable results

# Duration range (default varies by benchmark)
DURATIONS = [1, 10, 30, 60, 120]  # Custom duration range

# Channel counts (default varies by benchmark)
CHANNELS = [1, 2, 4, 8, 16]  # Custom channel counts
```

### Coefficient Pre-computation

For fair comparison, filter coefficients should be pre-computed:

```python
# Pre-compute coefficients before timing
for f in fchain:
    f.compute_coefficients()

# Now time only the filtering operation
time = timeit.timeit(lambda: fchain(wave.ys), number=REP)
```

This ensures timing measures filtering performance, not coefficient design.

## Best Practices

### Fair Comparisons

```python
# ✅ GOOD: Pre-compute coefficients
for f in fchain:
    f.compute_coefficients()
time = timeit.timeit(lambda: fchain(wave.ys), number=REP)

# ❌ BAD: Include coefficient design in timing
time = timeit.timeit(lambda: fchain(wave.ys), number=REP)
```

### Sufficient Repetitions

```python
# ✅ GOOD: Use 50+ repetitions
REP = 50
time = timeit.timeit(func, number=REP) / REP

# ❌ BAD: Too few repetitions (high variance)
REP = 5
time = timeit.timeit(func, number=REP) / REP
```

### Realistic Test Data

```python
# ✅ GOOD: Normalized random noise
audio = np.random.randn(channels, samples)
audio = audio / np.max(np.abs(audio))  # [-1, 1]

# ❌ BAD: Unrealistic data
audio = np.ones((channels, samples))  # All ones
```

## Related Resources

- {doc}`/guides/developer/testing` - Testing infrastructure
- {doc}`/guides/advanced/performance` - Performance optimization
- {doc}`/guides/advanced/gpu-acceleration` - GPU acceleration guide
- {doc}`/guides/developer/project-structure` - Project organization
