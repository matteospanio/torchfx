# Core Concepts

Understanding TorchFX's architecture is essential for building effective audio processing pipelines. This section explains the fundamental building blocks that make TorchFX powerful and easy to use.

## Overview

TorchFX is built around four core concepts that work together to create an intuitive and powerful audio processing framework:

1. **{class}`~torchfx.Wave`** - Digital audio representation with metadata
2. **{class}`~torchfx.FX`** - Base class for all audio effects and filters
3. **Pipeline Operator (`|`)** - Functional composition of effects
4. **Type System** - Time representations and musical notation

```{mermaid}
graph TB
    subgraph "TorchFX Architecture"
        Wave["Wave<br/>Audio container with<br/>tensor and sample rate"]
        FX["FX<br/>Base class for<br/>effects & filters"]
        Pipeline["Pipeline Operator |<br/>Chains effects together"]
        TypeSystem["Type System<br/>Musical time, units,<br/>devices"]

        Wave -->|"applies"| FX
        Wave -->|"uses"| Pipeline
        Pipeline -->|"chains"| FX
        FX -->|"uses"| TypeSystem
        Wave -->|"uses"| TypeSystem
    end

    subgraph "Built on PyTorch"
        Tensor["torch.Tensor<br/>Audio data"]
        Module["nn.Module<br/>Effect base"]
    end

    Wave -.->|"wraps"| Tensor
    FX -.->|"inherits"| Module

    style Wave fill:#e1f5ff
    style FX fill:#fff5e1
    style Pipeline fill:#e8f5e1
    style TypeSystem fill:#f5e1ff
```

## What You'll Learn

### Wave - Digital Audio Representation

The {class}`~torchfx.Wave` class wraps PyTorch tensors to represent digital audio signals. It provides:

- Sample rate ({term}`Sample Rate`) management
- Device handling (CPU/GPU)
- Audio file I/O (load/save)
- Metadata tracking
- Channel manipulation

**Learn more:** [Wave Concept](wave.md)

### FX - Effect Base Class

The {class}`~torchfx.FX` abstract base class defines the interface for all audio effects and filters. It:

- Inherits from {class}`torch.nn.Module` for PyTorch integration
- Provides automatic {term}`sample rate` configuration
- Enables gradient computation when needed
- Supports both real-time and batch processing

Effects built on FX include filters, dynamics, modulation, and more.

**Learn more:** [FX Base Class](fx.md)

### Pipeline Operator - Functional Composition

The pipeline operator (`|`) allows you to chain effects together in an intuitive, readable way:

```python
import torchfx as fx

wave = fx.Wave.from_file("audio.wav")
processed = wave | LoPass(1000) | Reverb(delay=4410) | Normalize()
```

This operator:
- Automatically configures effect parameters from the Wave
- Chains multiple effects sequentially
- Maintains immutability (returns new Wave objects)
- Works with both single effects and effect chains

**Learn more:** [Pipeline Operator](pipeline-operator.md)

### Type System - Musical Time and Units

TorchFX provides a rich type system for audio processing:

- **Musical Time**: BPM-synced timing (quarter notes, eighth notes, triplets, dotted notes)
- **Time Units**: Seconds, milliseconds, samples
- **Devices**: CPU and CUDA device management
- **Bit Rates**: Audio quality specifications

This enables writing effects that work musically rather than just technically.

**Learn more:** [Type System](type-system.md)

## Design Philosophy

TorchFX follows these design principles:

1. **PyTorch Native**: Built on {class}`torch.Tensor` and {class}`torch.nn.Module` for seamless integration
2. **Functional Style**: Immutable operations that return new objects
3. **Readable Code**: Pipeline operator creates self-documenting effect chains
4. **Musical Thinking**: Time represented in musical terms (BPM, note divisions) not just samples
5. **GPU Acceleration**: Automatic device management for high-performance processing

```{seealso}
- {doc}`/guides/tutorials/index` - Practical examples using these concepts
- {doc}`/api/index` - Complete API reference
- {cite:t}`spanio2025torchfx` - Academic paper on TorchFX design
```

## Architecture Diagram

```{mermaid}
classDiagram
    class Wave {
        +Tensor ys
        +int fs
        +dict metadata
        +Device device
        +from_file(path) Wave
        +save(path) None
        +transform(func) Wave
        +__or__(fx) Wave
        +channels() int
        +duration(unit) float
    }

    class FX {
        <<abstract>>
        +int fs
        +forward(x) Tensor*
    }

    class Filter {
        <<abstract>>
        +compute_coefficients()*
    }

    class IIRFilter {
        +Tensor b
        +Tensor a
        +forward(x) Tensor
    }

    class FIRFilter {
        +Tensor h
        +forward(x) Tensor
    }

    class Effect {
        +forward(x) Tensor
    }

    class Delay {
        +int delay_samples
        +float feedback
        +DelayStrategy strategy
        +forward(x) Tensor
    }

    class Reverb {
        +int delay
        +float decay
        +forward(x) Tensor
    }

    class Normalize {
        +float peak
        +NormalizationStrategy strategy
        +forward(x) Tensor
    }

    FX <|-- Filter
    FX <|-- Effect
    Filter <|-- IIRFilter
    Filter <|-- FIRFilter
    Effect <|-- Delay
    Effect <|-- Reverb
    Effect <|-- Normalize

    Wave --> FX : uses via |

    note for Wave "Container for digital audio\nwith sample rate and metadata"
    note for FX "Base class inheriting\nfrom torch.nn.Module"
    note for Filter "Abstract filter base\nfor IIR and FIR filters"
```

## Next Steps

Start with understanding the {class}`~torchfx.Wave` class, as it's the foundation of all audio processing in TorchFX:

```{toctree}
:maxdepth: 1

wave
fx
pipeline-operator
type-system
```

## External Resources

- [PyTorch Module Documentation](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) - Understanding the base class for FX
- [Digital Signal Processing on Wikipedia](https://en.wikipedia.org/wiki/Digital_signal_processing) - DSP fundamentals
- [PyTorch Tensor Operations](https://pytorch.org/docs/stable/tensors.html) - Working with audio tensors

## References

```{bibliography}
:filter: docname in docnames
:style: alpha
```
