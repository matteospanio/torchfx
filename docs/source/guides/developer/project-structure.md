# Project Structure

Comprehensive overview of the TorchFX repository structure, including source code organization, module dependencies, and package layout.

## Overview

The TorchFX repository follows a standard Python package structure with clear separation between source code, tests, documentation, and supporting tools. Understanding this structure is essential for contributing to the project and navigating the codebase effectively.

```{seealso}
{doc}`/guides/developer/testing` - Testing infrastructure
{doc}`/guides/developer/documentation` - Documentation system
```

## Repository Layout

The repository is organized with the following top-level structure:

```{mermaid}
graph TB
    Root["Repository Root"]

    Root --> Src["src/<br/>Source code"]
    Root --> Tests["tests/<br/>Test suite"]
    Root --> Docs["docs/<br/>Sphinx documentation"]
    Root --> Examples["examples/<br/>Usage examples"]
    Root --> Benchmark["benchmark/<br/>Performance tests"]
    Root --> CLI["cli/<br/>Command-line interface"]

    Root --> PyProject["pyproject.toml<br/>Project configuration"]
    Root --> UVLock["uv.lock<br/>Locked dependencies"]
    Root --> PreCommit[".pre-commit-config.yaml<br/>Code quality hooks"]
    Root --> GitHub[".github/<br/>CI/CD workflows"]
    Root --> README["README.md<br/>Project overview"]
    Root --> License["LICENSE<br/>GPLv3"]
    Root --> Changelog["CHANGELOG<br/>Version history"]

    Src --> TorchFX["torchfx/<br/>Main package"]
    Tests --> TestEffects["test_effects.py<br/>Effect tests"]
    Docs --> Source["source/<br/>Documentation source"]
    Examples --> Ex1["series_and_parallel_filters.py"]
    Examples --> Ex2["multi_channel_effect.py"]
    Examples --> Ex3["delay.py"]
    Benchmark --> B1["api_bench.py"]
    Benchmark --> B2["fir_bench.py"]
    Benchmark --> B3["iir_bench.py"]
    CLI --> Main["__main__.py<br/>CLI entry point"]

    GitHub --> CIWorkflow["workflows/ci.yml"]
    GitHub --> DocsWorkflow["workflows/docs.yml"]
```

### Directory Purpose Summary

| Directory | Purpose | Key Contents |
|-----------|---------|--------------|
| `src/torchfx/` | Main library package | Core modules: wave, effect, filter, typing |
| `tests/` | Test suite | Unit and integration tests |
| `docs/` | Documentation | Sphinx source files and configuration |
| `examples/` | Usage examples | Practical demonstrations of library features |
| `benchmark/` | Performance tests | GPU vs CPU performance comparisons |
| `cli/` | Command-line interface | CLI implementation (placeholder) |
| `.github/workflows/` | CI/CD pipelines | GitHub Actions workflow definitions |

## Source Package Structure

The main library is located in `src/torchfx/` and organized as a modular Python package. The structure emphasizes clear separation of concerns and explicit public API exports.

### Package Layout

```{mermaid}
graph TD
    TorchFX["src/torchfx/"]

    TorchFX --> Init["__init__.py<br/>Public API exports"]
    TorchFX --> Wave["wave.py<br/>Wave class"]
    TorchFX --> Effect["effect.py<br/>FX, Gain, Normalize<br/>Reverb, Delay"]
    TorchFX --> Typing["typing.py<br/>MusicalTime<br/>Type aliases"]
    TorchFX --> FilterDir["filter/<br/>Filter subpackage"]

    FilterDir --> FilterInit["__init__.py<br/>Filter exports"]
    FilterDir --> FilterBase["__base.py<br/>AbstractFilter<br/>ParallelFilterCombination"]
    FilterDir --> FilterIIR["iir.py<br/>IIR implementations"]
    FilterDir --> FilterFIR["fir.py<br/>FIR implementations"]

    Init -.->|exports| Wave
    Init -.->|exports| Effect
    Init -.->|exports| Typing
    Init -.->|exports| FilterInit

    FilterInit -.->|aggregates| FilterBase
    FilterInit -.->|aggregates| FilterIIR
    FilterInit -.->|aggregates| FilterFIR
```

### Module Organization

The package exposes a clean public API through `src/torchfx/__init__.py`:

```
torchfx/
├── __init__.py          # Public API gateway
├── wave.py              # Wave data container
├── effect.py            # Effects and FX base class
├── typing.py            # Custom type definitions
└── filter/              # Filter subpackage
    ├── __init__.py      # Filter module exports
    ├── __base.py        # Abstract base classes
    ├── iir.py           # IIR filter implementations
    └── fir.py           # FIR filter implementations
```

## Core Modules

### wave.py - Audio Data Container

The `wave.py` module contains the {class}`~torchfx.Wave` class, which is the primary data structure for audio in TorchFX.

**Key responsibilities**:
- Audio tensor data storage (`ys: torch.Tensor`)
- Sample rate management (`fs: int`)
- Device management methods (CPU/GPU)
- File I/O functionality
- Pipe operator support for chaining effects

**Example usage**:

```python
from torchfx import Wave

# Load audio file
wave = Wave.from_file("audio.wav")

# Access properties
print(wave.fs)  # Sample rate
print(wave.ys.shape)  # Tensor shape

# Move to GPU
wave.to("cuda")
```

```{seealso}
{doc}`/guides/core-concepts/wave` - Complete Wave documentation
```

### effect.py - Effects and Base Class

The `effect.py` module contains the {class}`~torchfx.FX` abstract base class and built-in effect implementations.

**Classes**:
- {class}`~torchfx.FX` - Abstract base class inheriting from `torch.nn.Module`
- {class}`~torchfx.effect.Gain` - Amplitude adjustment effect
- {class}`~torchfx.effect.Normalize` - Audio normalization with multiple strategies
- {class}`~torchfx.effect.Reverb` - Reverberation effect
- {class}`~torchfx.effect.Delay` - BPM-synchronized delay effect

All effects support the pipe operator (`|`) for chaining.

**Example usage**:

```python
from torchfx import Wave
from torchfx.effect import Gain, Normalize

wave = Wave.from_file("audio.wav")

# Chain effects using pipe operator
processed = wave | Gain(gain=2.0) | Normalize(peak=0.9)
```

```{seealso}
{doc}`/guides/tutorials/custom-effects` - Creating custom effects
{doc}`/guides/tutorials/effects-design` - Effect design patterns
```

### typing.py - Type System

The `typing.py` module defines custom types for audio DSP operations.

**Key types**:
- {class}`~torchfx.typing.MusicalTime` - Type for BPM-synchronized timing
- Type aliases for audio parameters (e.g., `Decibel`, `Second`)
- Annotations for improved type safety

**Example usage**:

```python
from torchfx.typing import MusicalTime

# Parse musical time notation
quarter_note = MusicalTime.from_string("1/4")
dotted_eighth = MusicalTime.from_string("1/8d")

# Convert to seconds
duration = quarter_note.duration_seconds(bpm=120, beats_per_bar=4)
print(f"Duration: {duration} seconds")  # 0.5 seconds
```

```{seealso}
{doc}`/guides/core-concepts/type-system` - Complete type system documentation
```

### filter/ - Filter Subpackage

The filter subpackage is organized into three main files:

#### `__base.py` - Abstract Interfaces

Contains abstract base classes for all filters:

- {class}`~torchfx.filter.AbstractFilter` - Base class for all filters
- {class}`~torchfx.filter.ParallelFilterCombination` - Combines multiple filters in parallel (supports `+` operator)

#### `iir.py` - IIR Filter Implementations

Contains specific IIR filter types:

- Butterworth filters: `HiButterworth`, `LoButterworth`, `BandButterworth`, `BandStopButterworth`
- Chebyshev filters: `HiChebyshev1`, `LoChebyshev1`, `HiChebyshev2`, `LoChebyshev2`
- Shelving filters: `LoShelving`, `HiShelving`
- Peaking filters: `PeakingEQ`
- Notch filters: `NotchFilter`
- AllPass filters: `AllPassFilter`
- Linkwitz-Riley crossover filters: `LinkwitzRiley`

#### `fir.py` - FIR Filter Implementations

Contains FIR filter classes:

- {class}`~torchfx.filter.FIR` - Basic FIR filter with coefficient input
- {class}`~torchfx.filter.DesignableFIR` - FIR filter with automatic coefficient design

```{seealso}
{doc}`/guides/tutorials/filters-design` - Filter design guide
{doc}`/guides/tutorials/series-parallel-filters` - Combining filters
```

## Module Dependencies

The following diagram illustrates dependency relationships between modules and external packages:

```{mermaid}
graph TB
    subgraph "Public API (torchfx.*)"
        Wave["Wave<br/>(from wave)"]
        FX["FX<br/>(from effect)"]
        EffectModule["effect module<br/>(Gain, Normalize, Reverb, Delay)"]
        FilterModule["filter module<br/>(AbstractFilter, IIR, FIR, Parallel)"]
        TypingModule["typing module<br/>(MusicalTime, type aliases)"]
    end

    subgraph "Internal Modules"
        WaveImpl["wave.py<br/>Wave class implementation"]
        EffectImpl["effect.py<br/>FX + effect implementations"]
        TypingImpl["typing.py<br/>Type definitions"]

        subgraph "filter/ subpackage"
            FilterBase["__base.py<br/>AbstractFilter<br/>ParallelFilterCombination"]
            FilterIIR["iir.py<br/>Butterworth, Chebyshev<br/>Shelving, Peaking, etc."]
            FilterFIR["fir.py<br/>FIR, DesignableFIR"]
        end
    end

    subgraph "External Dependencies"
        PyTorch["torch<br/>torch.nn.Module<br/>Tensor operations"]
        TorchAudio["torchaudio<br/>Audio I/O<br/>Transforms"]
        SciPy["scipy.signal<br/>Filter coefficient design"]
        NumPy["numpy<br/>Array operations"]
    end

    Wave -.->|implemented in| WaveImpl
    FX -.->|implemented in| EffectImpl
    EffectModule -.->|implemented in| EffectImpl
    FilterModule -.->|aggregates| FilterBase
    FilterModule -.->|aggregates| FilterIIR
    FilterModule -.->|aggregates| FilterFIR
    TypingModule -.->|implemented in| TypingImpl

    WaveImpl -->|depends on| PyTorch
    WaveImpl -->|depends on| TorchAudio
    EffectImpl -->|depends on| PyTorch
    EffectImpl -->|depends on| TorchAudio
    EffectImpl -->|uses| TypingImpl
    FilterBase -->|extends| PyTorch
    FilterIIR -->|extends| FilterBase
    FilterIIR -->|uses| SciPy
    FilterFIR -->|extends| FilterBase
    FilterFIR -->|uses| SciPy
```

### Public API Exports

The public API is explicitly controlled through `__all__` declarations in `__init__.py`:

| Export | Source Module | Description |
|--------|---------------|-------------|
| `Wave` | `torchfx.wave` | Audio data container class |
| `FX` | `torchfx.effect` | Abstract base class for all effects/filters |
| `effect` | `torchfx.effect` | Module containing effect implementations |
| `filter` | `torchfx.filter` | Subpackage containing filter implementations |
| `typing` | `torchfx.typing` | Module with custom type definitions |

## Import Patterns

### Recommended Import Patterns

The following import patterns are recommended for clean, maintainable code:

```python
# Core classes - direct import
from torchfx import Wave, FX

# Module imports for namespace organization
import torchfx.effect as effect
import torchfx.filter as filter
import torchfx.typing

# Direct class imports for convenience
from torchfx.effect import Gain, Normalize, Delay
from torchfx.filter import Butterworth, FIR
from torchfx.typing import MusicalTime
```

### Import Hierarchy

```{mermaid}
graph LR
    User["User Code"]

    subgraph "Top-Level Imports"
        TopWave["from torchfx import Wave"]
        TopFX["from torchfx import FX"]
        TopEffect["import torchfx.effect"]
        TopFilter["import torchfx.filter"]
        TopTyping["import torchfx.typing"]
    end

    subgraph "Submodule Imports"
        EffectClasses["from torchfx.effect import Gain, Normalize, Reverb, Delay"]
        FilterClasses["from torchfx.filter import Butterworth, Chebyshev, FIR"]
        TypeDefs["from torchfx.typing import MusicalTime"]
    end

    User --> TopWave
    User --> TopFX
    User --> TopEffect
    User --> TopFilter
    User --> TopTyping

    User --> EffectClasses
    User --> FilterClasses
    User --> TypeDefs

    TopEffect -.->|contains| EffectClasses
    TopFilter -.->|contains| FilterClasses
    TopTyping -.->|contains| TypeDefs
```

## Supporting Infrastructure

### Examples Directory

The `examples/` directory contains practical demonstrations of library features:

| File | Purpose |
|------|---------|
| `series_and_parallel_filters.py` | Demonstrates filter chaining (`\|`) and parallel combination (`+`) |
| `multi_channel_effect.py` | Shows custom multi-channel effect implementation |
| `delay.py` | Demonstrates BPM-synchronized delay with musical timing |

**Running examples**:

```bash
python examples/series_and_parallel_filters.py
python examples/multi_channel_effect.py
python examples/delay.py
```

### Tests Directory

The `tests/` directory contains the test suite with unit and integration tests. Test configuration is specified in `pyproject.toml`:

- Test discovery in `tests/` directory
- Python path includes `src/` for imports
- Coverage reporting targets `src/torchfx`

```{seealso}
{doc}`/guides/developer/testing` - Complete testing documentation
```

### Documentation Directory

The `docs/` directory contains Sphinx documentation:

```
docs/
├── source/
│   ├── api.rst          # API reference
│   ├── conf.py          # Sphinx configuration
│   ├── guides/          # User and developer guides
│   └── ...              # Other documentation pages
├── Makefile             # Build automation (Unix)
└── make.bat             # Build automation (Windows)
```

Documentation is automatically built and deployed to GitHub Pages via `.github/workflows/docs.yml`.

```{seealso}
{doc}`/guides/developer/documentation` - Documentation build system
```

### Benchmark Directory

The `benchmark/` directory contains performance measurement scripts:

| File | Purpose |
|------|---------|
| `api_bench.py` | Compares different API patterns (FilterChain, Sequential, pipe operator) |
| `fir_bench.py` | Measures FIR filter performance (GPU vs CPU vs SciPy) |
| `iir_bench.py` | Measures IIR filter performance (GPU vs CPU vs SciPy) |
| `draw3.py` | Visualizes benchmark results as PNG images |

```{seealso}
{doc}`/guides/developer/benchmarking` - Benchmarking guide
```

### CLI Directory

The `cli/` directory contains the command-line interface:

```
cli/
└── __main__.py          # CLI entry point
```

The CLI is configured as a project script in `pyproject.toml`:

```toml
[project.scripts]
torchfx = "cli.__main__:main"
```

Currently serves as a placeholder for future CLI functionality.

## Configuration Files

### pyproject.toml - Central Configuration Hub

The `pyproject.toml` file serves as the central configuration for the entire project, containing:

#### Project Metadata

- Name, version, description
- License (GPLv3) and authors
- Python version requirement (>=3.10)
- Dependencies and classifiers
- Project URLs (repository, documentation, changelog)

#### Tool Configurations

| Tool | Purpose | Section |
|------|---------|---------|
| `uv` | Package manager with PyTorch source configuration | `[tool.uv]` |
| `mypy` | Type checking with strict mode | `[tool.mypy]` |
| `ruff` | Fast Python linter and formatter | `[tool.ruff]` |
| `black` | Code formatting | `[tool.black]` |
| `coverage` | Test coverage measurement | `[tool.coverage]` |
| `pytest` | Testing framework | `[tool.pytest]` |
| `docformatter` | Docstring formatting | `[tool.docformatter]` |

#### Build System

Uses `hatchling` as the build backend:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

#### Dependency Groups

```toml
[dependency-groups]
cli = ["typer"]
dev = ["black", "mypy", "pytest", "ruff", ...]
docs = ["sphinx", "sphinx-immaterial", ...]
```

### uv.lock - Dependency Lock File

The `uv.lock` file provides:

- Complete dependency resolution with exact versions
- Reproducible builds across environments
- Platform-specific dependency tracking

Generated and managed by the `uv` package manager.

### .pre-commit-config.yaml - Code Quality Hooks

Configures pre-commit hooks that run before each commit:

- Type checking with `mypy`
- Linting with `ruff`
- Formatting with `black`
- Docstring formatting with `docformatter`

**Installation**:

```bash
pre-commit install
```

### .github/workflows/ - CI/CD Configuration

Contains GitHub Actions workflow definitions:

| Workflow | Purpose | File |
|----------|---------|------|
| CI | Runs tests, linting, and type checking in parallel across Python 3.10-3.13 | `ci.yml` |
| Docs | Builds Sphinx documentation and deploys to GitHub Pages | `docs.yml` |

## Package Distribution

The project is configured for distribution as a Python package:

### Build Artifacts

- **Wheel package**: `torchfx-*.whl` built using hatchling
- **Source distribution**: Generated from project source
- **Documentation site**: Deployed to GitHub Pages at [https://matteospanio.github.io/torchfx/](https://matteospanio.github.io/torchfx/)

### Installation Methods

The package can be installed via:

```bash
# From PyPI
pip install torchfx

# From source (development)
pip install -e .

# With extras (CLI support)
pip install torchfx[cli]
```

## Best Practices

### Module Organization

- **Single Responsibility**: Each module has a clear, focused purpose
- **Explicit Exports**: Public API is controlled via `__all__` declarations
- **Shallow Hierarchy**: Avoid deeply nested package structures
- **Clear Dependencies**: Minimize coupling between modules

### Code Structure

- **Type Hints**: Use type hints for all public functions and methods
- **Docstrings**: Document all public APIs with NumPy-style docstrings
- **Testing**: Write tests for all new functionality
- **Linting**: Follow ruff and black formatting standards

```{seealso}
{doc}`/guides/developer/style_guide` - Complete style guide
```

## Related Resources

- {doc}`/guides/developer/testing` - Testing infrastructure
- {doc}`/guides/developer/benchmarking` - Performance benchmarking
- {doc}`/guides/developer/documentation` - Documentation system
- {doc}`/guides/developer/style_guide` - Coding standards
