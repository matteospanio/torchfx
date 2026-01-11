# Installation

TorchFX can be installed either from the Python Package Index (PyPI) or by cloning the repository from GitHub. This guide covers system requirements, installation methods, platform-specific configuration, and dependency management.

:::{admonition} Quick Start
:class: tip

For most users, a simple `pip install torchfx` is all you need to get started. For development or advanced configurations, see the sections below.
:::

## Purpose and Scope

This document describes the installation procedures for TorchFX, including system requirements, installation methods, platform-specific PyTorch configuration, and dependency management. For a quick start guide with minimal setup, see {doc}`getting_started`. For details on the development environment setup, see {doc}`../developer/index`.

## System Requirements

TorchFX requires **Python 3.10 or higher**. The library is designed to run on multiple platforms with different hardware acceleration capabilities:

| Platform | PyTorch Backend | CUDA Support |
|----------|----------------|--------------|
| Linux | CUDA 12.4 | Yes |
| macOS | CPU | No |
| Windows | CPU | No |

:::{note}
The platform-specific PyTorch configuration is automatically handled using the `uv` package manager's source selection mechanism based on your operating system.
:::

### Core Dependencies

TorchFX requires the following Python packages:

- `torch>=2.6.0` - PyTorch tensor library
- `torchaudio>=2.6.0` - Audio I/O and transforms
- `numpy>=2.2.4` - Numerical computing
- `scipy>=1.15.2` - Signal processing algorithms
- `soundfile>=0.13.1` - Audio file reading/writing
- `annotated-types>=0.7.0` - Type annotation support

:::{tip}
These dependencies are installed automatically when you install TorchFX via pip or uv.
:::

## Installation Methods

### Installing from PyPI

The simplest installation method is through PyPI:

```bash
pip install torchfx
```

This installs the latest stable release published on PyPI, along with all required dependencies. On Linux systems, this will attempt to install CUDA 12.4-enabled PyTorch; on macOS and Windows, it will install CPU-only versions.

### Installing from Source

To install the latest development version of TorchFX directly from GitHub, follow these steps:

```bash
git clone https://github.com/matteospanio/torchfx
cd torchfx
pip install -e .
```

This approach is useful if you plan to contribute to the project or want access to the latest features and updates that may not yet be available on PyPI.

### Installation from Source with uv

For development or to use the latest unreleased features with reproducible builds, install from source using the `uv` package manager:

```bash
# Install uv if not already installed
pip install uv

# Clone the repository
git clone https://github.com/matteospanio/torchfx
cd torchfx

# Install in editable mode with all dependencies
uv sync
```

The `uv sync` command reads `pyproject.toml` and `uv.lock` to install all dependencies with exact versions for reproducible builds.

### Installation Workflow

The following diagram illustrates the complete installation workflow for both PyPI and source-based installations:

```{mermaid}
graph TB
    User["User"]

    subgraph "Installation Methods"
        PipInstall["pip install torchfx"]
        SourceInstall["Clone + uv sync"]
    end

    subgraph "PyPI Distribution"
        PyPI["PyPI Registry"]
        Wheel["torchfx-0.2.1-*.whl"]
    end

    subgraph "Source Installation"
        GitRepo["GitHub Repository<br/>matteospanio/torchfx"]
        PyProject["pyproject.toml<br/>Dependencies definition"]
        UVLock["uv.lock<br/>Locked versions"]
        UV["uv package manager"]
    end

    subgraph "Dependency Resolution"
        DepResolver["Platform Detection"]
        LinuxPath["Linux:<br/>pytorch-cu124 index"]
        NonLinuxPath["macOS/Windows:<br/>pytorch-cpu index"]
        PyTorchCUDA["torch 2.6.0+cu124"]
        PyTorchCPU["torch 2.6.0+cpu"]
        OtherDeps["numpy, scipy<br/>soundfile, torchaudio"]
    end

    subgraph "Installed Package"
        TorchFXPkg["torchfx package<br/>in site-packages"]
    end

    User -->|pip install| PipInstall
    User -->|git clone + uv sync| SourceInstall

    PipInstall --> PyPI
    PyPI --> Wheel
    Wheel --> DepResolver

    SourceInstall --> GitRepo
    GitRepo --> PyProject
    GitRepo --> UVLock
    PyProject --> UV
    UVLock --> UV
    UV --> DepResolver

    DepResolver -->|sys_platform == 'linux'| LinuxPath
    DepResolver -->|sys_platform != 'linux'| NonLinuxPath

    LinuxPath --> PyTorchCUDA
    NonLinuxPath --> PyTorchCPU

    PyTorchCUDA --> TorchFXPkg
    PyTorchCPU --> TorchFXPkg
    DepResolver --> OtherDeps
    OtherDeps --> TorchFXPkg
```

## Platform-Specific PyTorch Configuration

The `uv` package manager uses **platform markers** to select the appropriate PyTorch distribution. This configuration ensures that Linux users get CUDA-enabled builds while macOS and Windows users receive CPU-only builds.

### PyTorch Source Selection Logic

The following diagram shows how PyTorch sources are selected based on the platform:

```{mermaid}
graph TD
    Start["Dependency Resolution Start"]

    PlatformCheck{"sys_platform == 'linux'?"}

    subgraph "Linux Path"
        CUDAIndex["[[tool.uv.index]]<br/>name = 'pytorch-cu124'<br/>url = download.pytorch.org/whl/cu124"]
        CUDASource["torch source:<br/>index = 'pytorch-cu124'<br/>marker = sys_platform == 'linux'"]
        CUDAPackage["torch 2.6.0+cu124<br/>torchaudio 2.6.0+cu124<br/>CUDA 12.4 support"]
    end

    subgraph "Non-Linux Path"
        CPUIndex["[[tool.uv.index]]<br/>name = 'pytorch-cpu'<br/>url = download.pytorch.org/whl/cpu"]
        CPUSource["torch source:<br/>index = 'pytorch-cpu'<br/>marker = sys_platform != 'linux'"]
        CPUPackage["torch 2.6.0+cpu<br/>torchaudio 2.6.0+cpu<br/>CPU-only"]
    end

    Result["Resolved torch package"]

    Start --> PlatformCheck
    PlatformCheck -->|Yes| CUDASource
    PlatformCheck -->|No| CPUSource

    CUDASource --> CUDAIndex
    CUDAIndex --> CUDAPackage
    CPUSource --> CPUIndex
    CPUIndex --> CPUPackage

    CUDAPackage --> Result
    CPUPackage --> Result
```

The platform detection uses Python's `sys_platform` to determine which PyTorch index to use:

- **Linux systems**: Resolves `torch` and `torchaudio` from the `pytorch-cu124` index at `https://download.pytorch.org/whl/cu124`, providing CUDA 12.4 GPU acceleration.
- **macOS and Windows**: Resolves from the `pytorch-cpu` index at `https://download.pytorch.org/whl/cpu`, providing CPU-only builds.

The `explicit = true` flag in the configuration ensures these indices are only used when explicitly specified by the source configuration.

### GPU Support

TorchFX is built on top of **PyTorch**, which means GPU support depends on your local PyTorch installation. To enable GPU acceleration:

1. Make sure you have a compatible NVIDIA GPU
2. Ensure you're running on a Linux system (CUDA support is currently Linux-only in the default configuration)
3. Install PyTorch with CUDA support

If you need a specific CUDA version different from 12.4, you can manually install PyTorch:

```bash
# Example: Installing PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Replace `cu121` with the correct CUDA version for your system. Visit the [official PyTorch website](https://pytorch.org/get-started/locally/) for the correct installation command for your configuration.

## Dependency Management with uv

The project uses **uv** as its primary dependency manager, which provides faster dependency resolution and reproducible builds through lock files.

### Dependency Resolution Components

The following diagram illustrates how dependencies are resolved and installed:

```{mermaid}
graph TB
    subgraph "Configuration Files"
        PyProject["pyproject.toml"]
        UVLock["uv.lock"]
    end

    subgraph "Dependency Specifications"
        CoreDeps["[project.dependencies]<br/>torch>=2.6.0<br/>torchaudio>=2.6.0<br/>numpy>=2.2.4<br/>scipy>=1.15.2<br/>soundfile>=0.13.1<br/>annotated-types>=0.7.0"]

        DevGroups["[dependency-groups]<br/>cli: typer<br/>dev: black, mypy, pytest, ruff<br/>docs: sphinx, sphinx-immaterial"]

        UVSources["[tool.uv.sources]<br/>Platform-specific torch sources"]

        UVIndices["[[tool.uv.index]]<br/>pytorch-cpu<br/>pytorch-cu124"]
    end

    subgraph "Resolution Process"
        UVResolver["uv resolver engine"]
        PlatformDetect["Platform detection<br/>sys_platform"]
        VersionConstraints["Version constraint solver"]
    end

    subgraph "Lock File"
        LockedVersions["Exact package versions<br/>with hashes"]
        ResolvedDeps["Resolved dependency tree"]
    end

    subgraph "Installation"
        Download["Download packages"]
        Install["Install to environment"]
    end

    PyProject --> CoreDeps
    PyProject --> DevGroups
    PyProject --> UVSources
    PyProject --> UVIndices

    CoreDeps --> UVResolver
    DevGroups --> UVResolver
    UVSources --> PlatformDetect
    UVIndices --> PlatformDetect

    UVResolver --> VersionConstraints
    PlatformDetect --> VersionConstraints

    VersionConstraints --> LockedVersions
    VersionConstraints --> ResolvedDeps

    LockedVersions --> UVLock
    ResolvedDeps --> UVLock

    UVLock --> Download
    Download --> Install
```

### Dependency Groups

The project defines optional dependency groups for different use cases:

| Group | Purpose | Key Packages |
|-------|---------|--------------|
| `cli` | Command-line interface | `typer>=0.16.0` |
| `dev` | Development tools | `black`, `mypy`, `pytest`, `ruff`, `coverage`, `scalene` |
| `docs` | Documentation building | `sphinx>=8.1.3`, `sphinx-immaterial>=0.13.5` |

To install with specific groups:

```bash
# Install with dev dependencies
uv sync --group dev

# Install with all groups
uv sync --all-groups

# Install only docs dependencies
uv sync --group docs
```

:::{note}
For contributors and developers, we recommend using `uv sync --all-groups` to install all development dependencies including testing, documentation, and CLI tools.
:::

## Developers

If you are a developer and want to contribute to the TorchFX project, you can set up a development environment by following these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/matteospanio/torchfx
   ```

2. Navigate to the project directory:

   ```bash
   cd torchfx
   ```

3. Create a virtual environment (optional but recommended). The project is built using `uv`, hence we suggest using it:

   ```bash
   # The flag --all-groups will install also dev dependencies
   uv sync --all-groups
   ```

For more information on the development workflow, see {doc}`../developer/index`.

## Build System

The project uses **hatchling** as its build backend. The build system configuration is defined in `pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

To build a distribution package:

```bash
# Using uv
uv build

# Or using standard tools
python -m build
```

This produces a wheel file (`.whl`) in the `dist/` directory that can be distributed or installed with pip.

## Verification

After installation, verify that TorchFX is correctly installed and can import all core components.

### Basic Installation Check

To verify that the package has been correctly installed, run the following command in Python:

```python
import torchfx
print(torchfx.__version__)  # Should print '0.2.1'
```

### Comprehensive Verification

For a more thorough verification, check that all core components can be imported and that PyTorch is correctly configured:

```python
import torchfx
from torchfx import Wave, FX, Gain, Normalize
from torchfx.filter import Butterworth, LowPass

# Check version
print(torchfx.__version__)  # Should print '0.2.1'

# Verify PyTorch backend
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch version: {torch.__version__}")
```

On Linux systems with NVIDIA GPUs, `torch.cuda.is_available()` should return `True`. On macOS and Windows, it will return `False`, indicating CPU-only operation.

### Audio I/O Verification

To verify audio I/O capabilities:

```python
import torchaudio
print(f"torchaudio version: {torchaudio.__version__}")
print(f"Available backends: {torchaudio.list_audio_backends()}")
```

Expected output should show the available audio backends for your platform (e.g., `soundfile`, `sox`, `ffmpeg`).

## Checking Installation

You can perform a quick functional test to ensure TorchFX is working correctly:

```python
import torchfx as fx

# Create a simple sine wave
wave = fx.Wave.from_function(lambda t: fx.sin(2 * fx.pi * 440 * t), duration=1.0, fs=44100)

# Apply a gain effect
gained = wave | fx.Gain(db=6.0)

# Verify the operation succeeded
print(f"Original wave: {wave.shape}")
print(f"Processed wave: {gained.shape}")
print(f"Sample rate: {gained.fs} Hz")
```

If this runs without errors, your installation is working correctly.

## Troubleshooting

### PyTorch CUDA Version Mismatch

If you have an existing PyTorch installation with a different CUDA version, you may need to uninstall it first:

```bash
pip uninstall torch torchaudio torchvision
uv sync --reinstall-package torch
```

This ensures that the correct PyTorch version is installed according to the platform-specific configuration.

### Platform Detection Issues

If the wrong PyTorch variant is installed, verify your platform:

```python
import sys
print(f"Platform: {sys.platform}")
```

The marker `sys_platform == 'linux'` should be `True` on Linux systems. If you're on Linux but getting the CPU-only version, check your Python installation and ensure `sys.platform` returns `'linux'`.

### Dependency Conflicts

If you encounter dependency conflicts when using uv, regenerate the lock file:

```bash
rm uv.lock
uv lock
uv sync
```

This will resolve dependencies from scratch and create a new lock file with compatible versions.

### Import Errors

If you encounter import errors after installation, ensure that:

1. You're using the correct Python environment where TorchFX was installed
2. All dependencies were installed successfully
3. There are no naming conflicts with other packages

You can verify your Python environment:

```bash
which python
pip list | grep torchfx
```

### Audio Backend Issues

If you encounter issues loading audio files, ensure that `soundfile` is properly installed:

```bash
pip install soundfile --upgrade
```

On some systems, you may need to install system-level audio libraries:

```bash
# Ubuntu/Debian
sudo apt-get install libsndfile1

# macOS
brew install libsndfile

# Windows
# Usually included with soundfile package
```

:::{admonition} Still Having Issues?
:class: tip

If you continue to experience problems, please:
- Check the [GitHub Issues](https://github.com/matteospanio/torchfx/issues) for similar problems
- Create a new issue with details about your system, Python version, and the error message
- Include the output of `pip list` to show installed packages and versions
:::

## Next Steps

Now that you have TorchFX installed, you can:

- Follow the {doc}`getting_started` guide to learn the basics
- Explore {doc}`../core-concepts/index` to understand the fundamental concepts
- Try the {doc}`../tutorials/index` for hands-on examples
- Learn about {doc}`../advanced/gpu-acceleration` for performance optimization

:::{seealso}
- {doc}`getting_started` - Your first steps with TorchFX
- {doc}`../developer/index` - Contributing to TorchFX development
- {doc}`../advanced/performance` - Optimizing performance
:::
