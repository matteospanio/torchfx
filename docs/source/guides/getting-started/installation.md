# Installation

TorchFX can be installed from the Python Package Index (PyPI) or from source. This guide covers system requirements, build-time system dependencies, installation methods, and troubleshooting.

:::{admonition} Quick Start
:class: tip

For most users on Linux x86_64, macOS (Intel or Apple Silicon), or Windows x86_64, a simple `pip install torchfx` is all you need --- prebuilt CPU wheels for Python 3.10–3.14 mean no compiler is required. For GPU builds, less common platforms, or development setups, see the sections below.
:::

## Purpose and Scope

This document describes the installation procedures for TorchFX, including system requirements, the build system, installation methods, and dependency management. For a quick start guide with minimal setup, see {doc}`getting_started`. For details on the development environment, see {doc}`../developer/index`.

## System Requirements

### Runtime requirements

| Requirement | Version |
|-------------|---------|
| Python | 3.10, 3.11, 3.12, 3.13, or 3.14 |
| Operating system | Linux x86_64, macOS (Intel + Apple Silicon), or Windows x86_64 |
| GPU (optional) | NVIDIA GPU with CUDA 12.4 or 12.8. Prebuilt CUDA wheels exist for Linux x86_64 only --- see [GPU (CUDA) wheels](#gpu-wheels). Other configurations build from source. |

**Prebuilt CPU wheels** are published to PyPI for every supported Python version on:

| Platform | Wheel tag |
|----------|-----------|
| Linux x86_64 | `manylinux_2_28_x86_64` |
| macOS x86_64 (Intel) | `macosx_*_x86_64` |
| macOS arm64 (Apple Silicon) | `macosx_*_arm64` |
| Windows x86_64 | `win_amd64` |

On any of these targets, `pip install torchfx` downloads a wheel and does **not** require a C++ compiler or CMake. The wheels are produced by the [`wheels.yml`](https://github.com/matteospanio/torchfx/blob/master/.github/workflows/wheels.yml) workflow on every tagged release.

On other platforms (musllinux, aarch64 Linux, etc.), pip falls back to the source distribution and triggers a CMake build --- see [System dependencies (source builds)](#system-dependencies-source-builds) below.

CUDA wheels for Linux x86_64 (CUDA 12.4 and 12.8) are published to a separate index --- see [GPU (CUDA) wheels](#gpu-wheels). On unsupported configurations (Windows + CUDA, other CUDA versions, etc.) install from source with `nvcc` on `PATH`; the build detects CUDA automatically.

### Core Python dependencies

The following packages are installed automatically by pip or uv:

- `torch>=2.6.0` --- PyTorch tensor library
- `numpy>=2.2.4` --- numerical computing
- `scipy>=1.15.2` --- signal-processing algorithms (filter design)
- `soundfile>=0.13.1` --- audio file reading and writing
- `annotated-types>=0.7.0` --- type-annotation support

(system-dependencies-source-builds)=
### System dependencies (source builds)

When pip cannot find a matching wheel, the source distribution is built locally. The build invokes CMake (configured by [CMakeLists.txt](https://github.com/matteospanio/torchfx/blob/master/CMakeLists.txt)) to compile the C++/CUDA extension. You will need:

| Dependency | Minimum version | Purpose |
|------------|----------------|---------|
| C++17 compiler | GCC ≥ 9, Clang ≥ 10, or MSVC 2019+ | Compile the native extension |
| CMake | 3.18 | Drive the build |
| Python development headers | Matching your Python (e.g. `python3-dev` on Debian/Ubuntu) | Required by CMake's `find_package(Python ... Development.Module)` |
| `torch>=2.6.0` (already installed) | 2.6.0 | The build queries `torch.utils.cpp_extension` for include and library paths and links against `torch_cpu` / `c10` |
| OpenMP *(optional)* | --- | Linked automatically if found, enables multi-channel parallelism in the CPU kernels |
| CUDA toolkit + `nvcc` *(optional)* | matching your `torch` build | Required to compile the CUDA kernels; auto-detected by CMake |

Set `TORCHFX_NO_CUDA=1` in the environment to force a CPU-only build even when `nvcc` is available --- this matches what the wheel-publishing CI does and is the simplest way to get a fast, reproducible build without GPU support.

Per-platform installation of the build toolchain:

```bash
# Debian / Ubuntu
sudo apt-get install build-essential cmake python3-dev

# Fedora / RHEL
sudo dnf install gcc-c++ cmake python3-devel

# macOS (Xcode Command Line Tools provide clang; CMake via Homebrew)
xcode-select --install
brew install cmake

# Windows (PowerShell, with winget)
# Install Visual Studio 2022 Build Tools with the "Desktop development with C++"
# workload, then:
winget install Kitware.CMake
```

## Installation Methods

### From PyPI (recommended)

```bash
pip install torchfx
```

On Linux x86_64, macOS (Intel and Apple Silicon), or Windows x86_64 with Python 3.10–3.14, pip downloads a prebuilt CPU wheel and installation finishes in seconds. The C++ extension is precompiled inside the wheel; no compiler or CMake is needed.

On other platforms (musllinux, aarch64 Linux, etc.), pip falls through to the source distribution and runs the CMake build. Make sure the [system dependencies](#system-dependencies-source-builds) above are installed first.

(from-source-with-pip)=
### From source with pip

```bash
git clone https://github.com/matteospanio/torchfx
cd torchfx
pip install .
```

Use this path if you are on a platform without a published wheel, or want a custom build. Add `-e` for an editable install when developing.

(gpu-wheels)=
### GPU (CUDA) wheels

Prebuilt CUDA wheels for **Linux x86_64 / Python 3.10–3.14** are published to a project-managed PEP 503 index hosted on GitHub Pages, one index per CUDA toolkit version:

| CUDA | Index URL |
|------|-----------|
| 12.4 | `https://matteospanio.github.io/torchfx/wheels/cu124/` |
| 12.8 | `https://matteospanio.github.io/torchfx/wheels/cu128/` |

Pick the index whose CUDA version matches your installed `torch`, install `torch` first, then point `pip` at the matching index for `torchfx`:

```bash
# 1. PyTorch with the CUDA toolkit you want to target
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 2. TorchFX from the matching CUDA index
pip install torchfx \
    --index-url https://matteospanio.github.io/torchfx/wheels/cu124/ \
    --extra-index-url https://pypi.org/simple
```

The wheels carry a PEP 440 local-version segment (`torchfx==0.5.3+cu124`) so they will not collide with the CPU wheels on PyPI, and they link against the CUDA shared libraries that `torch` already provides --- the wheel itself stays small. CUDA wheels are not on PyPI; PyPI does not accept locally-versioned platform-specific wheels.

For platforms or CUDA versions not on the table above (Windows + CUDA, CUDA 11, etc.), build from source --- see [From source with pip](#from-source-with-pip) and make sure `nvcc` is on `PATH`. CMake auto-detects CUDA and links the CUDA kernels into `torchfx_ext`.

### From source with uv

For development with reproducible builds:

```bash
# Install uv if not already installed
pip install uv

# Clone and sync
git clone https://github.com/matteospanio/torchfx
cd torchfx
uv sync
```

`uv sync` reads [pyproject.toml](https://github.com/matteospanio/torchfx/blob/master/pyproject.toml) and `uv.lock` and installs dependencies at exact versions. The CMake build runs the same way as with pip.

### Installation Workflow

```{mermaid}
graph TB
    User["User"]

    subgraph "Installation Methods"
        PipPyPI["pip install torchfx"]
        PipSource["pip install . (source)"]
        UvSync["uv sync (source)"]
    end

    subgraph "PyPI prebuilt wheels (CPU)"
        ManyLinux["manylinux x86_64<br/>cp310 .. cp314"]
        MacIntel["macOS x86_64<br/>cp310 .. cp314"]
        MacArm["macOS arm64<br/>cp310 .. cp314"]
        Win["Windows x86_64<br/>cp310 .. cp314"]
        Sdist["torchfx-X.Y.Z.tar.gz<br/>(source)"]
    end

    subgraph "GitHub Pages CUDA wheels (Linux only)"
        CU124["manylinux x86_64<br/>+cu124, cp310 .. cp314"]
        CU128["manylinux x86_64<br/>+cu128, cp310 .. cp314"]
    end

    subgraph "Source build (scikit-build-core + CMake)"
        SBC["scikit-build-core"]
        CMake["CMake ≥ 3.18"]
        Cxx["C++17 compiler<br/>(GCC / Clang / MSVC)"]
        Nvcc["nvcc<br/>(optional, CUDA kernels)"]
        Ext["torchfx_ext<br/>(compiled .so / .dylib / .pyd)"]
    end

    Installed["torchfx package<br/>in site-packages"]

    User --> PipPyPI
    User --> PipSource
    User --> UvSync

    PipPyPI -->|matching wheel| ManyLinux
    PipPyPI -->|matching wheel| MacIntel
    PipPyPI -->|matching wheel| MacArm
    PipPyPI -->|matching wheel| Win
    PipPyPI -->|"--index-url cu124"| CU124
    PipPyPI -->|"--index-url cu128"| CU128
    PipPyPI -->|no wheel for platform| Sdist
    ManyLinux --> Installed
    MacIntel --> Installed
    MacArm --> Installed
    Win --> Installed
    CU124 --> Installed
    CU128 --> Installed

    Sdist --> SBC
    PipSource --> SBC
    UvSync --> SBC

    SBC --> CMake
    CMake --> Cxx
    CMake -. optional .-> Nvcc
    Cxx --> Ext
    Nvcc --> Ext
    Ext --> Installed
```

(build-system)=
## Build System

The build backend is **scikit-build-core**, configured in [pyproject.toml](https://github.com/matteospanio/torchfx/blob/master/pyproject.toml):

```toml
[build-system]
requires = ["scikit-build-core>=0.10", "torch>=2.6.0"]
build-backend = "scikit_build_core.build"

[tool.scikit-build]
wheel.packages = ["src/torchfx", "src/cli"]
cmake.build-type = "Release"
```

```{versionchanged} 0.5.3
Build backend migrated to scikit-build-core; the native extension is compiled
at install time and bundled into the wheel. Older releases JIT-compiled it on
first import via ``torch.utils.cpp_extension.load``.
```

### Building locally

To build a distribution package:

```bash
# CPU-only wheel (matches the CI flow)
TORCHFX_NO_CUDA=1 python -m build --wheel

# Full build (CUDA if nvcc is found)
python -m build

# Or with uv
uv build
```

Wheels land in `dist/`. The CI workflow at [`.github/workflows/wheels.yml`](https://github.com/matteospanio/torchfx/blob/master/.github/workflows/wheels.yml) runs the CPU-only command above for Python 3.10–3.13 on every tagged release.

### Installing a specific PyTorch build

The build links the extension against whichever `torch` is installed in the build environment, so to use a particular CUDA version:

1. Install `torch` first from the [official PyTorch install matrix](https://pytorch.org/get-started/locally/), e.g.
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```
2. Then install TorchFX from source without rebuilding the build environment:
   ```bash
   pip install torchfx --no-build-isolation
   ```

## Dependency Management with uv

The project uses **uv** as its primary dependency manager, which provides faster dependency resolution and reproducible builds through lock files.

### Dependency Groups

The optional dependency groups defined in [pyproject.toml](https://github.com/matteospanio/torchfx/blob/master/pyproject.toml) cover different use cases:

| Group | Purpose | Key packages |
|-------|---------|--------------|
| `cli` | Command-line interface | `typer`, `rich`, `prompt-toolkit`, `watchdog` |
| `dev` | Development tools | `black`, `mypy`, `pytest`, `pytest-benchmark`, `ruff`, `coverage`, `numba`, `scalene` |
| `realtime` | Real-time audio I/O | `sounddevice` |
| `cuda` | CUDA build helpers | `ninja` |
| `docs` | Documentation building | `sphinx`, `pydata-sphinx-theme`, `myst-parser`, `ablog`, `sphinx-design`, `sphinxcontrib-bibtex`, `sphinxcontrib-mermaid` |

To install with specific groups:

```bash
# Install with dev dependencies
uv sync --group dev

# Install only CLI dependencies
uv sync --group cli

# Install everything
uv sync --all-groups
```

:::{note}
For contributors and developers, we recommend `uv sync --all-groups` to install all development dependencies including testing, documentation, CLI, and realtime tools.
:::

## Developers

If you want to contribute to TorchFX, set up a development environment with:

```bash
git clone https://github.com/matteospanio/torchfx
cd torchfx
uv sync --all-groups
uv run pre-commit install
```

This installs all dependency groups and registers the project's pre-commit hooks. For more on the development workflow, see {doc}`../developer/index`.

## Verification

After installation, confirm that TorchFX is correctly installed and that the native extension loaded.

### Basic check

```python
from importlib.metadata import version
import torchfx

print(version("torchfx"))              # e.g. '0.5.3'
print(torchfx.is_native_available())   # True if torchfx_ext compiled successfully
```

`is_native_available()` is the canonical way to check that the C++ extension is loaded. Since 0.5.3 there is no Python fallback, so a `False` return value means the install is broken --- see [Troubleshooting](#troubleshooting).

### Comprehensive check

```python
from importlib.metadata import version
import torch
import torchfx
from torchfx import Wave, FX
from torchfx.effect import Gain, Normalize
from torchfx.filter import LoButterworth, HiButterworth, Butterworth

print(f"torchfx version:   {version('torchfx')}")
print(f"PyTorch version:   {torch.__version__}")
print(f"Native extension:  {torchfx.is_native_available()}")
print(f"CUDA available:    {torch.cuda.is_available()}")
```

On a Linux machine with an NVIDIA GPU and a CUDA-enabled `torch`, `torch.cuda.is_available()` should return `True`. On macOS, Windows, or any CPU-only install, it returns `False`.

### Functional smoke test

A quick test that exercises the full pipeline:

```python
import torch
import torchfx as fx

# 1 second of 440 Hz mono sine at 44.1 kHz
t = torch.linspace(0, 1, 44100)
ys = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)  # shape (1, 44100)
wave = fx.Wave(ys, fs=44100)

# Apply a -6 dB gain
gained = wave | fx.effect.Gain(0.5, gain_type="amplitude")

print(f"Original: shape={wave.ys.shape}, fs={wave.fs} Hz")
print(f"Processed: shape={gained.ys.shape}")
```

If this runs without errors, your installation is working correctly.

(troubleshooting)=
## Troubleshooting

### `ImportError: cannot import name 'torchfx_ext'` (or `is_native_available()` returns `False`)

The native C++ extension is missing. The extension is required to use TorchFX. Possible causes:

- **You installed from sdist on a platform where the build silently skipped a step.** Reinstall with `pip install --force-reinstall --no-binary torchfx torchfx` and watch the build log for errors.
- **You're missing system dependencies.** Install a C++17 compiler, CMake ≥ 3.18, and Python development headers (see [System dependencies](#system-dependencies-source-builds)).
- **PyTorch is not installed in the build environment.** The build links the extension against `torch`. If you use build isolation, the build env installs its own `torch` automatically; if you use `--no-build-isolation`, install `torch` first.

### Source install fails with `cmake: command not found`

CMake ≥ 3.18 is required for source builds. Install it with your system package manager:

```bash
# Debian/Ubuntu
sudo apt-get install cmake

# macOS (Homebrew)
brew install cmake

# Windows
# Install from https://cmake.org/download/ and add to PATH
```

### CUDA kernels not compiled (CPU-only build, GPU available)

CMake reports `TORCHFX_USE_CUDA = OFF` if it cannot find `nvcc` or the CUDA toolkit. Verify:

- `nvcc --version` works in your shell.
- The CUDA toolkit version matches your `torch` CUDA version.
- The `TORCHFX_NO_CUDA` env var is **not** set (`echo $TORCHFX_NO_CUDA` should print empty).

Then rebuild from source: `pip install torchfx --no-build-isolation --force-reinstall`.

### Dependency conflicts with uv

If `uv sync` fails to resolve dependencies, regenerate the lock file:

```bash
rm uv.lock
uv lock
uv sync
```

### Audio backend issues

If you hit `RuntimeError` reading audio files, ensure `soundfile` and its system library are installed:

```bash
pip install --upgrade soundfile

# Linux: libsndfile system library
sudo apt-get install libsndfile1

# macOS
brew install libsndfile

# Windows: usually bundled with the soundfile wheel
```

:::{admonition} Still Having Issues?
:class: tip

If you continue to experience problems, please:
- Check the [GitHub Issues](https://github.com/matteospanio/torchfx/issues) for similar reports.
- Open a new issue with your OS, Python version, full error traceback, and the output of `pip list` and `python -c "import torchfx; print(torchfx.is_native_available())"`.
:::

## Next Steps

Now that you have TorchFX installed, you can:

- Follow the {doc}`getting_started` guide to learn the basics
- Explore {doc}`../core-concepts/index` to understand the fundamental concepts
- Try the {doc}`../tutorials/index` for hands-on examples
- Learn about {doc}`../advanced/gpu-acceleration` for performance optimization

:::{seealso}
- {doc}`getting_started` --- Your first steps with TorchFX
- {doc}`../developer/index` --- Contributing to TorchFX development
- {doc}`../advanced/performance` --- Optimizing performance
:::
