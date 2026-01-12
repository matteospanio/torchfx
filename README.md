<div align="center">

# TorchFX

### GPU-Accelerated Audio DSP with PyTorch

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![arXiv](https://img.shields.io/badge/arXiv-2504.08624-b31b1b.svg)](https://arxiv.org/abs/2504.08624)
[![PyPI version](https://badge.fury.io/py/torchfx.svg)](https://badge.fury.io/py/torchfx)
![PyPI - Status](https://img.shields.io/pypi/status/torchfx)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/matteospanio/torchfx)

**[Documentation](https://matteospanio.github.io/torchfx/)** | **[Getting Started](https://matteospanio.github.io/torchfx/guides/getting-started/getting_started.html)** | **[API Reference](https://matteospanio.github.io/torchfx/api/index.html)** | **[Blog](https://matteospanio.github.io/torchfx/blog/index.html)**

</div>

---

TorchFX is a modern Python library for **high-performance digital signal processing** in audio, leveraging PyTorch and GPU acceleration. Built for researchers, engineers, and developers who need fast, flexible, and differentiable audio processing.

## ✨ Key Features

- ⚡ **GPU Acceleration** - Built on PyTorch for high-performance audio processing on CUDA-enabled devices
- 🔗 **Composable Pipelines** - Chain filters with the pipe operator (`|`) for sequential processing
- ➕ **Parallel Processing** - Combine filters with the add operator (`+`) for parallel filter combination
- 🧠 **PyTorch Native** - All filters are `torch.nn.Module` subclasses, enabling gradient-based optimization
- 🎯 **Simple & Intuitive** - Clean, object-oriented API designed for ease of use
- ⚙️ **Highly Extensible** - Create custom filters and effects by extending base classes
- 📊 **Performance Optimized** - Substantial performance gains over SciPy for long and multichannel signals

## 🚀 Quick Start

### Installation

```bash
pip install torchfx
```

Or install from source:

```bash
git clone https://github.com/matteospanio/torchfx
cd torchfx
pip install -e .
```

### Basic Example

```python
import torch
from torchfx import Wave
from torchfx.filter import LoButterworth, ParametricEQ

# Load audio
wave = Wave.from_file("audio.wav")

# Create filters
lowpass = LoButterworth(cutoff=5000, order=4, fs=wave.fs)
eq = ParametricEQ(frequency=1000, q=2.0, gain=3.0, fs=wave.fs)

# Sequential processing with pipe operator (|)
processed = wave | lowpass | eq

# Parallel processing with add operator (+)
stereo_enhancer = lowpass + eq
enhanced = wave | stereo_enhancer

# Save result
processed.save("output.wav")
```

## 📚 Documentation

- **[Full Documentation](https://matteospanio.github.io/torchfx/)** - Complete guides and API reference
- **[Getting Started](https://matteospanio.github.io/torchfx/guides/getting-started/getting_started.html)** - Installation and first steps
- **[Tutorials](https://matteospanio.github.io/torchfx/guides/tutorials/index.html)** - Practical examples and use cases
- **[API Reference](https://matteospanio.github.io/torchfx/api/index.html)** - Detailed API documentation
- **[Blog](https://matteospanio.github.io/torchfx/blog/index.html)** - Updates, releases, and insights

## 🛠️ Development

- **[Contributing Guide](CONTRIBUTING.md)** - How to contribute to TorchFX
- **[Style Guide](https://matteospanio.github.io/torchfx/guides/developer/style_guide.html)** - Coding standards and conventions
- **[Roadmap](https://matteospanio.github.io/torchfx/guides/developer/roadmap.html)** - Future plans and priorities

We welcome contributions from everyone! Please read our [Contributing Guide](CONTRIBUTING.md) to get started.

## 📖 Citation

If you use TorchFX in your research, please cite our paper:

```bibtex
@misc{spanio2025torchfxmodernapproachaudio,
  title={TorchFX: A modern approach to Audio DSP with PyTorch and GPU acceleration},
  author={Matteo Spanio and Antonio Rodà},
  year={2025},
  eprint={2504.08624},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  url={https://arxiv.org/abs/2504.08624},
}
```

## License

This project is licensed under the terms of the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

## Third-Party Acknowledgments

This project uses the following third-party libraries:

- [PyTorch](https://pytorch.org/) – BSD-style license
- [Torchaudio](https://pytorch.org/audio/) – BSD 2-Clause License
- [NumPy](https://numpy.org/) – BSD 3-Clause License
- [SciPy](https://scipy.org/) – BSD 3-Clause License
- [SoundFile](https://pysoundfile.readthedocs.io/) – BSD 3-Clause License

Their respective license texts are included in the `licenses/` directory.
