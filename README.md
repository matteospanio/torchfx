# torchfx

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![arXiv](https://img.shields.io/badge/arXiv-2504.08624-b31b1b.svg)](https://arxiv.org/abs/2504.08624)
[![PyPI version](https://badge.fury.io/py/torchfx.svg)](https://badge.fury.io/py/torchfx)
![PyPI - Status](https://img.shields.io/pypi/status/torchfx)

TorchFX is a Python library that provides a modern approach to audio digital signal processing (DSP) using PyTorch. It allows you to create and manipulate audio processing pipelines in a flexible and efficient way, leveraging the power of PyTorch for GPU acceleration.
TorchFX is designed to be easy to use and integrate with existing PyTorch workflows, making it a great choice for researchers and developers working in the field of audio processing.

## Features
- **GPU Acceleration**: Leverage the power of GPUs for real-time audio processing.
- **Flexible Pipelines**: Create complex audio processing pipelines using a simple and intuitive API.
- **Pytorch Integration**: Seamlessly integrate with existing PyTorch workflows and models.
- **Pipe operator**: Use the pipe operator (`|`) to create and manipulate audio processing pipelines in a more readable and concise way.

## Installation

To install TorchFX, you can use pip:

```bash
pip install torchfx
```
or clone the repository and install it manually:

```bash
git clone https://github.com/matteospanio/torchfx
cd torchfx
pip install -e .
```

## Usage

TorchFX provides a simple and intuitive API for creating and manipulating audio processing pipelines. Here is a basic example of how to use TorchFX:

```python
import torch
import torchfx as fx

# Create a simple audio processing pipeline
filtered_out = (
    fx.Wave.from_file("path_to_audio.wav")
    | fx.filter.LoButterworth(8000)
    | fx.filter.HiShelving(2000)
)
```

This example demonstrates how to create a simple audio processing pipeline using the `|` operator. The pipeline reads an audio file, applies a low-pass Butterworth filter, and then applies a high-shelving filter.
`torchfx` provides a `Wave` class that embeds the audio signal and its sampling rate in a single object. This allows you to easily manipulate the audio signal and apply various transformations using the provided filters. This class provides the bitwise or operator overloading, which allows you to chain multiple filters (and any kind of nn.Module) together in a single pipeline.

## API

At the moment the API is not fully documented, but you can find the list of available filters in the [filters.py](src/torchfx/filter/__init__.py) file. The filters are implemented as subclasses of `torch.nn.Module`, and you can use them just like any other PyTorch module.

## How to cite

If you use this code in your research, please cite the following paper:

```
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
