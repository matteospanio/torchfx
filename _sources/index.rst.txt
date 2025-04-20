:hero: A GPU accelerated and torch based audio DSP library.

.. torchfx documentation master file, created by
   sphinx-quickstart on Fri Apr 11 14:20:59 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Audio DSP with the GPU
======================

.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0
   :alt: License: GPL v3

.. image:: https://img.shields.io/badge/arXiv-2504.08624-b31b1b.svg
   :target: https://arxiv.org/abs/2504.08624
   :alt: arXiv

.. image:: https://badge.fury.io/py/torchfx.svg
   :target: https://badge.fury.io/py/torchfx
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/status/torchfx
   :alt: PyPI - Status

TorchFX is a modern Python library for digital signal processing (DSP) in audio, designed to leverage the power of **PyTorch** and **GPU acceleration**. It provides a clean and flexible API to build and compose audio processing pipelines using PyTorch modules, making it ideal for researchers, engineers, and developers working on modern audio applications.

TorchFX integrates seamlessly into PyTorch workflows, enabling real-time, differentiable audio processing and model-based DSP pipelines.

Motivation
----------

While many DSP libraries exist in Python (e.g., `scipy`, `librosa`, `torchaudio`), they often fail to address the full spectrum of needs in contemporary audio processing, such as:

- Full exploitation of **GPU acceleration**,
- Native integration with **deep learning models**,
- A clean, **object-oriented**, and modular interface,
- Efficient support for **multichannel audio signals**.

As a result, researchers and developers frequently end up building custom, ad-hoc solutions that are hard to maintain, reuse, or scale—especially when real-time performance or AI integration is required.

Benefits of TorchFX
-------------------

TorchFX is designed from the ground up to provide a modern, powerful, and flexible framework for audio DSP in Python:

- ✅ **GPU Acceleration**: Built on PyTorch, it allows high-performance, real-time audio processing on CUDA-enabled devices.
- ✅ **Functional Chaining**: The overloaded pipe operator (`|`) enables clean and readable composition of processing pipelines.
- ✅ **Extensible API**: Create custom filters and effects with ease using OOP principles.
- ✅ **PyTorch-Compatible**: All filters are subclasses of `torch.nn.Module`, making them seamlessly integrable with PyTorch models and training loops.
- ✅ **High Performance**: Benchmarks show substantial performance gains over SciPy, especially with long and multichannel signals.
- ✅ **Object-Oriented DSP**: Filters are designed as individual classes, promoting modularity, reusability, and testability.

Performance Insights
--------------------

Benchmarks included in the original paper demonstrate that TorchFX outperforms traditional DSP tools like SciPy in multi-channel contexts, thanks to its efficient use of **parallel GPU computation**. Even on CPU, TorchFX performs competitively due to effective multi-threading and optimized PyTorch internals.

.. image:: _static/3.png
   :alt: Benchmark performance
   :align: center

How to cite
-----------

If you use TorchFX in your research or project, please cite the following publication:

.. code-block:: bibtex

   @misc{spanio2025torchfxmodernapproachaudio,
     title={TorchFX: A modern approach to Audio DSP with PyTorch and GPU acceleration},
     author={Matteo Spanio and Antonio Rodà},
     year={2025},
     eprint={2504.08624},
     archivePrefix={arXiv},
     primaryClass={eess.AS},
     url={https://arxiv.org/abs/2504.08624},
   }

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   getting_started
   api
