API Reference
=============

TorchFX provides a comprehensive API for audio signal processing with PyTorch.

.. toctree::
   :maxdepth: 2
   :caption: Modules

   core
   filters
   effects
   validation

Overview
--------

The library is organized into four main modules:

* **Core** (:doc:`core`) - Base classes :py:class:`~torchfx.Wave` and :py:class:`~torchfx.FX`
* **Filters** (:doc:`filters`) - IIR and FIR filters for audio processing
* **Effects** (:doc:`effects`) - Built-in audio effects like Reverb and Delay
* **Validation** (:doc:`validation`) - Custom exceptions and parameter validation utilities

Quick Start
-----------

Import the main classes:

.. code-block:: python

   from torchfx import Wave, FX
   from torchfx.filter import LoButterworth
   from torchfx.effect import Reverb
   from torchfx.validation import TorchFXError, validate_sample_rate

For detailed documentation of each module, see the sections below.
