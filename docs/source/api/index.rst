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
   logging

Overview
--------

The library is organized into five main modules:

* **Core** (:doc:`core`) - Base classes :py:class:`~torchfx.Wave` and :py:class:`~torchfx.FX`
* **Filters** (:doc:`filters`) - IIR and FIR filters for audio processing
* **Effects** (:doc:`effects`) - Built-in audio effects like Reverb and Delay
* **Validation** (:doc:`validation`) - Custom exceptions and parameter validation utilities
* **Logging** (:doc:`logging`) - Structured logging and performance profiling

Quick Start
-----------

Import the main classes:

.. code-block:: python

   from torchfx import Wave, FX
   from torchfx.filter import LoButterworth
   from torchfx.effect import Reverb
   from torchfx.validation import TorchFXError, validate_sample_rate
   from torchfx.logging import enable_debug_logging, log_performance

For detailed documentation of each module, see the sections below.
