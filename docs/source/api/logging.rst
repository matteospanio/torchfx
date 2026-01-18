Logging
=======

The logging module provides structured logging capabilities for TorchFX with support for debug logging, performance profiling, and custom handlers.

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

TorchFX follows Python logging best practices:

* **NullHandler by default** - No log output unless explicitly enabled
* **Convenience functions** - Easy configuration for common use cases
* **Hierarchical loggers** - Fine-grained control over log output
* **Performance logging** - Built-in tools for profiling pipelines

Quick Start
-----------

Enable debug logging:

.. code-block:: python

   import torchfx
   torchfx.logging.enable_debug_logging()

Profile a filter chain:

.. code-block:: python

   from torchfx.logging import log_performance

   with log_performance("filter_chain"):
       result = wave | filter1 | filter2
   # Logs: "filter_chain completed in 0.045s"

Configuration Functions
-----------------------

.. autofunction:: torchfx.logging.get_logger

.. autofunction:: torchfx.logging.enable_logging

.. autofunction:: torchfx.logging.enable_debug_logging

.. autofunction:: torchfx.logging.disable_logging

Performance Logging
-------------------

Context Manager
^^^^^^^^^^^^^^^

.. autofunction:: torchfx.logging.log_performance

Decorator
^^^^^^^^^

.. autoclass:: torchfx.logging.LogPerformance
   :members:
   :special-members: __init__, __call__

Constants
---------

.. autodata:: torchfx.logging.DEFAULT_FORMAT
   :annotation: = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

.. autodata:: torchfx.logging.DEFAULT_DATE_FORMAT
   :annotation: = "%Y-%m-%d %H:%M:%S"

Usage Examples
--------------

Basic Logging Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import torchfx

   # Enable INFO level logging (default)
   torchfx.logging.enable_logging()

   # Enable DEBUG level logging
   torchfx.logging.enable_debug_logging()

   # Enable WARNING level only
   torchfx.logging.enable_logging(level="WARNING")

   # Disable logging
   torchfx.logging.disable_logging()

Custom Log Format
^^^^^^^^^^^^^^^^^

.. code-block:: python

   import torchfx

   # Simple format with just level and message
   torchfx.logging.enable_logging(
       level="DEBUG",
       format_string="%(levelname)s: %(message)s"
   )

   # Output to a file
   with open("torchfx.log", "w") as f:
       torchfx.logging.enable_logging(level="DEBUG", stream=f)

Performance Profiling
^^^^^^^^^^^^^^^^^^^^^

Using the context manager:

.. code-block:: python

   from torchfx.logging import log_performance, enable_logging

   enable_logging()

   # Time a code block
   with log_performance("audio_processing"):
       wave = Wave.from_file("input.wav")
       result = wave | filter1 | filter2 | reverb
       result.save("output.wav")

   # Capture timing information
   with log_performance("filter_chain") as timing:
       result = wave | complex_filter
   print(f"Processing took {timing['elapsed_seconds']:.3f}s")

Using the decorator:

.. code-block:: python

   from torchfx.logging import LogPerformance

   @LogPerformance("process_audio")
   def process_audio(wave):
       return wave | filter1 | filter2

   # Each call logs execution time
   result = process_audio(wave)

Module-Specific Logging
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from torchfx.logging import get_logger

   # Get logger for a specific module
   wave_logger = get_logger("wave")
   filter_logger = get_logger("filter.iir")

   # These inherit the root logger's level
   wave_logger.debug("Loading audio file")
   filter_logger.info("Computing coefficients")

Standard Python Logging
^^^^^^^^^^^^^^^^^^^^^^^

TorchFX integrates with Python's standard logging:

.. code-block:: python

   import logging

   # Configure using standard Python logging
   logging.getLogger("torchfx").setLevel(logging.DEBUG)

   # Add custom handler
   handler = logging.FileHandler("torchfx.log")
   handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
   logging.getLogger("torchfx").addHandler(handler)

Logger Hierarchy
----------------

TorchFX uses hierarchical loggers for fine-grained control:

.. code-block:: text

   torchfx                    # Root logger
   ├── torchfx.wave          # Wave class operations
   ├── torchfx.effect        # Effect processing
   ├── torchfx.filter        # Filter operations
   │   ├── torchfx.filter.iir
   │   └── torchfx.filter.fir
   ├── torchfx.validation    # Validation messages
   └── torchfx.performance   # Performance timing

You can enable logging for specific subsystems:

.. code-block:: python

   import logging

   # Only log filter operations at DEBUG level
   logging.getLogger("torchfx.filter").setLevel(logging.DEBUG)

   # Only log performance timing
   logging.getLogger("torchfx.performance").setLevel(logging.INFO)
