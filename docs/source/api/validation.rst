Validation
==========

The validation module provides custom exceptions and validation utilities for parameter validation across TorchFX.

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

TorchFX provides a comprehensive validation system with:

* **Custom exception hierarchy** for specific error handling
* **Validator functions** for common validation patterns
* **Context-aware error messages** with suggestions for fixes

Import validation utilities:

.. code-block:: python

   from torchfx.validation import (
       TorchFXError,
       InvalidParameterError,
       validate_sample_rate,
       validate_range,
   )

Exception Hierarchy
-------------------

All exceptions inherit from :py:class:`~torchfx.validation.TorchFXError`, enabling users to catch all library-specific errors with a single except clause:

.. code-block:: python

   try:
       # TorchFX operations
       wave = Wave.from_file("audio.wav")
       processed = wave | some_filter
   except TorchFXError as e:
       print(f"TorchFX error: {e}")

The exception hierarchy is organized as follows:

.. code-block:: text

   TorchFXError (base)
   ├── InvalidParameterError
   │   ├── InvalidSampleRateError
   │   ├── InvalidRangeError
   │   ├── InvalidShapeError
   │   └── InvalidTypeError
   └── AudioProcessingError
       ├── CoefficientComputationError
       └── FilterInstabilityError

Base Exception
^^^^^^^^^^^^^^

.. autoclass:: torchfx.validation.TorchFXError
   :members:
   :show-inheritance:

Parameter Validation Exceptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchfx.validation.InvalidParameterError
   :members:
   :show-inheritance:

.. autoclass:: torchfx.validation.InvalidSampleRateError
   :members:
   :show-inheritance:

.. autoclass:: torchfx.validation.InvalidRangeError
   :members:
   :show-inheritance:

.. autoclass:: torchfx.validation.InvalidShapeError
   :members:
   :show-inheritance:

.. autoclass:: torchfx.validation.InvalidTypeError
   :members:
   :show-inheritance:

Audio Processing Exceptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchfx.validation.AudioProcessingError
   :members:
   :show-inheritance:

.. autoclass:: torchfx.validation.CoefficientComputationError
   :members:
   :show-inheritance:

.. autoclass:: torchfx.validation.FilterInstabilityError
   :members:
   :show-inheritance:

Validator Functions
-------------------

Sample Rate Validation
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchfx.validation.validate_sample_rate

Range Validation
^^^^^^^^^^^^^^^^

.. autofunction:: torchfx.validation.validate_positive

.. autofunction:: torchfx.validation.validate_range

.. autofunction:: torchfx.validation.validate_in_set

Tensor Shape Validation
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchfx.validation.validate_tensor_ndim

.. autofunction:: torchfx.validation.validate_audio_tensor

Type Validation
^^^^^^^^^^^^^^^

.. autofunction:: torchfx.validation.validate_type

Audio-Specific Validation
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchfx.validation.validate_cutoff_frequency

.. autofunction:: torchfx.validation.validate_filter_order

.. autofunction:: torchfx.validation.validate_q_factor

Constants
---------

.. autodata:: torchfx.validation.COMMON_SAMPLE_RATES
   :annotation: = (8000, 11025, 16000, 22050, 44100, 48000, 88200, 96000, 176400, 192000)

Usage Examples
--------------

Validating Parameters in Custom Effects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from torchfx import FX
   from torchfx.validation import (
       validate_positive,
       validate_range,
       validate_sample_rate,
   )

   class CustomEffect(FX):
       def __init__(
           self,
           gain: float,
           mix: float = 0.5,
           fs: int | None = None,
       ) -> None:
           super().__init__()

           # Validate parameters
           validate_positive(gain, "gain")
           validate_range(mix, "mix", min_value=0, max_value=1)
           validate_sample_rate(fs, allow_none=True)

           self.gain = gain
           self.mix = mix
           self.fs = fs

Validating Audio Tensors
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import torch
   from torchfx.validation import validate_audio_tensor

   def process_audio(waveform: torch.Tensor) -> torch.Tensor:
       # Ensure tensor is valid audio shape
       validate_audio_tensor(
           waveform,
           min_channels=1,
           max_channels=8,
           min_samples=100,
       )
       # Process...
       return waveform

Catching Specific Errors
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from torchfx.validation import (
       InvalidRangeError,
       InvalidSampleRateError,
       TorchFXError,
   )

   try:
       # Some operation
       pass
   except InvalidSampleRateError as e:
       print(f"Invalid sample rate: {e.actual_value}")
   except InvalidRangeError as e:
       print(f"Value out of range: {e.parameter_name} = {e.actual_value}")
   except TorchFXError as e:
       print(f"Other TorchFX error: {e}")
