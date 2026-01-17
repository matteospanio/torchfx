"""Input validation utilities for TorchFX.

This module provides custom exceptions and validation functions for
parameter validation across TorchFX.

Exceptions
----------
TorchFXError
    Base exception for all TorchFX errors.
InvalidParameterError
    Raised when a parameter value is invalid.
InvalidSampleRateError
    Raised when sample rate is invalid.
InvalidRangeError
    Raised when a value is out of range.
InvalidShapeError
    Raised when tensor shape is invalid.
InvalidTypeError
    Raised when a parameter has wrong type.
AudioProcessingError
    Raised during audio processing failures.
CoefficientComputationError
    Raised when filter coefficient computation fails.
FilterInstabilityError
    Raised when a filter is numerically unstable.

Validators
----------
validate_sample_rate
    Validate sample rate parameters.
validate_positive
    Validate positive values.
validate_range
    Validate values within a range.
validate_in_set
    Validate values from a set of options.
validate_tensor_ndim
    Validate tensor dimensionality.
validate_audio_tensor
    Validate audio tensor shapes.
validate_type
    Validate parameter types.
validate_cutoff_frequency
    Validate filter cutoff frequencies.
validate_filter_order
    Validate filter order parameters.
validate_q_factor
    Validate Q factor parameters.

Constants
---------
COMMON_SAMPLE_RATES
    Tuple of common audio sample rates for reference.

Examples
--------
Catch all TorchFX errors:

>>> from torchfx.validation import TorchFXError
>>> try:
...     # TorchFX operations
...     pass
... except TorchFXError as e:
...     print(f"Error: {e}")

Validate parameters:

>>> from torchfx.validation import validate_sample_rate, validate_range
>>> validate_sample_rate(44100)  # OK
>>> validate_range(0.5, "decay", min_value=0, max_value=1)  # OK

Validate audio tensors:

>>> import torch
>>> from torchfx.validation import validate_audio_tensor
>>> waveform = torch.randn(2, 44100)
>>> validate_audio_tensor(waveform)  # OK

"""

from torchfx.validation.exceptions import (
    AudioProcessingError,
    CoefficientComputationError,
    FilterInstabilityError,
    InvalidParameterError,
    InvalidRangeError,
    InvalidSampleRateError,
    InvalidShapeError,
    InvalidTypeError,
    TorchFXError,
)
from torchfx.validation.validators import (
    COMMON_SAMPLE_RATES,
    validate_audio_tensor,
    validate_cutoff_frequency,
    validate_filter_order,
    validate_in_set,
    validate_positive,
    validate_q_factor,
    validate_range,
    validate_sample_rate,
    validate_tensor_ndim,
    validate_type,
)

__all__ = [
    # Exceptions
    "TorchFXError",
    "InvalidParameterError",
    "InvalidSampleRateError",
    "InvalidRangeError",
    "InvalidShapeError",
    "InvalidTypeError",
    "AudioProcessingError",
    "CoefficientComputationError",
    "FilterInstabilityError",
    # Validators
    "validate_sample_rate",
    "validate_positive",
    "validate_range",
    "validate_in_set",
    "validate_tensor_ndim",
    "validate_audio_tensor",
    "validate_type",
    "validate_cutoff_frequency",
    "validate_filter_order",
    "validate_q_factor",
    # Constants
    "COMMON_SAMPLE_RATES",
]
