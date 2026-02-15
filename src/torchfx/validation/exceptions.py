"""Custom exceptions for TorchFX validation and error handling.

This module provides a hierarchical exception system for TorchFX, enabling
specific error handling and context-aware error messages.

Exception Hierarchy
-------------------
TorchFXError (base)
    Base exception for all TorchFX library errors.
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

Examples
--------
Catch all TorchFX errors:

>>> try:
...     # TorchFX operations
...     pass
... except TorchFXError as e:
...     print(f"TorchFX error: {e}")

Catch specific parameter errors:

>>> try:
...     # Filter operations
...     pass
... except InvalidParameterError as e:
...     print(f"Invalid parameter: {e.parameter_name} = {e.actual_value}")

"""

from __future__ import annotations

from typing import Any


class TorchFXError(Exception):
    """Base exception for all TorchFX library errors.

    All custom exceptions in TorchFX inherit from this class, enabling
    users to catch all library-specific errors with a single except clause.

    Parameters
    ----------
    message : str
        Human-readable error message.
    parameter_name : str | None, optional
        Name of the parameter that caused the error, if applicable.
    actual_value : Any | None, optional
        The actual value that caused the error.
    suggestion : str | None, optional
        A suggestion for fixing the error.

    Examples
    --------
    >>> try:
    ...     raise TorchFXError("Something went wrong")
    ... except TorchFXError as e:
    ...     print(f"Error: {e}")
    Error: Something went wrong

    """

    def __init__(
        self,
        message: str,
        parameter_name: str | None = None,
        actual_value: Any | None = None,
        suggestion: str | None = None,
    ) -> None:
        self.message = message
        self.parameter_name = parameter_name
        self.actual_value = actual_value
        self.suggestion = suggestion
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the full error message with context."""
        parts = [self.message]
        if self.parameter_name is not None:
            parts.append(f"Parameter: {self.parameter_name}")
        if self.actual_value is not None:
            parts.append(f"Got: {self.actual_value!r}")
        if self.suggestion is not None:
            parts.append(f"Suggestion: {self.suggestion}")
        return " | ".join(parts)


class InvalidParameterError(TorchFXError):
    """Exception raised when a parameter value is invalid.

    This is the base class for all parameter validation errors.
    Use more specific subclasses when possible.

    Parameters
    ----------
    message : str
        Human-readable error message.
    parameter_name : str
        Name of the invalid parameter.
    actual_value : Any
        The actual value that was provided.
    expected : str | None, optional
        Description of what was expected.
    suggestion : str | None, optional
        A suggestion for fixing the error.

    Examples
    --------
    >>> raise InvalidParameterError(
    ...     "Cutoff frequency must be positive",
    ...     parameter_name="cutoff",
    ...     actual_value=-100,
    ...     expected="positive float",
    ... )
    Traceback (most recent call last):
        ...
    torchfx.validation.exceptions.InvalidParameterError: ...

    """

    def __init__(
        self,
        message: str,
        parameter_name: str,
        actual_value: Any,
        expected: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        self.expected = expected
        super().__init__(
            message=message,
            parameter_name=parameter_name,
            actual_value=actual_value,
            suggestion=suggestion,
        )

    def _format_message(self) -> str:
        """Format the full error message with context."""
        parts = [self.message]
        parts.append(f"Parameter: {self.parameter_name}")
        parts.append(f"Got: {self.actual_value!r}")
        if self.expected is not None:
            parts.append(f"Expected: {self.expected}")
        if self.suggestion is not None:
            parts.append(f"Suggestion: {self.suggestion}")
        return " | ".join(parts)


class InvalidSampleRateError(InvalidParameterError):
    """Exception raised when sample rate is invalid.

    Sample rate must be a positive integer within reasonable bounds
    for audio processing (typically 1 Hz to 384000 Hz).

    Parameters
    ----------
    actual_value : int | None
        The invalid sample rate value.
    suggestion : str | None, optional
        A suggestion for fixing the error.

    Examples
    --------
    >>> raise InvalidSampleRateError(actual_value=-44100)
    Traceback (most recent call last):
        ...
    torchfx.validation.exceptions.InvalidSampleRateError: ...

    """

    def __init__(
        self,
        actual_value: int | None,
        suggestion: str | None = None,
    ) -> None:
        msg = "Sample rate must be a positive integer"
        super().__init__(
            message=msg,
            parameter_name="fs",
            actual_value=actual_value,
            expected="positive integer (e.g., 44100, 48000)",
            suggestion=suggestion or "Common sample rates: 44100, 48000, 96000 Hz",
        )


class InvalidRangeError(InvalidParameterError):
    """Exception raised when a value is outside expected bounds.

    Parameters
    ----------
    parameter_name : str
        Name of the parameter.
    actual_value : float | int
        The actual value that was provided.
    min_value : float | int | None, optional
        Minimum allowed value. None means no minimum.
    max_value : float | int | None, optional
        Maximum allowed value. None means no maximum.
    min_inclusive : bool, optional
        If True, min_value is included in the range. Default is True.
    max_inclusive : bool, optional
        If True, max_value is included in the range. Default is True.

    Examples
    --------
    >>> raise InvalidRangeError("decay", 1.5, min_value=0, max_value=1)
    Traceback (most recent call last):
        ...
    torchfx.validation.exceptions.InvalidRangeError: ...

    """

    def __init__(
        self,
        parameter_name: str,
        actual_value: float | int,
        min_value: float | int | None = None,
        max_value: float | int | None = None,
        min_inclusive: bool = True,
        max_inclusive: bool = True,
    ) -> None:
        self.min_value = min_value
        self.max_value = max_value

        # Build expected range string
        left = "[" if min_inclusive else "("
        right = "]" if max_inclusive else ")"
        min_str = str(min_value) if min_value is not None else "-inf"
        max_str = str(max_value) if max_value is not None else "inf"
        expected = f"{left}{min_str}, {max_str}{right}"

        super().__init__(
            message=f"Value out of range for {parameter_name}",
            parameter_name=parameter_name,
            actual_value=actual_value,
            expected=expected,
        )


class InvalidShapeError(InvalidParameterError):
    """Exception raised when tensor shape is invalid.

    Parameters
    ----------
    parameter_name : str
        Name of the parameter.
    actual_shape : tuple[int, ...]
        The actual shape of the tensor.
    expected_ndim : int | None, optional
        Expected number of dimensions.
    expected_shape : tuple[int | None, ...] | None, optional
        Expected shape (None elements are wildcards).
    suggestion : str | None, optional
        A suggestion for fixing the error.

    Examples
    --------
    >>> raise InvalidShapeError(
    ...     "waveform",
    ...     actual_shape=(4, 2, 1000),
    ...     expected_ndim=2,
    ...     suggestion="Audio should be (channels, samples)"
    ... )
    Traceback (most recent call last):
        ...
    torchfx.validation.exceptions.InvalidShapeError: ...

    """

    def __init__(
        self,
        parameter_name: str,
        actual_shape: tuple[int, ...],
        expected_ndim: int | None = None,
        expected_shape: tuple[int | None, ...] | None = None,
        suggestion: str | None = None,
    ) -> None:
        if expected_ndim is not None:
            expected = f"{expected_ndim}D tensor"
        elif expected_shape is not None:
            expected = f"shape {expected_shape}"
        else:
            expected = "valid tensor shape"

        super().__init__(
            message=f"Invalid tensor shape for {parameter_name}",
            parameter_name=parameter_name,
            actual_value=f"shape {actual_shape}",
            expected=expected,
            suggestion=suggestion,
        )


class InvalidTypeError(InvalidParameterError):
    """Exception raised when a parameter has an invalid type.

    Parameters
    ----------
    parameter_name : str
        Name of the parameter.
    actual_type : type
        The actual type of the value.
    expected_types : tuple[type, ...]
        Tuple of expected types.

    Examples
    --------
    >>> raise InvalidTypeError(
    ...     "gain",
    ...     actual_type=str,
    ...     expected_types=(int, float),
    ... )
    Traceback (most recent call last):
        ...
    torchfx.validation.exceptions.InvalidTypeError: ...

    """

    def __init__(
        self,
        parameter_name: str,
        actual_type: type,
        expected_types: tuple[type, ...],
    ) -> None:
        expected_names = ", ".join(t.__name__ for t in expected_types)
        super().__init__(
            message=f"Invalid type for {parameter_name}",
            parameter_name=parameter_name,
            actual_value=actual_type.__name__,
            expected=expected_names,
        )


class AudioProcessingError(TorchFXError):
    """Exception raised during audio processing operations.

    This error indicates a problem during the actual processing of audio,
    not during parameter validation.

    Parameters
    ----------
    message : str
        Human-readable error message.
    suggestion : str | None, optional
        A suggestion for fixing the error.

    Examples
    --------
    >>> raise AudioProcessingError(
    ...     "Failed to process audio buffer",
    ...     suggestion="Check input tensor dimensions"
    ... )
    Traceback (most recent call last):
        ...
    torchfx.validation.exceptions.AudioProcessingError: ...

    """

    pass


class CoefficientComputationError(AudioProcessingError):
    """Exception raised when filter coefficient computation fails.

    This typically occurs when filter parameters result in mathematically
    invalid or numerically unstable coefficients.

    Parameters
    ----------
    filter_type : str
        The type of filter that failed.
    reason : str
        The reason for the failure.
    suggestion : str | None, optional
        A suggestion for fixing the error.

    Examples
    --------
    >>> raise CoefficientComputationError(
    ...     filter_type="LoButterworth",
    ...     reason="Cutoff frequency exceeds Nyquist",
    ... )
    Traceback (most recent call last):
        ...
    torchfx.validation.exceptions.CoefficientComputationError: ...

    """

    def __init__(
        self,
        filter_type: str,
        reason: str,
        suggestion: str | None = None,
    ) -> None:
        super().__init__(
            message=f"Failed to compute coefficients for {filter_type}: {reason}",
            suggestion=suggestion,
        )


class FilterInstabilityError(AudioProcessingError):
    """Exception raised when a filter is numerically unstable.

    Unstable filters can produce infinite or NaN values and should not
    be used for audio processing.

    Parameters
    ----------
    filter_type : str
        The type of filter that is unstable.
    suggestion : str | None, optional
        A suggestion for fixing the error.

    Examples
    --------
    >>> raise FilterInstabilityError(
    ...     filter_type="HiChebyshev1",
    ...     suggestion="Try reducing filter order",
    ... )
    Traceback (most recent call last):
        ...
    torchfx.validation.exceptions.FilterInstabilityError: ...

    """

    def __init__(
        self,
        filter_type: str,
        suggestion: str | None = None,
    ) -> None:
        super().__init__(
            message=f"Filter {filter_type} is numerically unstable",
            suggestion=suggestion or "Try reducing filter order or using a different filter type",
        )
