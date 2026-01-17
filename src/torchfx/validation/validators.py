"""Reusable validation utilities for TorchFX.

This module provides validator functions for common validation patterns:
- Sample rate validation
- Parameter range validation
- Tensor shape validation
- Type validation
- Audio-specific validators

All validators raise appropriate exceptions from torchfx.validation.exceptions.

Examples
--------
Validate a sample rate:

>>> from torchfx.validation import validate_sample_rate
>>> validate_sample_rate(44100)  # OK
>>> validate_sample_rate(-1)  # Raises InvalidSampleRateError

Validate a parameter range:

>>> from torchfx.validation import validate_range
>>> validate_range(0.5, "decay", min_value=0, max_value=1)  # OK
>>> validate_range(1.5, "decay", min_value=0, max_value=1)  # Raises InvalidRangeError

Validate an audio tensor:

>>> import torch
>>> from torchfx.validation import validate_audio_tensor
>>> waveform = torch.randn(2, 44100)
>>> validate_audio_tensor(waveform)  # OK

"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeVar

from torch import Tensor

from torchfx.validation.exceptions import (
    InvalidParameterError,
    InvalidRangeError,
    InvalidSampleRateError,
    InvalidShapeError,
    InvalidTypeError,
)

T = TypeVar("T", int, float)


# =============================================================================
# Constants
# =============================================================================

#: Common audio sample rates for reference and suggestions
COMMON_SAMPLE_RATES = (
    8000,
    11025,
    16000,
    22050,
    44100,
    48000,
    88200,
    96000,
    176400,
    192000,
)


# =============================================================================
# Sample Rate Validation
# =============================================================================


def validate_sample_rate(
    fs: int | None,
    *,
    allow_none: bool = False,
    min_rate: int = 1,
    max_rate: int = 384000,
) -> None:
    """Validate a sample rate parameter.

    Parameters
    ----------
    fs : int | None
        The sample rate to validate.
    allow_none : bool, optional
        If True, None is a valid value (for lazy initialization).
        Default is False.
    min_rate : int, optional
        Minimum allowed sample rate in Hz. Default is 1.
    max_rate : int, optional
        Maximum allowed sample rate in Hz. Default is 384000.

    Raises
    ------
    InvalidSampleRateError
        If the sample rate is invalid.

    Examples
    --------
    >>> validate_sample_rate(44100)  # OK
    >>> validate_sample_rate(None, allow_none=True)  # OK
    >>> validate_sample_rate(-1)  # Raises InvalidSampleRateError

    """
    if fs is None:
        if not allow_none:
            raise InvalidSampleRateError(
                actual_value=fs,
                suggestion="Sample rate must be set before processing",
            )
        return

    if not isinstance(fs, int):
        raise InvalidSampleRateError(
            actual_value=fs,
            suggestion=f"Got {type(fs).__name__}, expected int",
        )

    if fs < min_rate or fs > max_rate:
        raise InvalidSampleRateError(
            actual_value=fs,
            suggestion=f"Must be between {min_rate} and {max_rate} Hz",
        )


# =============================================================================
# Range Validation
# =============================================================================


def validate_positive(
    value: T,
    parameter_name: str,
    *,
    allow_zero: bool = False,
) -> None:
    """Validate that a value is positive.

    Parameters
    ----------
    value : int | float
        The value to validate.
    parameter_name : str
        Name of the parameter (for error messages).
    allow_zero : bool, optional
        If True, zero is allowed. Default is False.

    Raises
    ------
    InvalidRangeError
        If the value is not positive (or non-negative if allow_zero=True).

    Examples
    --------
    >>> validate_positive(1.0, "cutoff")  # OK
    >>> validate_positive(0, "cutoff", allow_zero=True)  # OK
    >>> validate_positive(-1, "cutoff")  # Raises InvalidRangeError

    """
    if allow_zero:
        if value < 0:
            raise InvalidRangeError(
                parameter_name=parameter_name,
                actual_value=value,
                min_value=0,
                max_value=None,
                min_inclusive=True,
            )
    else:
        if value <= 0:
            raise InvalidRangeError(
                parameter_name=parameter_name,
                actual_value=value,
                min_value=0,
                max_value=None,
                min_inclusive=False,
            )


def validate_range(
    value: T,
    parameter_name: str,
    *,
    min_value: T | None = None,
    max_value: T | None = None,
    min_inclusive: bool = True,
    max_inclusive: bool = True,
) -> None:
    """Validate that a value falls within a specified range.

    Parameters
    ----------
    value : int | float
        The value to validate.
    parameter_name : str
        Name of the parameter (for error messages).
    min_value : int | float | None, optional
        Minimum allowed value. None means no minimum.
    max_value : int | float | None, optional
        Maximum allowed value. None means no maximum.
    min_inclusive : bool, optional
        If True, min_value is included in the range. Default is True.
    max_inclusive : bool, optional
        If True, max_value is included in the range. Default is True.

    Raises
    ------
    InvalidRangeError
        If the value is outside the specified range.

    Examples
    --------
    >>> validate_range(0.5, "decay", min_value=0, max_value=1)  # OK
    >>> validate_range(0, "decay", min_value=0, max_value=1, min_inclusive=False)  # Raises

    """
    if min_value is not None:
        if min_inclusive:
            if value < min_value:
                raise InvalidRangeError(
                    parameter_name=parameter_name,
                    actual_value=value,
                    min_value=min_value,
                    max_value=max_value,
                    min_inclusive=min_inclusive,
                    max_inclusive=max_inclusive,
                )
        else:
            if value <= min_value:
                raise InvalidRangeError(
                    parameter_name=parameter_name,
                    actual_value=value,
                    min_value=min_value,
                    max_value=max_value,
                    min_inclusive=min_inclusive,
                    max_inclusive=max_inclusive,
                )

    if max_value is not None:
        if max_inclusive:
            if value > max_value:
                raise InvalidRangeError(
                    parameter_name=parameter_name,
                    actual_value=value,
                    min_value=min_value,
                    max_value=max_value,
                    min_inclusive=min_inclusive,
                    max_inclusive=max_inclusive,
                )
        else:
            if value >= max_value:
                raise InvalidRangeError(
                    parameter_name=parameter_name,
                    actual_value=value,
                    min_value=min_value,
                    max_value=max_value,
                    min_inclusive=min_inclusive,
                    max_inclusive=max_inclusive,
                )


def validate_in_set(
    value: Any,
    parameter_name: str,
    valid_values: Sequence[Any],
) -> None:
    """Validate that a value is in a set of allowed values.

    Parameters
    ----------
    value : Any
        The value to validate.
    parameter_name : str
        Name of the parameter (for error messages).
    valid_values : Sequence[Any]
        Sequence of allowed values.

    Raises
    ------
    InvalidParameterError
        If the value is not in the set of valid values.

    Examples
    --------
    >>> validate_in_set("amplitude", "gain_type", ["amplitude", "db", "power"])  # OK
    >>> validate_in_set("invalid", "gain_type", ["amplitude", "db", "power"])  # Raises

    """
    if value not in valid_values:
        raise InvalidParameterError(
            message=f"Invalid value for {parameter_name}",
            parameter_name=parameter_name,
            actual_value=value,
            expected=f"one of {list(valid_values)}",
        )


# =============================================================================
# Tensor Shape Validation
# =============================================================================


def validate_tensor_ndim(
    tensor: Tensor,
    parameter_name: str,
    *,
    expected_ndim: int | Sequence[int],
) -> None:
    """Validate tensor dimensionality.

    Parameters
    ----------
    tensor : Tensor
        The tensor to validate.
    parameter_name : str
        Name of the parameter (for error messages).
    expected_ndim : int | Sequence[int]
        Expected number of dimensions, or sequence of allowed dimensions.

    Raises
    ------
    InvalidShapeError
        If the tensor has wrong number of dimensions.

    Examples
    --------
    >>> t = torch.randn(2, 1000)
    >>> validate_tensor_ndim(t, "waveform", expected_ndim=2)  # OK
    >>> validate_tensor_ndim(t, "waveform", expected_ndim=[1, 2, 3])  # OK

    """
    expected_ndims = (expected_ndim,) if isinstance(expected_ndim, int) else tuple(expected_ndim)

    if tensor.ndim not in expected_ndims:
        if len(expected_ndims) == 1:
            suggestion = f"Expected {expected_ndims[0]}D tensor"
        else:
            suggestion = f"Expected tensor with {expected_ndims} dimensions"

        raise InvalidShapeError(
            parameter_name=parameter_name,
            actual_shape=tuple(tensor.shape),
            expected_ndim=expected_ndims[0] if len(expected_ndims) == 1 else None,
            suggestion=suggestion,
        )


def validate_audio_tensor(
    tensor: Tensor,
    parameter_name: str = "waveform",
    *,
    allow_mono: bool = True,
    min_channels: int | None = None,
    max_channels: int | None = None,
    min_samples: int | None = None,
) -> None:
    """Validate an audio tensor has correct shape.

    Audio tensors in TorchFX should have shape:
    - [T] for mono (if allow_mono=True)
    - [C, T] for multi-channel
    - [B, C, T] for batched multi-channel

    Parameters
    ----------
    tensor : Tensor
        The audio tensor to validate.
    parameter_name : str, optional
        Name of the parameter. Default is "waveform".
    allow_mono : bool, optional
        If True, allow 1D mono tensors. Default is True.
    min_channels : int | None, optional
        Minimum number of channels required.
    max_channels : int | None, optional
        Maximum number of channels allowed.
    min_samples : int | None, optional
        Minimum number of samples required.

    Raises
    ------
    InvalidShapeError
        If the tensor shape is invalid for audio.

    Examples
    --------
    >>> mono = torch.randn(1000)
    >>> stereo = torch.randn(2, 1000)
    >>> validate_audio_tensor(mono)  # OK
    >>> validate_audio_tensor(stereo)  # OK
    >>> validate_audio_tensor(stereo, min_channels=2)  # OK

    """
    valid_ndims = [2, 3]
    if allow_mono:
        valid_ndims.insert(0, 1)

    if tensor.ndim not in valid_ndims:
        suggestion = "Audio tensors should have shape [T], [C, T], or [B, C, T]"
        if not allow_mono:
            suggestion = "Audio tensors should have shape [C, T] or [B, C, T]"
        raise InvalidShapeError(
            parameter_name=parameter_name,
            actual_shape=tuple(tensor.shape),
            suggestion=suggestion,
        )

    # Get channel and sample dimensions based on ndim
    if tensor.ndim == 1:
        num_samples = tensor.shape[0]
        num_channels = 1
    elif tensor.ndim == 2:
        num_channels, num_samples = tensor.shape
    else:  # ndim == 3
        _, num_channels, num_samples = tensor.shape

    # Validate channels
    if min_channels is not None and num_channels < min_channels:
        raise InvalidShapeError(
            parameter_name=parameter_name,
            actual_shape=tuple(tensor.shape),
            suggestion=f"Expected at least {min_channels} channels, got {num_channels}",
        )

    if max_channels is not None and num_channels > max_channels:
        raise InvalidShapeError(
            parameter_name=parameter_name,
            actual_shape=tuple(tensor.shape),
            suggestion=f"Expected at most {max_channels} channels, got {num_channels}",
        )

    # Validate samples
    if min_samples is not None and num_samples < min_samples:
        raise InvalidShapeError(
            parameter_name=parameter_name,
            actual_shape=tuple(tensor.shape),
            suggestion=f"Expected at least {min_samples} samples, got {num_samples}",
        )


# =============================================================================
# Type Validation
# =============================================================================


def validate_type(
    value: Any,
    parameter_name: str,
    expected_types: type | tuple[type, ...],
) -> None:
    """Validate that a value has the expected type.

    Parameters
    ----------
    value : Any
        The value to validate.
    parameter_name : str
        Name of the parameter (for error messages).
    expected_types : type | tuple[type, ...]
        Expected type or tuple of expected types.

    Raises
    ------
    InvalidTypeError
        If the value is not of the expected type.

    Examples
    --------
    >>> validate_type(1.0, "gain", (int, float))  # OK
    >>> validate_type("hello", "gain", (int, float))  # Raises InvalidTypeError

    """
    if isinstance(expected_types, type):
        expected_types = (expected_types,)

    if not isinstance(value, expected_types):
        raise InvalidTypeError(
            parameter_name=parameter_name,
            actual_type=type(value),
            expected_types=expected_types,
        )


# =============================================================================
# Audio-Specific Validators
# =============================================================================


def validate_cutoff_frequency(
    cutoff: float,
    fs: int | None,
    parameter_name: str = "cutoff",
) -> None:
    """Validate a filter cutoff frequency.

    The cutoff must be positive and below the Nyquist frequency (fs/2).

    Parameters
    ----------
    cutoff : float
        Cutoff frequency in Hz.
    fs : int | None
        Sample rate in Hz. If None, only validates that cutoff is positive.
    parameter_name : str, optional
        Parameter name for error messages. Default is "cutoff".

    Raises
    ------
    InvalidRangeError
        If cutoff is invalid.

    Examples
    --------
    >>> validate_cutoff_frequency(1000, 44100)  # OK
    >>> validate_cutoff_frequency(30000, 44100)  # Raises (above Nyquist)

    """
    validate_positive(cutoff, parameter_name)

    if fs is not None:
        nyquist = fs / 2
        if cutoff >= nyquist:
            raise InvalidRangeError(
                parameter_name=parameter_name,
                actual_value=cutoff,
                min_value=0,
                max_value=nyquist,
                min_inclusive=False,
                max_inclusive=False,
            )


def validate_filter_order(
    order: int,
    parameter_name: str = "order",
    *,
    min_order: int = 1,
    max_order: int | None = None,
) -> None:
    """Validate a filter order parameter.

    Parameters
    ----------
    order : int
        Filter order to validate.
    parameter_name : str, optional
        Parameter name for error messages. Default is "order".
    min_order : int, optional
        Minimum allowed order. Default is 1.
    max_order : int | None, optional
        Maximum allowed order. None means no maximum.

    Raises
    ------
    InvalidRangeError
        If order is invalid.
    InvalidTypeError
        If order is not an integer.

    Examples
    --------
    >>> validate_filter_order(4)  # OK
    >>> validate_filter_order(0)  # Raises InvalidRangeError

    """
    validate_type(order, parameter_name, int)
    validate_range(
        order,
        parameter_name,
        min_value=min_order,
        max_value=max_order,
    )


def validate_q_factor(
    q: float,
    parameter_name: str = "q",
    *,
    min_q: float = 0.001,
    max_q: float | None = None,
) -> None:
    """Validate a Q factor (quality factor) parameter.

    Parameters
    ----------
    q : float
        Q factor to validate.
    parameter_name : str, optional
        Parameter name for error messages. Default is "q".
    min_q : float, optional
        Minimum allowed Q. Default is 0.001.
    max_q : float | None, optional
        Maximum allowed Q. None means no maximum.

    Raises
    ------
    InvalidRangeError
        If Q is invalid.

    Examples
    --------
    >>> validate_q_factor(0.707)  # OK
    >>> validate_q_factor(0)  # Raises InvalidRangeError

    """
    validate_positive(q, parameter_name)
    if min_q is not None or max_q is not None:
        validate_range(q, parameter_name, min_value=min_q, max_value=max_q)
