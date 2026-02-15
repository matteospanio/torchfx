"""Real-time audio processing exceptions for TorchFX.

This module provides exception classes specific to real-time audio
processing, extending the TorchFX exception hierarchy.

Exception Hierarchy
-------------------
RealtimeError (TorchFXError)
    Base exception for all real-time processing errors.
BackendNotAvailableError (RealtimeError)
    Raised when a requested audio backend is not installed.
StreamError (RealtimeError)
    Raised for audio stream lifecycle errors.
BufferOverrunError (AudioProcessingError)
    Raised when the ring buffer overflows.
BufferUnderrunError (AudioProcessingError)
    Raised when the ring buffer underflows.

Examples
--------
Catch all real-time errors:

>>> from torchfx.realtime.exceptions import RealtimeError
>>> try:
...     pass  # real-time operations
... except RealtimeError as e:
...     print(f"Real-time error: {e}")

"""

from __future__ import annotations

from torchfx.validation.exceptions import AudioProcessingError, TorchFXError


class RealtimeError(TorchFXError):
    """Base exception for all real-time processing errors.

    Parameters
    ----------
    message : str
        Human-readable error message.
    suggestion : str | None, optional
        A suggestion for fixing the error.

    Examples
    --------
    >>> raise RealtimeError("Processing failed")
    Traceback (most recent call last):
        ...
    torchfx.realtime.exceptions.RealtimeError: Processing failed

    """

    pass


class BackendNotAvailableError(RealtimeError):
    """Raised when a requested audio backend is not installed or available.

    Parameters
    ----------
    backend_name : str
        Name of the backend that is not available.
    suggestion : str | None, optional
        Installation instructions or alternative.

    Examples
    --------
    >>> raise BackendNotAvailableError("sounddevice")
    Traceback (most recent call last):
        ...
    torchfx.realtime.exceptions.BackendNotAvailableError: ...

    """

    def __init__(
        self,
        backend_name: str,
        suggestion: str | None = None,
    ) -> None:
        super().__init__(
            message=f"Audio backend '{backend_name}' is not available",
            suggestion=suggestion or f"Install with: pip install {backend_name}",
        )
        self.backend_name = backend_name


class StreamError(RealtimeError):
    """Raised for audio stream lifecycle errors.

    Parameters
    ----------
    message : str
        Human-readable error message.
    suggestion : str | None, optional
        A suggestion for fixing the error.

    Examples
    --------
    >>> raise StreamError("Failed to open audio stream")
    Traceback (most recent call last):
        ...
    torchfx.realtime.exceptions.StreamError: Failed to open audio stream

    """

    pass


class BufferOverrunError(AudioProcessingError):
    """Raised when the ring buffer overflows.

    This occurs when the producer writes data faster than the consumer
    reads it, causing data loss.

    Parameters
    ----------
    message : str
        Human-readable error message.
    suggestion : str | None, optional
        A suggestion for fixing the error.

    Examples
    --------
    >>> raise BufferOverrunError("Ring buffer overflow")
    Traceback (most recent call last):
        ...
    torchfx.realtime.exceptions.BufferOverrunError: Ring buffer overflow

    """

    pass


class BufferUnderrunError(AudioProcessingError):
    """Raised when the ring buffer underflows.

    This occurs when the consumer attempts to read more data than
    is available, typically due to processing being faster than input.

    Parameters
    ----------
    message : str
        Human-readable error message.
    suggestion : str | None, optional
        A suggestion for fixing the error.

    Examples
    --------
    >>> raise BufferUnderrunError("Ring buffer underrun")
    Traceback (most recent call last):
        ...
    torchfx.realtime.exceptions.BufferUnderrunError: Ring buffer underrun

    """

    pass
