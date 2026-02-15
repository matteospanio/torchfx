"""Abstract audio backend interface for real-time I/O.

This module defines the abstract base class for audio backends and
the configuration types used across all backend implementations.

Classes
-------
StreamDirection
    Audio stream direction enum.
StreamState
    Audio stream lifecycle state enum.
StreamConfig
    Configuration dataclass for audio streams.
AudioBackend
    Abstract base class for audio I/O backends.

Examples
--------
>>> from torchfx.realtime.backend import StreamConfig, StreamDirection
>>> config = StreamConfig(sample_rate=48000, buffer_size=512, channels_in=2, channels_out=2)
>>> config.direction
<StreamDirection.DUPLEX: 'duplex'>
>>> config.latency_ms
10.666666666666666

"""

from __future__ import annotations

import abc
import enum
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from torch import Tensor


class StreamDirection(enum.Enum):
    """Audio stream direction."""

    INPUT = "input"
    OUTPUT = "output"
    DUPLEX = "duplex"


class StreamState(enum.Enum):
    """Audio stream lifecycle state."""

    CLOSED = "closed"
    OPEN = "open"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


# Type alias for the audio callback function.
#
# Signature: ``callback(input_data, output_data, frame_count) -> None``
#
# - ``input_data``: shape ``(channels_in, buffer_size)`` or empty tensor
# - ``output_data``: shape ``(channels_out, buffer_size)`` -- write processed audio here
# - ``frame_count``: number of frames in this callback
AudioCallback = Callable[[Tensor, Tensor, int], None]


@dataclass(frozen=True)
class StreamConfig:
    """Configuration for an audio stream.

    Parameters
    ----------
    sample_rate : int
        Sample rate in Hz (e.g., 44100, 48000). Default is 48000.
    buffer_size : int
        Number of samples per buffer/callback frame. Default is 512.
    channels_in : int
        Number of input channels (0 for output-only). Default is 0.
    channels_out : int
        Number of output channels (0 for input-only). Default is 2.
    dtype : str
        Sample format string. Default is ``"float32"``.
    device_in : int | str | None
        Input device identifier. None for system default.
    device_out : int | str | None
        Output device identifier. None for system default.
    latency : str | float
        Latency hint: ``"low"``, ``"high"``, or seconds as float.
        Default is ``"low"``.

    Examples
    --------
    >>> config = StreamConfig(sample_rate=44100, buffer_size=256)
    >>> config.direction
    <StreamDirection.OUTPUT: 'output'>
    >>> config.latency_ms
    5.804988662131519

    """

    sample_rate: int = 48000
    buffer_size: int = 512
    channels_in: int = 0
    channels_out: int = 2
    dtype: str = "float32"
    device_in: int | str | None = None
    device_out: int | str | None = None
    latency: str | float = "low"

    @property
    def direction(self) -> StreamDirection:
        """Infer stream direction from channel counts.

        Returns
        -------
        StreamDirection
            DUPLEX if both in and out channels > 0,
            INPUT if only in > 0, OUTPUT otherwise.

        """
        if self.channels_in > 0 and self.channels_out > 0:
            return StreamDirection.DUPLEX
        elif self.channels_in > 0:
            return StreamDirection.INPUT
        return StreamDirection.OUTPUT

    @property
    def latency_ms(self) -> float:
        """Calculate theoretical minimum latency in milliseconds.

        Returns
        -------
        float
            Latency based on buffer_size / sample_rate.

        """
        return (self.buffer_size / self.sample_rate) * 1000.0


class AudioBackend(abc.ABC):
    """Abstract base class for audio I/O backends.

    All audio backends must implement this interface. The backend manages
    the lifecycle of audio streams and routes audio data through callbacks.

    Subclasses should implement all abstract methods to provide a complete
    audio I/O solution.

    Examples
    --------
    Implementing a custom backend:

    >>> class MyBackend(AudioBackend):
    ...     # Implement all abstract methods
    ...     pass

    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable name of the backend."""
        ...

    @property
    @abc.abstractmethod
    def is_available(self) -> bool:
        """Whether this backend's dependencies are available."""
        ...

    @abc.abstractmethod
    def get_devices(self) -> list[dict[str, Any]]:
        """List available audio devices.

        Returns
        -------
        list[dict[str, Any]]
            List of device info dicts with keys: ``"name"``, ``"index"``,
            ``"max_input_channels"``, ``"max_output_channels"``,
            ``"default_sample_rate"``.

        """
        ...

    @abc.abstractmethod
    def get_default_device(self, direction: StreamDirection) -> int | str:
        """Get the default device for a given direction.

        Parameters
        ----------
        direction : StreamDirection
            The stream direction to query.

        Returns
        -------
        int | str
            Device identifier.

        """
        ...

    @abc.abstractmethod
    def open_stream(
        self,
        config: StreamConfig,
        callback: AudioCallback | None = None,
    ) -> None:
        """Open an audio stream with the given configuration.

        Parameters
        ----------
        config : StreamConfig
            Stream configuration.
        callback : AudioCallback | None
            Callback for real-time processing. If None, use blocking API.

        """
        ...

    @abc.abstractmethod
    def start(self) -> None:
        """Start the audio stream."""
        ...

    @abc.abstractmethod
    def stop(self) -> None:
        """Stop the audio stream."""
        ...

    @abc.abstractmethod
    def close(self) -> None:
        """Close the audio stream and release resources."""
        ...

    @property
    @abc.abstractmethod
    def state(self) -> StreamState:
        """Current stream state."""
        ...

    @abc.abstractmethod
    def read(self, num_frames: int) -> Tensor:
        """Read audio frames (blocking API).

        Parameters
        ----------
        num_frames : int
            Number of frames to read.

        Returns
        -------
        Tensor
            Shape ``(channels_in, num_frames)``.

        """
        ...

    @abc.abstractmethod
    def write(self, data: Tensor) -> None:
        """Write audio frames (blocking API).

        Parameters
        ----------
        data : Tensor
            Shape ``(channels_out, num_frames)``.

        """
        ...
