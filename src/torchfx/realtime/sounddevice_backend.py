"""PortAudio audio backend via sounddevice.

This module provides a concrete audio backend implementation using the
sounddevice library, which wraps PortAudio for cross-platform audio I/O.

Classes
-------
SoundDeviceBackend
    Audio backend using sounddevice (PortAudio).

Notes
-----
The ``sounddevice`` library is an optional dependency. Install it with:

.. code-block:: bash

    pip install sounddevice

Examples
--------
>>> from torchfx.realtime import SoundDeviceBackend, StreamConfig
>>> backend = SoundDeviceBackend()  # doctest: +SKIP
>>> config = StreamConfig(sample_rate=48000, buffer_size=512, channels_out=2)
>>> backend.open_stream(config)  # doctest: +SKIP
>>> backend.start()  # doctest: +SKIP

"""

from __future__ import annotations

from typing import Any, cast

import torch
from numpy.typing import NDArray
from torch import Tensor

from torchfx.realtime._compat import get_sounddevice
from torchfx.realtime.backend import (
    AudioBackend,
    AudioCallback,
    StreamConfig,
    StreamDirection,
    StreamState,
)
from torchfx.realtime.exceptions import StreamError


class SoundDeviceBackend(AudioBackend):
    """Audio backend using sounddevice (PortAudio).

    Provides real-time audio I/O through PortAudio with support for
    input, output, and duplex streams. Supports both callback-based
    and blocking APIs.

    The sounddevice module is imported lazily on instantiation.

    Raises
    ------
    BackendNotAvailableError
        If sounddevice is not installed.

    Examples
    --------
    >>> backend = SoundDeviceBackend()  # doctest: +SKIP
    >>> devices = backend.get_devices()  # doctest: +SKIP

    """

    def __init__(self) -> None:
        self._sd = get_sounddevice()
        self._stream: Any = None
        self._state: StreamState = StreamState.CLOSED
        self._config: StreamConfig | None = None
        self._callback: AudioCallback | None = None

    @property
    def name(self) -> str:
        """Return backend name."""
        return "sounddevice (PortAudio)"

    @property
    def is_available(self) -> bool:
        """Return True since instantiation succeeded."""
        return True

    def get_devices(self) -> list[dict[str, Any]]:
        """List available audio devices.

        Returns
        -------
        list[dict[str, Any]]
            Device information dictionaries.

        """
        devices = self._sd.query_devices()
        result: list[dict[str, Any]] = []
        if isinstance(devices, dict):
            devices = [devices]
        for i, dev in enumerate(devices):
            result.append(
                {
                    "name": dev["name"],
                    "index": i,
                    "max_input_channels": dev["max_input_channels"],
                    "max_output_channels": dev["max_output_channels"],
                    "default_sample_rate": dev["default_samplerate"],
                }
            )
        return result

    def get_default_device(self, direction: StreamDirection) -> int | str:
        """Get the default device for a given direction.

        Parameters
        ----------
        direction : StreamDirection
            The stream direction.

        Returns
        -------
        int | str
            Default device index or name.

        """
        default = self._sd.default.device
        if direction == StreamDirection.INPUT:
            return cast(int | str, default[0] if isinstance(default, (list, tuple)) else default)
        elif direction == StreamDirection.OUTPUT:
            return cast(int | str, default[1] if isinstance(default, (list, tuple)) else default)
        # DUPLEX: return input device
        return cast(int | str, default[0] if isinstance(default, (list, tuple)) else default)

    def open_stream(
        self,
        config: StreamConfig,
        callback: AudioCallback | None = None,
    ) -> None:
        """Open an audio stream.

        Parameters
        ----------
        config : StreamConfig
            Stream configuration.
        callback : AudioCallback | None
            Audio processing callback. If None, blocking API is used.

        Raises
        ------
        StreamError
            If stream is already open or opening fails.

        """
        if self._state not in (StreamState.CLOSED, StreamState.STOPPED):
            raise StreamError(
                "Cannot open stream: stream is already open or running",
                suggestion="Close the existing stream first",
            )

        self._config = config
        self._callback = callback

        sd_callback = self._make_sd_callback(callback) if callback else None

        dtype_map = {
            "float32": "float32",
            "int16": "int16",
            "int24": "int24",
            "int32": "int32",
        }
        sd_dtype = dtype_map.get(config.dtype, "float32")

        try:
            direction = config.direction
            if direction == StreamDirection.DUPLEX:
                self._stream = self._sd.Stream(
                    samplerate=config.sample_rate,
                    blocksize=config.buffer_size,
                    device=(config.device_in, config.device_out),
                    channels=(config.channels_in, config.channels_out),
                    dtype=sd_dtype,
                    latency=config.latency,
                    callback=sd_callback,
                )
            elif direction == StreamDirection.INPUT:
                self._stream = self._sd.InputStream(
                    samplerate=config.sample_rate,
                    blocksize=config.buffer_size,
                    device=config.device_in,
                    channels=config.channels_in,
                    dtype=sd_dtype,
                    latency=config.latency,
                    callback=sd_callback,
                )
            else:
                self._stream = self._sd.OutputStream(
                    samplerate=config.sample_rate,
                    blocksize=config.buffer_size,
                    device=config.device_out,
                    channels=config.channels_out,
                    dtype=sd_dtype,
                    latency=config.latency,
                    callback=sd_callback,
                )
            self._state = StreamState.OPEN
        except Exception as e:
            self._state = StreamState.ERROR
            raise StreamError(
                f"Failed to open audio stream: {e}",
                suggestion="Check audio device availability and configuration",
            ) from e

    def _make_sd_callback(self, callback: AudioCallback) -> Any:
        """Create a sounddevice-compatible callback wrapper.

        The wrapper converts numpy arrays to/from PyTorch tensors,
        transposing between sounddevice's ``(frames, channels)`` format
        and torchfx's ``(channels, frames)`` format.

        Parameters
        ----------
        callback : AudioCallback
            The user's audio callback.

        Returns
        -------
        callable
            A sounddevice-compatible callback function.

        """
        config = self._config
        assert config is not None

        def sd_callback(
            indata: NDArray[Any] | None,
            outdata: NDArray[Any] | None,
            frames: int,
            time: Any,  # noqa: ARG001
            status: Any,
        ) -> None:
            if status:
                from torchfx.logging import get_logger

                logger = get_logger("realtime.sounddevice")
                logger.warning("sounddevice status: %s", status)

            # Convert input: numpy (frames, channels) -> tensor (channels, frames)
            if indata is not None:
                input_tensor = torch.from_numpy(indata.T.copy())
            else:
                input_tensor = torch.empty(0)

            # Create output tensor
            if outdata is not None:
                output_tensor = torch.zeros(outdata.shape[1], outdata.shape[0], dtype=torch.float32)
            else:
                output_tensor = torch.empty(0)

            callback(input_tensor, output_tensor, frames)

            # Copy output back: tensor (channels, frames) -> numpy (frames, channels)
            if outdata is not None:
                outdata[:] = output_tensor.numpy().T

        return sd_callback

    def start(self) -> None:
        """Start the audio stream.

        Raises
        ------
        StreamError
            If stream is not open.

        """
        if self._stream is None or self._state not in (StreamState.OPEN, StreamState.STOPPED):
            raise StreamError(
                "Cannot start stream: stream is not open",
                suggestion="Call open_stream() first",
            )
        self._stream.start()
        self._state = StreamState.RUNNING

    def stop(self) -> None:
        """Stop the audio stream.

        Raises
        ------
        StreamError
            If stream is not running.

        """
        if self._stream is None or self._state != StreamState.RUNNING:
            raise StreamError(
                "Cannot stop stream: stream is not running",
                suggestion="Call start() first",
            )
        self._stream.stop()
        self._state = StreamState.STOPPED

    def close(self) -> None:
        """Close the audio stream and release resources."""
        if self._stream is not None:
            try:
                if self._state == StreamState.RUNNING:
                    self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self._state = StreamState.CLOSED

    @property
    def state(self) -> StreamState:
        """Current stream state."""
        return self._state

    def read(self, num_frames: int) -> Tensor:
        """Read audio frames using blocking API.

        Parameters
        ----------
        num_frames : int
            Number of frames to read.

        Returns
        -------
        Tensor
            Shape ``(channels_in, num_frames)``.

        Raises
        ------
        StreamError
            If stream is not configured for input.

        """
        if self._config is None:
            raise StreamError("Stream not configured", suggestion="Call open_stream() first")
        if self._config.channels_in == 0:
            raise StreamError(
                "Cannot read from output-only stream",
                suggestion="Configure channels_in > 0",
            )

        data, _overflowed = self._sd.rec(
            frames=num_frames,
            samplerate=self._config.sample_rate,
            channels=self._config.channels_in,
            dtype=self._config.dtype,
            blocking=True,
        )
        return torch.from_numpy(data.T.copy())

    def write(self, data: Tensor) -> None:
        """Write audio frames using blocking API.

        Parameters
        ----------
        data : Tensor
            Shape ``(channels_out, num_frames)``.

        Raises
        ------
        StreamError
            If stream is not configured for output.

        """
        if self._config is None:
            raise StreamError("Stream not configured", suggestion="Call open_stream() first")
        if self._config.channels_out == 0:
            raise StreamError(
                "Cannot write to input-only stream",
                suggestion="Configure channels_out > 0",
            )

        # Convert tensor (channels, frames) -> numpy (frames, channels)
        np_data = data.numpy().T
        self._sd.play(
            np_data,
            samplerate=self._config.sample_rate,
            blocking=True,
        )
