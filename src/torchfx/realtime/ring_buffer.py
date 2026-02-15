"""Lock-free SPSC ring buffer for real-time audio tensor transfer.

This module provides a single-producer single-consumer (SPSC) ring buffer
backed by pre-allocated PyTorch tensors. It is designed for real-time audio
processing where an audio callback thread produces data and a processing
thread consumes it.

Classes
-------
TensorRingBuffer
    Lock-free SPSC ring buffer operating on PyTorch tensors.

Examples
--------
>>> import torch
>>> from torchfx.realtime.ring_buffer import TensorRingBuffer
>>> buf = TensorRingBuffer(capacity=1024, channels=2)
>>> data = torch.randn(2, 256)
>>> written = buf.write(data)
>>> output = buf.read(256)

"""

from __future__ import annotations

import math

import torch
from torch import Tensor


class TensorRingBuffer:
    """Lock-free SPSC ring buffer for real-time audio tensor transfer.

    Uses separate read/write indices with a pre-allocated tensor backing
    store. In the SPSC model, only the producer writes ``_write_idx`` and
    only the consumer writes ``_read_idx``, so no locks are needed.

    The capacity must be a power of 2 for efficient modular arithmetic
    (bitwise AND instead of modulo). If a non-power-of-2 value is provided,
    it is rounded up to the next power of 2.

    Parameters
    ----------
    capacity : int
        Total capacity in samples per channel. Rounded up to next power of 2.
    channels : int
        Number of audio channels. Default is 1.
    dtype : torch.dtype
        Data type for the buffer tensor. Default is ``torch.float32``.
    device : str
        Device for the buffer tensor. Default is ``"cpu"``.

    Attributes
    ----------
    buffer : Tensor
        Pre-allocated backing tensor of shape ``(channels, capacity)``.

    Examples
    --------
    >>> buf = TensorRingBuffer(capacity=512, channels=2)
    >>> buf.capacity
    512
    >>> buf.available_read
    0
    >>> buf.available_write
    512

    """

    def __init__(
        self,
        capacity: int,
        channels: int = 1,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")
        if channels <= 0:
            raise ValueError(f"Channels must be positive, got {channels}")

        # Round up to next power of 2
        self._capacity = 1 << math.ceil(math.log2(capacity)) if capacity > 0 else 1
        self._channels = channels
        self._mask = self._capacity - 1
        self.buffer = torch.zeros(channels, self._capacity, dtype=dtype, device=device)
        self._write_idx = 0
        self._read_idx = 0

    @property
    def capacity(self) -> int:
        """Total buffer capacity in samples per channel."""
        return self._capacity

    @property
    def channels(self) -> int:
        """Number of audio channels."""
        return self._channels

    @property
    def available_read(self) -> int:
        """Number of samples available for reading."""
        return self._write_idx - self._read_idx

    @property
    def available_write(self) -> int:
        """Number of samples available for writing."""
        return self._capacity - self.available_read

    def write(self, data: Tensor) -> int:
        """Write samples into the buffer.

        Parameters
        ----------
        data : Tensor
            Audio data of shape ``(channels, samples)`` or ``(samples,)`` for
            single-channel buffers.

        Returns
        -------
        int
            Number of samples actually written. May be less than requested
            if the buffer is near full.

        Examples
        --------
        >>> buf = TensorRingBuffer(capacity=256, channels=1)
        >>> data = torch.ones(1, 100)
        >>> buf.write(data)
        100
        >>> buf.available_read
        100

        """
        if data.ndim == 1:
            data = data.unsqueeze(0)

        if data.shape[0] != self._channels:
            raise ValueError(
                f"Channel mismatch: buffer has {self._channels} channels, "
                f"data has {data.shape[0]} channels"
            )

        num_samples = data.shape[1]
        space = self.available_write
        to_write = min(num_samples, space)

        if to_write == 0:
            return 0

        write_pos = self._write_idx & self._mask
        first_chunk = min(to_write, self._capacity - write_pos)
        second_chunk = to_write - first_chunk

        self.buffer[:, write_pos : write_pos + first_chunk] = data[:, :first_chunk]
        if second_chunk > 0:
            self.buffer[:, :second_chunk] = data[:, first_chunk : first_chunk + second_chunk]

        self._write_idx += to_write
        return to_write

    def read(self, num_samples: int) -> Tensor:
        """Read samples from the buffer.

        Parameters
        ----------
        num_samples : int
            Number of samples to read.

        Returns
        -------
        Tensor
            Audio data of shape ``(channels, num_samples)``.

        Raises
        ------
        BufferUnderrunError
            If fewer samples are available than requested.

        Examples
        --------
        >>> buf = TensorRingBuffer(capacity=256, channels=1)
        >>> _ = buf.write(torch.ones(1, 100))
        >>> output = buf.read(50)
        >>> output.shape
        torch.Size([1, 50])
        >>> buf.available_read
        50

        """
        available = self.available_read
        if num_samples > available:
            from torchfx.realtime.exceptions import BufferUnderrunError

            raise BufferUnderrunError(
                f"Requested {num_samples} samples but only {available} available",
                suggestion="Reduce read size or wait for more data",
            )

        read_pos = self._read_idx & self._mask
        first_chunk = min(num_samples, self._capacity - read_pos)
        second_chunk = num_samples - first_chunk

        result = torch.empty(
            self._channels, num_samples, dtype=self.buffer.dtype, device=self.buffer.device
        )
        result[:, :first_chunk] = self.buffer[:, read_pos : read_pos + first_chunk]
        if second_chunk > 0:
            result[:, first_chunk:] = self.buffer[:, :second_chunk]

        self._read_idx += num_samples
        return result

    def peek(self, num_samples: int) -> Tensor:
        """Read samples without advancing the read pointer.

        Useful for overlap-add processing where the consumer needs
        to read overlapping frames.

        Parameters
        ----------
        num_samples : int
            Number of samples to peek.

        Returns
        -------
        Tensor
            Audio data of shape ``(channels, num_samples)``.

        Raises
        ------
        BufferUnderrunError
            If fewer samples are available than requested.

        """
        available = self.available_read
        if num_samples > available:
            from torchfx.realtime.exceptions import BufferUnderrunError

            raise BufferUnderrunError(
                f"Requested {num_samples} samples but only {available} available",
                suggestion="Reduce peek size or wait for more data",
            )

        read_pos = self._read_idx & self._mask
        first_chunk = min(num_samples, self._capacity - read_pos)
        second_chunk = num_samples - first_chunk

        result = torch.empty(
            self._channels, num_samples, dtype=self.buffer.dtype, device=self.buffer.device
        )
        result[:, :first_chunk] = self.buffer[:, read_pos : read_pos + first_chunk]
        if second_chunk > 0:
            result[:, first_chunk:] = self.buffer[:, :second_chunk]

        return result

    def advance_read(self, num_samples: int) -> None:
        """Advance the read pointer without reading data.

        Use after ``peek()`` to advance by the hop size in
        overlap-add processing.

        Parameters
        ----------
        num_samples : int
            Number of samples to advance.

        Raises
        ------
        BufferUnderrunError
            If advancing would go past the write pointer.

        """
        if num_samples > self.available_read:
            from torchfx.realtime.exceptions import BufferUnderrunError

            raise BufferUnderrunError(
                f"Cannot advance {num_samples} samples, only {self.available_read} available",
            )
        self._read_idx += num_samples

    def clear(self) -> None:
        """Reset the buffer to empty state.

        Examples
        --------
        >>> buf = TensorRingBuffer(capacity=256, channels=1)
        >>> _ = buf.write(torch.ones(1, 100))
        >>> buf.clear()
        >>> buf.available_read
        0

        """
        self._write_idx = 0
        self._read_idx = 0
        self.buffer.zero_()
