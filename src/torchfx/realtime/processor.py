"""Real-time audio processor for TorchFX.

This module provides the ``RealtimeProcessor`` class that orchestrates
audio I/O through a backend, buffering through a ring buffer, and
processing through a chain of FX effects.

Classes
-------
RealtimeProcessor
    Real-time audio processor connecting backend to effect chain.

Examples
--------
>>> from torchfx.realtime import RealtimeProcessor, StreamConfig
>>> from torchfx.effect import Gain
>>> config = StreamConfig(sample_rate=48000, buffer_size=512,
...                       channels_in=2, channels_out=2)
>>> # processor = RealtimeProcessor(
>>> #     effects=[Gain(0.5)],
>>> #     backend=SoundDeviceBackend(),
>>> #     config=config,
>>> # )

"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Sequence
from typing import Any, cast

import torch
from torch import Tensor, nn

from torchfx.effect import FX
from torchfx.filter.__base import AbstractFilter
from torchfx.logging import get_logger
from torchfx.realtime.backend import AudioBackend, StreamConfig
from torchfx.realtime.exceptions import RealtimeError
from torchfx.realtime.ring_buffer import TensorRingBuffer
from torchfx.validation import validate_positive, validate_sample_rate

_logger = get_logger("realtime.processor")


class RealtimeProcessor:
    """Real-time audio processor connecting an audio backend to an effect chain.

    Orchestrates audio I/O through a backend and processing through
    a chain of FX effects. Supports thread-safe parameter updates
    during processing.

    Parameters
    ----------
    effects : Sequence[FX] | nn.Sequential
        Chain of effects to apply in order.
    backend : AudioBackend
        Audio backend for I/O.
    config : StreamConfig
        Stream configuration.
    buffer_capacity : int
        Ring buffer capacity in samples per channel. Default is 8192.

    Examples
    --------
    >>> from torchfx.realtime import RealtimeProcessor, StreamConfig
    >>> from torchfx.effect import Gain
    >>> config = StreamConfig(sample_rate=48000, buffer_size=512,
    ...                       channels_in=1, channels_out=1)

    """

    def __init__(
        self,
        effects: Sequence[FX] | nn.Sequential,
        backend: AudioBackend,
        config: StreamConfig,
        buffer_capacity: int = 8192,
    ) -> None:
        validate_sample_rate(config.sample_rate)
        validate_positive(config.buffer_size, "buffer_size")

        self._effects: list[FX] = self._normalize_effects(effects)
        self._backend = backend
        self._config = config
        self._running = False

        # Configure effects with sample rate (same pattern as Wave.__or__)
        for effect in self._effects:
            if hasattr(effect, "fs") and effect.fs is None:
                effect.fs = config.sample_rate
            if isinstance(effect, AbstractFilter) and not effect._has_computed_coeff:
                effect.compute_coefficients()

        # Ring buffers for input and output
        ch_in = max(config.channels_in, 1)
        ch_out = max(config.channels_out, 1)
        self._input_buffer = TensorRingBuffer(buffer_capacity, ch_in)
        self._output_buffer = TensorRingBuffer(buffer_capacity, ch_out)

        # Thread-safe parameter updates: double-buffered
        self._pending_params: dict[str, Any] = {}
        self._param_lock = threading.Lock()

    def __enter__(self) -> RealtimeProcessor:
        """Start processing and return self for use as context manager.

        Examples
        --------
        >>> with RealtimeProcessor(effects, backend, config) as processor:
        ...     processor.set_parameter("0.gain", 0.5)
        ...     # processing runs until the block exits

        """
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Stop processing on context exit."""
        if self._running:
            self.stop()

    @staticmethod
    def _normalize_effects(effects: Sequence[FX] | nn.Sequential) -> list[FX]:
        modules: Iterable[FX] = (
            cast(Iterable[FX], effects) if isinstance(effects, nn.Sequential) else effects
        )

        normalized: list[FX] = []
        for effect in modules:
            if not isinstance(effect, FX):
                raise TypeError("All effects must inherit from FX when used in RealtimeProcessor")
            normalized.append(effect)

        return normalized

    def start(self) -> None:
        """Start real-time processing.

        Opens and starts the audio stream. The backend callback
        routes audio through the effect chain.

        Raises
        ------
        RealtimeError
            If the processor is already running.

        """
        if self._running:
            raise RealtimeError(
                "Processor is already running",
                suggestion="Call stop() before starting again",
            )

        _logger.info(
            "Starting real-time processor: %dHz, %d buffer, %din/%dout",
            self._config.sample_rate,
            self._config.buffer_size,
            self._config.channels_in,
            self._config.channels_out,
        )

        self._backend.open_stream(self._config, callback=self._audio_callback)
        self._backend.start()
        self._running = True

    def stop(self) -> None:
        """Stop real-time processing and close the stream.

        Raises
        ------
        RealtimeError
            If the processor is not running.

        """
        if not self._running:
            raise RealtimeError(
                "Processor is not running",
                suggestion="Call start() first",
            )

        _logger.info("Stopping real-time processor")
        self._running = False
        self._backend.stop()
        self._backend.close()

    def set_parameter(self, name: str, value: Any) -> None:
        """Thread-safe parameter update.

        Parameters are staged in a pending dict and applied at the
        next processing boundary (start of next audio callback).
        This avoids locks in the audio processing path.

        Parameters
        ----------
        name : str
            Dot-separated parameter path, e.g., ``"0.cutoff"`` for
            effect index 0, attribute ``cutoff``.
        value : Any
            New parameter value.

        Examples
        --------
        >>> # processor.set_parameter("0.cutoff", 2000)
        >>> # processor.set_parameter("1.gain", 0.8)

        """
        with self._param_lock:
            self._pending_params[name] = value

    def _apply_pending_params(self) -> None:
        """Swap pending parameters into active effects.

        Called at the start of each audio callback (buffer boundary).

        """
        if not self._pending_params:
            return

        with self._param_lock:
            params = self._pending_params.copy()
            self._pending_params.clear()

        for key, value in params.items():
            parts = key.split(".", 1)
            effect_idx = int(parts[0])
            attr_name = parts[1] if len(parts) > 1 else None

            if effect_idx < 0 or effect_idx >= len(self._effects):
                _logger.warning("Invalid effect index: %d", effect_idx)
                continue

            effect = self._effects[effect_idx]
            if attr_name:
                setattr(effect, attr_name, value)
                # Recompute coefficients if it's a filter parameter
                if isinstance(effect, AbstractFilter):
                    effect.compute_coefficients()
                    move_coeff = getattr(effect, "move_coeff", None)
                    if callable(move_coeff):
                        move_coeff(torch.device("cpu"))
                    reset_state = getattr(effect, "reset_state", None)
                    if callable(reset_state):
                        reset_state()
            else:
                _logger.warning("No attribute specified for effect %d", effect_idx)

    @torch.no_grad()
    def _audio_callback(
        self,
        input_data: Tensor,
        output_data: Tensor,
        frame_count: int,  # noqa: ARG002
    ) -> None:
        """Audio callback invoked by the backend for each buffer.

        1. Apply pending parameter updates
        2. Process input through effect chain
        3. Write processed audio to output tensor

        Parameters
        ----------
        input_data : Tensor
            Input audio of shape ``(channels_in, buffer_size)``.
        output_data : Tensor
            Output tensor to write into, shape ``(channels_out, buffer_size)``.
        frame_count : int
            Number of frames in this callback.

        """
        self._apply_pending_params()

        # Process through effect chain (all effects validated as FX/nn.Module at init)
        x = input_data
        for effect in self._effects:
            x = effect(x)

        # Write processed audio to output
        if output_data.numel() > 0:
            # Handle potential channel mismatch
            if x.shape[0] != output_data.shape[0]:
                # Simple mono-to-stereo or truncation
                if x.shape[0] == 1 and output_data.shape[0] > 1:
                    x = x.expand(output_data.shape[0], -1)
                else:
                    x = x[: output_data.shape[0]]
            output_data.copy_(x)

    def reset_state(self) -> None:
        """Reset all internal state (filter states, ring buffers).

        Useful after seeking in a file or switching audio sources.

        """
        self._input_buffer.clear()
        self._output_buffer.clear()
        for effect in self._effects:
            reset_state = getattr(effect, "reset_state", None)
            if callable(reset_state):
                reset_state()

    @property
    def latency_ms(self) -> float:
        """Estimated total latency in milliseconds."""
        return self._config.latency_ms

    @property
    def is_running(self) -> bool:
        """Whether the processor is currently running."""
        return self._running

    @property
    def effects(self) -> list[FX]:
        """The current effect chain."""
        return self._effects

    @property
    def config(self) -> StreamConfig:
        """The stream configuration."""
        return self._config
