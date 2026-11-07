"""Real-time audio processor for TorchFX.

This module provides the ``RealtimeProcessor`` class that orchestrates
audio I/O through a backend and processing through a chain of FX
effects.

Architecture
------------
The realtime path is split into two threads to keep DSP work out of the
audio callback's deadline-critical context:

* **Audio I/O thread** (driven by the backend, e.g. PortAudio). The
  callback only moves samples between the backend's interleaved buffers
  and the processor's two ring buffers; no effect processing happens
  here. The callback records its own wall-clock duration into
  ``latency_log`` and increments ``xrun_count`` whenever the backend
  reports an xrun or the output ring underflows.
* **DSP worker thread**. Reads input chunks from the input ring buffer,
  runs the effect chain, writes the result into the output ring buffer.
  Wakes on a ``threading.Event`` signalled by the audio callback or runs
  on a short timeout so it can be joined cleanly on stop.

The output ring is *primed* with one chunk of silence at start so the
first callback always finds data ready. This costs one extra block of
latency in exchange for zero startup xruns.

Streaming filter state is preserved across chunks because effect
instances are long-lived; the DSP worker calls them once per block in
order.

Classes
-------
RealtimeProcessor
    Real-time audio processor with worker-thread DSP and instrumented
    callback path.

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

import collections
import math
import threading
import time
from collections.abc import Iterable, Sequence
from typing import Any, cast

import torch
from torch import Tensor, nn

from torchfx.effect import FX
from torchfx.filter.__base import AbstractFilter
from torchfx.logging import get_logger
from torchfx.realtime.backend import AudioBackend, BackendStatus, StreamConfig
from torchfx.realtime.exceptions import RealtimeError
from torchfx.realtime.ring_buffer import TensorRingBuffer
from torchfx.validation import validate_positive, validate_sample_rate

_logger = get_logger("realtime.processor")

# Default latency log size — enough for ~10 minutes at 512-sample /
# 48 kHz callbacks (~94 callbacks/s).
_DEFAULT_LATENCY_LOG_SIZE = 65536

# How many buffer_size chunks the input/output ring buffers should hold.
# Larger values trade memory for tolerance against worker-thread jitter.
_DEFAULT_RING_BLOCKS = 4


class RealtimeProcessor:
    """Real-time audio processor connecting an audio backend to an effect chain.

    Parameters
    ----------
    effects : Sequence[FX] | nn.Sequential
        Chain of effects to apply in order. Each effect must preserve
        chunk length; effects that change the time dimension (e.g.,
        ``Delay`` with feedback that grows the buffer) are rejected at
        runtime.
    backend : AudioBackend
        Audio backend driving the I/O thread.
    config : StreamConfig
        Stream configuration. ``config.buffer_size`` defines the DSP
        chunk size.
    ring_blocks : int
        Capacity of each ring buffer in units of ``buffer_size``. Default
        is 4 (≈40 ms tolerance at 512-sample / 48 kHz).
    latency_log_size : int
        Maximum number of per-callback latency samples retained. Default
        is 65 536 (≈ 11 min at 512-sample / 48 kHz). Older entries are
        evicted FIFO.
    prime_output : bool
        If True (default), write one chunk of silence into the output
        ring at start so the first callback never underflows.
    start_worker : bool
        If True (default), ``start()`` spawns a dedicated DSP worker
        thread that drains the input ring continuously. If False, the
        caller is responsible for calling ``process_pending()`` after
        each callback round-trip. The latter mode is intended for
        deterministic testing and for embedding the processor inside a
        host-driven scheduler.

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
        ring_blocks: int = _DEFAULT_RING_BLOCKS,
        latency_log_size: int = _DEFAULT_LATENCY_LOG_SIZE,
        prime_output: bool = True,
        start_worker: bool = True,
    ) -> None:
        validate_sample_rate(config.sample_rate)
        validate_positive(config.buffer_size, "buffer_size")
        if ring_blocks < 2:
            raise ValueError(
                f"ring_blocks must be at least 2 (got {ring_blocks}); "
                f"need at least one block for the producer and one for the consumer"
            )

        self._effects: list[FX] = self._normalize_effects(effects)
        self._backend = backend
        self._config = config
        self._running = False
        self._ring_blocks = ring_blocks
        self._prime_output = prime_output
        self._start_worker = start_worker

        # Configure effects with sample rate (same pattern as Wave.__or__)
        for effect in self._effects:
            if hasattr(effect, "fs") and effect.fs is None:
                effect.fs = config.sample_rate
            if isinstance(effect, AbstractFilter) and not effect._has_computed_coeff:
                effect.compute_coefficients()

        # Ring buffers. Capacity is rounded up to next power of 2 by
        # TensorRingBuffer; we request ring_blocks * buffer_size so the
        # producer can stay ahead of the consumer for ring_blocks - 1
        # callbacks before underflowing.
        ch_in = max(config.channels_in, 1)
        ch_out = max(config.channels_out, 1)
        capacity = ring_blocks * config.buffer_size
        self._input_buffer = TensorRingBuffer(capacity, ch_in)
        self._output_buffer = TensorRingBuffer(capacity, ch_out)

        # Thread-safe parameter updates: pending dicts drained at each DSP worker
        # iteration. ``_pending_params`` are instant; ``_pending_ramps`` request a
        # smoothed change. ``_active_ramps`` is worker-owned (no lock needed).
        self._pending_params: dict[str, Any] = {}
        self._pending_ramps: dict[str, tuple[float, float]] = {}  # name -> (target, ramp_ms)
        self._active_ramps: dict[str, dict[str, float]] = {}
        self._param_lock = threading.Lock()

        # Worker thread coordination.
        self._stop_event = threading.Event()
        self._input_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._worker_error: BaseException | None = None

        # Instrumentation. Counters are int-valued and only the audio
        # callback writes to them, so plain ``+=`` is safe under CPython
        # GIL. ``latency_log`` is a bounded deque; ``deque.append`` is
        # atomic so the callback can append without locking.
        self._latency_log: collections.deque[int] = collections.deque(maxlen=latency_log_size)
        self._xrun_count: int = 0
        self._input_overflow_count: int = 0
        self._output_underflow_count: int = 0
        self._backend_xrun_count: int = 0
        self._callback_count: int = 0

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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start real-time processing.

        Spawns the DSP worker thread, primes the output ring, and starts
        the audio backend. The audio callback is registered with the
        backend; once running, sample movement between the backend's
        buffers and the ring buffers happens on every callback while the
        effect chain runs in the worker thread.

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
            "Starting real-time processor: %dHz, %d buffer, %din/%dout, ring=%d blocks",
            self._config.sample_rate,
            self._config.buffer_size,
            self._config.channels_in,
            self._config.channels_out,
            self._ring_blocks,
        )

        self._stop_event.clear()
        self._input_event.clear()
        self._worker_error = None

        # Prime the output ring with one block of silence so the first
        # audio callback never has to underflow.
        if self._prime_output and self._config.channels_out > 0:
            silence = torch.zeros(
                self._output_buffer.channels,
                self._config.buffer_size,
                dtype=self._output_buffer.buffer.dtype,
            )
            self._output_buffer.write(silence)

        # Start the worker BEFORE the audio backend so the first
        # callback has a thread waiting to service it. Skipped in
        # ``start_worker=False`` mode, where the caller drives DSP via
        # ``process_pending()``.
        if self._start_worker:
            self._worker_thread = threading.Thread(
                target=self._dsp_worker_loop,
                name="torchfx-realtime-dsp",
                daemon=True,
            )
            self._worker_thread.start()

        try:
            self._backend.open_stream(self._config, callback=self._audio_callback)
            self._backend.start()
        except BaseException:
            # If backend setup fails, tear the worker thread down.
            self._stop_event.set()
            self._input_event.set()
            if self._worker_thread is not None:
                self._worker_thread.join(timeout=1.0)
                self._worker_thread = None
            raise

        self._running = True

    def stop(self) -> None:
        """Stop real-time processing and close the stream.

        Stops the audio backend first (so no more callbacks fire) and
        then signals the worker thread to exit. Any exception captured
        from the worker thread is re-raised here so callers see DSP
        errors at a known point in their code.

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

        _logger.info(
            "Stopping real-time processor (%d callbacks, %d xruns)",
            self._callback_count,
            self._xrun_count,
        )

        self._running = False
        try:
            self._backend.stop()
        finally:
            self._backend.close()

        self._stop_event.set()
        self._input_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=1.0)
            self._worker_thread = None

        if self._worker_error is not None:
            err = self._worker_error
            self._worker_error = None
            raise err

    # ------------------------------------------------------------------
    # Parameter updates
    # ------------------------------------------------------------------

    def set_parameter(self, name: str, value: Any, ramp_ms: float = 0.0) -> None:
        """Thread-safe parameter update, optionally smoothed over ``ramp_ms``.

        Parameters are staged and applied at the next DSP worker iteration
        boundary, *not* in the audio callback (keeping the callback alloc-free).
        With ``ramp_ms > 0`` a numeric parameter is linearly ramped from its
        current value to ``value`` over that many milliseconds (at block
        granularity), avoiding the zipper noise of a discrete jump. Filter
        coefficients are recomputed each block during the ramp **without**
        resetting DF1 state, so a cutoff sweep stays continuous.

        Parameters
        ----------
        name : str
            Dot-separated parameter path, e.g., ``"0.cutoff"`` for effect
            index 0, attribute ``cutoff``.
        value : Any
            New parameter value.
        ramp_ms : float, optional
            Ramp duration in milliseconds (default ``0`` = instant).

        Examples
        --------
        >>> # processor.set_parameter("0.cutoff", 2000)
        >>> # processor.set_parameter("1.gain", 0.8, ramp_ms=20)

        """
        with self._param_lock:
            if ramp_ms and ramp_ms > 0:
                self._pending_ramps[name] = (float(value), float(ramp_ms))
                self._pending_params.pop(name, None)  # ramp supersedes a pending instant set
            else:
                self._pending_params[name] = value
                self._pending_ramps.pop(name, None)

    def _resolve_effect_attr(self, key: str) -> tuple[FX, str] | None:
        """Resolve a ``"<idx>.<attr>"`` key to ``(effect, attr)`` or ``None``."""
        parts = key.split(".", 1)
        try:
            idx = int(parts[0])
        except ValueError:
            _logger.warning("Invalid effect index in parameter key: %r", key)
            return None
        if idx < 0 or idx >= len(self._effects):
            _logger.warning("Invalid effect index: %d", idx)
            return None
        if len(parts) < 2 or not parts[1]:
            _logger.warning("No attribute specified for effect %d", idx)
            return None
        return self._effects[idx], parts[1]

    def _set_effect_attr(self, effect: FX, attr: str, value: Any, *, reset: bool) -> None:
        """Set ``effect.attr`` and (for filters) recompute coefficients.

        ``reset=True`` clears DF1 state (a deliberate one-off change); ``reset=False``
        preserves it (a ramp step) so a swept filter does not restart each block.

        """
        setattr(effect, attr, value)
        if isinstance(effect, AbstractFilter):
            effect.compute_coefficients()
            effect._coeff_fs = getattr(effect, "fs", None)
            effect._coeff_fingerprint = effect._design_fingerprint()
            # _sos_device_cache lives on the IIR/Biquad subclasses (not AbstractFilter);
            # invalidate it so the next forward re-caches the recomputed coefficients.
            cast(Any, effect)._sos_device_cache = None
            if reset:
                reset_state = getattr(effect, "reset_state", None)
                if callable(reset_state):
                    reset_state()

    def _apply_pending_params(self) -> None:
        """Swap pending parameters into active effects, stepping ramps (worker
        thread)."""
        params: dict[str, Any] = {}
        ramps: dict[str, tuple[float, float]] = {}
        if self._pending_params or self._pending_ramps:
            with self._param_lock:
                params, self._pending_params = self._pending_params, {}
                ramps, self._pending_ramps = self._pending_ramps, {}
        elif not self._active_ramps:
            return

        # Instant updates (reset filter state, the existing one-off semantics).
        for key, value in params.items():
            res = self._resolve_effect_attr(key)
            if res is None:
                continue
            self._set_effect_attr(*res, value, reset=True)
            self._active_ramps.pop(key, None)  # an instant set cancels any ramp

        # Start new ramps, capturing the current value as the ramp origin.
        block_ms = self._config.buffer_size / self._config.sample_rate * 1000.0
        for key, (target, ramp_ms) in ramps.items():
            res = self._resolve_effect_attr(key)
            if res is None:
                continue
            effect, attr = res
            try:
                start = float(getattr(effect, attr))
            except (TypeError, ValueError):
                self._set_effect_attr(effect, attr, target, reset=True)  # non-numeric: instant
                continue
            total = max(1, math.ceil(ramp_ms / block_ms))
            self._active_ramps[key] = {"start": start, "target": target, "total": total, "block": 0}

        # Advance active ramps by one block.
        # ponytail: block-rate ramp (coeffs recomputed per block). Per-sample gain
        # interpolation is the upgrade path if block-rate audibly steps.
        finished: list[str] = []
        for key, r in self._active_ramps.items():
            res = self._resolve_effect_attr(key)
            if res is None:
                finished.append(key)
                continue
            effect, attr = res
            r["block"] += 1
            if r["block"] >= r["total"]:
                self._set_effect_attr(effect, attr, r["target"], reset=False)
                finished.append(key)
            else:
                t = r["block"] / r["total"]
                value = r["start"] + (r["target"] - r["start"]) * t
                self._set_effect_attr(effect, attr, value, reset=False)
        for key in finished:
            self._active_ramps.pop(key, None)

    # ------------------------------------------------------------------
    # Audio callback (runs in backend thread)
    # ------------------------------------------------------------------

    def _audio_callback(
        self,
        input_data: Tensor,
        output_data: Tensor,
        frame_count: int,
        status: BackendStatus | None = None,
    ) -> None:
        """Audio callback: ring buffer I/O only.

        Invoked by the backend on every buffer. This routine must
        complete well within ``buffer_size / sample_rate`` seconds; it
        therefore does **no** effect processing, only sample movement
        between the backend's tensors and the processor's ring buffers.

        Parameters
        ----------
        input_data : Tensor
            Input audio of shape ``(channels_in, frame_count)`` or empty
            tensor for output-only streams.
        output_data : Tensor
            Output tensor to fill, shape ``(channels_out, frame_count)``
            or empty for input-only streams.
        frame_count : int
            Number of frames in this callback.
        status : BackendStatus | None
            Backend-reported status flags. May be ``None`` for backends
            that don't surface them; this is treated as "no xruns".

        """
        t_start = time.perf_counter_ns()
        try:
            # Track backend-reported xruns separately so the user can
            # tell whether dropouts came from us (output ring underflow)
            # or from the backend / OS (PortAudio overflow).
            if status is not None and status.has_xrun:
                self._backend_xrun_count += 1
                self._xrun_count += 1

            # Push input into the input ring. Truncated writes are a
            # ring overflow — the worker isn't draining fast enough.
            if input_data.numel() > 0:
                written = self._input_buffer.write(input_data)
                if written < input_data.shape[-1]:
                    self._input_overflow_count += 1
                    self._xrun_count += 1

            # Pull processed output from the output ring. If less than
            # frame_count samples are available, copy what we have and
            # zero the rest — an audible underflow.
            if output_data.numel() > 0:
                self._fill_output(output_data, frame_count)

            self._callback_count += 1
            # Signal the worker that there's input to drain. The event
            # is edge-triggered: clearing happens in the worker.
            self._input_event.set()
        finally:
            t_end = time.perf_counter_ns()
            self._latency_log.append(t_end - t_start)

    def _fill_output(self, output_data: Tensor, frame_count: int) -> None:
        """Copy processed samples from the output ring into ``output_data``.

        Always advances the read pointer by ``frame_count`` to keep the
        backend's stream in lock-step with the DSP worker. When the
        output ring has fewer than ``frame_count`` samples ready, the
        tail of ``output_data`` is zeroed and an xrun is recorded.

        """
        available = self._output_buffer.available_read
        out_channels = output_data.shape[0]
        in_channels = self._output_buffer.channels

        if available >= frame_count:
            chunk = self._output_buffer.read(frame_count)
            self._copy_with_channel_adapt(chunk, output_data)
            return

        # Underflow: copy what we have, zero the rest.
        if available > 0:
            chunk = self._output_buffer.read(available)
            head = output_data[:, :available]
            self._copy_with_channel_adapt(chunk, head)
        output_data[:, available:].zero_()
        self._output_underflow_count += 1
        self._xrun_count += 1
        # Silence the unused-variable lints from out/in_channels.
        del out_channels, in_channels

    @staticmethod
    def _copy_with_channel_adapt(src: Tensor, dst: Tensor) -> None:
        """Copy ``src`` into ``dst`` with simple channel adaptation.

        - Equal channel counts: direct copy.
        - Mono source, multi-channel destination: broadcast mono into
          every destination channel.
        - Multi-channel source, narrower destination: truncate to the
          destination's channel count.

        """
        if src.shape[0] == dst.shape[0]:
            dst.copy_(src)
        elif src.shape[0] == 1 and dst.shape[0] > 1:
            dst.copy_(src.expand(dst.shape[0], -1))
        else:
            dst.copy_(src[: dst.shape[0]])

    # ------------------------------------------------------------------
    # DSP worker (runs in worker thread)
    # ------------------------------------------------------------------

    def _dsp_worker_loop(self) -> None:
        """DSP worker thread: drain input → effects → output, repeat."""
        try:
            while not self._stop_event.is_set():
                # Wait for the callback to signal new input. Time out so
                # that even with no audio activity, the worker can still
                # observe ``_stop_event`` and exit promptly.
                self._input_event.wait(timeout=0.05)
                self._input_event.clear()
                self._dsp_drain_available()
        except BaseException as exc:  # noqa: BLE001
            # Capture and surface from ``stop()``.
            self._worker_error = exc
            _logger.error("DSP worker thread terminated: %s", exc)

    def process_pending(self) -> int:
        """Drive one DSP draining pass synchronously.

        Intended for tests and host-scheduled deployments using
        ``start_worker=False``. Reads as many ``buffer_size``-aligned
        chunks as are available in the input ring, runs the effect
        chain on each, and writes results to the output ring.

        Returns
        -------
        int
            Number of chunks processed.

        """
        return self._dsp_drain_available()

    @torch.no_grad()
    def _dsp_drain_available(self) -> int:
        """Process every full chunk currently in the input ring.

        Returns
        -------
        int
            Number of chunks processed.

        """
        chunk_size = self._config.buffer_size
        chunks_processed = 0

        while True:
            if self._input_buffer.available_read < chunk_size:
                break
            if self._output_buffer.available_write < chunk_size:
                # Output ring is full — the audio callback isn't
                # draining (e.g., backend stopped). Yield to avoid a
                # busy loop.
                break

            self._apply_pending_params()
            chunk = self._input_buffer.read(chunk_size)

            x = chunk
            for effect in self._effects:
                x = effect(x)

            if x.shape[-1] != chunk_size:
                raise RealtimeError(
                    f"RealtimeProcessor requires chunk-length-preserving effects, "
                    f"but {type(effect).__name__} changed length from "
                    f"{chunk_size} to {x.shape[-1]}.",
                    suggestion="Avoid effects that change time dimension (e.g. Delay with "
                    "taps>1) in the realtime chain, or pre-process them with StreamProcessor.",
                )

            self._output_buffer.write(x)
            chunks_processed += 1

        return chunks_processed

    # ------------------------------------------------------------------
    # State management & introspection
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """Reset internal state: ring buffers and stateful effects.

        Useful after seeking in a file or switching audio sources. Safe
        to call while running, but typically called between sessions.

        """
        self._input_buffer.clear()
        self._output_buffer.clear()
        for effect in self._effects:
            reset_state = getattr(effect, "reset_state", None)
            if callable(reset_state):
                reset_state()

    def reset_metrics(self) -> None:
        """Reset latency log and xrun counters.

        Useful between benchmark phases to discard startup transients.

        """
        self._latency_log.clear()
        self._xrun_count = 0
        self._input_overflow_count = 0
        self._output_underflow_count = 0
        self._backend_xrun_count = 0
        self._callback_count = 0

    def latency_log_ns(self) -> list[int]:
        """Snapshot of per-callback wall-clock durations in nanoseconds.

        The returned list is a copy; safe to consume from any thread.

        Examples
        --------
        >>> # samples = processor.latency_log_ns()
        >>> # p99 = sorted(samples)[int(len(samples) * 0.99)] / 1e6  # ms

        """
        return list(self._latency_log)

    def latency_stats_ms(self) -> dict[str, float]:
        """Summary statistics of per-callback latency in milliseconds.

        Returns
        -------
        dict
            Keys ``count``, ``min``, ``median``, ``mean``, ``p95``,
            ``p99``, ``max``. Empty log returns zeros except ``count``.

        Examples
        --------
        >>> # stats = processor.latency_stats_ms()
        >>> # print(stats["p99"], stats["count"])

        """
        samples = self.latency_log_ns()
        if not samples:
            return {
                "count": 0.0,
                "min": 0.0,
                "median": 0.0,
                "mean": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "max": 0.0,
            }
        sorted_ms = sorted(s / 1_000_000.0 for s in samples)
        n = len(sorted_ms)

        def pct(p: float) -> float:
            idx = min(n - 1, max(0, int(p * n)))
            return sorted_ms[idx]

        return {
            "count": float(n),
            "min": sorted_ms[0],
            "median": pct(0.5),
            "mean": sum(sorted_ms) / n,
            "p95": pct(0.95),
            "p99": pct(0.99),
            "max": sorted_ms[-1],
        }

    @property
    def latency_ms(self) -> float:
        """Estimated total latency in milliseconds.

        Computed as ``(buffer_size + 1 block of output priming) /
        sample_rate``. With output priming enabled this is two buffers
        of latency; without priming it is one.

        """
        blocks = 2 if self._prime_output else 1
        return (blocks * self._config.buffer_size / self._config.sample_rate) * 1000.0

    @property
    def xrun_count(self) -> int:
        """Total number of xruns seen since start (or last ``reset_metrics``).

        Includes input ring overflows, output ring underflows, and backend-reported
        xruns. Use the more granular properties below to attribute by source.

        """
        return self._xrun_count

    @property
    def input_overflow_count(self) -> int:
        """Times the audio callback couldn't push all input samples into the ring."""
        return self._input_overflow_count

    @property
    def output_underflow_count(self) -> int:
        """Times the audio callback found the output ring empty."""
        return self._output_underflow_count

    @property
    def backend_xrun_count(self) -> int:
        """Xruns reported by the audio backend itself (e.g. PortAudio
        paInputOverflow)."""
        return self._backend_xrun_count

    @property
    def callback_count(self) -> int:
        """Number of audio callbacks serviced since start (or last
        ``reset_metrics``)."""
        return self._callback_count

    @property
    def deadline_ms(self) -> float:
        """Per-callback deadline in milliseconds (``buffer_size / sample_rate``)."""
        return (self._config.buffer_size / self._config.sample_rate) * 1000.0

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
