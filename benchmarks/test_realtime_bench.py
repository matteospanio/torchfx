"""Realtime processor latency and jitter benchmarks.

These benchmarks measure ``RealtimeProcessor`` per-callback wall time
and the percentile-tail of latency over many callbacks. They are
deterministic: they use a ``DeterministicMockBackend`` that drives the
processor synchronously (no PortAudio, no real audio I/O), so results
depend only on the effect chain and the CPU.

What is measured
----------------
* ``test_realtime_one_round_trip`` — wall time of one push+drain+pull
  round-trip, the unit pytest-benchmark times. Useful for tracking
  changes in callback path overhead.
* ``test_realtime_latency_distribution`` — drives N round-trips, then
  reports p50 / p95 / p99 of the callback latencies recorded by the
  processor itself (`latency_stats_ms`). Printed via the
  ``extra_info`` mechanism so pytest-benchmark JSON output captures
  them. **Not a competition timing**: the actual measurement is the
  per-round-trip wall time; the extras add the tail-distribution
  context.

Run
---
``uv run pytest benchmarks/test_realtime_bench.py --benchmark-enable``

"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import Tensor

from torchfx.effect import FX, Gain
from torchfx.filter.iir import HiButterworth, LoButterworth
from torchfx.realtime.backend import (
    AudioBackend,
    AudioCallback,
    BackendStatus,
    StreamConfig,
    StreamDirection,
    StreamState,
)
from torchfx.realtime.processor import RealtimeProcessor


class DeterministicMockBackend(AudioBackend):
    """Minimal MockBackend for benchmarks (no real audio I/O).

    Same idea as the test ``MockBackend`` but slimmed down: it carries
    a back-reference to the processor and synchronously drains DSP
    between the push and pull callbacks. The benchmarked unit of work
    is ``run_one_round_trip``.

    """

    def __init__(self) -> None:
        self._state: StreamState = StreamState.CLOSED
        self._config: StreamConfig | None = None
        self._callback: AudioCallback | None = None
        self._processor: Any = None

    @property
    def name(self) -> str:
        return "deterministic-mock"

    @property
    def is_available(self) -> bool:
        return True

    def get_devices(self) -> list[dict[str, Any]]:
        return []

    def get_default_device(self, direction: StreamDirection) -> int | str:  # noqa: ARG002
        return 0

    def open_stream(
        self,
        config: StreamConfig,
        callback: AudioCallback | None = None,
    ) -> None:
        self._config = config
        self._callback = callback
        self._state = StreamState.OPEN

    def start(self) -> None:
        self._state = StreamState.RUNNING

    def stop(self) -> None:
        self._state = StreamState.STOPPED

    def close(self) -> None:
        self._state = StreamState.CLOSED

    @property
    def state(self) -> StreamState:
        return self._state

    def read(self, num_frames: int) -> Tensor:
        ch = self._config.channels_in if self._config else 1
        return torch.zeros(ch, num_frames)

    def write(self, data: Tensor) -> None:
        pass

    def run_one_round_trip(
        self, input_chunk: Tensor, output_chunk: Tensor, status: BackendStatus | None = None
    ) -> None:
        """One push → drain → pull cycle, reusing the same tensors."""
        assert self._callback is not None
        assert self._processor is not None
        frames = input_chunk.shape[-1]
        scratch = torch.zeros_like(output_chunk)
        self._callback(input_chunk, scratch, frames, status)
        self._processor.process_pending()
        self._callback(input_chunk, output_chunk, frames, status)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

# 48 kHz buffer of 512 samples — 10.67 ms deadline, the common low-latency
# default on Linux. Keep the same axes as the existing hotpath benchmarks so
# headlines line up.
SAMPLE_RATE = 48_000
BUFFER_SIZES = [128, 256, 512, 1024]
CHANNELS = [1, 2]
CASCADE_DEPTHS = [0, 2, 5, 10]


def _build_chain(depth: int, fs: int) -> list[FX]:
    """Construct an effect chain of approximate ``depth`` SOS sections."""
    if depth == 0:
        return [Gain(1.0)]  # near-trivial
    # Alternate Hi/Lo Butterworth at moderate orders to mimic mastering chains.
    chain: list[FX] = []
    for i in range(depth):
        if i % 2 == 0:
            chain.append(HiButterworth(60.0 + 10.0 * i, order=2, fs=fs))
        else:
            chain.append(LoButterworth(8000.0 + 100.0 * i, order=4, fs=fs))
    return chain


def _make_processor(
    buffer_size: int, channels: int, depth: int
) -> tuple[RealtimeProcessor, DeterministicMockBackend, Tensor, Tensor]:
    config = StreamConfig(
        sample_rate=SAMPLE_RATE,
        buffer_size=buffer_size,
        channels_in=channels,
        channels_out=channels,
    )
    backend = DeterministicMockBackend()
    effects = _build_chain(depth, SAMPLE_RATE)
    proc = RealtimeProcessor(
        effects=effects,
        backend=backend,
        config=config,
        start_worker=False,
        latency_log_size=4096,
    )
    backend._processor = proc
    in_chunk = torch.randn(channels, buffer_size, dtype=torch.float32) * 0.1
    out_chunk = torch.zeros(channels, buffer_size, dtype=torch.float32)
    return proc, backend, in_chunk, out_chunk


# ──────────────────────────────────────────────────────────────────────────────
# Benchmarks
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.benchmark(group="realtime-roundtrip")
@pytest.mark.parametrize("buffer_size", BUFFER_SIZES)
@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("depth", CASCADE_DEPTHS)
def test_realtime_one_round_trip(
    benchmark: Any, buffer_size: int, channels: int, depth: int
) -> None:
    """Wall-clock of one audio-callback round trip.

    This is the headline number for "can a 512-sample buffer at 48 kHz fit within its
    10.67 ms deadline". Reported as min/median/max by pytest-benchmark in the standard
    table.

    """
    proc, backend, in_chunk, out_chunk = _make_processor(buffer_size, channels, depth)
    proc.start()
    # Warm-up beyond pytest-benchmark's own warmup so coefficient
    # caches and any first-call dtype churn are out of the way.
    for _ in range(8):
        backend.run_one_round_trip(in_chunk, out_chunk)
    proc.reset_metrics()

    def run() -> None:
        backend.run_one_round_trip(in_chunk, out_chunk)

    benchmark.pedantic(run, rounds=50, warmup_rounds=5, iterations=1)
    proc.stop()

    # Surface the deadline so JSON consumers can compute "budget used".
    benchmark.extra_info["deadline_ms"] = proc.deadline_ms
    benchmark.extra_info["channels"] = channels
    benchmark.extra_info["depth"] = depth
    benchmark.extra_info["buffer_size"] = buffer_size


@pytest.mark.benchmark(group="realtime-jitter")
@pytest.mark.parametrize("buffer_size", [256, 512])
@pytest.mark.parametrize("depth", [2, 5, 10])
def test_realtime_latency_distribution(benchmark: Any, buffer_size: int, depth: int) -> None:
    """Percentile tail of per-callback latency over many round-trips.

    Drives ``N`` round trips, then reports the latency distribution
    recorded by the processor itself. The deadline is
    ``buffer_size / sample_rate``; ``p99 / deadline`` is the budget
    fraction at the 99th percentile.

    """
    channels = 2
    proc, backend, in_chunk, out_chunk = _make_processor(buffer_size, channels, depth)
    proc.start()
    # Warm-up
    for _ in range(16):
        backend.run_one_round_trip(in_chunk, out_chunk)
    proc.reset_metrics()

    N = 1024

    def run() -> None:
        for _ in range(N):
            backend.run_one_round_trip(in_chunk, out_chunk)

    benchmark.pedantic(run, rounds=3, warmup_rounds=1, iterations=1)
    stats = proc.latency_stats_ms()
    deadline = proc.deadline_ms
    proc.stop()

    # extra_info ends up in pytest-benchmark's JSON output.
    benchmark.extra_info["latency_p50_ms"] = stats["median"]
    benchmark.extra_info["latency_p95_ms"] = stats["p95"]
    benchmark.extra_info["latency_p99_ms"] = stats["p99"]
    benchmark.extra_info["latency_max_ms"] = stats["max"]
    benchmark.extra_info["deadline_ms"] = deadline
    benchmark.extra_info["budget_p99"] = stats["p99"] / deadline if deadline > 0 else 0.0
    benchmark.extra_info["budget_max"] = stats["max"] / deadline if deadline > 0 else 0.0
    benchmark.extra_info["xrun_count"] = proc.xrun_count
    benchmark.extra_info["callback_count"] = proc.callback_count


@pytest.mark.benchmark(group="realtime-xrun-detection")
def test_xrun_counter_increments_under_synthetic_overflow() -> None:
    """Smoke benchmark: confirm xrun bookkeeping is wired end-to-end.

    Not timed — runs a few callbacks with synthetic
    ``BackendStatus(input_overflow=True)`` and verifies the counter
    monotonically increases.
    """
    proc, backend, in_chunk, out_chunk = _make_processor(512, 2, 2)
    proc.start()
    for _ in range(4):
        backend.run_one_round_trip(in_chunk, out_chunk, status=BackendStatus(input_overflow=True))
    assert proc.backend_xrun_count >= 4
    assert proc.xrun_count >= proc.backend_xrun_count
    proc.stop()
