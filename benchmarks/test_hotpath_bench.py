"""Targeted hot-path benchmarks for realtime and batch throughput.

These benchmarks are intentionally smaller/faster than the full suite and focus on
recently critical paths:

- deferred Wave pipe materialization vs direct sequential execution
- Delay/Reverb effect runtime (including native delay-line dispatch)
- realtime-style 512-sample chunk processing

"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from torchfx import Wave
from torchfx.effect import Delay, Reverb
from torchfx.filter import HiButterworth, LoButterworth

from .conftest import DEVICES, SAMPLE_RATE, create_signal_torch

HOTPATH_REP = 15
HOTPATH_WARMUP = 3


def _make_iir_chain(fs: int) -> nn.Sequential:
    chain = nn.Sequential(
        HiButterworth(cutoff=80, order=2, fs=fs),
        LoButterworth(cutoff=8000, order=4, fs=fs),
    )
    for filt in chain:
        filt.compute_coefficients()
    return chain


@pytest.mark.benchmark(group="hotpath-wave-pipe")
@pytest.mark.parametrize("device", DEVICES)
def test_wave_pipe_materialized_iir_chain(cuda_sync_benchmark, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    x = create_signal_torch(2, 5.0, device)
    wave = Wave(x, SAMPLE_RATE, device=device)
    chain = _make_iir_chain(SAMPLE_RATE)
    if device == "cuda":
        chain = chain.to("cuda")

    cuda_sync_benchmark.pedantic(
        lambda: (wave | chain).ys,
        rounds=HOTPATH_REP,
        warmup_rounds=HOTPATH_WARMUP,
    )


@pytest.mark.benchmark(group="hotpath-wave-pipe")
@pytest.mark.parametrize("device", DEVICES)
def test_direct_sequential_iir_chain(cuda_sync_benchmark, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    x = create_signal_torch(2, 5.0, device)
    chain = _make_iir_chain(SAMPLE_RATE)
    if device == "cuda":
        chain = chain.to("cuda")

    cuda_sync_benchmark.pedantic(
        lambda: chain(x),
        rounds=HOTPATH_REP,
        warmup_rounds=HOTPATH_WARMUP,
    )


@pytest.mark.benchmark(group="hotpath-delay")
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("channels", [1, 2])
def test_delay_effect(cuda_sync_benchmark, device, channels):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    x = create_signal_torch(channels, 1.0, device)
    delay = Delay(delay_samples=1024, feedback=0.4, mix=0.3, taps=3)
    if device == "cuda":
        delay = delay.to("cuda")

    cuda_sync_benchmark.pedantic(
        lambda: delay(x),
        rounds=HOTPATH_REP,
        warmup_rounds=HOTPATH_WARMUP,
    )


@pytest.mark.benchmark(group="hotpath-delay")
@pytest.mark.parametrize("device", DEVICES)
def test_reverb_effect(cuda_sync_benchmark, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    x = create_signal_torch(2, 1.0, device)
    reverb = Reverb(delay=2048, decay=0.5, mix=0.3)
    if device == "cuda":
        reverb = reverb.to("cuda")

    cuda_sync_benchmark.pedantic(
        lambda: reverb(x),
        rounds=HOTPATH_REP,
        warmup_rounds=HOTPATH_WARMUP,
    )


@pytest.mark.benchmark(group="hotpath-realtime-chunk")
@pytest.mark.parametrize("device", DEVICES)
def test_realtime_chunk_chain(cuda_sync_benchmark, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # 512 samples at 48kHz ~= 10.67 ms callback period.
    chunk = torch.randn(2, 512, device=device, dtype=torch.float32)

    chain = _make_iir_chain(SAMPLE_RATE)
    if device == "cuda":
        chain = chain.to("cuda")

    cuda_sync_benchmark.pedantic(
        lambda: chain(chunk),
        rounds=HOTPATH_REP,
        warmup_rounds=HOTPATH_WARMUP,
    )
