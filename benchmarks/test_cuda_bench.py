"""CUDA performance benchmarks for torchfx biquad/SOS/pipeline processing."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from torchfx.filter import HiButterworth, LoButterworth
from torchfx.filter.biquad import BiquadLPF

from .conftest import SAMPLE_RATE, create_signal_torch

BIQUAD_DURATIONS = [0.1, 1.0, 5.0, 30.0]
BIQUAD_CHANNELS = [1, 2]
DEVICES = ["cpu", "cuda"]


@pytest.mark.benchmark(group="biquad-stateful")
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("channels", BIQUAD_CHANNELS)
@pytest.mark.parametrize("duration", BIQUAD_DURATIONS)
def test_biquad_stateful(cuda_sync_benchmark, duration, channels, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    filt = BiquadLPF(cutoff=1000, q=0.707, fs=SAMPLE_RATE)
    filt.compute_coefficients()

    x = create_signal_torch(channels, duration, device)

    if device == "cuda":
        filt.move_coeff("cuda")

    # Force stateful mode (call twice + reset, matching original benchmark)
    _ = filt(x)
    filt.reset_state()
    _ = filt(x)
    filt.reset_state()
    _ = filt(x)

    cuda_sync_benchmark.pedantic(
        lambda f=filt, sig=x: f(sig),
        rounds=20,
        warmup_rounds=5,
    )


SOS_ORDERS = [4, 8]


@pytest.mark.benchmark(group="sos-cascade")
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("order", SOS_ORDERS)
def test_sos_cascade(cuda_sync_benchmark, order, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    dur = 5.0
    filt = LoButterworth(cutoff=2000, order=order, fs=SAMPLE_RATE)
    filt.compute_coefficients()

    x = create_signal_torch(2, dur, device)

    if device == "cuda":
        filt.move_coeff("cuda")

    # Force stateful
    _ = filt(x)
    filt.reset_state()
    _ = filt(x)
    filt.reset_state()
    _ = filt(x)

    cuda_sync_benchmark.pedantic(
        lambda f=filt, sig=x: f(sig),
        rounds=20,
        warmup_rounds=5,
    )


@pytest.mark.benchmark(group="pipeline")
@pytest.mark.parametrize("device", DEVICES)
def test_pipeline(cuda_sync_benchmark, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    dur = 10.0
    chain = nn.Sequential(
        HiButterworth(cutoff=100, order=2, fs=SAMPLE_RATE),
        LoButterworth(cutoff=8000, order=4, fs=SAMPLE_RATE),
    )
    for f in chain:
        f.compute_coefficients()
        if device == "cuda":
            f.move_coeff("cuda")

    x = create_signal_torch(2, dur, device)

    cuda_sync_benchmark.pedantic(
        lambda c=chain, sig=x: c(sig),
        rounds=20,
        warmup_rounds=5,
    )
