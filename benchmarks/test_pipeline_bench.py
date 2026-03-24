"""Pipeline benchmarks: SOS cascade and multi-filter chains (CPU vs CUDA)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from torchfx.filter import HiButterworth, LoButterworth

from .conftest import DEVICES, REP, SAMPLE_RATE, WARMUP, create_signal_torch

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

    # Force stateful mode
    _ = filt(x)
    filt.reset_state()
    _ = filt(x)
    filt.reset_state()
    _ = filt(x)

    cuda_sync_benchmark.pedantic(
        lambda f=filt, sig=x: f(sig),
        rounds=REP,
        warmup_rounds=WARMUP,
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
        rounds=REP,
        warmup_rounds=WARMUP,
    )
