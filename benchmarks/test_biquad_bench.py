"""Single biquad filter benchmarks: CPU vs CUDA (stateful processing)."""

from __future__ import annotations

import pytest
import torch

from torchfx.filter.biquad import BiquadLPF

from .conftest import (
    BIQUAD_CHANNELS,
    BIQUAD_DURATIONS,
    DEVICES,
    REP,
    SAMPLE_RATE,
    WARMUP,
    create_signal_torch,
)


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
        filt = filt.to("cuda")

    # Force stateful mode (call twice + reset)
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
