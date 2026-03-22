"""IIR filter benchmarks: GPU vs CPU vs SciPy across duration x channel matrix."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from scipy.signal import butter, cheby1, lfilter

from torchfx import Wave
from torchfx.filter import HiButterworth, HiChebyshev1, LoButterworth, LoChebyshev1

from .conftest import SAMPLE_RATE, create_signal_numpy

DURATIONS = [1, 5, 180, 300, 600]
CHANNELS = [1, 2, 4, 8, 12]
REP = 50


def _make_iir_chain(fs):
    return nn.Sequential(
        HiButterworth(cutoff=1000, order=2, fs=fs),
        LoButterworth(cutoff=5000, order=2, fs=fs),
        HiChebyshev1(cutoff=1500, order=2, fs=fs),
        LoChebyshev1(cutoff=1800, order=2, fs=fs),
    )


@pytest.mark.benchmark(group="iir")
@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("duration", DURATIONS)
def test_iir_gpu(cuda_sync_benchmark, duration, channels):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    signal = create_signal_numpy(channels, duration)
    wave = Wave(signal, SAMPLE_RATE)
    fchain = _make_iir_chain(SAMPLE_RATE)

    wave.to("cuda")
    fchain.to("cuda")

    for f in fchain:
        f.compute_coefficients()
        f.move_coeff("cuda")

    cuda_sync_benchmark.pedantic(
        lambda: fchain(wave.ys),
        rounds=REP,
        warmup_rounds=5,
    )


@pytest.mark.benchmark(group="iir")
@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("duration", DURATIONS)
def test_iir_cpu(benchmark, duration, channels):
    signal = create_signal_numpy(channels, duration)
    wave = Wave(signal, SAMPLE_RATE)
    fchain = _make_iir_chain(SAMPLE_RATE)

    for f in fchain:
        f.compute_coefficients()

    benchmark.pedantic(
        lambda: fchain(wave.ys),
        rounds=REP,
        warmup_rounds=5,
    )


@pytest.mark.benchmark(group="iir")
@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("duration", DURATIONS)
def test_iir_scipy(benchmark, duration, channels):
    signal = create_signal_numpy(channels, duration)

    b1, a1 = butter(2, 1000, btype="high", fs=SAMPLE_RATE)
    b2, a2 = butter(2, 5000, btype="low", fs=SAMPLE_RATE)
    b3, a3 = cheby1(2, 0.5, 1500, btype="high", fs=SAMPLE_RATE)
    b4, a4 = cheby1(2, 0.5, 1800, btype="low", fs=SAMPLE_RATE)

    def run():
        x = lfilter(b1, a1, signal)
        x = lfilter(b2, a2, x)
        x = lfilter(b3, a3, x)
        x = lfilter(b4, a4, x)
        return x

    benchmark.pedantic(run, rounds=REP, warmup_rounds=5)
