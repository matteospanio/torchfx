"""FIR filter benchmarks: GPU vs CPU vs SciPy across duration x channel matrix."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from scipy.signal import firwin, lfilter

from torchfx import Wave
from torchfx.filter import DesignableFIR

from .conftest import SAMPLE_RATE, create_signal_numpy

DURATIONS = [5, 60, 180, 300, 600]
CHANNELS = [1, 2, 4, 8, 12]
REP = 50


def _make_fir_chain(fs):
    return nn.Sequential(
        DesignableFIR(num_taps=101, cutoff=1000, fs=fs),
        DesignableFIR(num_taps=102, cutoff=5000, fs=fs),
        DesignableFIR(num_taps=103, cutoff=1500, fs=fs),
        DesignableFIR(num_taps=104, cutoff=1800, fs=fs),
        DesignableFIR(num_taps=105, cutoff=1850, fs=fs),
    )


@pytest.mark.benchmark(group="fir")
@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("duration", DURATIONS)
def test_fir_gpu(cuda_sync_benchmark, duration, channels):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    signal = create_signal_numpy(channels, duration)
    wave = Wave(signal, SAMPLE_RATE)
    fchain = _make_fir_chain(SAMPLE_RATE)

    for f in fchain:
        f.compute_coefficients()

    wave.to("cuda")
    fchain.to("cuda")

    cuda_sync_benchmark.pedantic(
        lambda: wave | fchain,
        rounds=REP,
        warmup_rounds=5,
    )


@pytest.mark.benchmark(group="fir")
@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("duration", DURATIONS)
def test_fir_cpu(benchmark, duration, channels):
    signal = create_signal_numpy(channels, duration)
    wave = Wave(signal, SAMPLE_RATE)
    fchain = _make_fir_chain(SAMPLE_RATE)

    for f in fchain:
        f.compute_coefficients()

    benchmark.pedantic(
        lambda: wave | fchain,
        rounds=REP,
        warmup_rounds=5,
    )


@pytest.mark.benchmark(group="fir")
@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("duration", DURATIONS)
def test_fir_scipy(benchmark, duration, channels):
    signal = create_signal_numpy(channels, duration)

    b1 = firwin(101, 1000, fs=SAMPLE_RATE)
    b2 = firwin(102, 5000, fs=SAMPLE_RATE)
    b3 = firwin(103, 1500, fs=SAMPLE_RATE)
    b4 = firwin(104, 1800, fs=SAMPLE_RATE)
    b5 = firwin(105, 1850, fs=SAMPLE_RATE)

    def run():
        a = [1]
        x = lfilter(b1, a, signal)
        x = lfilter(b2, a, x)
        x = lfilter(b3, a, x)
        x = lfilter(b4, a, x)
        x = lfilter(b5, a, x)
        return x

    benchmark.pedantic(run, rounds=REP, warmup_rounds=5)
