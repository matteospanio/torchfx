"""API style comparison benchmarks: FilterChain vs Sequential vs Pipe vs SciPy."""

from __future__ import annotations

import pytest
from scipy.signal import butter, cheby1, lfilter
from torch import nn
from torch.nn import Sequential

from torchfx import Wave
from torchfx.filter import HiChebyshev1, LoButterworth

from .conftest import REP, SAMPLE_RATE, WARMUP, create_signal_numpy

DURATION = 120
NUM_CHANNELS = 8


class FilterChain(nn.Module):
    def __init__(self, fs):
        super().__init__()
        self.f1 = HiChebyshev1(20, fs=fs)
        self.f2 = HiChebyshev1(60, fs=fs)
        self.f3 = HiChebyshev1(65, fs=fs)
        self.f4 = LoButterworth(5000, fs=fs)
        self.f5 = LoButterworth(4900, fs=fs)
        self.f6 = LoButterworth(4850, fs=fs)

    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        x = self.f5(x)
        x = self.f6(x)
        return x


@pytest.fixture(scope="module")
def api_bench_data():
    """Shared signal data for all API benchmarks."""
    signal_np = create_signal_numpy(NUM_CHANNELS, DURATION)
    wave = Wave(signal_np, SAMPLE_RATE)
    return signal_np, wave


@pytest.mark.benchmark(group="api-comparison")
def test_filter_chain(benchmark, api_bench_data):
    _, wave = api_bench_data

    def run():
        fchain = FilterChain(wave.fs)
        return fchain(wave.ys)

    benchmark.pedantic(run, rounds=REP, warmup_rounds=WARMUP)


@pytest.mark.benchmark(group="api-comparison")
def test_sequential(benchmark, api_bench_data):
    _, wave = api_bench_data

    def run():
        fchain = Sequential(
            HiChebyshev1(20, fs=wave.fs),
            HiChebyshev1(60, fs=wave.fs),
            HiChebyshev1(65, fs=wave.fs),
            LoButterworth(5000, fs=wave.fs),
            LoButterworth(4900, fs=wave.fs),
            LoButterworth(4850, fs=wave.fs),
        )
        return fchain(wave.ys)

    benchmark.pedantic(run, rounds=REP, warmup_rounds=WARMUP)


@pytest.mark.benchmark(group="api-comparison")
def test_pipe(benchmark, api_bench_data):
    _, wave = api_bench_data

    def run():
        return (
            wave
            | HiChebyshev1(20)
            | HiChebyshev1(60)
            | HiChebyshev1(65)
            | LoButterworth(5000)
            | LoButterworth(4900)
            | LoButterworth(4850)
        ).ys

    benchmark.pedantic(run, rounds=REP, warmup_rounds=WARMUP)


@pytest.mark.benchmark(group="api-comparison")
def test_scipy(benchmark, api_bench_data):
    signal_np, _ = api_bench_data

    b1, a1 = cheby1(2, 0.5, 20, btype="high", fs=SAMPLE_RATE)
    b2, a2 = cheby1(2, 0.5, 60, btype="high", fs=SAMPLE_RATE)
    b3, a3 = cheby1(2, 0.5, 65, btype="high", fs=SAMPLE_RATE)
    b4, a4 = butter(2, 5000, btype="low", fs=SAMPLE_RATE)
    b5, a5 = butter(2, 4900, btype="low", fs=SAMPLE_RATE)
    b6, a6 = butter(2, 4850, btype="low", fs=SAMPLE_RATE)

    def run():
        x = lfilter(b1, a1, signal_np)
        x = lfilter(b2, a2, x)
        x = lfilter(b3, a3, x)
        x = lfilter(b4, a4, x)
        x = lfilter(b5, a5, x)
        x = lfilter(b6, a6, x)
        return x

    benchmark.pedantic(run, rounds=REP, warmup_rounds=WARMUP)
