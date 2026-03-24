"""FIR filter benchmarks: torchfx GPU vs CPU vs SciPy vs numba CUDA."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from scipy.signal import firwin, lfilter

from torchfx import Wave
from torchfx.filter import DesignableFIR

from .conftest import (
    CHANNELS,
    DURATIONS,
    REP,
    SAMPLE_RATE,
    WARMUP,
    create_signal_numpy,
    numba_cuda_available,
)

# ── Filter setup ──────────────────────────────────────────────────────────────

FIR_TAPS = [101, 102, 103, 104, 105]
FIR_CUTOFFS = [1000, 5000, 1500, 1800, 1850]


def _make_fir_chain(fs):
    return nn.Sequential(
        *[
            DesignableFIR(num_taps=t, cutoff=c, fs=fs)
            for t, c in zip(FIR_TAPS, FIR_CUTOFFS, strict=False)
        ]
    )


def _scipy_fir_coefficients():
    """Pre-compute scipy FIR filter coefficients."""
    return [firwin(t, c, fs=SAMPLE_RATE) for t, c in zip(FIR_TAPS, FIR_CUTOFFS, strict=False)]


SCIPY_FIR_COEFFS = _scipy_fir_coefficients()


# ── numba CUDA kernel ─────────────────────────────────────────────────────────


def _get_numba_fir_kernel():
    """Lazily compile a numba CUDA FIR convolution kernel."""
    from numba import cuda, float64, int64, void

    @cuda.jit(
        void(float64[:], int64, float64[:, :], float64[:, :]),
        cache=True,
    )
    def _numba_fir_conv(h, h_len, x, y):
        """Per-channel FIR filtering on GPU.

        One thread per channel.

        """
        c = cuda.grid(1)
        if c >= x.shape[0]:
            return
        T = x.shape[1]
        for n in range(T):
            acc = 0.0
            for k in range(h_len):
                idx = n - k
                if idx >= 0:
                    acc += h[k] * x[c, idx]
            y[c, n] = acc

    return _numba_fir_conv


# ── Benchmarks ────────────────────────────────────────────────────────────────


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
        warmup_rounds=WARMUP,
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
        warmup_rounds=WARMUP,
    )


@pytest.mark.benchmark(group="fir")
@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("duration", DURATIONS)
def test_fir_scipy(benchmark, duration, channels):
    signal = create_signal_numpy(channels, duration)

    def run():
        x = signal
        for b in SCIPY_FIR_COEFFS:
            x = lfilter(b, [1], x)
        return x

    benchmark.pedantic(run, rounds=REP, warmup_rounds=WARMUP)


@pytest.mark.benchmark(group="fir")
@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("duration", DURATIONS)
def test_fir_numba_cuda(cuda_sync_benchmark, duration, channels):
    if not numba_cuda_available():
        pytest.skip("numba CUDA not available")

    from numba import cuda as numba_cuda

    kernel = _get_numba_fir_kernel()
    signal = create_signal_numpy(channels, duration).astype(np.float64)

    # Transfer signal to GPU
    d_x = numba_cuda.to_device(signal)
    d_y = numba_cuda.device_array_like(d_x)
    d_tmp = numba_cuda.device_array_like(d_x)

    # Transfer filter coefficients
    d_coeffs = [numba_cuda.to_device(b.astype(np.float64)) for b in SCIPY_FIR_COEFFS]

    threads_per_block = min(256, channels)
    blocks = (channels + threads_per_block - 1) // threads_per_block

    def run():
        src = d_x
        for i, (d_h, b) in enumerate(zip(d_coeffs, SCIPY_FIR_COEFFS, strict=False)):
            dst = d_y if i % 2 == 0 else d_tmp
            kernel[blocks, threads_per_block](d_h, len(b), src, dst)
            src = dst
        numba_cuda.synchronize()

    # Warmup
    for _ in range(3):
        run()

    cuda_sync_benchmark.pedantic(run, rounds=REP, warmup_rounds=WARMUP)
