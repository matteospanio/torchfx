"""IIR filter benchmarks: torchfx GPU vs CPU vs SciPy vs numba CUDA."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from scipy.signal import butter, cheby1, lfilter

from torchfx import Wave
from torchfx.filter import HiButterworth, HiChebyshev1, LoButterworth, LoChebyshev1

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


def _make_iir_chain(fs):
    return nn.Sequential(
        HiButterworth(cutoff=1000, order=2, fs=fs),
        LoButterworth(cutoff=5000, order=2, fs=fs),
        HiChebyshev1(cutoff=1500, order=2, fs=fs),
        LoChebyshev1(cutoff=1800, order=2, fs=fs),
    )


def _scipy_coefficients():
    """Pre-compute scipy filter coefficients (shared across tests)."""
    b1, a1 = butter(2, 1000, btype="high", fs=SAMPLE_RATE)
    b2, a2 = butter(2, 5000, btype="low", fs=SAMPLE_RATE)
    b3, a3 = cheby1(2, 0.5, 1500, btype="high", fs=SAMPLE_RATE)
    b4, a4 = cheby1(2, 0.5, 1800, btype="low", fs=SAMPLE_RATE)
    return [(b1, a1), (b2, a2), (b3, a3), (b4, a4)]


SCIPY_FILTERS = _scipy_coefficients()


# ── numba CUDA kernel ─────────────────────────────────────────────────────────


def _get_numba_iir_kernel():
    """Lazily compile the numba CUDA IIR kernel."""
    from numba import cuda, float64, void

    @cuda.jit(
        void(float64, float64, float64, float64, float64, float64[:, :], float64[:, :]),
        cache=True,
    )
    def _numba_iir_df1(b0, b1, b2, a1, a2, x, y):
        """Per-channel DF1 IIR biquad on GPU.

        One thread per channel.

        """
        c = cuda.grid(1)
        if c >= x.shape[0]:
            return
        T = x.shape[1]
        sx0 = 0.0
        sx1 = 0.0
        sy0 = 0.0
        sy1 = 0.0
        for n in range(T):
            xn = x[c, n]
            yn = b0 * xn + b1 * sx0 + b2 * sx1 - a1 * sy0 - a2 * sy1
            y[c, n] = yn
            sx1 = sx0
            sx0 = xn
            sy1 = sy0
            sy0 = yn

    return _numba_iir_df1


# ── Benchmarks ────────────────────────────────────────────────────────────────


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
        warmup_rounds=WARMUP,
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
        warmup_rounds=WARMUP,
    )


@pytest.mark.benchmark(group="iir")
@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("duration", DURATIONS)
def test_iir_scipy(benchmark, duration, channels):
    signal = create_signal_numpy(channels, duration)

    def run():
        x = signal
        for b, a in SCIPY_FILTERS:
            x = lfilter(b, a, x)
        return x

    benchmark.pedantic(run, rounds=REP, warmup_rounds=WARMUP)


@pytest.mark.benchmark(group="iir")
@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("duration", DURATIONS)
def test_iir_numba_cuda(cuda_sync_benchmark, duration, channels):
    if not numba_cuda_available():
        pytest.skip("numba CUDA not available")

    from numba import cuda as numba_cuda

    kernel = _get_numba_iir_kernel()
    signal = create_signal_numpy(channels, duration)

    # Pre-compute filter coefficients (same as scipy)
    filters = SCIPY_FILTERS

    # Transfer signal to GPU
    d_x = numba_cuda.to_device(signal.astype(np.float64))
    d_y = numba_cuda.device_array_like(d_x)
    d_tmp = numba_cuda.device_array_like(d_x)

    threads_per_block = min(256, channels)
    blocks = (channels + threads_per_block - 1) // threads_per_block

    def run():
        # Cascade: x → filter1 → filter2 → ... → filterN
        src = d_x
        for i, (b, a) in enumerate(filters):
            dst = d_y if i % 2 == 0 else d_tmp
            b0, b1, b2 = float(b[0]), float(b[1]), float(b[2])
            a1, a2 = float(a[1]), float(a[2])
            kernel[blocks, threads_per_block](b0, b1, b2, a1, a2, src, dst)
            src = dst
        numba_cuda.synchronize()

    # Warmup (compile kernel + warm caches)
    for _ in range(3):
        run()

    cuda_sync_benchmark.pedantic(run, rounds=REP, warmup_rounds=WARMUP)
