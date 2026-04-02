"""IIR filter benchmarks: torchfx GPU vs CPU vs SciPy vs numba JIT vs numba CUDA."""

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
    numba_available,
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


# ── numba JIT lfilter ────────────────────────────────────────────────────────


def _get_numba_iir_lfilter():
    """Lazily compile the numba @njit IIR lfilter."""
    from numba import njit, prange

    @njit(cache=True, fastmath=True, parallel=True)
    def _numba_iir_lfilter(x, filters_b, filters_a):
        C, T = x.shape
        for f in range(filters_b.shape[0]):
            b0 = filters_b[f, 0]
            b1 = filters_b[f, 1]
            b2 = filters_b[f, 2]
            a1 = filters_a[f, 1]
            a2 = filters_a[f, 2]
            y = np.empty_like(x)
            for c in prange(C):
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
            x = y
        return x

    return _numba_iir_lfilter


def _pack_scipy_iir_coefficients():
    """Pack scipy coefficients into 2D arrays for numba compatibility."""
    filters_b = np.array([b for b, _ in SCIPY_FILTERS])
    filters_a = np.array([a for _, a in SCIPY_FILTERS])
    return filters_b, filters_a


# ── numba CUDA kernels ───────────────────────────────────────────────────────


def _get_numba_iir_cuda_kernels():
    """Lazily compile the numba CUDA IIR kernels.

    Returns a (forcing_kernel, recurrence_kernel) pair.  The forcing kernel
    computes f[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] with one thread per
    sample (embarrassingly parallel).  The recurrence kernel then solves
    y[n] = f[n] - a1*y[n-1] - a2*y[n-2] with one thread per channel
    (sequential — inherent IIR dependency).

    """
    from numba import cuda, float64, void

    @cuda.jit(
        void(float64, float64, float64, float64[:, :], float64[:, :]),
        cache=True,
        fastmath=True,
    )
    def _numba_forcing(b0, b1, b2, x, f):
        """Compute forcing function.

        One thread per sample.

        """
        idx = cuda.grid(1)
        C = x.shape[0]
        T = x.shape[1]
        if idx >= C * T:
            return
        c = idx // T
        n = idx % T
        xn = x[c, n]
        xn1 = x[c, n - 1] if n >= 1 else 0.0
        xn2 = x[c, n - 2] if n >= 2 else 0.0
        f[c, n] = b0 * xn + b1 * xn1 + b2 * xn2

    @cuda.jit(
        void(float64, float64, float64[:, :], float64[:, :]),
        cache=True,
        fastmath=True,
    )
    def _numba_iir_recurrence(a1, a2, f, y):
        """Solve IIR recurrence.

        One thread per channel (sequential in time).

        """
        c = cuda.grid(1)
        if c >= f.shape[0]:
            return
        T = f.shape[1]
        ym1 = 0.0
        ym2 = 0.0
        for n in range(T):
            yn = f[c, n] - a1 * ym1 - a2 * ym2
            y[c, n] = yn
            ym2 = ym1
            ym1 = yn

    return _numba_forcing, _numba_iir_recurrence


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
def test_iir_numba(benchmark, duration, channels):
    if not numba_available():
        pytest.skip("numba not available")

    lfilter_jit = _get_numba_iir_lfilter()
    signal = create_signal_numpy(channels, duration).astype(np.float64)
    filters_b, filters_a = _pack_scipy_iir_coefficients()

    # Warmup (trigger JIT compilation)
    lfilter_jit(signal, filters_b, filters_a)

    benchmark.pedantic(
        lambda: lfilter_jit(signal, filters_b, filters_a),
        rounds=REP,
        warmup_rounds=WARMUP,
    )


@pytest.mark.benchmark(group="iir")
@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("duration", DURATIONS)
def test_iir_numba_cuda(cuda_sync_benchmark, duration, channels):
    if not numba_cuda_available():
        pytest.skip("numba CUDA not available")

    from numba import cuda as numba_cuda

    forcing_kernel, recurrence_kernel = _get_numba_iir_cuda_kernels()
    signal = create_signal_numpy(channels, duration)

    # Pre-compute filter coefficients (same as scipy)
    filters = SCIPY_FILTERS

    # Transfer signal to GPU
    d_x = numba_cuda.to_device(signal.astype(np.float64))
    d_y = numba_cuda.device_array_like(d_x)
    d_f = numba_cuda.device_array_like(d_x)
    d_tmp = numba_cuda.device_array_like(d_x)

    # Grid for forcing kernel: one thread per sample (embarrassingly parallel)
    total_elements = channels * signal.shape[1]
    forcing_tpb = 256
    forcing_blocks = (total_elements + forcing_tpb - 1) // forcing_tpb

    # Grid for recurrence kernel: one thread per channel
    recurrence_tpb = min(256, channels)
    recurrence_blocks = (channels + recurrence_tpb - 1) // recurrence_tpb

    def run():
        src = d_x
        for i, (b, a) in enumerate(filters):
            dst = d_y if i % 2 == 0 else d_tmp
            b0, b1, b2 = float(b[0]), float(b[1]), float(b[2])
            a1, a2 = float(a[1]), float(a[2])
            forcing_kernel[forcing_blocks, forcing_tpb](b0, b1, b2, src, d_f)
            recurrence_kernel[recurrence_blocks, recurrence_tpb](a1, a2, d_f, dst)
            src = dst
        numba_cuda.synchronize()

    # Warmup (compile kernels + warm caches)
    for _ in range(3):
        run()

    cuda_sync_benchmark.pedantic(run, rounds=REP, warmup_rounds=WARMUP)
