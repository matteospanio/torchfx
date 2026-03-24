"""Shared fixtures, constants, and helpers for torchfx benchmarks."""

from __future__ import annotations

import numpy as np
import pytest
import torch

# ── Shared constants ──────────────────────────────────────────────────────────

SAMPLE_RATE = 44100

# Filter benchmark grid
DURATIONS = [1, 5, 30, 60, 120, 300]  # seconds
CHANNELS = [1, 2, 4, 8, 12]

# Biquad-specific (shorter signals, fewer channels)
BIQUAD_DURATIONS = [0.1, 1.0, 5.0, 30.0]
BIQUAD_CHANNELS = [1, 2]

# FFT convolution benchmark grid
FFTCONV_DURATIONS = [1, 10, 60]
FFTCONV_KERNELS = [64, 128, 256, 512, 1024]

# Benchmark repetitions
REP = 30
WARMUP = 5

DEVICES = ["cpu", "cuda"]


# ── Helpers ───────────────────────────────────────────────────────────────────


def numba_available() -> bool:
    """Check if numba is available."""
    try:
        import numba  # noqa: F401

        return True
    except ImportError:
        return False


def numba_cuda_available() -> bool:
    """Check if numba CUDA is available (import + device check)."""
    try:
        from numba import cuda

        return cuda.is_available()
    except (ImportError, Exception):
        return False


def pytest_configure(config):  # noqa: ARG001
    """Print native extension status at the start of benchmark runs."""
    from torchfx._ops import is_native_available

    available = is_native_available()
    cuda = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda else "N/A"
    print(f"\n[torchfx] native extension: {'YES' if available else 'NO (using slow fallback!)'}")
    print(f"[torchfx] CUDA available: {cuda} ({gpu_name})")
    print(f"[torchfx] numba CUDA available: {numba_cuda_available()}")


# ── Signal generators ─────────────────────────────────────────────────────────


def create_signal_torch(channels: int, duration_sec: float, device: str = "cpu") -> torch.Tensor:
    """Create a normalized random test signal as a torch Tensor."""
    T = int(SAMPLE_RATE * duration_sec)
    x = torch.randn(channels, T, device=device, dtype=torch.float32)
    return x / x.abs().max()


def create_signal_numpy(channels: int, duration_sec: float) -> np.ndarray:
    """Create a normalized random test signal as a NumPy array."""
    T = int(SAMPLE_RATE * duration_sec)
    signal = np.random.randn(channels, T).astype(np.float32)
    signal /= np.max(np.abs(signal), axis=1, keepdims=True)
    return signal


# ── CUDA sync benchmark fixture ──────────────────────────────────────────────


class CudaSyncBenchmark:
    """Wraps pytest-benchmark with CUDA synchronization."""

    def __init__(self, bench):
        self._bench = bench

    def __call__(self, func, *args, **kwargs):
        if torch.cuda.is_available():

            def wrapper(*a, **kw):
                torch.cuda.synchronize()
                result = func(*a, **kw)
                torch.cuda.synchronize()
                return result

            self._bench(wrapper, *args, **kwargs)
        else:
            self._bench(func, *args, **kwargs)

    def pedantic(self, func, args=(), kwargs=None, **pedantic_kwargs):
        kwargs = kwargs or {}
        if torch.cuda.is_available():

            def wrapper(*a, **kw):
                torch.cuda.synchronize()
                result = func(*a, **kw)
                torch.cuda.synchronize()
                return result

            self._bench.pedantic(wrapper, args=args, kwargs=kwargs, **pedantic_kwargs)
        else:
            self._bench.pedantic(func, args=args, kwargs=kwargs, **pedantic_kwargs)


@pytest.fixture
def cuda_sync_benchmark(benchmark):
    """Benchmark fixture with CUDA synchronization before/after each iteration."""
    return CudaSyncBenchmark(benchmark)
