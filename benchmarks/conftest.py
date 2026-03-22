"""Shared fixtures and helpers for torchfx benchmarks."""

from __future__ import annotations

import numpy as np
import pytest
import torch

SAMPLE_RATE = 44100


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
