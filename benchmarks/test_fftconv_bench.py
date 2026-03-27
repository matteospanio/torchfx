"""FFT convolution vs direct conv1d benchmarks (CPU and CUDA)."""

from __future__ import annotations

import gc

import pytest
import torch
import torch.nn.functional as F

from torchfx.filter._fftconv import fft_conv1d

from .conftest import FFTCONV_DURATIONS, FFTCONV_KERNELS, REP, SAMPLE_RATE, WARMUP

CHANNELS = 2  # fixed stereo for FFT conv benchmarks


# ── CPU benchmarks ────────────────────────────────────────────────────────────


@pytest.mark.benchmark(group="fftconv-cpu")
@pytest.mark.parametrize("kernel_size", FFTCONV_KERNELS)
@pytest.mark.parametrize("duration", FFTCONV_DURATIONS)
def test_fftconv_cpu(benchmark, duration, kernel_size):
    T = SAMPLE_RATE * duration
    x = torch.randn(1, CHANNELS, T)
    w = torch.randn(1, 1, kernel_size)
    pad = kernel_size - 1

    benchmark.pedantic(
        lambda x=x, w=w, pad=pad: fft_conv1d(x, w, padding=(pad, 0)),
        rounds=REP,
        warmup_rounds=WARMUP,
    )

    del x, w
    gc.collect()


@pytest.mark.benchmark(group="fftconv-cpu")
@pytest.mark.parametrize("kernel_size", FFTCONV_KERNELS)
@pytest.mark.parametrize("duration", FFTCONV_DURATIONS)
def test_direct_conv1d_cpu(benchmark, duration, kernel_size):
    T = SAMPLE_RATE * duration
    x = torch.randn(1, CHANNELS, T)
    w = torch.randn(CHANNELS, 1, kernel_size)
    pad = kernel_size - 1

    benchmark.pedantic(
        lambda x=x, w=w, pad=pad: F.conv1d(F.pad(x, (pad, 0)), w, groups=CHANNELS),
        rounds=REP,
        warmup_rounds=WARMUP,
    )

    del x, w
    gc.collect()


# ── CUDA benchmarks ──────────────────────────────────────────────────────────


@pytest.mark.benchmark(group="fftconv-cuda")
@pytest.mark.parametrize("kernel_size", FFTCONV_KERNELS)
@pytest.mark.parametrize("duration", FFTCONV_DURATIONS)
def test_fftconv_cuda(cuda_sync_benchmark, duration, kernel_size):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    T = SAMPLE_RATE * duration
    x = torch.randn(1, CHANNELS, T, device="cuda")
    w = torch.randn(1, 1, kernel_size, device="cuda")
    pad = kernel_size - 1

    cuda_sync_benchmark.pedantic(
        lambda x=x, w=w, pad=pad: fft_conv1d(x, w, padding=(pad, 0)),
        rounds=REP,
        warmup_rounds=WARMUP,
    )

    del x, w
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.benchmark(group="fftconv-cuda")
@pytest.mark.parametrize("kernel_size", FFTCONV_KERNELS)
@pytest.mark.parametrize("duration", FFTCONV_DURATIONS)
def test_direct_conv1d_cuda(cuda_sync_benchmark, duration, kernel_size):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    T = SAMPLE_RATE * duration
    x = torch.randn(1, CHANNELS, T, device="cuda")
    w = torch.randn(CHANNELS, 1, kernel_size, device="cuda")
    pad = kernel_size - 1

    cuda_sync_benchmark.pedantic(
        lambda x=x, w=w, pad=pad: F.conv1d(F.pad(x, (pad, 0)), w, groups=CHANNELS),
        rounds=REP,
        warmup_rounds=WARMUP,
    )

    del x, w
    gc.collect()
    torch.cuda.empty_cache()
