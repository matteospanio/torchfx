"""Torchaudio comparator benchmarks.

Pairs against the industry-standard PyTorch audio library so the IS²
2026 paper can show TorchFX vs torchaudio on identical workloads.

Three head-to-heads at the moment:

* **Single biquad** — ``torchaudio.functional.biquad`` vs TorchFX
  ``Biquad`` (DF1, native C++ kernel). Both consume the same
  ``(b0, b1, b2, a0, a1, a2)`` Audio-EQ-Cookbook coefficients.
* **High-order IIR cascade** — torchaudio has no SOS cascade primitive,
  so the comparator chains ``functional.biquad`` K times in Python.
  Pairs against TorchFX's ``FusedSOSCascade``, which collapses K
  sections into one kernel call.
* **FIR via FFT convolution** — ``torchaudio.functional.fftconvolve``
  vs TorchFX ``FIR(conv_mode='fft')``.

The comparator is fair: same input tensor, same dtype, same device,
both warmed up, same number of measured rounds. ``torchaudio`` is
treated as optional — if missing the whole module skips with a clear
reason.

Run
---
``uv run pytest benchmarks/test_torchaudio_bench.py --benchmark-enable``

"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from scipy.signal import butter

from torchfx.filter import BiquadLPF, FusedSOSCascade
from torchfx.filter._design import design_butterworth_sos
from torchfx.filter.fir import FIR

from .conftest import REP, SAMPLE_RATE, WARMUP, create_signal_torch

# ── Optional dependency guard ─────────────────────────────────────────────────


def _torchaudio_available() -> bool:
    try:
        import torchaudio  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _torchaudio_available(),
    reason="torchaudio not installed (optional comparator dependency)",
)


# Lazy import — only fires when the skip condition above passes.
def _ta() -> Any:
    import torchaudio

    return torchaudio


# ── Benchmark axes ────────────────────────────────────────────────────────────

# Keep aligned with the existing biquad/IIR benchmarks so headlines plot
# directly against them. Two channel counts is enough for a comparator —
# the existing IIR benchmark already characterises channel scaling.
DURATIONS_BIQUAD = [1.0, 5.0]
DURATIONS_CASCADE = [1.0, 5.0, 30.0]
DURATIONS_FIR = [1.0, 10.0]
CHANNELS = [1, 2]
CASCADE_ORDERS = [4, 8, 16]  # Butterworth order → K=order/2 SOS sections
FIR_KERNELS = [127, 511, 2047]


# ── Helper: design a Butterworth biquad (single SOS section) ──────────────────


def _butter_biquad_coeffs() -> tuple[float, float, float, float, float, float]:
    """RBJ-cookbook coefficients for a 2nd-order Butterworth LPF at 5 kHz.

    Designed via scipy (the same backend both libraries used historically) and split
    into (b0, b1, b2, 1, a1, a2) form. a0 is normalised to 1.

    """
    b, a = butter(2, 5000.0, btype="low", fs=SAMPLE_RATE)
    b0, b1, b2 = float(b[0]), float(b[1]), float(b[2])
    a0, a1, a2 = float(a[0]), float(a[1]), float(a[2])
    return (b0, b1, b2, a0, a1, a2)


# ── 1. Single biquad ──────────────────────────────────────────────────────────


@pytest.mark.benchmark(group="biquad-vs-torchaudio")
@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("duration", DURATIONS_BIQUAD)
def test_biquad_torchaudio(benchmark: Any, duration: float, channels: int) -> None:
    """torchaudio.functional.biquad — single 2nd-order section."""
    ta = _ta()
    x = create_signal_torch(channels, duration)
    b0, b1, b2, a0, a1, a2 = _butter_biquad_coeffs()

    def run() -> torch.Tensor:
        return ta.functional.biquad(x, b0, b1, b2, a0, a1, a2)  # type: ignore[no-any-return]

    benchmark.pedantic(run, rounds=REP, warmup_rounds=WARMUP)


@pytest.mark.benchmark(group="biquad-vs-torchaudio")
@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("duration", DURATIONS_BIQUAD)
def test_biquad_torchfx(benchmark: Any, duration: float, channels: int) -> None:
    """TorchFX Biquad with identical coefficients."""
    x = create_signal_torch(channels, duration)
    biq = BiquadLPF(cutoff=5000.0, q=1.0 / np.sqrt(2.0), fs=SAMPLE_RATE)
    biq.compute_coefficients()

    def run() -> torch.Tensor:
        return biq(x)  # type: ignore[no-any-return]

    benchmark.pedantic(run, rounds=REP, warmup_rounds=WARMUP)


# ── 2. High-order IIR cascade ────────────────────────────────────────────────


def _butter_sos_sections(order: int) -> list[tuple[float, ...]]:
    """Pre-design a high-order Butterworth LPF as a list of biquad tuples.

    Used by the torchaudio cascade benchmark — torchaudio has no SOS
    primitive so the cascade is a Python loop over its ``biquad``.

    """
    # Use TorchFX's native design (same numerics scipy produces) so the
    # two paths are numerically equivalent up to FP rounding.
    # cutoff_norm = cutoff_hz / (fs/2): 5000 / 22050 ≈ 0.2268.
    cutoff_norm = 5000.0 / (SAMPLE_RATE / 2)
    sos = design_butterworth_sos(order=order, cutoff_norm=cutoff_norm, btype="low")
    # design_*: shape [K, 6] = [b0, b1, b2, a0, a1, a2]; convert to tuples
    return [tuple(float(v) for v in row) for row in sos.tolist()]


@pytest.mark.benchmark(group="iir-cascade-vs-torchaudio")
@pytest.mark.parametrize("order", CASCADE_ORDERS)
@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("duration", DURATIONS_CASCADE)
def test_iir_cascade_torchaudio(benchmark: Any, duration: float, channels: int, order: int) -> None:
    """High-order IIR via repeated torchaudio.functional.biquad calls.

    torchaudio has no SOS-cascade primitive, so the realistic baseline is a Python loop
    over the K = order/2 second-order sections. Each section is a separate kernel
    dispatch.

    """
    ta = _ta()
    x = create_signal_torch(channels, duration)
    sections = _butter_sos_sections(order)

    def run() -> torch.Tensor:
        y = x
        for b0, b1, b2, a0, a1, a2 in sections:
            y = ta.functional.biquad(y, b0, b1, b2, a0, a1, a2)
        return y  # type: ignore[no-any-return]

    benchmark.pedantic(run, rounds=REP, warmup_rounds=WARMUP)


@pytest.mark.benchmark(group="iir-cascade-vs-torchaudio")
@pytest.mark.parametrize("order", CASCADE_ORDERS)
@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("duration", DURATIONS_CASCADE)
def test_iir_cascade_torchfx_fused(
    benchmark: Any, duration: float, channels: int, order: int
) -> None:
    """High-order IIR via TorchFX FusedSOSCascade (single kernel call)."""
    x = create_signal_torch(channels, duration)
    # Pre-design biquads with the same per-section coefficients, then
    # cascade-fuse them. Importantly we use ``FusedSOSCascade`` so this
    # benchmark targets the kernel-fusion path the paper claims wins.
    # ``Biquad`` itself is abstract, so we use ``BiquadLPF`` as a concrete
    # subclass and overwrite its SOS row.
    biquads = []
    for b0, b1, b2, _a0, a1, a2 in _butter_sos_sections(order):
        b = BiquadLPF(cutoff=5000.0, q=0.707, fs=SAMPLE_RATE)
        # Override the lazy coefficients with our pre-designed values.
        b._sos = torch.tensor([[b0, b1, b2, 1.0, a1, a2]], dtype=torch.float64)
        b._a1 = a1
        b._a2 = a2
        biquads.append(b)
    fused = FusedSOSCascade(*biquads)

    def run() -> torch.Tensor:
        return fused(x)  # type: ignore[no-any-return]

    benchmark.pedantic(run, rounds=REP, warmup_rounds=WARMUP)


# ── 3. FIR via FFT convolution ────────────────────────────────────────────────


@pytest.mark.benchmark(group="fir-fft-vs-torchaudio")
@pytest.mark.parametrize("kernel_size", FIR_KERNELS)
@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("duration", DURATIONS_FIR)
def test_fir_fft_torchaudio(
    benchmark: Any, duration: float, channels: int, kernel_size: int
) -> None:
    """torchaudio.functional.fftconvolve as FIR baseline."""
    ta = _ta()
    x = create_signal_torch(channels, duration)
    # Simple sinc-windowed lowpass as the kernel — same numerics on both sides.
    kernel = torch.hann_window(kernel_size, dtype=torch.float32)
    kernel = kernel / kernel.sum()
    # torchaudio.functional.fftconvolve requires kernel.ndim == x.ndim;
    # broadcast the 1-D kernel to (1, K) so the same filter is applied to
    # every channel.
    kernel_2d = kernel.unsqueeze(0)

    def run() -> torch.Tensor:
        return ta.functional.fftconvolve(x, kernel_2d, mode="same")  # type: ignore[no-any-return]

    benchmark.pedantic(run, rounds=REP, warmup_rounds=WARMUP)


@pytest.mark.benchmark(group="fir-fft-vs-torchaudio")
@pytest.mark.parametrize("kernel_size", FIR_KERNELS)
@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("duration", DURATIONS_FIR)
def test_fir_fft_torchfx(benchmark: Any, duration: float, channels: int, kernel_size: int) -> None:
    """TorchFX FIR(conv_mode='fft') with the same kernel."""
    x = create_signal_torch(channels, duration)
    kernel = torch.hann_window(kernel_size, dtype=torch.float32)
    kernel = kernel / kernel.sum()
    # FIR(b, conv_mode=...): b is the impulse-response taps. ``fs`` is
    # not a constructor argument — FIR is stateless and doesn't need it.
    fir = FIR(kernel.tolist(), conv_mode="fft")

    def run() -> torch.Tensor:
        return fir(x)  # type: ignore[no-any-return]

    benchmark.pedantic(run, rounds=REP, warmup_rounds=WARMUP)
