"""Representative torchfx workloads used for profiling and cross-environment
benchmarking.

Each scenario is a zero-arg callable returning a torch.nn.Module-like pipeline and
a pre-built input tensor. The caller decides what to measure (scalene, torch.profiler,
pytest-benchmark) and on which device. The same scenarios are used both locally on
CPU and on the SLURM cluster on CUDA so results are comparable.

Scenarios:
    offline_filter_chain_cpu : 10-min stereo @ 48 kHz, Butterworth | Chebyshev1 | Gain
    offline_batch_gpu        : (B=32, C=2, T=480000) through a 6-filter chain
    realtime_chunks_cpu      : 512-sample chunks through BiquadBPF | Delay (CPU realtime)
    realtime_chunks_gpu      : 512-sample chunks through BiquadBPF | Delay | Reverb (GPU realtime)

The scenarios do NOT pick a device; the caller passes ``device=`` so the same code
path runs locally on CPU and on the cluster on CUDA.

"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn

from torchfx.effect import Delay, Gain, Reverb
from torchfx.filter import BiquadBPF, HiChebyshev1, LoButterworth
from torchfx.wave import Wave

SAMPLE_RATE = 48_000


@dataclass
class Scenario:
    """A single profiling workload.

    Attributes
    ----------
    name : str
        Short identifier used as filename prefix for profile artifacts.
    build : Callable[[str], nn.Module]
        Builds the filter chain on the given device.
    make_input : Callable[[str], torch.Tensor]
        Produces the input signal on the given device.
    description : str
        One-line human summary.

    """

    name: str
    build: Callable[[str], nn.Module]
    make_input: Callable[[str], torch.Tensor]
    description: str


# ── Offline: single long signal through a filter chain ───────────────────────


def _build_offline_chain(device: str) -> nn.Module:
    chain = nn.Sequential(
        LoButterworth(cutoff=4_000, order=8, fs=SAMPLE_RATE),
        HiChebyshev1(cutoff=80, order=6, ripple=0.5, fs=SAMPLE_RATE),
        Gain(gain=-3.0, gain_type="db"),
    )
    for m in chain:
        if hasattr(m, "compute_coefficients"):
            m.compute_coefficients()
    return chain.to(device)


def _make_offline_input(device: str) -> torch.Tensor:
    # 10 minutes stereo
    T = SAMPLE_RATE * 60 * 10
    return torch.randn(2, T, device=device, dtype=torch.float32) * 0.1


offline_filter_chain = Scenario(
    name="offline_filter_chain",
    build=_build_offline_chain,
    make_input=_make_offline_input,
    description="10-min stereo @ 48kHz | LoButter(8) | HiCheby1(6) | Gain",
)


# ── Offline batch: (B, C, T) through 6 filters (GPU-oriented) ────────────────


def _build_batch_chain(device: str) -> nn.Module:
    chain = nn.Sequential(
        LoButterworth(cutoff=6_000, order=4, fs=SAMPLE_RATE),
        LoButterworth(cutoff=3_000, order=4, fs=SAMPLE_RATE),
        HiChebyshev1(cutoff=60, order=4, ripple=0.5, fs=SAMPLE_RATE),
        HiChebyshev1(cutoff=120, order=4, ripple=0.5, fs=SAMPLE_RATE),
        BiquadBPF(cutoff=1_000, q=1.0, fs=SAMPLE_RATE),
        Gain(gain=-6.0, gain_type="db"),
    )
    for m in chain:
        if hasattr(m, "compute_coefficients"):
            m.compute_coefficients()
    return chain.to(device)


def _make_batch_input(device: str) -> torch.Tensor:
    # 10 s of stereo, batch of 32
    B, C, T = 32, 2, SAMPLE_RATE * 10
    return torch.randn(B, C, T, device=device, dtype=torch.float32) * 0.1


offline_batch_chain = Scenario(
    name="offline_batch_chain",
    build=_build_batch_chain,
    make_input=_make_batch_input,
    description="(B=32, C=2, T=480k) through 6-filter chain",
)


# ── Realtime: 512-sample chunks, stateful path ───────────────────────────────


def _build_realtime_cpu(device: str) -> nn.Module:
    chain = nn.Sequential(
        BiquadBPF(cutoff=1_000, q=1.2, fs=SAMPLE_RATE),
        Delay(delay_samples=int(SAMPLE_RATE * 0.25), feedback=0.3, mix=0.4, fs=SAMPLE_RATE),
    )
    for m in chain:
        if hasattr(m, "compute_coefficients"):
            m.compute_coefficients()
    return chain.to(device)


def _build_realtime_gpu(device: str) -> nn.Module:
    chain = nn.Sequential(
        BiquadBPF(cutoff=1_000, q=1.2, fs=SAMPLE_RATE),
        Delay(delay_samples=int(SAMPLE_RATE * 0.25), feedback=0.3, mix=0.4, fs=SAMPLE_RATE),
        Reverb(decay=0.5, mix=0.3, delay=int(SAMPLE_RATE * 0.05)),
    )
    for m in chain:
        if hasattr(m, "compute_coefficients"):
            m.compute_coefficients()
    return chain.to(device)


def _make_realtime_chunk(device: str) -> torch.Tensor:
    # 512-sample stereo chunk
    return torch.randn(2, 512, device=device, dtype=torch.float32) * 0.1


realtime_chunks_cpu = Scenario(
    name="realtime_chunks_cpu",
    build=_build_realtime_cpu,
    make_input=_make_realtime_chunk,
    description="512-sample stereo chunks | BiquadBPF | Delay (CPU-realtime)",
)

realtime_chunks_gpu = Scenario(
    name="realtime_chunks_gpu",
    build=_build_realtime_gpu,
    make_input=_make_realtime_chunk,
    description="512-sample stereo chunks | BiquadBPF | Delay | Reverb (GPU-realtime)",
)


ALL_SCENARIOS: list[Scenario] = [
    offline_filter_chain,
    offline_batch_chain,
    realtime_chunks_cpu,
    realtime_chunks_gpu,
]


__all__ = [
    "ALL_SCENARIOS",
    "SAMPLE_RATE",
    "Scenario",
    "Wave",
    "offline_batch_chain",
    "offline_filter_chain",
    "realtime_chunks_cpu",
    "realtime_chunks_gpu",
]
