"""FP32-vs-FP64 numerical validation for the native IIR kernels (Roadmap Epic B3).

The CUDA (and CPU) SOS kernels are templated on ``scalar_t``: a float32 input runs
the FP32 path, a float64 input runs FP64. These tests verify that:

1. the FP64 path matches ``scipy.signal.sosfilt`` to ~double precision, and
2. the FP32 path is *correct to float32 precision* — i.e. it tracks the FP64/scipy
   reference within a documented bound, rather than being silently wrong.

The per-(filter, order) errors are collected and printed so the boundary between
"FP32-safe" and "needs FP64" designs is explicit and tracked over time. Run on CPU
always; on CUDA when a GPU is present (that is the path Epic B exists to validate).

"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.signal as sps
import torch

from torchfx.filter import Chebyshev1, LoButterworth, LoChebyshev1  # noqa: F401
from torchfx.filter.iir import IIR

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

# (label, scipy-design callable -> sos, torchfx filter factory) for matched designs.
# fs is fixed; cutoff well inside the band so designs are well-conditioned.
FS = 48_000
CUTOFF = 4_000.0


def _scipy_butter_sos(order: int) -> np.ndarray:
    return sps.butter(order, CUTOFF / (0.5 * FS), btype="low", output="sos")


def _scipy_cheby1_sos(order: int) -> np.ndarray:
    return sps.cheby1(order, 0.5, CUTOFF / (0.5 * FS), btype="low", output="sos")


CASES = [
    ("butterworth", _scipy_butter_sos, lambda o: LoButterworth(CUTOFF, order=o, fs=FS)),
    ("cheby1", _scipy_cheby1_sos, lambda o: LoChebyshev1(CUTOFF, ripple=0.5, order=o, fs=FS)),
]
ORDERS = [2, 4, 8, 16]


def _signal(n: int = 96_000) -> torch.Tensor:
    g = torch.Generator().manual_seed(7)
    return torch.randn(2, n, generator=g, dtype=torch.float64)


def _run(filt: IIR, x: torch.Tensor) -> torch.Tensor:
    """Run a torchfx IIR once (stateless from zero initial state) and return CPU f64."""
    return filt(x).detach().to(device="cpu", dtype=torch.float64)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("name,scipy_design,make", CASES, ids=[c[0] for c in CASES])
def test_fp64_path_matches_scipy(device, order, name, scipy_design, make):
    """FP64 native path must match scipy.sosfilt to ~double precision."""
    x = _signal().to(device)
    sos = scipy_design(order)
    ref = sps.sosfilt(sos, x.cpu().numpy(), axis=-1)

    y64 = _run(make(order), x.to(torch.float64)).numpy()

    np.testing.assert_allclose(
        y64, ref, rtol=1e-7, atol=1e-9, err_msg=f"{name} order={order} on {device}"
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("name,scipy_design,make", CASES, ids=[c[0] for c in CASES])
def test_fp32_path_tracks_reference(device, order, name, scipy_design, make, record_property):
    """FP32 native path must track the FP64 reference within float32 precision.

    A real logic bug shows up at O(0.1+); float32 rounding over ~10^5 samples and a few
    cascade sections lands well under 1e-2 absolute for these designs.

    """
    x = _signal().to(device)
    sos = scipy_design(order)
    ref = sps.sosfilt(sos, x.cpu().numpy(), axis=-1)

    y32 = _run(make(order), x.to(torch.float32)).numpy()

    abs_err = np.abs(y32 - ref)
    max_abs = float(abs_err.max())
    # RMS error relative to signal RMS: a global, near-zero-robust precision metric
    # (per-sample relative error blows up next to zero crossings and is meaningless).
    rel_rms = float(np.sqrt(np.mean(abs_err**2)) / np.sqrt(np.mean(ref**2)))

    record_property(f"{name}_o{order}_{device}_max_abs", max_abs)
    record_property(f"{name}_o{order}_{device}_rel_rms", rel_rms)

    # FP32 is correct-to-precision, not garbage. max_abs catches a localized logic
    # bug (O(0.1+)); rel_rms catches a global precision regression. Both are loose
    # enough to absorb honest float32 accumulation through an order-16 cascade.
    assert max_abs < 1e-2, f"{name} order={order} on {device}: max_abs={max_abs:.3e}"
    assert rel_rms < 2e-3, f"{name} order={order} on {device}: rel_rms={rel_rms:.3e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("order", [2, 8])
def test_cuda_fp32_matches_cuda_fp64(order):
    """On GPU, the FP32 and FP64 kernels must agree to float32 precision."""
    x = _signal().cuda()
    y64 = _run(LoButterworth(CUTOFF, order=order, fs=FS), x.to(torch.float64)).numpy()
    y32 = _run(LoButterworth(CUTOFF, order=order, fs=FS), x.to(torch.float32)).numpy()
    np.testing.assert_allclose(y32, y64, rtol=1e-3, atol=1e-4)
