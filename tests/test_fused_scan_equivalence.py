"""Equivalence of the fused per-section SOS path (Roadmap Epic 4.6 / paper C1).

``TORCHFX_FUSED_SCAN=1`` switches ``sos_forward_cuda`` to the fused path that folds
the FIR forcing into the scan (one kernel per section instead of forcing + scan).
This checks the fused path reproduces the 3-phase oracle (same float order, so it
should match very tightly) and ``scipy.signal.sosfilt``, across section count,
channels, dtype, and both the sequential (``T <= threshold``) and parallel
(``T > threshold``) branches — selected deterministically via ``threshold``.

"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from scipy.signal import butter, sosfilt

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# Force one branch or the other regardless of signal length.
FORCE_SEQ = 1 << 30  # T <= huge -> sequential kernel
FORCE_PAR = 0  # T <= 0 never true -> parallel scan


def _cascade_sos(k: int) -> np.ndarray:
    """K well-conditioned 2nd-order sections (varied cutoffs) -> [k, 6]."""
    cutoffs = np.linspace(0.08, 0.45, k)
    return np.concatenate([butter(2, float(c), output="sos") for c in cutoffs], axis=0)


def _run(x: torch.Tensor, sos: torch.Tensor, threshold: int, fused: bool):
    from torchfx._ops import parallel_iir_forward

    prev = os.environ.get("TORCHFX_FUSED_SCAN")
    os.environ["TORCHFX_FUSED_SCAN"] = "1" if fused else "0"
    try:
        y, _, _ = parallel_iir_forward(x, sos, None, None, sos_cpu=sos.cpu(), threshold=threshold)
    finally:
        if prev is None:
            os.environ.pop("TORCHFX_FUSED_SCAN", None)
        else:
            os.environ["TORCHFX_FUSED_SCAN"] = prev
    return y


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("k", [1, 3, 6])
@pytest.mark.parametrize("c", [1, 4])
@pytest.mark.parametrize("threshold", [FORCE_SEQ, FORCE_PAR])
def test_fused_matches_oracle_and_scipy(dtype, k, c, threshold):
    t = 3000  # spans both branches depending on `threshold`
    g = torch.Generator().manual_seed(k * 100 + c)
    x = torch.randn(c, t, generator=g, dtype=dtype)
    sos_np = _cascade_sos(k)
    sos = torch.tensor(sos_np, dtype=dtype, device="cuda")
    xc = x.cuda()

    oracle = _run(xc, sos, threshold, fused=False)
    fused = _run(xc, sos, threshold, fused=True)

    # Fused vs oracle: identical math in identical float order -> very tight.
    tol = {"rtol": 1e-5, "atol": 1e-6} if dtype == torch.float32 else {"rtol": 1e-12, "atol": 1e-13}
    torch.testing.assert_close(fused, oracle, **tol)

    # Both vs scipy reference (CPU float64), looser float32 bound.
    ref = torch.tensor(sosfilt(sos_np, x.double().numpy(), axis=-1), dtype=dtype)
    stol = {"rtol": 2e-3, "atol": 2e-4} if dtype == torch.float32 else {"rtol": 1e-9, "atol": 1e-10}
    torch.testing.assert_close(fused.cpu(), ref, **stol)
