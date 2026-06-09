"""Cross-channel SIMD CPU SOS path equivalence (issue #24).

``TORCHFX_FORCE_SIMD=1`` engages the cache-blocked cross-channel SIMD kernel at any
channel count; ``TORCHFX_NO_SIMD=1`` forces the scalar path. This checks the SIMD path
reproduces the scalar kernel and ``scipy.signal.sosfilt`` across channel counts
(including partial SIMD groups), section counts, and dtypes, and that it carries
streaming state correctly across chunks.

"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from scipy.signal import butter, sosfilt


def _run(x, sos, simd: bool):
    from torchfx._ops import parallel_iir_forward

    keep = {k: os.environ.get(k) for k in ("TORCHFX_NO_SIMD", "TORCHFX_FORCE_SIMD")}
    os.environ.pop("TORCHFX_NO_SIMD", None)
    os.environ.pop("TORCHFX_FORCE_SIMD", None)
    os.environ["TORCHFX_FORCE_SIMD" if simd else "TORCHFX_NO_SIMD"] = "1"
    try:
        y, sx, sy = parallel_iir_forward(x, sos, None, None, sos_cpu=sos)
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return y, sx, sy


def _cascade(k: int) -> np.ndarray:
    return np.concatenate([butter(2, c, output="sos") for c in np.linspace(0.1, 0.4, k)], axis=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("k", [1, 3, 6])
@pytest.mark.parametrize("c", [1, 2, 3, 4, 5, 7, 8, 9, 16, 17])
def test_simd_matches_scalar_and_scipy(dtype, k, c):
    t = 2000
    sos_np = _cascade(k)
    g = torch.Generator().manual_seed(k * 100 + c)
    x = torch.randn(c, t, generator=g, dtype=dtype)
    sos = torch.tensor(sos_np, dtype=dtype)

    scalar, _, _ = _run(x, sos, simd=False)
    simd, _, _ = _run(x, sos, simd=True)

    # Same math, same order -> tight; both within a float bound of the scipy reference.
    tol = {"rtol": 1e-5, "atol": 1e-6} if dtype == torch.float32 else {"rtol": 1e-12, "atol": 1e-13}
    torch.testing.assert_close(simd, scalar, **tol)
    ref = torch.tensor(sosfilt(sos_np, x.double().numpy(), axis=-1), dtype=dtype)
    stol = {"rtol": 2e-3, "atol": 2e-4} if dtype == torch.float32 else {"rtol": 1e-9, "atol": 1e-10}
    torch.testing.assert_close(simd.cpu(), ref, **stol)


def test_simd_streaming_state_continuity():
    """Chunked streaming through the SIMD path == whole-signal processing."""
    from torchfx._ops import parallel_iir_forward

    c, k, t = 16, 3, 4096  # C > typical core count so the SIMD path is exercised
    sos_np = _cascade(k)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(c, t, dtype=torch.float64, generator=g)
    sos = torch.tensor(sos_np, dtype=torch.float64)

    whole, _, _ = _run(x, sos, simd=True)

    prev = os.environ.get("TORCHFX_FORCE_SIMD")
    os.environ["TORCHFX_FORCE_SIMD"] = "1"
    try:
        sx = sy = None
        outs = []
        for chunk in x.split(700, dim=1):
            y, sx, sy = parallel_iir_forward(chunk, sos, sx, sy, sos_cpu=sos)
            outs.append(y)
        streamed = torch.cat(outs, dim=1)
    finally:
        if prev is None:
            os.environ.pop("TORCHFX_FORCE_SIMD", None)
        else:
            os.environ["TORCHFX_FORCE_SIMD"] = prev

    torch.testing.assert_close(streamed, whole, rtol=1e-12, atol=1e-13)
