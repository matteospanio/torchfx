"""Cross-channel SIMD CPU SOS path correctness (Roadmap Epic F1).

The SIMD kernel (used at high channel counts) must match both scipy and the scalar
kernel exactly-to-precision. C >= 16 selects the SIMD path by default.

"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.signal as sps
import torch

from torchfx.filter import LoButterworth

FS = 48000


@pytest.mark.parametrize("order", [2, 4, 8])
@pytest.mark.parametrize("channels", [16, 32])
def test_cpu_simd_matches_scipy(channels, order):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(channels, 4096, generator=g, dtype=torch.float64)  # SIMD path
    sos = sps.butter(order, 4000 / (0.5 * FS), btype="low", output="sos")
    ref = sps.sosfilt(sos, x.numpy(), axis=-1)

    y = LoButterworth(4000, order=order, fs=FS)(x).numpy()

    np.testing.assert_allclose(y, ref, rtol=1e-9, atol=1e-11)


def test_cpu_simd_matches_scalar_path():
    """The SIMD path (32 ch) and the scalar path (8 ch) agree on identical channels."""
    g = torch.Generator().manual_seed(1)
    base = torch.randn(8, 4096, generator=g, dtype=torch.float64)

    x_simd = base.repeat(4, 1)  # 32 channels -> SIMD path
    y_simd = LoButterworth(4000, order=8, fs=FS)(x_simd)
    y_scalar = LoButterworth(4000, order=8, fs=FS)(base)  # 8 channels -> scalar path

    torch.testing.assert_close(y_simd[:8], y_scalar, rtol=1e-10, atol=1e-12)
