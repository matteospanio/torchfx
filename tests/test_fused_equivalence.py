"""Fused-vs-unfused equivalence gate (Roadmap Epic A4).

Locks in that a ``FusedSOSCascade`` is observationally identical to the
equivalent unfused ``nn.Sequential`` of stateful filters, in both batch and
chunked-streaming modes. This is the permanent regression gate that every later
kernel change (FP32 templating, single-kernel fusion, single-pass scan) must
keep green.

"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from torchfx.filter import HiButterworth, LoButterworth
from torchfx.filter.biquad import BiquadLPF
from torchfx.filter.fused import FusedSOSCascade


def _make_chain(fs: int) -> list[nn.Module]:
    """Three independent, freshly-designed mixed IIR/biquad sections."""
    return [
        HiButterworth(cutoff=80, order=2, fs=fs),
        LoButterworth(cutoff=4000, order=4, fs=fs),
        BiquadLPF(cutoff=2000, q=0.707, fs=fs),
    ]


def _tol(dtype: torch.dtype) -> dict[str, float]:
    return {"rtol": 1e-4, "atol": 1e-5} if dtype == torch.float32 else {"rtol": 1e-9, "atol": 1e-11}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("channels", [1, 2, 8])
def test_fused_matches_unfused_batch(dtype, channels):
    fs = 48000
    g = torch.Generator().manual_seed(0)
    x = torch.randn(channels, 4096, generator=g).to(dtype)

    unfused = nn.Sequential(*_make_chain(fs))
    fused = FusedSOSCascade(*_make_chain(fs))

    out_unfused = unfused(x.clone())
    out_fused = fused(x.clone())

    torch.testing.assert_close(out_fused, out_unfused, **_tol(dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("channels", [1, 2, 8])
def test_fused_matches_unfused_chunked(dtype, channels):
    """Chunked streaming: fused == unfused == single-pass, with state continuity."""
    fs = 48000
    g = torch.Generator().manual_seed(1)
    x = torch.randn(channels, 4096, generator=g).to(dtype)

    unfused = nn.Sequential(*_make_chain(fs))
    fused = FusedSOSCascade(*_make_chain(fs))
    fused_single = FusedSOSCascade(*_make_chain(fs))

    chunk = 512
    outs_unfused, outs_fused = [], []
    for i in range(0, x.shape[1], chunk):
        seg = x[:, i : i + chunk]
        outs_unfused.append(unfused(seg.clone()))
        outs_fused.append(fused(seg.clone()))

    out_unfused = torch.cat(outs_unfused, dim=1)
    out_fused = torch.cat(outs_fused, dim=1)
    out_single = fused_single(x.clone())

    tol = _tol(dtype)
    # Chunk-to-chunk continuity of the fused kernel itself.
    torch.testing.assert_close(out_fused, out_single, **tol)
    # Fused cascade is observationally identical to the unfused sequential.
    torch.testing.assert_close(out_fused, out_unfused, **tol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("channels", [1, 8])
def test_fused_matches_unfused_chunked_cuda(channels):
    fs = 48000
    g = torch.Generator().manual_seed(2)
    x = torch.randn(channels, 4096, generator=g, dtype=torch.float64).cuda()

    unfused = nn.Sequential(*_make_chain(fs))
    fused = FusedSOSCascade(*_make_chain(fs))
    fused_single = FusedSOSCascade(*_make_chain(fs))

    chunk = 512
    outs_unfused, outs_fused = [], []
    for i in range(0, x.shape[1], chunk):
        seg = x[:, i : i + chunk]
        outs_unfused.append(unfused(seg.clone()))
        outs_fused.append(fused(seg.clone()))

    out_unfused = torch.cat(outs_unfused, dim=1)
    out_fused = torch.cat(outs_fused, dim=1)
    out_single = fused_single(x.clone())

    torch.testing.assert_close(out_fused, out_single, rtol=1e-9, atol=1e-11)
    torch.testing.assert_close(out_fused, out_unfused, rtol=1e-9, atol=1e-11)
