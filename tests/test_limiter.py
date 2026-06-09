"""Tests for the look-ahead brick-wall Limiter (issue #31)."""

from __future__ import annotations

import pytest
import torch

from torchfx.effect import Limiter

FS = 48000


def _thr_lin(db: float) -> float:
    return 10 ** (db / 20)


# --------------------------------------------------------------------------- #
# Brick-wall guarantee: |y| never exceeds the ceiling
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("thr_db", [-1.0, -6.0, -12.0])
@pytest.mark.parametrize("scale", [0.5, 1.0, 3.0, 10.0])
def test_brickwall_never_exceeds_threshold(thr_db, scale):
    lim = Limiter(threshold=thr_db, lookahead=0.005, release=0.05, fs=FS)
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(2, 24000, generator=gen, dtype=torch.float64) * scale
    y = lim(x)
    assert y.abs().max().item() <= _thr_lin(thr_db) + 1e-9


def test_brickwall_on_transients():
    lim = Limiter(threshold=-6.0, lookahead=0.003, release=0.02, fs=FS)
    x = torch.full((1, 10000), 0.2, dtype=torch.float64)
    x[0, 5000] = 8.0  # single-sample spike far over full scale
    x[0, 2000:2100] = 3.0  # a loud burst
    y = lim(x)
    assert y.abs().max().item() <= _thr_lin(-6.0) + 1e-9


def test_brickwall_on_sine_over_full_scale():
    lim = Limiter(threshold=-1.0, lookahead=0.005, release=0.05, fs=FS)
    t = torch.arange(48000, dtype=torch.float64) / FS
    x = (3.0 * torch.sin(2 * torch.pi * 220 * t)).unsqueeze(0)
    y = lim(x)
    assert y.abs().max().item() <= _thr_lin(-1.0) + 1e-9


def test_lookahead_zero_still_brickwall():
    lim = Limiter(threshold=-3.0, lookahead=0.0, release=0.05, fs=FS)
    gen = torch.Generator().manual_seed(2)
    x = torch.randn(1, 8000, generator=gen, dtype=torch.float64) * 2.0
    y = lim(x)
    assert y.abs().max().item() <= _thr_lin(-3.0) + 1e-9


# --------------------------------------------------------------------------- #
# Transparency + look-ahead behaviour
# --------------------------------------------------------------------------- #
def test_below_threshold_passes_through():
    lim = Limiter(threshold=-1.0, lookahead=0.005, release=0.05, fs=FS)
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(1, 8000, generator=gen, dtype=torch.float64) * 0.1  # peaks well below ceiling
    assert x.abs().max().item() < _thr_lin(-1.0)
    torch.testing.assert_close(lim(x), x, rtol=1e-9, atol=1e-12)


def test_lookahead_reduces_gain_before_peak():
    L = round(0.005 * FS)  # 240
    spike_at = 2000
    base = 0.3  # below the -6 dB ceiling (~0.501)
    x = torch.full((1, 4000), base, dtype=torch.float64)
    x[0, spike_at] = 5.0
    y_la = Limiter(threshold=-6.0, lookahead=0.005, release=0.05, fs=FS)(x)
    y_no = Limiter(threshold=-6.0, lookahead=0.0, release=0.05, fs=FS)(x)
    # With look-ahead the gain is already pulling down within the window before the spike;
    # without it the pre-spike samples (below threshold) are untouched.
    assert y_la[0, spike_at - L // 2].abs().item() < 0.9 * base
    assert y_no[0, spike_at - L // 2].abs().item() == pytest.approx(base, rel=1e-9)


def test_release_recovers_after_peak():
    lim = Limiter(threshold=-6.0, lookahead=0.002, release=0.03, fs=FS)
    loud = torch.full((1, 2000), 4.0, dtype=torch.float64)
    quiet = torch.full((1, 16000), 0.2, dtype=torch.float64)  # below ceiling
    x = torch.cat([loud, quiet], dim=1)
    y = lim(x)
    g_tail = y[0, 2000:].abs() / x[0, 2000:].abs()
    assert g_tail[0].item() < 0.9  # still attenuated right after the loud part
    assert g_tail[-1].item() == pytest.approx(1.0, abs=1e-3)  # recovered (one-pole asymptote)


# --------------------------------------------------------------------------- #
# Shapes, dtypes, channels
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", [(4096,), (2, 4096), (3, 2, 4096)])
def test_shapes_preserved(shape):
    lim = Limiter(threshold=-1.0, fs=FS)
    x = torch.randn(*shape, dtype=torch.float64) * 2.0
    y = lim(x)
    assert y.shape == x.shape
    assert y.abs().max().item() <= _thr_lin(-1.0) + 1e-9


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_preserved_and_brickwall(dtype):
    lim = Limiter(threshold=-2.0, fs=FS)
    x = torch.randn(2, 8000, dtype=dtype) * 2.0
    y = lim(x)
    assert y.dtype == dtype
    eps = 1e-5 if dtype == torch.float32 else 1e-9
    assert y.abs().max().item() <= _thr_lin(-2.0) + eps


def test_identical_channels_identical_output():
    lim = Limiter(threshold=-3.0, fs=FS)
    gen = torch.Generator().manual_seed(4)
    mono = torch.randn(1, 8000, generator=gen, dtype=torch.float64) * 2.0
    y = lim(mono.repeat(2, 1))
    torch.testing.assert_close(y[0], y[1])


# --------------------------------------------------------------------------- #
# Lazy fs + validation
# --------------------------------------------------------------------------- #
def test_fs_required():
    with pytest.raises(ValueError, match="Sample rate"):
        Limiter(threshold=-1.0)(torch.randn(1, 1000))


def test_fs_recompute_on_change():
    lim = Limiter(threshold=-1.0, fs=FS)
    x = torch.randn(1, 4000, dtype=torch.float64) * 2.0
    lim(x)
    lim.fs = 96000
    lim(x)
    assert lim._last_fs == 96000
    assert lim._lookahead_samples == round(0.005 * 96000)


@pytest.mark.parametrize("kwargs", [{"lookahead": -1.0}, {"release": -1.0}, {"fs": 0}])
def test_invalid_params_raise(kwargs):
    with pytest.raises(ValueError):
        Limiter(**kwargs)


# --------------------------------------------------------------------------- #
# CPU / CUDA parity
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("lookahead", [0.0, 0.005])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cpu_cuda_parity(lookahead, dtype):
    lim_cpu = Limiter(threshold=-3.0, lookahead=lookahead, release=0.05, fs=FS)
    lim_cuda = Limiter(threshold=-3.0, lookahead=lookahead, release=0.05, fs=FS)
    gen = torch.Generator().manual_seed(7)
    x = torch.randn(8, 16000, generator=gen, dtype=dtype) * 2.0
    y_cpu = lim_cpu(x)
    y_cuda = lim_cuda(x.cuda()).cpu()
    tol = {"rtol": 1e-4, "atol": 1e-5} if dtype == torch.float32 else {"rtol": 1e-9, "atol": 1e-10}
    torch.testing.assert_close(y_cpu, y_cuda, **tol)
