"""Tests for the Freeverb-style Reverb effect (issue #18)."""

from __future__ import annotations

import pytest
import torch

from torchfx.effect import Reverb

FS = 48000


def _impulse(n: int = FS, c: int = 1, dtype=torch.float64) -> torch.Tensor:
    x = torch.zeros(c, n, dtype=dtype)
    x[:, 0] = 1.0
    return x


# --------------------------------------------------------------------------- #
# Stability + reverb tail
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("room", [0.0, 0.5, 0.95])
@pytest.mark.parametrize("damp", [0.0, 0.5, 1.0])
def test_stable_and_finite(room, damp):
    rev = Reverb(room_size=room, damping=damp, mix=1.0, fs=FS)
    y = rev(_impulse())
    assert torch.isfinite(y).all()
    assert y.abs().max() < 10.0  # bounded (no runaway feedback)


def test_impulse_produces_decaying_tail():
    rev = Reverb(room_size=0.85, damping=0.3, mix=1.0, fs=FS)
    y = rev(_impulse())[0]
    assert (y[2000:] ** 2).sum() > 0  # there is a tail
    assert y[40000:].abs().max() < y[:4000].abs().max()  # and it decays


def test_larger_room_has_longer_tail():
    small = Reverb(room_size=0.2, damping=0.2, mix=1.0, fs=FS)(_impulse())[0]
    large = Reverb(room_size=0.95, damping=0.2, mix=1.0, fs=FS)(_impulse())[0]
    # Energy in the late tail grows with room size (longer decay).
    assert (large[24000:] ** 2).sum() > (small[24000:] ** 2).sum()


def test_damping_changes_output():
    bright = Reverb(room_size=0.8, damping=0.0, mix=1.0, fs=FS)(_impulse())
    dark = Reverb(room_size=0.8, damping=1.0, mix=1.0, fs=FS)(_impulse())
    assert not torch.allclose(bright, dark)


# --------------------------------------------------------------------------- #
# Wet/dry mix
# --------------------------------------------------------------------------- #
def test_mix_zero_is_dry():
    x = torch.randn(2, 8000, dtype=torch.float64)
    torch.testing.assert_close(Reverb(mix=0.0, fs=FS)(x), x, rtol=1e-12, atol=1e-13)


def test_mix_interpolates_linearly():
    x = _impulse(n=8000)
    full = Reverb(room_size=0.8, mix=1.0, fs=FS)(x)
    half = Reverb(room_size=0.8, mix=0.5, fs=FS)(x)
    expected_half = 0.5 * x + 0.5 * full  # dry=1-mix, wet=mix on the same wet signal
    torch.testing.assert_close(half, expected_half, rtol=1e-9, atol=1e-12)


# --------------------------------------------------------------------------- #
# Shapes, dtypes, channels
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", [(8000,), (2, 8000), (3, 2, 8000)])
def test_shapes_preserved(shape):
    y = Reverb(fs=FS)(torch.randn(*shape, dtype=torch.float64))
    assert y.shape == shape


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_preserved(dtype):
    y = Reverb(fs=FS)(torch.randn(2, 8000, dtype=dtype))
    assert y.dtype == dtype


def test_identical_channels_identical_output():
    mono = _impulse(n=8000)
    y = Reverb(room_size=0.8, fs=FS)(mono.repeat(2, 1))
    torch.testing.assert_close(y[0], y[1])


# --------------------------------------------------------------------------- #
# Lazy fs + validation
# --------------------------------------------------------------------------- #
def test_fs_required():
    with pytest.raises(ValueError, match="Sample rate"):
        Reverb()(torch.randn(1, 1000))


@pytest.mark.parametrize(
    "kwargs",
    [{"room_size": -0.1}, {"room_size": 1.1}, {"damping": 2.0}, {"mix": -1.0}, {"fs": 0}],
)
def test_invalid_params_raise(kwargs):
    with pytest.raises(ValueError):
        Reverb(**kwargs)


# --------------------------------------------------------------------------- #
# CPU / CUDA parity
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cpu_cuda_parity(dtype):
    rev_cpu = Reverb(room_size=0.8, damping=0.4, mix=0.3, fs=FS)
    rev_cuda = Reverb(room_size=0.8, damping=0.4, mix=0.3, fs=FS)
    gen = torch.Generator().manual_seed(5)
    x = torch.randn(4, 16000, generator=gen, dtype=dtype)
    y_cpu = rev_cpu(x)
    y_cuda = rev_cuda(x.cuda()).cpu()
    tol = {"rtol": 2e-3, "atol": 2e-4} if dtype == torch.float32 else {"rtol": 1e-9, "atol": 1e-10}
    torch.testing.assert_close(y_cpu, y_cuda, **tol)
