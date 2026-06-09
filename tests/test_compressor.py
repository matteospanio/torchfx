"""Tests for the Compressor dynamics effect (issue #30)."""

from __future__ import annotations

import math

import pytest
import torch

from torchfx.effect import Compressor

FS = 48000


def _static_gain_db(level_db: float, threshold: float, ratio: float, knee: float) -> float:
    """Reference static gain-reduction curve (matches the kernel, in dB)."""
    inv_ratio = 0.0 if math.isinf(ratio) else 1.0 / ratio
    over = level_db - threshold
    if knee > 0 and 2 * abs(over) <= knee:
        t = over + knee / 2
        lsc = level_db + (inv_ratio - 1) * t * t / (2 * knee)
    elif over > 0:
        lsc = threshold + over * inv_ratio
    else:
        lsc = level_db
    return lsc - level_db


def _const(amp: float, n: int = 2048, c: int = 1, dtype=torch.float64) -> torch.Tensor:
    return torch.full((c, n), amp, dtype=dtype)


# --------------------------------------------------------------------------- #
# Static gain curve (attack=release=0 -> instantaneous detector = |x|)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("level_db", [-30.0, -20.0, -12.0, -6.0, 0.0])
@pytest.mark.parametrize("ratio", [2.0, 4.0, 10.0])
@pytest.mark.parametrize("knee", [0.0, 6.0])
def test_static_curve_matches_reference(level_db, ratio, knee):
    threshold = -18.0
    amp = 10 ** (level_db / 20)
    comp = Compressor(threshold=threshold, ratio=ratio, attack=0.0, release=0.0, knee=knee, fs=FS)
    y = comp(_const(amp))
    g_db = _static_gain_db(level_db, threshold, ratio, knee)
    expected = amp * 10 ** (g_db / 20)
    torch.testing.assert_close(y, torch.full_like(y, expected), rtol=1e-5, atol=1e-7)


def test_below_threshold_is_unity():
    comp = Compressor(threshold=-12.0, ratio=4.0, attack=0.0, release=0.0, knee=0.0, fs=FS)
    x = _const(10 ** (-24.0 / 20))  # well below threshold
    torch.testing.assert_close(comp(x), x, rtol=1e-6, atol=1e-9)


def test_limiter_clamps_at_threshold():
    threshold = -10.0
    comp = Compressor(
        threshold=threshold, ratio=float("inf"), attack=0.0, release=0.0, knee=0.0, fs=FS
    )
    for level_db in (-5.0, 0.0, 6.0):
        amp = 10 ** (level_db / 20)
        y = comp(_const(amp))
        out_db = 20 * math.log10(float(y.abs().max()))
        assert out_db == pytest.approx(threshold, abs=1e-4)


def test_makeup_gain_on_uncompressed():
    comp = Compressor(
        threshold=-6.0, ratio=4.0, attack=0.0, release=0.0, knee=0.0, makeup_gain=6.0, fs=FS
    )
    x = _const(10 ** (-30.0 / 20))  # below threshold -> only makeup applies
    torch.testing.assert_close(comp(x), x * 10 ** (6.0 / 20), rtol=1e-5, atol=1e-8)


def test_soft_knee_continuous():
    # Sweep input level across the knee; the gain-reduction curve must be continuous.
    threshold, ratio, knee = -18.0, 4.0, 8.0
    comp = Compressor(threshold=threshold, ratio=ratio, attack=0.0, release=0.0, knee=knee, fs=FS)
    prev = None
    for level_db in [threshold + d for d in torch.linspace(-6, 6, 25).tolist()]:
        amp = 10 ** (level_db / 20)
        out_db = 20 * math.log10(float(comp(_const(amp)).abs().max()))
        gr = out_db - level_db
        if prev is not None:
            assert abs(gr - prev) < 0.6  # no discontinuity across the knee
        prev = gr


# --------------------------------------------------------------------------- #
# Ballistics
# --------------------------------------------------------------------------- #
def test_attack_smoothing_ramps_in():
    # Loud step: gain reduction engages over ~attack, so the onset sample is louder
    # (less reduction) than the settled region.
    comp = Compressor(threshold=-20.0, ratio=8.0, attack=0.02, release=0.05, knee=0.0, fs=FS)
    x = _const(10 ** (-3.0 / 20), n=4096)
    y = comp(x).abs()
    assert float(y[0, 0]) > float(y[0, -1]) + 1e-4  # onset less compressed than settled


def test_release_recovers_after_loud():
    # loud (compressed) -> silence: the held gain reduction recovers over ~release.
    comp = Compressor(threshold=-20.0, ratio=8.0, attack=0.001, release=0.05, knee=0.0, fs=FS)
    loud = 10 ** (-3.0 / 20)
    # Tail long enough for the held level (-3 dBFS) to decay below threshold (-20 dB).
    x = torch.cat([_const(loud, n=2000), _const(1e-4, n=16000)], dim=1)
    y = comp(x)
    quiet = y[:, 2000:]
    # gain (out/in) over the quiet tail rises back toward unity as the hold releases.
    gain = (quiet.abs() / 1e-4)[0]
    assert float(gain[-1]) > float(gain[0])  # recovers
    assert float(gain[0]) < 0.5  # starts compressed (~8:1 at -3 dBFS over -20 dB)
    assert float(gain[-1]) == pytest.approx(1.0, abs=1e-2)  # fully recovered


def test_coefficient_formula():
    comp = Compressor(attack=0.01, release=0.1, fs=FS)
    comp(_const(0.1))  # triggers coeff computation
    assert comp._aA == pytest.approx(math.exp(-1.0 / (0.01 * FS)))
    assert comp._aR == pytest.approx(math.exp(-1.0 / (0.1 * FS)))


# --------------------------------------------------------------------------- #
# Detector modes
# --------------------------------------------------------------------------- #
def test_rms_less_reduction_than_peak_on_sine():
    t = torch.arange(FS, dtype=torch.float64) / FS
    x = (0.7 * torch.sin(2 * math.pi * 1000 * t)).unsqueeze(0)
    peak = Compressor(threshold=-20.0, ratio=4.0, detector="peak", fs=FS)(x)
    rms = Compressor(threshold=-20.0, ratio=4.0, detector="rms", fs=FS)(x)
    # RMS of a sine is ~3 dB below its peak, so it compresses less (louder output).
    assert float(rms.abs().mean()) > float(peak.abs().mean())


# --------------------------------------------------------------------------- #
# Shapes / dtypes / device
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", [(2048,), (2, 2048), (3, 2, 2048)])
def test_shape_preserved(shape):
    comp = Compressor(threshold=-12.0, ratio=4.0, fs=FS)
    x = torch.randn(*shape) * 0.5
    assert comp(x).shape == x.shape


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_preserved(dtype):
    comp = Compressor(threshold=-12.0, ratio=4.0, fs=FS)
    x = (torch.randn(2, 1024) * 0.5).to(dtype)
    assert comp(x).dtype == dtype


def test_identical_channels_identical_output():
    comp = Compressor(threshold=-15.0, ratio=4.0, attack=0.003, release=0.05, fs=FS)
    ch = torch.randn(1, 4096) * 0.6
    y = comp(torch.cat([ch, ch], dim=0))
    torch.testing.assert_close(y[0], y[1])


def test_silent_input_silent_output():
    comp = Compressor(threshold=-20.0, ratio=4.0, fs=FS)
    x = torch.zeros(2, 1024)
    assert torch.all(comp(x) == 0)


# --------------------------------------------------------------------------- #
# fs propagation
# --------------------------------------------------------------------------- #
def test_fs_required_without_pipeline():
    comp = Compressor(threshold=-12.0, ratio=4.0)  # no fs
    with pytest.raises(ValueError, match=r"Sample rate \(fs\) is required"):
        comp(torch.randn(1, 256))


def test_fs_set_via_attribute_recomputes():
    comp = Compressor(attack=0.01, fs=44100)
    comp(_const(0.1))
    a1 = comp._aA
    comp.fs = 48000  # mimics Wave.__update_config
    comp(_const(0.1))
    assert comp._aA != a1
    assert comp._aA == pytest.approx(math.exp(-1.0 / (0.01 * 48000)))


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kwargs",
    [
        {"ratio": 0.5},
        {"attack": -0.1},
        {"release": -0.1},
        {"knee": -1.0},
        {"detector": "median"},
        {"fs": 0},
        {"fs": -48000},
    ],
)
def test_invalid_params_raise(kwargs):
    with pytest.raises(ValueError):
        Compressor(**kwargs)


# --------------------------------------------------------------------------- #
# CPU / CUDA parity
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("detector", ["peak", "rms"])
@pytest.mark.parametrize("ratio", [4.0, float("inf")])
def test_cpu_cuda_parity(detector, ratio):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(3, 4096, generator=g, dtype=torch.float32) * 0.6
    comp = Compressor(
        threshold=-18.0, ratio=ratio, attack=0.004, release=0.06, detector=detector, fs=FS
    )
    y_cpu = comp(x)
    y_cuda = comp(x.cuda()).cpu()
    torch.testing.assert_close(y_cuda, y_cpu, rtol=1e-4, atol=1e-4)
