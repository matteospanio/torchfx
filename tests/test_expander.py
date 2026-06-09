"""Tests for the Expander / Gate dynamics effect (issue #32)."""

from __future__ import annotations

import math

import pytest
import torch

from torchfx.effect import Expander, Gate

FS = 48000
_NO_FLOOR = -240.0


def _static_gain_db(
    level_db: float, threshold: float, ratio: float, knee: float, floor: float = _NO_FLOOR
) -> float:
    """Reference downward-expander static gain curve (matches the kernel, in dB)."""
    slope = 1.0e6 if math.isinf(ratio) else ratio - 1.0
    over = level_db - threshold
    if knee > 0 and 2 * abs(over) <= knee:
        t = over - knee / 2
        gdb = -slope * t * t / (2 * knee)
    elif over < 0:
        gdb = slope * over
    else:
        gdb = 0.0
    return max(gdb, floor)


def _const(amp: float, n: int = 2048, c: int = 1, dtype=torch.float64) -> torch.Tensor:
    return torch.full((c, n), amp, dtype=dtype)


# --------------------------------------------------------------------------- #
# Static gain curve (attack=release=0 -> instantaneous detector = |x|)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("level_db", [-50.0, -42.0, -36.0, -30.0, -24.0])
@pytest.mark.parametrize("ratio", [2.0, 4.0, 10.0])
@pytest.mark.parametrize("knee", [0.0, 6.0])
def test_static_curve_matches_reference(level_db, ratio, knee):
    threshold = -30.0
    amp = 10 ** (level_db / 20)
    exp = Expander(threshold=threshold, ratio=ratio, attack=0.0, release=0.0, knee=knee, fs=FS)
    y = exp(_const(amp))
    g_db = _static_gain_db(level_db, threshold, ratio, knee)
    expected = amp * 10 ** (g_db / 20)
    torch.testing.assert_close(y, torch.full_like(y, expected), rtol=1e-5, atol=1e-9)


def test_above_threshold_is_unity():
    exp = Expander(threshold=-40.0, ratio=4.0, attack=0.0, release=0.0, knee=0.0, fs=FS)
    x = _const(10 ** (-12.0 / 20))  # well above threshold
    torch.testing.assert_close(exp(x), x, rtol=1e-6, atol=1e-9)


def test_below_threshold_is_attenuated():
    exp = Expander(threshold=-30.0, ratio=4.0, attack=0.0, release=0.0, knee=0.0, fs=FS)
    amp = 10 ** (-42.0 / 20)  # 12 dB below threshold
    y = exp(_const(amp))
    # ratio 4 -> a further (4-1)*12 = 36 dB down: output = amp * 10^(-36/20)
    expected = amp * 10 ** (-36.0 / 20)
    torch.testing.assert_close(y, torch.full_like(y, expected), rtol=1e-5, atol=1e-12)
    assert y.abs().max().item() < amp  # genuinely quieter


def test_infinite_ratio_gates_to_floor():
    floor = -80.0
    exp = Expander(
        threshold=-30.0, ratio=float("inf"), attack=0.0, release=0.0, knee=0.0, floor=floor, fs=FS
    )
    amp = 10 ** (-36.0 / 20)  # below threshold
    y = exp(_const(amp))
    expected = amp * 10 ** (floor / 20)
    torch.testing.assert_close(y, torch.full_like(y, expected), rtol=1e-5, atol=1e-12)


def test_floor_clamps_attenuation():
    floor = -20.0
    exp = Expander(
        threshold=-20.0, ratio=10.0, attack=0.0, release=0.0, knee=0.0, floor=floor, fs=FS
    )
    amp = 10 ** (-60.0 / 20)  # 40 dB below -> would be (10-1)*40 = 360 dB down, clamped
    y = exp(_const(amp))
    expected = amp * 10 ** (floor / 20)
    torch.testing.assert_close(y, torch.full_like(y, expected), rtol=1e-5, atol=1e-12)


def test_soft_knee_is_continuous():
    threshold, knee = -30.0, 8.0
    exp = Expander(threshold=threshold, ratio=6.0, attack=0.0, release=0.0, knee=knee, fs=FS)
    levels = torch.linspace(threshold - knee, threshold + knee, 40)
    gains = []
    for lvl in levels.tolist():
        amp = 10 ** (lvl / 20)
        gains.append((exp(_const(amp)).mean() / amp).item())
    gains_t = torch.tensor(gains)
    # No jumps across the knee (gain is monotonic non-decreasing and smooth in level).
    assert torch.all(gains_t.diff() >= -1e-6)
    assert gains_t.diff().abs().max() < 0.1


# --------------------------------------------------------------------------- #
# Ballistics
# --------------------------------------------------------------------------- #
def test_release_closes_after_signal_drops():
    """A loud burst then silence: the gate closes (gain falls) over the release."""
    exp = Expander(
        threshold=-30.0, ratio=float("inf"), attack=0.0, release=0.05, floor=-90.0, fs=FS
    )
    loud = torch.full((1, 2000), 0.5, dtype=torch.float64)
    quiet = torch.full((1, 16000), 10 ** (-50.0 / 20), dtype=torch.float64)  # below threshold
    x = torch.cat([loud, quiet], dim=1)
    y = exp(x)
    g = y[0, 2000:].abs() / x[0, 2000:].abs()
    assert g[0].item() > 0.5  # still open right after the burst (release not elapsed)
    assert g[-1].item() < 1e-3  # fully closed by the end


def test_attack_opens_gradually():
    """Silence then a tone: with a slow attack the gate opens over time, not instantly."""
    exp = Expander(
        threshold=-30.0, ratio=float("inf"), attack=0.02, release=0.2, floor=-90.0, fs=FS
    )
    quiet = torch.full((1, 4000), 10 ** (-50.0 / 20), dtype=torch.float64)
    loud = torch.full((1, 8000), 0.5, dtype=torch.float64)
    x = torch.cat([quiet, loud], dim=1)
    y = exp(x)
    g = y[0, 4000:].abs() / x[0, 4000:].abs()
    assert g[0].item() < g[2000].item()  # opening up
    assert g[-1].item() > 0.9  # fully open at the end


def test_peak_vs_rms_differ():
    g_peak = Expander(threshold=-20.0, ratio=4.0, detector="peak", fs=FS)
    g_rms = Expander(threshold=-20.0, ratio=4.0, detector="rms", fs=FS)
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(1, 8000, generator=gen, dtype=torch.float64) * 0.02
    assert not torch.allclose(g_peak(x), g_rms(x))


# --------------------------------------------------------------------------- #
# Shapes, dtypes, channels
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", [(4096,), (2, 4096), (3, 2, 4096)])
def test_shapes_preserved(shape):
    exp = Expander(threshold=-30.0, ratio=3.0, fs=FS)
    x = torch.randn(*shape, dtype=torch.float64) * 0.1
    assert exp(x).shape == x.shape


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_preserved(dtype):
    exp = Expander(threshold=-30.0, ratio=4.0, fs=FS)
    x = torch.randn(2, 4096, dtype=dtype) * 0.1
    assert exp(x).dtype == dtype


def test_identical_channels_identical_output():
    exp = Expander(threshold=-30.0, ratio=4.0, fs=FS)
    gen = torch.Generator().manual_seed(1)
    mono = torch.randn(1, 8000, generator=gen, dtype=torch.float64) * 0.05
    stereo = mono.repeat(2, 1)
    y = exp(stereo)
    torch.testing.assert_close(y[0], y[1])


# --------------------------------------------------------------------------- #
# Lazy fs + validation
# --------------------------------------------------------------------------- #
def test_fs_required():
    exp = Expander(threshold=-30.0, ratio=4.0)
    with pytest.raises(ValueError, match="Sample rate"):
        exp(torch.randn(1, 1000))


def test_fs_recompute_on_change():
    exp = Expander(threshold=-30.0, ratio=4.0, fs=FS)
    x = torch.randn(1, 4000, dtype=torch.float64) * 0.05
    exp(x)
    exp.fs = 96000
    exp(x)  # recomputes coeffs without error
    assert exp._last_fs == 96000


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ratio": 0.5},
        {"attack": -1.0},
        {"release": -1.0},
        {"knee": -1.0},
        {"floor": 6.0},
        {"detector": "bogus"},
        {"fs": 0},
    ],
)
def test_invalid_params_raise(kwargs):
    with pytest.raises(ValueError):
        Expander(**kwargs)


# --------------------------------------------------------------------------- #
# Gate convenience subclass
# --------------------------------------------------------------------------- #
def test_gate_is_infinite_ratio_expander():
    # attack=0 so the gate opens instantly (its default attack ramps over ~1 ms).
    gate = Gate(threshold=-40.0, floor=-80.0, attack=0.0, fs=FS)
    assert math.isinf(gate.ratio)
    below = _const(10 ** (-50.0 / 20))
    above = _const(10 ** (-20.0 / 20))
    torch.testing.assert_close(gate(below), below * 10 ** (-80.0 / 20), rtol=1e-4, atol=1e-12)
    torch.testing.assert_close(gate(above), above, rtol=1e-5, atol=1e-9)


# --------------------------------------------------------------------------- #
# CPU / CUDA parity
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("detector", ["peak", "rms"])
@pytest.mark.parametrize(
    "ratio,dtype",
    [
        (4.0, torch.float32),
        (4.0, torch.float64),
        # A ratio=inf gate is a near-step transfer function; in float32 the detector's
        # rounding (CPU-sequential vs CUDA) flips a few boundary samples between unity
        # and floor, so per-sample parity is only meaningful in float64.
        (float("inf"), torch.float64),
    ],
)
def test_cpu_cuda_parity(ratio, dtype, detector):
    exp_cpu = Expander(threshold=-30.0, ratio=ratio, detector=detector, floor=-80.0, fs=FS)
    exp_cuda = Expander(threshold=-30.0, ratio=ratio, detector=detector, floor=-80.0, fs=FS)
    gen = torch.Generator().manual_seed(7)
    x = torch.randn(8, 16000, generator=gen, dtype=dtype) * 0.05
    y_cpu = exp_cpu(x)
    y_cuda = exp_cuda(x.cuda()).cpu()
    tol = {"rtol": 1e-4, "atol": 1e-5} if dtype == torch.float32 else {"rtol": 1e-9, "atol": 1e-10}
    torch.testing.assert_close(y_cpu, y_cuda, **tol)
