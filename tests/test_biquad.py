# ruff: noqa: A001
"""Tests for biquad filter implementations.

Tests cover coefficient correctness (matching AudioNoise formulas), forward-pass shape
preservation, stateful chunk processing, and basic frequency-domain behavior.

"""

import math

import pytest
import torch

from torchfx.filter import (
    BiquadAllPass,
    BiquadBPF,
    BiquadBPFPeak,
    BiquadHPF,
    BiquadLPF,
    BiquadNotch,
)


FS = 44100
CUTOFF = 1000.0
Q = 0.707


# ---------- helpers ----------


def omega_alpha(cutoff: float, q: float, fs: int):
    w0 = 2.0 * math.pi * cutoff / fs
    sin_w0 = math.sin(w0)
    cos_w0 = math.cos(w0)
    alpha = sin_w0 / (2.0 * q)
    return sin_w0, cos_w0, alpha


# ---------- coefficient tests ----------


class TestBiquadCoefficients:
    """Verify that each biquad type produces the correct AudioNoise coefficients."""

    def test_lpf_coefficients(self) -> None:
        f = BiquadLPF(cutoff=CUTOFF, q=Q, fs=FS)
        f.compute_coefficients()
        assert f.b is not None and f.a is not None

        _, cos_w0, alpha = omega_alpha(CUTOFF, Q, FS)
        a0_inv = 1.0 / (1.0 + alpha)
        b1 = (1.0 - cos_w0) * a0_inv

        assert f.b[0].item() == pytest.approx(b1 / 2.0, rel=1e-10)
        assert f.b[1].item() == pytest.approx(b1, rel=1e-10)
        assert f.b[2].item() == pytest.approx(b1 / 2.0, rel=1e-10)
        assert f.a[0].item() == pytest.approx(1.0, rel=1e-10)
        assert f.a[1].item() == pytest.approx(-2.0 * cos_w0 * a0_inv, rel=1e-10)
        assert f.a[2].item() == pytest.approx((1.0 - alpha) * a0_inv, rel=1e-10)

    def test_hpf_coefficients(self) -> None:
        f = BiquadHPF(cutoff=CUTOFF, q=Q, fs=FS)
        f.compute_coefficients()

        _, cos_w0, alpha = omega_alpha(CUTOFF, Q, FS)
        a0_inv = 1.0 / (1.0 + alpha)
        b1_raw = (1.0 + cos_w0) * a0_inv

        assert f.b is not None
        assert f.b[0].item() == pytest.approx(b1_raw / 2.0, rel=1e-10)
        assert f.b[1].item() == pytest.approx(-b1_raw, rel=1e-10)
        assert f.b[2].item() == pytest.approx(b1_raw / 2.0, rel=1e-10)

    def test_notch_coefficients(self) -> None:
        f = BiquadNotch(cutoff=CUTOFF, q=Q, fs=FS)
        f.compute_coefficients()

        _, cos_w0, alpha = omega_alpha(CUTOFF, Q, FS)
        a0_inv = 1.0 / (1.0 + alpha)
        common = -2.0 * cos_w0 * a0_inv

        assert f.b is not None and f.a is not None
        assert f.b[0].item() == pytest.approx(a0_inv, rel=1e-10)
        assert f.b[1].item() == pytest.approx(common, rel=1e-10)
        assert f.b[2].item() == pytest.approx(a0_inv, rel=1e-10)
        assert f.a[1].item() == pytest.approx(common, rel=1e-10)

    def test_bpf_coefficients(self) -> None:
        f = BiquadBPF(cutoff=CUTOFF, q=Q, fs=FS)
        f.compute_coefficients()

        _, cos_w0, alpha = omega_alpha(CUTOFF, Q, FS)
        a0_inv = 1.0 / (1.0 + alpha)

        assert f.b is not None
        assert f.b[0].item() == pytest.approx(alpha * a0_inv, rel=1e-10)
        assert f.b[1].item() == pytest.approx(0.0, abs=1e-15)
        assert f.b[2].item() == pytest.approx(-alpha * a0_inv, rel=1e-10)

    def test_bpf_peak_coefficients(self) -> None:
        f = BiquadBPFPeak(cutoff=CUTOFF, q=Q, fs=FS)
        f.compute_coefficients()

        _, cos_w0, alpha = omega_alpha(CUTOFF, Q, FS)
        a0_inv = 1.0 / (1.0 + alpha)

        assert f.b is not None
        assert f.b[0].item() == pytest.approx(Q * alpha * a0_inv, rel=1e-10)
        assert f.b[1].item() == pytest.approx(0.0, abs=1e-15)

    def test_allpass_coefficients(self) -> None:
        f = BiquadAllPass(cutoff=CUTOFF, q=Q, fs=FS)
        f.compute_coefficients()

        _, cos_w0, alpha = omega_alpha(CUTOFF, Q, FS)
        a0_inv = 1.0 / (1.0 + alpha)
        b0 = (1.0 - alpha) * a0_inv
        b1 = -2.0 * cos_w0 * a0_inv

        assert f.b is not None and f.a is not None
        assert f.b[0].item() == pytest.approx(b0, rel=1e-10)
        assert f.b[1].item() == pytest.approx(b1, rel=1e-10)
        assert f.b[2].item() == pytest.approx(1.0, rel=1e-10)
        assert f.a[1].item() == pytest.approx(b1, rel=1e-10)
        assert f.a[2].item() == pytest.approx(b0, rel=1e-10)


# ---------- forward / shape tests ----------


class TestBiquadForward:
    """Test forward-pass shape preservation and basic behavior."""

    @pytest.fixture
    def mono_signal(self) -> torch.Tensor:
        return torch.sin(torch.linspace(0, 2 * math.pi * 440, 4410))

    @pytest.fixture
    def stereo_signal(self) -> torch.Tensor:
        t = torch.linspace(0, 1.0, 4410)
        return torch.stack([torch.sin(2 * math.pi * 440 * t), torch.sin(2 * math.pi * 880 * t)])

    @pytest.fixture
    def batch_signal(self) -> torch.Tensor:
        return torch.randn(4, 2, 4410)

    def test_lpf_mono_shape(self, mono_signal: torch.Tensor) -> None:
        f = BiquadLPF(cutoff=1000, q=0.707, fs=44100)
        out = f(mono_signal)
        assert out.shape == mono_signal.shape

    def test_lpf_stereo_shape(self, stereo_signal: torch.Tensor) -> None:
        f = BiquadLPF(cutoff=1000, q=0.707, fs=44100)
        out = f(stereo_signal)
        assert out.shape == stereo_signal.shape

    def test_lpf_batch_shape(self, batch_signal: torch.Tensor) -> None:
        f = BiquadLPF(cutoff=1000, q=0.707, fs=44100)
        out = f(batch_signal)
        assert out.shape == batch_signal.shape

    def test_hpf_forward(self, mono_signal: torch.Tensor) -> None:
        f = BiquadHPF(cutoff=1000, q=0.707, fs=44100)
        out = f(mono_signal)
        assert out.shape == mono_signal.shape
        assert torch.isfinite(out).all()

    def test_notch_forward(self, mono_signal: torch.Tensor) -> None:
        f = BiquadNotch(cutoff=440, q=10.0, fs=44100)
        out = f(mono_signal)
        assert out.shape == mono_signal.shape
        # Notch at the signal's frequency should attenuate it significantly
        assert out.abs().mean() < mono_signal.abs().mean()

    def test_bpf_forward(self, mono_signal: torch.Tensor) -> None:
        f = BiquadBPF(cutoff=440, q=0.707, fs=44100)
        out = f(mono_signal)
        assert out.shape == mono_signal.shape
        assert torch.isfinite(out).all()

    def test_allpass_unity_magnitude(self, mono_signal: torch.Tensor) -> None:
        """AllPass should preserve energy (approximately)."""
        f = BiquadAllPass(cutoff=1000, q=0.707, fs=44100)
        out = f(mono_signal)
        assert out.shape == mono_signal.shape
        # RMS should be very close
        rms_in = mono_signal.pow(2).mean().sqrt()
        rms_out = out.pow(2).mean().sqrt()
        assert rms_out.item() == pytest.approx(rms_in.item(), rel=0.05)

    def test_no_fs_raises(self) -> None:
        f = BiquadLPF(cutoff=1000, q=0.707)
        with pytest.raises(ValueError, match="Sample rate"):
            f(torch.randn(1000))


# ---------- stateful tests ----------


class TestBiquadStateful:
    """Test stateful chunk-to-chunk processing."""

    def test_second_call_uses_stateful_path(self) -> None:
        f = BiquadLPF(cutoff=1000, q=0.707, fs=44100)
        x = torch.randn(4410)
        f(x)
        assert f._stateful is True

    def test_reset_state(self) -> None:
        f = BiquadLPF(cutoff=1000, q=0.707, fs=44100)
        x = torch.randn(4410)
        f(x)
        assert f._stateful is True
        f.reset_state()
        assert f._stateful is False
        assert f._state_x is None
        assert f._state_y is None

    def test_stateful_output_finite(self) -> None:
        """Multiple calls should produce finite output (no divergence)."""
        f = BiquadLPF(cutoff=1000, q=0.707, fs=44100)
        for _ in range(10):
            x = torch.randn(2, 1024)
            out = f(x)
            assert torch.isfinite(out).all(), "Stateful output diverged"

    def test_reset_then_process_matches_fresh(self) -> None:
        """After reset, output should match a freshly constructed filter."""
        f1 = BiquadLPF(cutoff=1000, q=0.707, fs=44100)
        f2 = BiquadLPF(cutoff=1000, q=0.707, fs=44100)

        x = torch.randn(4410)
        f1(torch.randn(4410))  # warm up f1 with some data
        f1.reset_state()

        out1 = f1(x)
        out2 = f2(x)
        torch.testing.assert_close(out1, out2, atol=1e-6, rtol=1e-5)
