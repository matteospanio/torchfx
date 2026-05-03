"""Tests for :mod:`torchfx._ops` — the native-extension dispatch layer.

The extension is compiled at build time via scikit-build-core / CMake. These tests
verify that the pre-built extension is importable, the dispatch functions produce
correct results, and state management works as expected.

"""

from __future__ import annotations

import numpy as np
import scipy.signal as sps
import torch

from torchfx import _ops

# ---------- module constants & availability ----------


class TestAvailability:
    def test_parallel_scan_threshold(self):
        """The threshold is a public knob — lock the default so tuning is deliberate."""
        assert _ops.PARALLEL_SCAN_THRESHOLD == 2048

    def test_is_native_available_returns_true(self):
        """Extension is always available — compiled at install time."""
        assert _ops.is_native_available() is True

    def test_extension_importable(self):
        """The pre-built extension module must be importable."""
        from torchfx import torchfx_ext

        assert hasattr(torchfx_ext, "biquad_forward")
        assert hasattr(torchfx_ext, "sos_forward")
        assert hasattr(torchfx_ext, "delay_line_forward")


# ---------- biquad dispatch ----------


class TestBiquadDispatch:
    def _coeffs(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Trivial pass-through biquad: b=[1,0,0], a=[1,0,0].
        b = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        a = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        return b, a

    def test_passthrough_biquad(self):
        b, a = self._coeffs()
        x = torch.randn(2, 256, dtype=torch.float64)
        y, sx, sy = _ops.biquad_forward(x, b, a, None, None)
        torch.testing.assert_close(y, x)

    def test_state_defaults_allocated(self):
        b, a = self._coeffs()
        x = torch.randn(2, 256, dtype=torch.float64)
        y, sx, sy = _ops.biquad_forward(x, b, a, None, None)
        assert y.shape == x.shape
        assert sx.shape == (2, 2)
        assert sy.shape == (2, 2)

    def test_state_roundtrip(self):
        """Feeding back the returned state must not raise and must preserve shape."""
        b, a = self._coeffs()
        x1 = torch.randn(2, 128, dtype=torch.float64)
        x2 = torch.randn(2, 128, dtype=torch.float64)

        _, sx, sy = _ops.biquad_forward(x1, b, a, None, None)
        y2, sx2, sy2 = _ops.biquad_forward(x2, b, a, sx, sy)
        assert y2.shape == x2.shape
        assert sx2.shape == sx.shape
        assert sy2.shape == sy.shape

    def test_nontrivial_matches_scipy(self):
        """A real lowpass biquad matches scipy.signal.sosfilt."""
        torch.manual_seed(42)
        sos = sps.butter(2, 2000 / (0.5 * 44100), btype="lowpass", output="sos")
        x = torch.randn(2, 1024, dtype=torch.float64)
        ref = sps.sosfilt(sos, x.numpy(), axis=-1)

        b = torch.tensor(sos[0, :3], dtype=torch.float64)
        a = torch.tensor(sos[0, 3:], dtype=torch.float64)
        y, _, _ = _ops.biquad_forward(x, b, a, None, None)

        np.testing.assert_allclose(y.numpy(), ref, atol=1e-6, rtol=1e-6)


# ---------- SOS cascade dispatch ----------


class TestSosDispatch:
    def _sos(self) -> torch.Tensor:
        # Two pass-through biquad sections.
        return torch.tensor(
            [
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
        )

    def test_passthrough_sos(self):
        sos = self._sos()
        x = torch.randn(2, 256, dtype=torch.float64)
        y, sx, sy = _ops.parallel_iir_forward(x, sos, None, None)
        torch.testing.assert_close(y, x)

    def test_state_defaults_allocated(self):
        sos = self._sos()
        x = torch.randn(2, 256, dtype=torch.float64)
        y, sx, sy = _ops.parallel_iir_forward(x, sos, None, None)
        assert y.shape == x.shape
        # state is (K, C, 2)
        assert sx.shape == (sos.shape[0], x.shape[0], 2)
        assert sy.shape == (sos.shape[0], x.shape[0], 2)

    def test_nontrivial_matches_scipy(self):
        """A 4th-order Butterworth matches scipy.signal.sosfilt."""
        torch.manual_seed(42)
        sos_np = sps.butter(4, 2000 / (0.5 * 44100), btype="lowpass", output="sos")
        x = torch.randn(2, 1024, dtype=torch.float64)
        ref = sps.sosfilt(sos_np, x.numpy(), axis=-1)

        sos = torch.tensor(sos_np, dtype=torch.float64)
        y, _, _ = _ops.parallel_iir_forward(x, sos, None, None)

        np.testing.assert_allclose(y.numpy(), ref, atol=1e-4, rtol=1e-4)


# ---------- delay-line dispatch ----------


class TestDelayDispatch:
    def test_cpu_delay_basic(self):
        """CPU delay kernel produces correct output."""
        x = torch.randn(2, 512, dtype=torch.float64)
        y = _ops.delay_line_forward(x, delay_samples=100, decay=0.5, mix=0.8)
        assert y.shape == x.shape
        # Pre-delay region should be unchanged
        torch.testing.assert_close(y[:, :100], x[:, :100])
        # Post-delay region should differ
        assert not torch.allclose(y[:, 100:], x[:, 100:])

    def test_cpu_delay_matches_formula(self):
        """Verify y[n] = x[n] + mix*decay*x[n-delay] for n >= delay."""
        x = torch.arange(10, dtype=torch.float64).unsqueeze(0)
        delay, decay, mix = 3, 0.5, 1.0
        y = _ops.delay_line_forward(x, delay_samples=delay, decay=decay, mix=mix)

        coeff = mix * decay
        expected = x.clone()
        expected[0, delay:] = x[0, delay:] + coeff * x[0, :-delay]
        torch.testing.assert_close(y, expected)

    def test_short_signal_returns_input(self):
        """Signal shorter than delay should be returned unchanged."""
        x = torch.randn(2, 50, dtype=torch.float64)
        y = _ops.delay_line_forward(x, delay_samples=100, decay=0.5, mix=0.5)
        torch.testing.assert_close(y, x)

    def test_1d_input(self):
        """1-D signal should work and return 1-D."""
        x = torch.randn(512, dtype=torch.float64)
        y = _ops.delay_line_forward(x, delay_samples=50, decay=0.5, mix=0.5)
        assert y.shape == x.shape
