"""Tests verifying the pure-PyTorch fallback when CUDA extension is unavailable.

These tests ensure that torchfx works correctly when:
1. CUDA is not available
2. The native extension fails to compile
3. Input tensors are on CPU

"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.signal import butter, lfilter

from torchfx.filter.biquad import BiquadLPF, BiquadHPF
from torchfx.filter import LoButterworth
from torchfx.filter.filterbank import LogFilterBank

SAMPLE_RATE = 44100
ATOL = 1e-4
RTOL = 1e-4


class TestCPUFallback:
    """Verify filters work correctly on CPU (fallback path)."""

    def test_biquad_cpu_works(self):
        """Biquad filter processes correctly on CPU."""
        filt = BiquadLPF(cutoff=2000, q=0.707, fs=SAMPLE_RATE)
        x = torch.randn(2, SAMPLE_RATE)
        y = filt(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()

    def test_biquad_stateful_cpu(self):
        """Stateful biquad path works on CPU."""
        filt = BiquadLPF(cutoff=2000, q=0.707, fs=SAMPLE_RATE)
        x = torch.randn(2, SAMPLE_RATE)

        # First call: stateless
        y1 = filt(x)
        # Second call: stateful
        y2 = filt(x)

        assert y1.shape == x.shape
        assert y2.shape == x.shape

    def test_sos_cascade_cpu(self):
        """SOS cascade works on CPU."""
        filt = LoButterworth(cutoff=2000, order=4, fs=SAMPLE_RATE)
        x = torch.randn(2, SAMPLE_RATE)
        y = filt(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()

    def test_sos_stateful_cpu(self):
        """Stateful SOS path works on CPU."""
        filt = LoButterworth(cutoff=2000, order=4, fs=SAMPLE_RATE)
        x = torch.randn(2, SAMPLE_RATE)

        y1 = filt(x)
        y2 = filt(x)

        assert y1.shape == x.shape
        assert y2.shape == x.shape


class TestExtensionLoadFailure:
    """Verify graceful handling of extension load failures."""

    def test_ops_returns_none_when_unavailable(self, monkeypatch):
        """_ops dispatch functions return None when extension unavailable."""
        import torchfx._ops as ops

        # Reset extension state
        monkeypatch.setattr(ops, "_ext", None)
        monkeypatch.setattr(ops, "_ext_load_attempted", False)

        # Mock torch.cuda.is_available to return False
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        x = torch.randn(2, 1000)
        b = torch.tensor([0.1, 0.2, 0.1])
        a = torch.tensor([1.0, -0.5, 0.1])

        result = ops.biquad_forward(x, b, a, None, None)
        assert result is None

    def test_filter_works_after_ops_failure(self, monkeypatch):
        """Filters still work when _ops returns None."""
        import torchfx._ops as ops

        monkeypatch.setattr(ops, "_ext", None)
        monkeypatch.setattr(ops, "_ext_load_attempted", True)

        filt = BiquadLPF(cutoff=2000, q=0.707, fs=SAMPLE_RATE)
        x = torch.randn(2, SAMPLE_RATE)

        # Force stateful mode
        _ = filt(x)
        y = filt(x)

        assert y.shape == x.shape
        assert torch.isfinite(y).all()


class TestLogFilterBankCPU:
    """Test LogFilterBank on CPU."""

    def test_basic_creation(self):
        """Create a filter bank with correct number of bands."""
        fb = LogFilterBank(n_bands=10, f_min=20.0, f_max=20000.0, fs=SAMPLE_RATE)
        assert len(fb.filters) == 10
        assert len(fb.center_frequencies) == 10

    def test_frequency_bounds(self):
        """Center frequencies span f_min to f_max."""
        fb = LogFilterBank(n_bands=5, f_min=100.0, f_max=10000.0, fs=SAMPLE_RATE)
        freqs = fb.center_frequencies
        assert abs(freqs[0] - 100.0) < 0.01
        assert abs(freqs[-1] - 10000.0) < 0.1

    def test_log_spacing(self):
        """Adjacent frequency ratios are constant (logarithmic spacing)."""
        fb = LogFilterBank(n_bands=8, f_min=100.0, f_max=10000.0, fs=SAMPLE_RATE)
        freqs = fb.center_frequencies
        ratios = [freqs[k + 1] / freqs[k] for k in range(len(freqs) - 1)]
        for r in ratios:
            assert abs(r - ratios[0]) < 0.01

    def test_forward_shape(self):
        """Output shape is [n_bands, C, T]."""
        fb = LogFilterBank(n_bands=5, f_min=100.0, f_max=5000.0, fs=SAMPLE_RATE)
        x = torch.randn(2, SAMPLE_RATE)
        y = fb(x)
        assert y.shape == (5, 2, SAMPLE_RATE)

    def test_fs_none_raises(self):
        """Forward raises when fs is not set."""
        fb = LogFilterBank(n_bands=5, f_min=100.0, f_max=5000.0)
        x = torch.randn(2, SAMPLE_RATE)
        with pytest.raises(ValueError, match="Sample rate"):
            fb(x)

    def test_invalid_params(self):
        """Invalid parameters raise assertions."""
        with pytest.raises(AssertionError):
            LogFilterBank(n_bands=1, f_min=100.0, f_max=10000.0)

        with pytest.raises(AssertionError):
            LogFilterBank(n_bands=5, f_min=-100.0, f_max=10000.0)

        with pytest.raises(AssertionError):
            LogFilterBank(n_bands=5, f_min=10000.0, f_max=100.0)
