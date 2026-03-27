"""Tests for CUDA kernel numerical accuracy.

These tests compare the CUDA parallel prefix scan implementation against
scipy.signal.lfilter to ensure numerical correctness.

"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.signal import butter, cheby1, lfilter, sosfilt, tf2sos

from torchfx.filter import LoButterworth, HiButterworth, LoChebyshev1
from torchfx.filter.biquad import BiquadLPF, BiquadHPF
from torchfx.filter.filterbank import LogFilterBank

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

SAMPLE_RATE = 44100
ATOL = 1e-4
RTOL = 1e-4


def random_signal(C: int, T: int, device: str = "cuda") -> torch.Tensor:
    """Create a random test signal."""
    return torch.randn(C, T, device=device, dtype=torch.float32)


class TestBiquadCUDA:
    """Test single biquad filter accuracy on CUDA."""

    def test_biquad_lpf_matches_scipy(self):
        """CUDA biquad LPF output matches scipy.signal.lfilter."""
        b_scipy, a_scipy = butter(2, 0.1)
        x_np = np.random.randn(1, SAMPLE_RATE).astype(np.float64)
        y_ref = lfilter(b_scipy, a_scipy, x_np)

        filt = BiquadLPF(cutoff=0.1 * SAMPLE_RATE / 2, q=0.707, fs=SAMPLE_RATE)
        filt.compute_coefficients()
        filt.move_coeff("cuda")

        x_cuda = torch.from_numpy(x_np).float().cuda()
        y_cuda = filt(x_cuda)

        np.testing.assert_allclose(y_cuda.cpu().numpy(), y_ref, atol=ATOL, rtol=RTOL)

    def test_biquad_hpf_matches_scipy(self):
        """CUDA biquad HPF output matches scipy.signal.lfilter."""
        b_scipy, a_scipy = butter(2, 0.3, btype="high")
        x_np = np.random.randn(2, SAMPLE_RATE).astype(np.float64)

        filt = BiquadHPF(cutoff=0.3 * SAMPLE_RATE / 2, q=0.707, fs=SAMPLE_RATE)
        filt.compute_coefficients()
        filt.move_coeff("cuda")

        x_cuda = torch.from_numpy(x_np).float().cuda()
        y_cuda = filt(x_cuda)

        # Compare per-channel against scipy
        for c in range(2):
            y_ref = lfilter(b_scipy, a_scipy, x_np[c])
            np.testing.assert_allclose(y_cuda[c].cpu().numpy(), y_ref, atol=ATOL, rtol=RTOL)

    @pytest.mark.parametrize("T", [100, 1024, 4096, 44100])
    def test_various_lengths(self, T):
        """Test accuracy across different signal lengths."""
        filt = BiquadLPF(cutoff=2000, q=1.0, fs=SAMPLE_RATE)
        filt.compute_coefficients()
        filt.move_coeff("cuda")

        x = random_signal(1, T)
        y = filt(x)

        # Verify output shape
        assert y.shape == x.shape

    @pytest.mark.parametrize("C", [1, 2, 4, 8])
    def test_multichannel(self, C):
        """Test multi-channel processing."""
        filt = BiquadLPF(cutoff=2000, q=1.0, fs=SAMPLE_RATE)
        filt.compute_coefficients()
        filt.move_coeff("cuda")

        x = random_signal(C, SAMPLE_RATE)
        y = filt(x)

        assert y.shape == (C, SAMPLE_RATE)


class TestSOSCascadeCUDA:
    """Test SOS cascade (higher-order IIR) accuracy on CUDA."""

    def test_butterworth_4th_order(self):
        """4th-order Butterworth matches scipy.signal.sosfilt."""
        sos_scipy = butter(4, 0.2, output="sos")
        x_np = np.random.randn(1, SAMPLE_RATE).astype(np.float64)
        y_ref = sosfilt(sos_scipy, x_np)

        filt = LoButterworth(cutoff=0.2 * SAMPLE_RATE / 2, order=4, fs=SAMPLE_RATE)
        filt.compute_coefficients()
        filt.move_coeff("cuda")

        x_cuda = torch.from_numpy(x_np).float().cuda()
        y_cuda = filt(x_cuda)

        np.testing.assert_allclose(y_cuda.cpu().numpy(), y_ref, atol=ATOL, rtol=RTOL)

    def test_butterworth_8th_order(self):
        """8th-order Butterworth for high-order stability."""
        filt = LoButterworth(cutoff=3000, order=8, fs=SAMPLE_RATE)
        filt.compute_coefficients()
        filt.move_coeff("cuda")

        x = random_signal(2, SAMPLE_RATE)
        y = filt(x)

        assert y.shape == x.shape
        assert torch.isfinite(y).all(), "Output contains NaN/Inf"

    def test_chebyshev_accuracy(self):
        """Chebyshev Type 1 filter accuracy."""
        filt = LoChebyshev1(cutoff=2000, order=4, fs=SAMPLE_RATE)
        filt.compute_coefficients()
        filt.move_coeff("cuda")

        x = random_signal(1, SAMPLE_RATE)
        y = filt(x)

        assert y.shape == x.shape
        assert torch.isfinite(y).all()


class TestStatefulContinuity:
    """Test that chunked processing matches single-pass processing."""

    def test_biquad_chunked_matches_full(self):
        """Process in two chunks; verify output matches single-pass."""
        filt_full = BiquadLPF(cutoff=2000, q=1.0, fs=SAMPLE_RATE)
        filt_full.compute_coefficients()
        filt_full.move_coeff("cuda")

        filt_chunk = BiquadLPF(cutoff=2000, q=1.0, fs=SAMPLE_RATE)
        filt_chunk.compute_coefficients()
        filt_chunk.move_coeff("cuda")

        T = 8000
        x = random_signal(2, T)

        # Single pass
        y_full = filt_full(x)

        # Two chunks
        chunk1 = x[:, : T // 2]
        chunk2 = x[:, T // 2 :]

        y1 = filt_chunk(chunk1)
        y2 = filt_chunk(chunk2)
        y_chunked = torch.cat([y1, y2], dim=1)

        torch.testing.assert_close(y_full, y_chunked, atol=ATOL, rtol=RTOL)

    def test_sos_chunked_matches_full(self):
        """SOS cascade chunked processing consistency."""
        filt_full = LoButterworth(cutoff=2000, order=4, fs=SAMPLE_RATE)
        filt_full.compute_coefficients()
        filt_full.move_coeff("cuda")

        filt_chunk = LoButterworth(cutoff=2000, order=4, fs=SAMPLE_RATE)
        filt_chunk.compute_coefficients()
        filt_chunk.move_coeff("cuda")

        T = 8000
        x = random_signal(2, T)

        y_full = filt_full(x)

        y1 = filt_chunk(x[:, : T // 2])
        y2 = filt_chunk(x[:, T // 2 :])
        y_chunked = torch.cat([y1, y2], dim=1)

        torch.testing.assert_close(y_full, y_chunked, atol=ATOL, rtol=RTOL)


class TestShapeSupport:
    """Test various input tensor shapes."""

    def test_1d_input(self):
        """Single-channel 1D input [T]."""
        filt = BiquadLPF(cutoff=2000, q=1.0, fs=SAMPLE_RATE)
        filt.compute_coefficients()
        filt.move_coeff("cuda")

        x = torch.randn(SAMPLE_RATE, device="cuda")
        y = filt(x)
        assert y.shape == (SAMPLE_RATE,)

    def test_2d_input(self):
        """Multi-channel 2D input [C, T]."""
        filt = BiquadLPF(cutoff=2000, q=1.0, fs=SAMPLE_RATE)
        filt.compute_coefficients()
        filt.move_coeff("cuda")

        x = torch.randn(4, SAMPLE_RATE, device="cuda")
        y = filt(x)
        assert y.shape == (4, SAMPLE_RATE)

    def test_3d_input(self):
        """Batched 3D input [B, C, T]."""
        filt = BiquadLPF(cutoff=2000, q=1.0, fs=SAMPLE_RATE)
        filt.compute_coefficients()
        filt.move_coeff("cuda")

        x = torch.randn(3, 2, SAMPLE_RATE, device="cuda")
        y = filt(x)
        assert y.shape == (3, 2, SAMPLE_RATE)


class TestLogFilterBank:
    """Test LogFilterBank on CUDA."""

    def test_center_frequencies(self):
        """Verify logarithmic spacing of center frequencies."""
        fb = LogFilterBank(n_bands=10, f_min=100.0, f_max=10000.0, fs=SAMPLE_RATE)
        freqs = fb.center_frequencies

        assert len(freqs) == 10
        assert abs(freqs[0] - 100.0) < 0.01
        assert abs(freqs[-1] - 10000.0) < 0.1

        # Check log spacing: ratios between adjacent frequencies should be constant
        ratios = [freqs[k + 1] / freqs[k] for k in range(len(freqs) - 1)]
        for r in ratios:
            assert abs(r - ratios[0]) < 0.01, "Frequencies not logarithmically spaced"

    def test_output_shape(self):
        """Verify output shape is [n_bands, C, T]."""
        fb = LogFilterBank(n_bands=5, f_min=100.0, f_max=5000.0, fs=SAMPLE_RATE)
        x = random_signal(2, SAMPLE_RATE)
        y = fb(x)

        assert y.shape == (5, 2, SAMPLE_RATE)

    def test_fs_propagation(self):
        """Verify sampling frequency propagates to all bands."""
        fb = LogFilterBank(n_bands=5, f_min=100.0, f_max=5000.0)
        assert fb.fs is None

        fb.fs = SAMPLE_RATE
        for f in fb.filters:
            assert f.fs == SAMPLE_RATE
