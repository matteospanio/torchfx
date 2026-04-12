"""Tests for :class:`torchfx.filter.fused.FusedSOSCascade`.

Baseline coverage for this module was 9% — the fused cascade had no dedicated
test file. These tests cover construction, the :meth:`from_chain` class method,
correctness against ``scipy.signal.sosfilt``, device/dtype handling, the
stateful-processing path, and error-handling branches.

"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.signal as sps
import torch
import torch.nn as nn

from torchfx.filter import HiButterworth, LoButterworth
from torchfx.filter.fused import FusedSOSCascade
from torchfx.filter.iir import IIR

SAMPLE_RATE = 44100
ATOL = 1e-4
RTOL = 1e-4


# ---------- helpers ----------


def _reference_sosfilt(filters: list[IIR], x: np.ndarray) -> np.ndarray:
    """Apply filters sequentially via scipy and return the result."""
    y = x.copy()
    for f in filters:
        assert f._sos is not None
        sos = f._sos.detach().cpu().numpy().astype(np.float64)
        y = sps.sosfilt(sos, y, axis=-1)
    return y


# ---------- construction ----------


class TestConstruction:
    def test_single_filter(self):
        f = LoButterworth(cutoff=2000, order=4, fs=SAMPLE_RATE)
        fused = FusedSOSCascade(f)
        assert fused._num_sections == f._sos.shape[0]
        assert fused.fs == SAMPLE_RATE
        assert fused._sos.dtype == torch.float64

    def test_multiple_filters(self):
        f1 = LoButterworth(cutoff=2000, order=4, fs=SAMPLE_RATE)
        f2 = HiButterworth(cutoff=200, order=2, fs=SAMPLE_RATE)
        fused = FusedSOSCascade(f1, f2)
        assert fused._num_sections == f1._sos.shape[0] + f2._sos.shape[0]

    def test_requires_at_least_one(self):
        with pytest.raises(ValueError, match="at least one"):
            FusedSOSCascade()

    def test_rejects_non_iir(self):
        class NotAFilter:
            pass

        with pytest.raises(TypeError, match="Expected filter with SOS coefficients"):
            FusedSOSCascade(NotAFilter())  # type: ignore[arg-type]

    def test_rejects_mismatched_fs(self):
        f1 = LoButterworth(cutoff=2000, order=4, fs=44100)
        f2 = LoButterworth(cutoff=2000, order=4, fs=48000)
        with pytest.raises(ValueError, match="different sample rates"):
            FusedSOSCascade(f1, f2)

    def test_requires_fs_when_coeffs_missing(self):
        f = LoButterworth(cutoff=2000, order=4)  # fs not set
        with pytest.raises(ValueError, match="no sampling frequency"):
            FusedSOSCascade(f)

    def test_computes_coefficients_eagerly(self):
        f = LoButterworth(cutoff=2000, order=4, fs=SAMPLE_RATE)
        assert f._sos is None
        FusedSOSCascade(f)
        assert f._sos is not None


# ---------- from_chain ----------


class TestFromChain:
    def test_from_sequential(self):
        chain = nn.Sequential(
            LoButterworth(cutoff=2000, order=4, fs=SAMPLE_RATE),
            HiButterworth(cutoff=200, order=2, fs=SAMPLE_RATE),
        )
        fused = FusedSOSCascade.from_chain(chain)
        assert isinstance(fused, FusedSOSCascade)
        assert fused._num_sections > 0

    def test_from_single_iir(self):
        f = LoButterworth(cutoff=2000, order=4, fs=SAMPLE_RATE)
        fused = FusedSOSCascade.from_chain(f)
        assert fused._num_sections == f._sos.shape[0]

    def test_rejects_bad_type(self):
        with pytest.raises(TypeError, match="Expected nn.Sequential or IIR/Biquad"):
            FusedSOSCascade.from_chain(nn.ReLU())  # type: ignore[arg-type]

    def test_empty_sequential_raises(self):
        with pytest.raises(ValueError, match="No IIR/Biquad filters"):
            FusedSOSCascade.from_chain(nn.Sequential(nn.Identity()))


# ---------- numerical correctness ----------


class TestNumericalCorrectness:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_single_filter_matches_sosfilt(self, seed):
        torch.manual_seed(seed)
        f = LoButterworth(cutoff=2000, order=4, fs=SAMPLE_RATE)
        fused = FusedSOSCascade(f)

        x = torch.randn(2, SAMPLE_RATE, dtype=torch.float64)
        y = fused(x)
        ref = _reference_sosfilt([f], x.numpy())
        np.testing.assert_allclose(y.numpy(), ref, atol=ATOL, rtol=RTOL)

    def test_multi_filter_matches_sequential_sosfilt(self):
        torch.manual_seed(42)
        f1 = LoButterworth(cutoff=4000, order=4, fs=SAMPLE_RATE)
        f2 = HiButterworth(cutoff=200, order=2, fs=SAMPLE_RATE)
        fused = FusedSOSCascade(f1, f2)

        x = torch.randn(2, SAMPLE_RATE, dtype=torch.float64)
        y = fused(x)
        ref = _reference_sosfilt([f1, f2], x.numpy())
        np.testing.assert_allclose(y.numpy(), ref, atol=ATOL, rtol=RTOL)


# ---------- shape handling ----------


class TestShapes:
    def _make(self) -> FusedSOSCascade:
        f = LoButterworth(cutoff=2000, order=4, fs=SAMPLE_RATE)
        return FusedSOSCascade(f)

    def test_1d_input(self):
        fused = self._make()
        x = torch.randn(SAMPLE_RATE)
        y = fused(x)
        assert y.shape == x.shape

    def test_2d_input(self):
        fused = self._make()
        x = torch.randn(2, SAMPLE_RATE)
        y = fused(x)
        assert y.shape == x.shape

    def test_3d_input(self):
        fused = self._make()
        x = torch.randn(4, 2, SAMPLE_RATE)
        y = fused(x)
        assert y.shape == x.shape


# ---------- stateful processing ----------


class TestStateful:
    def test_reset_state(self):
        f = LoButterworth(cutoff=2000, order=4, fs=SAMPLE_RATE)
        fused = FusedSOSCascade(f)
        x = torch.randn(2, 1024)

        _ = fused(x)
        assert fused._state_x is not None
        assert fused._stateful is True

        fused.reset_state()
        assert fused._state_x is None
        assert fused._state_y is None
        assert fused._stateful is False

    def test_chunked_equals_contiguous(self):
        """Processing a signal in two chunks must match a single-shot call."""
        torch.manual_seed(7)
        f = LoButterworth(cutoff=2000, order=4, fs=SAMPLE_RATE)

        x = torch.randn(2, 2048, dtype=torch.float64)

        fused_all = FusedSOSCascade(f)
        y_all = fused_all(x)

        fused_chunk = FusedSOSCascade(f)
        y1 = fused_chunk(x[:, :1024])
        y2 = fused_chunk(x[:, 1024:])
        y_chunked = torch.cat([y1, y2], dim=-1)

        torch.testing.assert_close(y_chunked, y_all, atol=ATOL, rtol=RTOL)

    def test_channel_resize_reallocates_state(self):
        f = LoButterworth(cutoff=2000, order=4, fs=SAMPLE_RATE)
        fused = FusedSOSCascade(f)

        y1 = fused(torch.randn(2, 512))
        assert fused._state_x is not None
        assert fused._state_x.shape[1] == 2

        # New call with different channel count re-allocates state.
        y2 = fused(torch.randn(4, 512))
        assert fused._state_x.shape[1] == 4
        assert y1.shape == (2, 512)
        assert y2.shape == (4, 512)


# ---------- device & dtype ----------


class TestDeviceDtype:
    def test_move_coeff_cpu_noop(self):
        f = LoButterworth(cutoff=2000, order=4, fs=SAMPLE_RATE)
        fused = FusedSOSCascade(f)
        fused.move_coeff("cpu")
        assert fused._sos.device.type == "cpu"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_preserves_input_dtype(self, dtype):
        f = LoButterworth(cutoff=2000, order=4, fs=SAMPLE_RATE)
        fused = FusedSOSCascade(f)
        x = torch.randn(2, 512, dtype=dtype)
        y = fused(x)
        assert y.dtype == dtype


# ---------- high-order stress ----------


class TestHighOrder:
    @pytest.mark.parametrize("order", [6, 12, 20])
    def test_high_order_stable(self, order):
        """High-order cascades must remain numerically stable and finite.

        This also doubles as a canary for the ``sec_sx0[16]`` stack-array limit
        in the native CPU kernel flagged in Phase 2 of the performance plan:
        an order-20 Butterworth is 10 biquad sections, comfortably under the
        current hard-coded 16 — bumping the order here when that limit is
        fixed will extend this guard.

        """
        torch.manual_seed(0)
        f = LoButterworth(cutoff=2000, order=order, fs=SAMPLE_RATE)
        fused = FusedSOSCascade(f)
        x = torch.randn(2, SAMPLE_RATE, dtype=torch.float64)
        y = fused(x)
        assert torch.isfinite(y).all()
        # Reference check against scipy.
        ref = _reference_sosfilt([f], x.numpy())
        np.testing.assert_allclose(y.numpy(), ref, atol=ATOL, rtol=RTOL)
