"""Targeted tests closing :mod:`torchfx.filter.iir` coverage gaps.

The Phase 0 baseline shows :file:`src/torchfx/filter/iir.py` at 71% line
coverage. :file:`tests/test_iir.py` already covers the Butterworth,
Chebyshev1/2, Shelving, ParametricEQ, and Elliptic families. This file fills
the remaining holes:

* the pure-PyTorch Direct-Form-1 fallback path in ``_forward_sos`` (triggered
  by forcing ``_ops.parallel_iir_forward`` to return ``None``)
* the ``_biquad_df1_fallback`` helper
* the 1-D and 3-D shape branches of ``_forward_sos``
* ``IIR.forward`` raising when ``fs`` is unset
* the ``LinkwitzRiley`` family (base class + convenience subclasses + the
  positive-even-order validation)
* the error branch in ``Shelving._omega`` when ``fs`` is unset

"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.signal as sps
import torch

from torchfx.filter import LoButterworth
from torchfx.filter.iir import (
    HiLinkwitzRiley,
    HiShelving,
    LinkwitzRiley,
    LoLinkwitzRiley,
    _biquad_df1_fallback,
)

SAMPLE_RATE = 44100
ATOL = 1e-5
RTOL = 1e-5


# ---------- IIR.forward error branch ----------


class TestForwardNoFs:
    def test_forward_without_fs_raises(self):
        f = LoButterworth(cutoff=1000, order=4)  # fs is None
        x = torch.randn(2, 512, dtype=torch.float64)
        with pytest.raises(ValueError, match="[Ss]ample rate"):
            _ = f(x)


# ---------- Pure-PyTorch fallback path ----------


def _force_pure_python_fallback(monkeypatch):
    """Make ``parallel_iir_forward`` return ``None`` so the fallback runs."""
    from torchfx import _ops

    def _none(*args, **kwargs):  # noqa: ARG001
        return None

    monkeypatch.setattr(_ops, "parallel_iir_forward", _none)


class TestPurePythonFallback:
    def test_fallback_matches_scipy(self, monkeypatch):
        _force_pure_python_fallback(monkeypatch)

        torch.manual_seed(0)
        f = LoButterworth(cutoff=2000, order=4, fs=SAMPLE_RATE)
        x = torch.randn(2, 1024, dtype=torch.float64)
        y = f(x)

        f.compute_coefficients()
        assert f._sos is not None
        sos_np = f._sos.numpy().astype(np.float64)
        ref = sps.sosfilt(sos_np, x.numpy(), axis=-1)

        np.testing.assert_allclose(y.numpy(), ref, atol=1e-4, rtol=1e-4)

    def test_fallback_1d_input(self, monkeypatch):
        _force_pure_python_fallback(monkeypatch)
        f = LoButterworth(cutoff=2000, order=4, fs=SAMPLE_RATE)
        x = torch.randn(512, dtype=torch.float64)
        y = f(x)
        assert y.shape == x.shape

    def test_fallback_3d_input(self, monkeypatch):
        _force_pure_python_fallback(monkeypatch)
        f = LoButterworth(cutoff=2000, order=4, fs=SAMPLE_RATE)
        x = torch.randn(4, 2, 512, dtype=torch.float64)
        y = f(x)
        assert y.shape == x.shape

    def test_fallback_preserves_dtype_f32(self, monkeypatch):
        _force_pure_python_fallback(monkeypatch)
        f = LoButterworth(cutoff=2000, order=4, fs=SAMPLE_RATE)
        x = torch.randn(2, 512, dtype=torch.float32)
        y = f(x)
        assert y.dtype == torch.float32


class TestDF1Helper:
    def test_biquad_df1_fallback_matches_scipy_sosfilt(self):
        """Direct call with a pass-through biquad returns the input unchanged."""
        torch.manual_seed(1)
        x = torch.randn(2, 512, dtype=torch.float64)
        # Pass-through biquad (b=[1,0,0], a=[1,0,0]).
        b0 = torch.tensor(1.0, dtype=torch.float64)
        b1 = torch.tensor(0.0, dtype=torch.float64)
        b2 = torch.tensor(0.0, dtype=torch.float64)
        a1 = torch.tensor(0.0, dtype=torch.float64)
        a2 = torch.tensor(0.0, dtype=torch.float64)
        sx = torch.zeros(2, 2, dtype=torch.float64)
        sy = torch.zeros(2, 2, dtype=torch.float64)
        y = _biquad_df1_fallback(x, b0, b1, b2, a1, a2, sx, sy)
        torch.testing.assert_close(y, x)

    def test_biquad_df1_fallback_nontrivial(self):
        """A real lowpass biquad matches scipy.signal.sosfilt."""
        torch.manual_seed(2)
        sos = sps.butter(2, 2000 / (0.5 * SAMPLE_RATE), btype="lowpass", output="sos")
        assert sos.shape == (1, 6)  # single section

        x = torch.randn(2, 1024, dtype=torch.float64)
        ref = sps.sosfilt(sos, x.numpy(), axis=-1)

        b0 = torch.tensor(float(sos[0, 0]), dtype=torch.float64)
        b1 = torch.tensor(float(sos[0, 1]), dtype=torch.float64)
        b2 = torch.tensor(float(sos[0, 2]), dtype=torch.float64)
        a1 = torch.tensor(float(sos[0, 4]), dtype=torch.float64)
        a2 = torch.tensor(float(sos[0, 5]), dtype=torch.float64)

        sx = torch.zeros(2, 2, dtype=torch.float64)
        sy = torch.zeros(2, 2, dtype=torch.float64)
        y = _biquad_df1_fallback(x, b0, b1, b2, a1, a2, sx, sy)

        np.testing.assert_allclose(y.numpy(), ref, atol=1e-6, rtol=1e-6)


# ---------- LinkwitzRiley family ----------


class TestLinkwitzRiley:
    def test_lowpass_forward_matches_stacked_butterworth(self):
        torch.manual_seed(3)
        f = LinkwitzRiley(btype="lowpass", cutoff=2000, order=4, fs=SAMPLE_RATE)
        x = torch.randn(2, 2048, dtype=torch.float64)
        y = f(x)

        # Reference: two cascaded 2nd-order Butterworth.
        sos = sps.butter(2, 2000 / (0.5 * SAMPLE_RATE), btype="lowpass", output="sos")
        sos_cascade = np.vstack([sos, sos])
        ref = sps.sosfilt(sos_cascade, x.numpy(), axis=-1)

        np.testing.assert_allclose(y.numpy(), ref, atol=1e-4, rtol=1e-4)

    def test_highpass_via_convenience(self):
        f = HiLinkwitzRiley(cutoff=2000, order=4, fs=SAMPLE_RATE)
        assert isinstance(f, LinkwitzRiley)
        assert f.btype == "highpass"
        x = torch.randn(2, 1024, dtype=torch.float64)
        y = f(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()

    def test_lowpass_via_convenience(self):
        f = LoLinkwitzRiley(cutoff=2000, order=4, fs=SAMPLE_RATE)
        assert f.btype == "lowpass"
        x = torch.randn(2, 1024, dtype=torch.float64)
        y = f(x)
        assert y.shape == x.shape

    def test_order_scale_db_halves_via_floor_div(self):
        # ``order=24`` with ``order_scale="db"`` becomes ``24 // 6 = 4``. That
        # is an even positive integer, so the validation passes.
        f = LinkwitzRiley(btype="lowpass", cutoff=1000, order=24, order_scale="db", fs=SAMPLE_RATE)
        assert f.order == 4

    @pytest.mark.parametrize("bad_order", [0, -2, 3, 5, 7])
    def test_rejects_non_positive_or_odd_order(self, bad_order):
        with pytest.raises(ValueError, match="positive even integer"):
            LinkwitzRiley(btype="lowpass", cutoff=1000, order=bad_order, fs=SAMPLE_RATE)

    def test_sum_of_lp_hp_is_approximately_flat(self):
        """Landmark property of Linkwitz-Riley: LP + HP is magnitude-flat.

        With identical order and cutoff, the magnitude of ``(lp(x) + hp(x))``
        should approximate the original signal for most frequencies. We test
        with a mid-band sine which both filters attenuate only slightly.
        """
        torch.manual_seed(4)
        fs = SAMPLE_RATE
        lp = LoLinkwitzRiley(cutoff=2000, order=4, fs=fs)
        hp = HiLinkwitzRiley(cutoff=2000, order=4, fs=fs)

        # Broadband noise — the spectrum-wide sum property matters here.
        x = torch.randn(2, 4096, dtype=torch.float64)
        y = lp(x) + hp(x)

        # Trim edges where transients dominate; the tail should track x closely.
        # LR4 is magnitude-flat but not amplitude-identity (there's a group
        # delay), so compare energies rather than samples.
        energy_in = (x[:, 1024:] ** 2).mean().item()
        energy_out = (y[:, 1024:] ** 2).mean().item()
        assert energy_out == pytest.approx(energy_in, rel=0.25)


# ---------- Shelving._omega error branch ----------


class TestShelvingOmegaNoFs:
    def test_omega_requires_fs(self):
        f = HiShelving(cutoff=4000, q=0.707, gain=3.0, gain_scale="db")
        with pytest.raises(ValueError, match="[Ss]ample rate"):
            _ = f._omega
