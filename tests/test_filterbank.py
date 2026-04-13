"""Targeted CPU tests for :class:`torchfx.filter.filterbank.LogFilterBank`.

There is already a `TestLogFilterBankCPU` class in ``tests/test_cuda_fallback.py``
covering the CUDA-fallback surface — this file fills in the branches those
tests don't touch:

* the :attr:`fs` getter (nobody reads ``fb.fs`` directly)
* the :attr:`fs` setter propagation loop to children
* :meth:`compute_coefficients` (explicit eager call)
* the per-forward ``f.fs = self._fs`` branch in :meth:`forward` when a child
  filter has ``fs = None``.

These are the lines flagged at 73% coverage in the Phase 0 baseline
(``src/torchfx/filter/filterbank.py``: 134, 138-141, 151-155, 181).

"""

from __future__ import annotations

import torch

from torchfx.filter.biquad import BiquadBPF
from torchfx.filter.filterbank import LogFilterBank

SAMPLE_RATE = 44100


# ---------- fs getter / setter ----------


class TestFsAccessors:
    def test_getter_reflects_init_value(self):
        fb = LogFilterBank(n_bands=4, f_min=100.0, f_max=10000.0, fs=SAMPLE_RATE)
        assert fb.fs == SAMPLE_RATE

    def test_getter_none_when_unset(self):
        fb = LogFilterBank(n_bands=4, f_min=100.0, f_max=10000.0)
        assert fb.fs is None

    def test_setter_propagates_to_children(self):
        fb = LogFilterBank(n_bands=4, f_min=100.0, f_max=10000.0)
        fb.fs = SAMPLE_RATE
        assert fb.fs == SAMPLE_RATE
        for f in fb.filters:
            assert isinstance(f, BiquadBPF)
            assert f.fs == SAMPLE_RATE

    def test_setter_none_does_not_touch_children(self):
        """Setting fs = None must not push ``None`` onto children that already have a
        valid fs — the setter's ``if value is not None`` guard is the branch under
        test."""
        fb = LogFilterBank(n_bands=4, f_min=100.0, f_max=10000.0, fs=SAMPLE_RATE)
        fb.fs = None
        assert fb.fs is None
        for f in fb.filters:
            assert f.fs == SAMPLE_RATE  # untouched


# ---------- compute_coefficients ----------


class TestComputeCoefficients:
    def test_walks_children(self):
        fb = LogFilterBank(n_bands=3, f_min=100.0, f_max=10000.0, fs=SAMPLE_RATE)
        # Children start with no coefficients.
        for f in fb.filters:
            assert not f._has_computed_coeff
        fb.compute_coefficients()
        for f in fb.filters:
            assert f._has_computed_coeff

    def test_sets_own_sentinel_tensors(self):
        """compute_coefficients sets ``self.a`` / ``self.b`` to sentinel tensors so
        ``_has_computed_coeff`` on the bank itself returns True."""
        fb = LogFilterBank(n_bands=3, f_min=100.0, f_max=10000.0, fs=SAMPLE_RATE)
        fb.compute_coefficients()
        assert isinstance(fb.a, torch.Tensor)
        assert isinstance(fb.b, torch.Tensor)


# ---------- forward per-call fs propagation ----------


class TestForwardFsPropagation:
    def test_forward_propagates_fs_to_lazy_children(self):
        """If a child filter has ``fs = None`` at forward time, the forward path patches
        it in-place from the bank's own fs.

        This covers the
        ``if f.fs is None`` branch in LogFilterBank.forward.

        """
        fb = LogFilterBank(n_bands=3, f_min=100.0, f_max=10000.0, fs=SAMPLE_RATE)

        # Construct a fresh BiquadBPF with no fs and drop it into the bank,
        # bypassing the fs setter so the branch actually fires at forward time.
        orphan = BiquadBPF(cutoff=1000.0, q=1.0)
        assert orphan.fs is None
        fb.filters[0] = orphan

        x = torch.randn(2, 1024, dtype=torch.float64)
        y = fb(x)

        assert orphan.fs == SAMPLE_RATE  # patched in-place
        assert y.shape == (fb.n_bands, *x.shape)
        assert torch.isfinite(y).all()
