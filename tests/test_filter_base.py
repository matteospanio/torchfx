"""Tests for :mod:`torchfx.filter.__base` — AbstractFilter + ParallelFilterCombination.

Baseline coverage was 44% and the entire ``ParallelFilterCombination`` class
was tested only indirectly. These tests cover the ``+``/``__radd__`` operators,
``_has_computed_coeff`` branching, fs propagation, and the
parallel-combination forward path (including nested topologies).

"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.signal as sps
import torch

from torchfx.filter import HiButterworth, LoButterworth
from torchfx.filter.__base import AbstractFilter, ParallelFilterCombination

SAMPLE_RATE = 44100
ATOL = 1e-5
RTOL = 1e-5


# ---------- _has_computed_coeff branches ----------


class TestHasComputedCoeff:
    def test_false_before_compute_iir(self):
        f = LoButterworth(cutoff=1000, order=4, fs=SAMPLE_RATE)
        assert f._sos is None
        assert f._has_computed_coeff is False

    def test_true_after_compute_iir(self):
        f = LoButterworth(cutoff=1000, order=4, fs=SAMPLE_RATE)
        f.compute_coefficients()
        assert f._has_computed_coeff is True

    def test_parallel_propagates_child_status(self):
        f1 = LoButterworth(cutoff=1000, order=4, fs=SAMPLE_RATE)
        f2 = HiButterworth(cutoff=200, order=4, fs=SAMPLE_RATE)
        parallel = f1 + f2
        assert parallel._has_computed_coeff is False

        f1.compute_coefficients()
        assert parallel._has_computed_coeff is False  # f2 still pending

        f2.compute_coefficients()
        assert parallel._has_computed_coeff is True


# ---------- + / __radd__ operators ----------


class TestAddOperator:
    def test_add_creates_parallel_combination(self):
        f1 = LoButterworth(cutoff=1000, order=4, fs=SAMPLE_RATE)
        f2 = HiButterworth(cutoff=200, order=4, fs=SAMPLE_RATE)
        parallel = f1 + f2
        assert isinstance(parallel, ParallelFilterCombination)
        assert len(parallel.filters) == 2
        assert parallel.filters[0] is f1
        assert parallel.filters[1] is f2

    def test_add_rejects_non_filter(self):
        f = LoButterworth(cutoff=1000, order=4, fs=SAMPLE_RATE)
        with pytest.raises(AssertionError, match="AbstractFilter"):
            _ = f + 42  # type: ignore[operator]

    def test_radd_preserves_order(self):
        """f1.__radd__(f2) yields ``f2 + f1`` — preserves left-to-right order."""
        f1 = LoButterworth(cutoff=1000, order=4, fs=SAMPLE_RATE)
        f2 = HiButterworth(cutoff=200, order=4, fs=SAMPLE_RATE)
        parallel = f1.__radd__(f2)
        assert isinstance(parallel, ParallelFilterCombination)
        assert parallel.filters[0] is f2
        assert parallel.filters[1] is f1

    def test_radd_rejects_non_filter(self):
        f = LoButterworth(cutoff=1000, order=4, fs=SAMPLE_RATE)
        with pytest.raises(AssertionError, match="AbstractFilter"):
            f.__radd__(42)  # type: ignore[arg-type]


# ---------- ParallelFilterCombination fs propagation ----------


class TestFsPropagation:
    def test_init_without_fs_does_not_set_children(self):
        f1 = LoButterworth(cutoff=1000, order=4)
        f2 = HiButterworth(cutoff=200, order=4)
        parallel = ParallelFilterCombination(f1, f2)
        assert parallel.fs is None
        assert f1.fs is None
        assert f2.fs is None

    def test_init_with_fs_propagates(self):
        f1 = LoButterworth(cutoff=1000, order=4)
        f2 = HiButterworth(cutoff=200, order=4)
        parallel = ParallelFilterCombination(f1, f2, fs=SAMPLE_RATE)
        assert parallel.fs == SAMPLE_RATE
        assert f1.fs == SAMPLE_RATE
        assert f2.fs == SAMPLE_RATE

    def test_setter_propagates(self):
        f1 = LoButterworth(cutoff=1000, order=4)
        f2 = HiButterworth(cutoff=200, order=4)
        parallel = f1 + f2
        parallel.fs = SAMPLE_RATE
        assert f1.fs == SAMPLE_RATE
        assert f2.fs == SAMPLE_RATE

    def test_setter_preserves_existing_child_fs(self):
        """Children that already have fs set are not overwritten."""
        f1 = LoButterworth(cutoff=1000, order=4, fs=48000)
        f2 = HiButterworth(cutoff=200, order=4)  # no fs
        parallel = ParallelFilterCombination(f1, f2)
        parallel.fs = SAMPLE_RATE
        assert f1.fs == 48000  # untouched
        assert f2.fs == SAMPLE_RATE  # propagated

    def test_setter_to_none_is_noop_on_children(self):
        f1 = LoButterworth(cutoff=1000, order=4, fs=SAMPLE_RATE)
        f2 = HiButterworth(cutoff=200, order=4, fs=SAMPLE_RATE)
        parallel = f1 + f2
        parallel.fs = None
        # Setting to None does not wipe child fs.
        assert f1.fs == SAMPLE_RATE
        assert f2.fs == SAMPLE_RATE
        assert parallel.fs is None


# ---------- compute_coefficients / forward ----------


class TestParallelForward:
    def test_compute_coefficients_walks_children(self):
        f1 = LoButterworth(cutoff=1000, order=4, fs=SAMPLE_RATE)
        f2 = HiButterworth(cutoff=200, order=4, fs=SAMPLE_RATE)
        parallel = f1 + f2
        parallel.compute_coefficients()
        assert f1._has_computed_coeff
        assert f2._has_computed_coeff

    def test_parallel_output_is_sum_of_branches(self):
        """``(f1 + f2)(x)`` must equal ``f1(x) + f2(x)`` element-wise."""
        torch.manual_seed(0)
        f1 = LoButterworth(cutoff=4000, order=4, fs=SAMPLE_RATE)
        f2 = HiButterworth(cutoff=200, order=4, fs=SAMPLE_RATE)

        x = torch.randn(2, SAMPLE_RATE, dtype=torch.float64)

        # Independent filters (no state-sharing with the parallel combo).
        f1_solo = LoButterworth(cutoff=4000, order=4, fs=SAMPLE_RATE)
        f2_solo = HiButterworth(cutoff=200, order=4, fs=SAMPLE_RATE)
        y_solo = f1_solo(x) + f2_solo(x)

        parallel = ParallelFilterCombination(f1, f2)
        y_parallel = parallel(x)

        torch.testing.assert_close(y_parallel, y_solo, atol=ATOL, rtol=RTOL)

    def test_matches_scipy_reference(self):
        torch.manual_seed(1)
        f1 = LoButterworth(cutoff=4000, order=4, fs=SAMPLE_RATE)
        f2 = HiButterworth(cutoff=200, order=4, fs=SAMPLE_RATE)
        parallel = f1 + f2
        parallel.compute_coefficients()

        x = torch.randn(2, SAMPLE_RATE, dtype=torch.float64)
        y = parallel(x)

        x_np = x.numpy()
        ref = sps.sosfilt(f1._sos.numpy().astype(np.float64), x_np) + sps.sosfilt(
            f2._sos.numpy().astype(np.float64), x_np
        )
        np.testing.assert_allclose(y.numpy(), ref, atol=1e-3, rtol=1e-3)

    def test_three_way_parallel(self):
        torch.manual_seed(2)
        f1 = LoButterworth(cutoff=500, order=2, fs=SAMPLE_RATE)
        f2 = HiButterworth(cutoff=8000, order=2, fs=SAMPLE_RATE)
        f3 = LoButterworth(cutoff=2000, order=2, fs=SAMPLE_RATE)
        parallel = ParallelFilterCombination(f1, f2, f3)

        x = torch.randn(2, 1024, dtype=torch.float64)
        y = parallel(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()

    def test_nested_series_parallel_topology(self):
        """Cover a mixed series/parallel topology using the ``|`` operator."""
        torch.manual_seed(3)
        f1 = LoButterworth(cutoff=4000, order=2, fs=SAMPLE_RATE)
        f2 = HiButterworth(cutoff=200, order=2, fs=SAMPLE_RATE)
        f3 = LoButterworth(cutoff=6000, order=2, fs=SAMPLE_RATE)

        chain = ParallelFilterCombination(f1, f2) | f3

        x = torch.randn(2, 1024, dtype=torch.float64)
        y = chain(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()


# ---------- typing / misc ----------


def test_abstract_filter_cannot_be_instantiated():
    with pytest.raises(TypeError):
        AbstractFilter()  # type: ignore[abstract]
