"""Tests for :mod:`torchfx.filter.utils`.

Baseline coverage for this module was 0% — it had no dedicated test file.
The entire public surface is :func:`compute_order`, which maps a user-facing
filter order (linear count or dB/octave) onto the internal linear order used
by SciPy filter designers.

Note: :mod:`torchfx.filter.iir` inlines ``order // 6`` instead of calling
``compute_order`` — tracked as a Phase 4 duplication cleanup. These tests lock
the current semantics in place so that consolidation PR is safe.

"""

from __future__ import annotations

import pytest

from torchfx.filter.utils import compute_order


class TestComputeOrder:
    def test_linear_identity(self):
        assert compute_order(4, "linear") == 4

    def test_linear_zero(self):
        assert compute_order(0, "linear") == 0

    @pytest.mark.parametrize("o", [1, 2, 5, 8, 12, 30])
    def test_linear_pass_through(self, o):
        assert compute_order(o, "linear") == o

    def test_db_canonical_docstring_example(self):
        """The docstring example: 24 dB/oct → order 4."""
        assert compute_order(24, "db") == 4

    @pytest.mark.parametrize(
        "db,expected",
        [
            (6, 1),
            (12, 2),
            (18, 3),
            (24, 4),
            (36, 6),
            (48, 8),
            (60, 10),
        ],
    )
    def test_db_standard_slopes(self, db, expected):
        assert compute_order(db, "db") == expected

    @pytest.mark.parametrize(
        "db,expected",
        [
            (5, 0),  # below one octave
            (7, 1),  # floors toward zero
            (11, 1),
            (13, 2),
        ],
    )
    def test_db_floors_below_next_step(self, db, expected):
        """``//`` is floor division — non-multiples of 6 round down."""
        assert compute_order(db, "db") == expected

    def test_db_zero(self):
        assert compute_order(0, "db") == 0
