# ruff: noqa: A001
"""Regression tests for stale-coefficient recomputation on sample-rate change.

A filter reused across sample rates must recompute its coefficients instead of
silently applying the ones designed for the previous ``fs`` (Roadmap Epic A1).
These cover the *direct* ``forward()`` call path; the ``Wave`` pipe path is
covered separately via ``Wave.__update_config``.

"""

from __future__ import annotations

import pytest
import torch

import torchfx as fx
from torchfx.filter import LoButterworth
from torchfx.filter.biquad import BiquadLPF


def _signal(n: int = 2000) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.randn(1, n, generator=g, dtype=torch.float64)


def test_iir_recomputes_on_fs_change_direct():
    """Reusing an IIR across sample rates yields the fresh-design output, not stale."""
    x = _signal()

    reused = LoButterworth(cutoff=100, order=4, fs=1000)
    reused(x)  # materialize coefficients at fs=1000
    assert reused._coeff_fs == 1000

    reused.fs = 2000
    out_reused = reused(x)
    assert reused._coeff_fs == 2000

    fresh = LoButterworth(cutoff=100, order=4, fs=2000)
    out_fresh = fresh(x)

    torch.testing.assert_close(out_reused, out_fresh)


def test_biquad_recomputes_on_fs_change_direct():
    """Same guarantee for a single biquad section."""
    x = _signal()

    reused = BiquadLPF(cutoff=100, q=0.707, fs=1000)
    reused(x)
    assert reused._coeff_fs == 1000

    reused.fs = 2000
    out_reused = reused(x)
    assert reused._coeff_fs == 2000

    fresh = BiquadLPF(cutoff=100, q=0.707, fs=2000)
    out_fresh = fresh(x)

    torch.testing.assert_close(out_reused, out_fresh)


def test_fs_change_resets_df1_state():
    """An fs change clears accumulated state so the next forward starts clean."""
    x = _signal()

    reused = LoButterworth(cutoff=100, order=4, fs=1000)
    for _ in range(3):
        reused(x)  # accumulate plenty of DF1 state at fs=1000
    assert reused._state_y is not None

    reused.fs = 2000
    out_after = reused(x)  # first post-change forward must ignore the old state

    fresh = LoButterworth(cutoff=100, order=4, fs=2000)
    torch.testing.assert_close(out_after, fresh(x))


def test_no_redundant_recompute_when_fs_unchanged():
    """Repeated forwards at a fixed fs must not redesign coefficients each call."""
    x = _signal()
    f = LoButterworth(cutoff=100, order=4, fs=1000)
    f(x)
    sos_first = f._sos
    f(x)
    # Same tensor object: coefficients were not recomputed on the second call.
    assert f._sos is sos_first


def test_wave_pipe_fs_change_recomputes():
    """Piping one filter instance into Waves of different fs is correct (Wave path)."""
    data = _signal()
    filt = LoButterworth(cutoff=100, order=4)

    w1 = fx.Wave(data.clone(), fs=1000)
    out1 = (w1 | filt).ys

    w2 = fx.Wave(data.clone(), fs=2000)
    out2 = (w2 | filt).ys

    fresh1 = LoButterworth(cutoff=100, order=4, fs=1000)
    fresh2 = LoButterworth(cutoff=100, order=4, fs=2000)
    torch.testing.assert_close(out1, fresh1(data.clone()))
    torch.testing.assert_close(out2, fresh2(data.clone()))


def test_offline_materialize_resets_state_across_waves():
    """Reusing one filter instance across two offline Waves must not leak state (A2)."""
    data = _signal()
    filt = BiquadLPF(cutoff=100, q=0.707)  # fs propagated by the Wave

    w_a = fx.Wave(data.clone(), fs=1000)
    out_a = (w_a | filt).ys

    # Same fs, same instance: without the offline reset, state from w_a would
    # bleed into w_b and corrupt its output.
    w_b = fx.Wave(data.clone(), fs=1000)
    out_b = (w_b | filt).ys

    fresh = BiquadLPF(cutoff=100, q=0.707, fs=1000)
    out_fresh = fresh(data.clone())

    torch.testing.assert_close(out_a, out_fresh)
    torch.testing.assert_close(out_b, out_fresh)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_precision_input_rejected(dtype):
    """Half-precision inputs must fail loudly, not be silently upcast (Roadmap A3)."""
    from torchfx import _ops

    with pytest.raises(TypeError, match="[Hh]alf-precision"):
        _ops._select_native_dtype(torch.zeros(4, dtype=dtype))
