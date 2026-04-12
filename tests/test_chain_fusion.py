"""Tests for deferred pipeline fusion and FilterChain."""

from __future__ import annotations

import torch
import torch.nn as nn

from torchfx import FilterChain, Wave
from torchfx.effect import Gain
from torchfx.filter.biquad import BiquadLPF
from torchfx.filter.iir import HiButterworth, LoButterworth

SAMPLE_RATE = 16000
DURATION = 0.1  # seconds — short signals for fast tests
N_SAMPLES = int(SAMPLE_RATE * DURATION)


def _make_wave() -> Wave:
    torch.manual_seed(42)
    return Wave(torch.randn(1, N_SAMPLES, dtype=torch.float64), SAMPLE_RATE)


def _apply_sequentially(wave: Wave, *filters: nn.Module) -> torch.Tensor:
    """Apply filters one by one (no fusion) and return the result tensor.

    Filters must already have ``fs`` set and coefficients computed.

    """
    data = wave.ys.clone()
    for f in filters:
        data = f(data)
    return data


def _prepare(fs: int, *filters: nn.Module) -> None:
    """Set fs and compute coefficients on IIR filters (mimics Wave pipeline config)."""
    from torchfx.effect import FX
    from torchfx.filter.__base import AbstractFilter

    for f in filters:
        if isinstance(f, FX):
            if hasattr(f, "fs") and f.fs is None:
                f.fs = fs
            if isinstance(f, AbstractFilter) and not f._has_computed_coeff:
                f.compute_coefficients()


# ---------- Deferred fusion ----------


class TestDeferredFusion:
    def test_deferred_pipeline_fuses_consecutive_iir(self):
        """Wave | iir1 | iir2 | iir3 fuses into a single FusedSOSCascade."""
        wave = _make_wave()
        # Deferred path
        f1a = LoButterworth(cutoff=4000, order=2)
        f2a = HiButterworth(cutoff=200, order=2)
        f3a = LoButterworth(cutoff=6000, order=2)
        result = wave | f1a | f2a | f3a

        # Reference (separate filter instances, no fusion)
        f1b = LoButterworth(cutoff=4000, order=2)
        f2b = HiButterworth(cutoff=200, order=2)
        f3b = LoButterworth(cutoff=6000, order=2)
        _prepare(SAMPLE_RATE, f1b, f2b, f3b)
        expected = _apply_sequentially(wave, f1b, f2b, f3b)

        torch.testing.assert_close(result.ys, expected, atol=1e-6, rtol=1e-6)

    def test_filterchain_fusion(self):
        """Wave | (iir1 | iir2 | iir3) — FilterChain is flattened then fused."""
        wave = _make_wave()
        f1a = LoButterworth(cutoff=4000, order=2)
        f2a = HiButterworth(cutoff=200, order=2)
        f3a = LoButterworth(cutoff=6000, order=2)
        result = wave | (f1a | f2a | f3a)

        f1b = LoButterworth(cutoff=4000, order=2)
        f2b = HiButterworth(cutoff=200, order=2)
        f3b = LoButterworth(cutoff=6000, order=2)
        _prepare(SAMPLE_RATE, f1b, f2b, f3b)
        expected = _apply_sequentially(wave, f1b, f2b, f3b)

        torch.testing.assert_close(result.ys, expected, atol=1e-6, rtol=1e-6)

    def test_sequential_fusion(self):
        """Wave | nn.Sequential(iir1, iir2, iir3) — Sequential children fused."""
        wave = _make_wave()
        f1a = LoButterworth(cutoff=4000, order=2)
        f2a = HiButterworth(cutoff=200, order=2)
        f3a = LoButterworth(cutoff=6000, order=2)
        result = wave | nn.Sequential(f1a, f2a, f3a)

        f1b = LoButterworth(cutoff=4000, order=2)
        f2b = HiButterworth(cutoff=200, order=2)
        f3b = LoButterworth(cutoff=6000, order=2)
        _prepare(SAMPLE_RATE, f1b, f2b, f3b)
        expected = _apply_sequentially(wave, f1b, f2b, f3b)

        torch.testing.assert_close(result.ys, expected, atol=1e-6, rtol=1e-6)

    def test_mixed_chain_fuses_iir_runs_separately(self):
        """IIR runs separated by non-IIR effects are fused independently."""
        wave = _make_wave()
        f1a = LoButterworth(cutoff=4000, order=2)
        f2a = HiButterworth(cutoff=200, order=2)
        gain_a = Gain(0.5)
        f3a = LoButterworth(cutoff=6000, order=2)
        f4a = HiButterworth(cutoff=100, order=2)
        result = wave | f1a | f2a | gain_a | f3a | f4a

        f1b = LoButterworth(cutoff=4000, order=2)
        f2b = HiButterworth(cutoff=200, order=2)
        gain_b = Gain(0.5)
        f3b = LoButterworth(cutoff=6000, order=2)
        f4b = HiButterworth(cutoff=100, order=2)
        _prepare(SAMPLE_RATE, f1b, f2b, gain_b, f3b, f4b)
        expected = _apply_sequentially(wave, f1b, f2b, gain_b, f3b, f4b)

        torch.testing.assert_close(result.ys, expected, atol=1e-6, rtol=1e-6)

    def test_single_iir_no_fusion(self):
        """A single IIR filter should not be wrapped in FusedSOSCascade."""
        wave = _make_wave()
        f1a = LoButterworth(cutoff=4000, order=2)
        gain_a = Gain(0.8)
        result = wave | f1a | gain_a

        f1b = LoButterworth(cutoff=4000, order=2)
        gain_b = Gain(0.8)
        _prepare(SAMPLE_RATE, f1b, gain_b)
        expected = _apply_sequentially(wave, f1b, gain_b)

        torch.testing.assert_close(result.ys, expected, atol=1e-6, rtol=1e-6)

    def test_no_mutation_of_original_filters(self):
        """Piping should not mutate the original filter objects' state tensors."""
        wave = _make_wave()
        f1 = LoButterworth(cutoff=4000, order=2, fs=SAMPLE_RATE)
        f1.compute_coefficients()
        sos_before = f1._sos.clone()

        _ = wave | f1
        torch.testing.assert_close(f1._sos, sos_before)


# ---------- Lazy materialization ----------


class TestLazyMaterialization:
    def test_pipeline_deferred_until_ys_access(self):
        """Pipeline should not execute until .ys is accessed."""
        wave = _make_wave()
        f1 = LoButterworth(cutoff=4000, order=2)

        w2 = wave | f1
        assert len(w2._pipeline) == 1

        _ = w2.ys  # triggers materialization
        assert len(w2._pipeline) == 0

    def test_direct_ys_assignment_clears_pipeline(self):
        """Setting .ys directly should clear any pending pipeline."""
        wave = _make_wave()
        f1 = LoButterworth(cutoff=4000, order=2)

        w2 = wave | f1
        assert len(w2._pipeline) == 1

        new_tensor = torch.randn(1, N_SAMPLES, dtype=torch.float64)
        w2.ys = new_tensor
        assert w2._pipeline == []
        torch.testing.assert_close(w2.ys, new_tensor)


# ---------- FilterChain ----------


class TestFilterChain:
    def test_flattening(self):
        """(f1 | f2) | f3 should produce a flat chain with 3 children."""
        f1 = LoButterworth(cutoff=4000, order=2, fs=SAMPLE_RATE)
        f2 = HiButterworth(cutoff=200, order=2, fs=SAMPLE_RATE)
        f3 = LoButterworth(cutoff=6000, order=2, fs=SAMPLE_RATE)

        chain = (f1 | f2) | f3
        assert isinstance(chain, FilterChain)
        children = list(chain.children())
        assert len(children) == 3

    def test_cross_type_chaining(self):
        """Gain | IIR should produce a FilterChain."""
        gain = Gain(0.5)
        f1 = LoButterworth(cutoff=4000, order=2, fs=SAMPLE_RATE)

        chain = gain | f1
        assert isinstance(chain, FilterChain)
        children = list(chain.children())
        assert len(children) == 2

    def test_mixed_biquad_iir_fusion(self):
        """Biquad + IIR filters in a chain should auto-fuse."""
        wave = _make_wave()
        bq_a = BiquadLPF(cutoff=3000, q=0.707)
        f1a = LoButterworth(cutoff=4000, order=2)
        result = wave | bq_a | f1a

        bq_b = BiquadLPF(cutoff=3000, q=0.707)
        f1b = LoButterworth(cutoff=4000, order=2)
        _prepare(SAMPLE_RATE, bq_b, f1b)
        expected = _apply_sequentially(wave, bq_b, f1b)

        torch.testing.assert_close(result.ys, expected, atol=1e-6, rtol=1e-6)

    def test_fx_or_returns_not_implemented_for_non_module(self):
        """FX.__or__ with a non-Module should return NotImplemented."""
        gain = Gain(0.5)
        result = gain.__or__(42)
        assert result is NotImplemented

    def test_filterchain_ror_returns_not_implemented(self):
        """FilterChain.__ror__ always returns NotImplemented."""
        f1 = LoButterworth(cutoff=4000, order=2, fs=SAMPLE_RATE)
        f2 = HiButterworth(cutoff=200, order=2, fs=SAMPLE_RATE)
        chain = f1 | f2
        assert chain.__ror__("anything") is NotImplemented
