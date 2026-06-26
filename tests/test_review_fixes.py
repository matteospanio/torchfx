# ruff: noqa: A001
"""Regression tests for the IS2-review fixes (issues #84, #85, #88, #90-#93).

Each test pins one verified finding from the 2026-06 code review:

- #84  FusedSOSCascade recomputes on fs change (the realtime ``effect.fs = ...``
  assignment path).
- #85  DesignableFIR redesigns on fs change and refuses to run undesigned.
- #88  IIR/Biquad/DesignableFIR redesign when a design parameter is mutated
  after the first forward.
- #90  fft path matches scipy.lfilter for signals shorter than the kernel
  (verified NOT a bug — pinned here so it stays correct).
- #91  MusicalTime.duration_seconds raises ValueError (not assert) on bad BPM.
- #92  batch_process rejects channel-coupled (global Normalize) effects.
- #93  Wave.merge validates devices like it validates fs.

"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy import signal

import torchfx as fx
from torchfx.filter import FIR, DesignableFIR, LoButterworth
from torchfx.filter.biquad import BiquadLPF
from torchfx.filter.fused import FusedSOSCascade
from torchfx.typing import MusicalTime


def _signal(n: int = 2000, ch: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.randn(ch, n, generator=g, dtype=torch.float64)


# --------------------------------------------------------------------------- #
# #84 — FusedSOSCascade fs-change guard.                                      #
# --------------------------------------------------------------------------- #


class TestFusedCascadeFsChange:
    def test_recomputes_on_fs_change(self):
        """A fused cascade reused across sample rates matches a fresh design."""
        x = _signal()
        fused = FusedSOSCascade(
            LoButterworth(cutoff=100, order=2, fs=1000),
            BiquadLPF(cutoff=200, q=0.707, fs=1000),
        )
        fused(x)  # materialize at fs=1000
        assert fused._coeff_fs == 1000

        # The RealtimeProcessor path: a plain attribute assignment.
        fused.fs = 2000
        out_reused = fused(x)
        assert fused._coeff_fs == 2000

        fresh = FusedSOSCascade(
            LoButterworth(cutoff=100, order=2, fs=2000),
            BiquadLPF(cutoff=200, q=0.707, fs=2000),
        )
        torch.testing.assert_close(out_reused, fresh(x))

    def test_fs_change_resets_state(self):
        """Accumulated DF1 state is dropped with the stale coefficients."""
        x = _signal()
        fused = FusedSOSCascade(LoButterworth(cutoff=100, order=4, fs=1000))
        for _ in range(3):
            fused(x)
        assert fused._state_y is not None

        fused.fs = 2000
        out_after = fused(x)

        fresh = FusedSOSCascade(LoButterworth(cutoff=100, order=4, fs=2000))
        torch.testing.assert_close(out_after, fresh(x))

    def test_gain_fold_survives_recompute(self):
        """The folded static gain is re-applied to the rebuilt SOS matrix."""
        x = _signal()
        fused = FusedSOSCascade(LoButterworth(cutoff=100, order=2, fs=1000), gain=0.5)
        fused(x)
        fused.fs = 2000
        out = fused(x)

        fresh = FusedSOSCascade(LoButterworth(cutoff=100, order=2, fs=2000), gain=0.5)
        torch.testing.assert_close(out, fresh(x))

    def test_no_rebuild_when_fs_unchanged(self):
        """Repeated forwards at a fixed fs must not rebuild the SOS matrix."""
        x = _signal()
        fused = FusedSOSCascade(LoButterworth(cutoff=100, order=2, fs=1000))
        fused(x)
        sos_first = fused._sos
        fused(x)
        assert fused._sos is sos_first


# --------------------------------------------------------------------------- #
# #85 — DesignableFIR fs guard + no silent passthrough.                       #
# --------------------------------------------------------------------------- #


class TestDesignableFIRFsChange:
    def test_redesigns_on_fs_change(self):
        x = _signal()
        reused = DesignableFIR(cutoff=100, num_taps=31, fs=1000)
        reused(x)
        assert reused._coeff_fs == 1000

        reused.fs = 2000
        out_reused = reused(x)
        assert reused._coeff_fs == 2000

        fresh = DesignableFIR(cutoff=100, num_taps=31, fs=2000)
        torch.testing.assert_close(out_reused, fresh(x))

    def test_undesigned_forward_raises(self):
        """Fs=None + no design must raise, not pass audio through unfiltered."""
        f = DesignableFIR(cutoff=100, num_taps=31)  # fs deferred
        with pytest.raises(ValueError, match="fs"):
            f(_signal())

    def test_wave_pipe_still_designs(self):
        """The Wave pipe path designs the deferred filter exactly once."""
        data = _signal()
        f = DesignableFIR(cutoff=100, num_taps=31)
        out = (fx.Wave(data.clone(), fs=1000) | f).ys
        fresh = DesignableFIR(cutoff=100, num_taps=31, fs=1000)
        torch.testing.assert_close(out, fresh(data.clone()))


# --------------------------------------------------------------------------- #
# #88 — design-parameter mutation triggers redesign.                          #
# --------------------------------------------------------------------------- #


class TestParameterMutation:
    def test_iir_cutoff_mutation(self):
        x = _signal()
        f = LoButterworth(cutoff=100, order=4, fs=1000)
        f(x)
        f.cutoff = 250
        out = f(x)
        fresh = LoButterworth(cutoff=250, order=4, fs=1000)
        torch.testing.assert_close(out, fresh(x))

    def test_iir_order_mutation(self):
        x = _signal()
        f = LoButterworth(cutoff=100, order=2, fs=1000)
        f(x)
        f.order = 6
        out = f(x)
        fresh = LoButterworth(cutoff=100, order=6, fs=1000)
        torch.testing.assert_close(out, fresh(x))

    def test_biquad_q_mutation(self):
        x = _signal()
        f = BiquadLPF(cutoff=100, q=0.707, fs=1000)
        f(x)
        f.q = 2.0
        out = f(x)
        fresh = BiquadLPF(cutoff=100, q=2.0, fs=1000)
        torch.testing.assert_close(out, fresh(x))

    def test_designable_fir_cutoff_mutation(self):
        x = _signal()
        f = DesignableFIR(cutoff=100, num_taps=31, fs=1000)
        f(x)
        f.cutoff = 250
        out = f(x)
        fresh = DesignableFIR(cutoff=250, num_taps=31, fs=1000)
        torch.testing.assert_close(out, fresh(x))

    def test_no_redesign_without_mutation(self):
        x = _signal()
        f = LoButterworth(cutoff=100, order=4, fs=1000)
        f(x)
        sos_first = f._sos
        f(x)
        assert f._sos is sos_first


# --------------------------------------------------------------------------- #
# #90 — short-signal FIR correctness (pinned: matches scipy.lfilter).         #
# --------------------------------------------------------------------------- #


class TestShortSignalFIR:
    @pytest.mark.parametrize("T,K", [(1, 100), (5, 16), (3, 4), (50, 64)])
    @pytest.mark.parametrize("conv_mode", ["fft", "direct"])
    def test_matches_lfilter_when_signal_shorter_than_kernel(self, T, K, conv_mode):
        b = np.ones(K) / K
        x = np.random.RandomState(0).randn(T)
        ref = signal.lfilter(b, [1.0], x)
        out = FIR(b=b, conv_mode=conv_mode)(torch.tensor(x, dtype=torch.float64))
        torch.testing.assert_close(
            out, torch.tensor(ref, dtype=torch.float64), atol=1e-8, rtol=1e-6
        )


# --------------------------------------------------------------------------- #
# #91 — MusicalTime contract.                                                 #
# --------------------------------------------------------------------------- #


class TestMusicalTimeValidation:
    @pytest.mark.parametrize("bpm", [0, -1, -120.5])
    def test_nonpositive_bpm_raises_value_error(self, bpm):
        mt = MusicalTime.from_string("1/4")
        with pytest.raises(ValueError, match="BPM must be positive"):
            mt.duration_seconds(bpm)


# --------------------------------------------------------------------------- #
# #92 — batch_process channel-coupling guard.                                 #
# --------------------------------------------------------------------------- #


class TestBatchProcessGuard:
    def _waves(self, n: int = 3) -> list[fx.Wave]:
        g = torch.Generator().manual_seed(0)
        return [fx.Wave(torch.randn(1, 512, generator=g), 48000) for _ in range(n)]

    def test_global_normalize_rejected(self):
        from torchfx.effect import Normalize

        with pytest.raises(ValueError, match="channel-coupled"):
            fx.batch_process(self._waves(), Normalize(peak=1.0))

    def test_per_channel_normalize_allowed(self):
        from torchfx.effect import Normalize, PerChannelNormalizationStrategy

        out = fx.batch_process(
            self._waves(), Normalize(peak=1.0, strategy=PerChannelNormalizationStrategy())
        )
        assert len(out) == 3

    def test_filters_still_allowed(self):
        out = fx.batch_process(self._waves(), LoButterworth(cutoff=4000, order=4))
        assert len(out) == 3


# --------------------------------------------------------------------------- #
# #93 — Wave.merge device validation.                                         #
# --------------------------------------------------------------------------- #


class TestMergeDeviceValidation:
    def test_same_device_ok(self):
        a = fx.Wave(torch.randn(1, 100), 48000)
        b = fx.Wave(torch.randn(1, 100), 48000)
        merged = fx.Wave.merge([a, b], split_channels=True)
        assert merged.channels() == 2

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a second device")
    def test_device_mismatch_raises(self):
        a = fx.Wave(torch.randn(1, 100), 48000)
        b = fx.Wave(torch.randn(1, 100), 48000, device="cuda")
        with pytest.raises(ValueError, match="[Dd]evice mismatch"):
            fx.Wave.merge([a, b], split_channels=True)
