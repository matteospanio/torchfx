# ruff: noqa: A001
"""Streaming-state continuity for the dynamics effects and the reverb (#72 / #86).

The native kernels now thread their per-channel recurrence state through the
effect instance, so processing a signal in chunks must equal processing it in one
shot — the property that makes ``StreamProcessor`` / ``RealtimeProcessor`` use of
these effects artifact-free at chunk boundaries.

Limiter caveat: its look-ahead window deliberately does not cross chunk
boundaries (an exact match would require L samples of output latency), so chunked
equality is only exact for ``lookahead=0``; with look-ahead we assert the
brick-wall property and gain-state continuity instead.

"""

from __future__ import annotations

import pytest
import torch

from torchfx.effect import Compressor, Expander, Gate, Limiter, Reverb


def _signal(n: int = 8192, ch: int = 2, scale: float = 1.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(42)
    return torch.randn(ch, n, generator=g) * scale


def _chunked(effect, x: torch.Tensor, n_chunks: int = 4) -> torch.Tensor:
    return torch.cat([effect(c) for c in torch.chunk(x, n_chunks, dim=1)], dim=1)


class TestCompressorStreaming:
    @pytest.mark.parametrize("detector", ["peak", "rms"])
    def test_chunked_equals_oneshot(self, detector):
        x = _signal(scale=2.0)
        one = Compressor(threshold=-10, ratio=4, detector=detector, fs=48000)
        chk = Compressor(threshold=-10, ratio=4, detector=detector, fs=48000)
        torch.testing.assert_close(_chunked(chk, x), one(x))

    def test_reset_state_restores_fresh_behavior(self):
        x = _signal(scale=2.0)
        eff = Compressor(threshold=-10, ratio=4, fs=48000)
        first = eff(x)
        eff.reset_state()
        torch.testing.assert_close(eff(x), first)

    def test_state_persists_without_reset(self):
        """Without a reset, the second call continues from the first call's state."""
        x = _signal(scale=2.0)
        eff = Compressor(threshold=-10, ratio=4, release=0.2, fs=48000)
        first = eff(x)
        second = eff(x)  # detector envelope is already charged -> different output
        assert not torch.allclose(first, second)

    def test_channel_count_change_self_heals(self):
        eff = Compressor(threshold=-10, ratio=4, fs=48000)
        eff(_signal(ch=2, scale=2.0))
        out = eff(_signal(ch=4, scale=2.0))  # stale [2,3] state must be dropped
        assert out.shape == (4, 8192)


class TestExpanderStreaming:
    @pytest.mark.parametrize("cls,kwargs", [(Expander, {"ratio": 3.0}), (Gate, {})])
    def test_chunked_equals_oneshot(self, cls, kwargs):
        x = _signal(scale=0.1)
        one = cls(threshold=-25, fs=48000, **kwargs)
        chk = cls(threshold=-25, fs=48000, **kwargs)
        torch.testing.assert_close(_chunked(chk, x), one(x))

    def test_reset_state_restores_fresh_behavior(self):
        x = _signal(scale=0.1)
        eff = Expander(threshold=-25, ratio=3.0, fs=48000)
        first = eff(x)
        eff.reset_state()
        torch.testing.assert_close(eff(x), first)


class TestLimiterStreaming:
    def test_chunked_equals_oneshot_no_lookahead(self):
        x = _signal(scale=3.0)
        one = Limiter(threshold=-1.0, lookahead=0.0, release=0.05, fs=48000)
        chk = Limiter(threshold=-1.0, lookahead=0.0, release=0.05, fs=48000)
        torch.testing.assert_close(_chunked(chk, x), one(x))

    def test_brick_wall_holds_across_chunks_with_lookahead(self):
        x = _signal(scale=3.0)
        eff = Limiter(threshold=-1.0, lookahead=0.005, release=0.05, fs=48000)
        out = _chunked(eff, x, n_chunks=8)
        ceiling = 10 ** (-1.0 / 20)
        assert out.abs().max() <= ceiling + 1e-4

    def test_gain_state_carried(self):
        """A loud first chunk leaves g < 1; a quiet second chunk shows the release
        tail."""
        loud = torch.ones(1, 4096) * 3.0
        quiet = torch.ones(1, 4096) * 0.1

        streaming = Limiter(threshold=-1.0, lookahead=0.0, release=0.5, fs=48000)
        streaming(loud)
        out_carried = streaming(quiet)

        fresh = Limiter(threshold=-1.0, lookahead=0.0, release=0.5, fs=48000)
        out_fresh = fresh(quiet)

        # With carried gain the quiet chunk starts attenuated (release still recovering).
        assert out_carried[0, 0] < out_fresh[0, 0]


class TestReverbStreaming:
    def test_chunked_equals_oneshot(self):
        x = _signal()
        one = Reverb(room_size=0.7, damping=0.4, mix=0.3, fs=48000)
        chk = Reverb(room_size=0.7, damping=0.4, mix=0.3, fs=48000)
        torch.testing.assert_close(_chunked(chk, x), one(x))

    def test_tail_flows_across_chunks(self):
        """After a burst, a silent chunk must contain the reverb tail (not silence)."""
        burst = torch.zeros(1, 4096)
        burst[0, 0] = 1.0
        silence = torch.zeros(1, 4096)

        eff = Reverb(room_size=0.8, damping=0.2, mix=1.0, fs=48000)
        eff(burst)
        tail = eff(silence)
        assert tail.abs().max() > 1e-4

    def test_reset_state_cuts_tail(self):
        burst = torch.zeros(1, 4096)
        burst[0, 0] = 1.0
        silence = torch.zeros(1, 4096)

        eff = Reverb(room_size=0.8, damping=0.2, mix=1.0, fs=48000)
        eff(burst)
        eff.reset_state()
        tail = eff(silence)
        torch.testing.assert_close(tail, silence)

    def test_fs_change_invalidates_state(self):
        """State from one fs (different scratch width) must not be reused at another."""
        x = _signal(n=4096, ch=1)
        eff = Reverb(room_size=0.7, fs=44100)
        eff(x)
        eff.fs = 48000
        out = eff(x)  # must not crash; fresh state at the new fs
        fresh = Reverb(room_size=0.7, fs=48000)
        torch.testing.assert_close(out, fresh(x))


class TestWaveIsolation:
    def test_offline_wave_reuse_does_not_leak_state(self):
        """Wave materialisation resets effect state, so reuse across Waves is clean."""
        import torchfx as fx

        data = _signal(scale=2.0)
        eff = Compressor(threshold=-10, ratio=4, fs=48000)

        out_a = (fx.Wave(data.clone(), fs=48000) | eff).ys
        out_b = (fx.Wave(data.clone(), fs=48000) | eff).ys
        torch.testing.assert_close(out_a, out_b)
