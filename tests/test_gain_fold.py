"""Static-gain folding in the fusion planner (Roadmap Epic E1).

A static linear ``Gain`` (clamp=False) between SOS filters is folded into the fused
cascade's numerator instead of breaking the fused run; a clamping ``Gain`` or a
dynamic ``Normalize`` is non-linear and stays as its own stage. Folding is exact
(a scalar commutes through a linear filter).

"""

from __future__ import annotations

import torch

import torchfx as fx
from torchfx.effect import Gain, Normalize
from torchfx.filter import HiButterworth, LoButterworth
from torchfx.filter.fused import FusedSOSCascade

FS = 48000


def _signal() -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.randn(2, 4096, generator=g, dtype=torch.float64)


# ---------- planner structure ----------


def test_static_gain_does_not_break_fusion():
    plan = fx.Wave._build_plan(
        [
            LoButterworth(4000, order=4, fs=FS),
            Gain(2.0),
            HiButterworth(200, order=4, fs=FS),
        ]
    )
    assert len(plan) == 1
    assert isinstance(plan[0], FusedSOSCascade)


def test_clamp_gain_breaks_fusion():
    plan = fx.Wave._build_plan(
        [
            LoButterworth(4000, order=4, fs=FS),
            Gain(2.0, clamp=True),  # non-linear -> not folded
            HiButterworth(200, order=4, fs=FS),
        ]
    )
    assert len(plan) == 3
    assert isinstance(plan[1], Gain)


def test_normalize_breaks_fusion():
    plan = fx.Wave._build_plan(
        [
            LoButterworth(4000, order=4, fs=FS),
            Normalize(peak=0.8),  # dynamic / non-linear -> not folded
            HiButterworth(200, order=4, fs=FS),
        ]
    )
    assert len(plan) == 3
    assert isinstance(plan[1], Normalize)


def test_standalone_gain_kept_as_op():
    plan = fx.Wave._build_plan([Gain(2.0)])
    assert len(plan) == 1
    assert isinstance(plan[0], Gain)


# ---------- numerical correctness ----------


def test_folded_gain_matches_unfused():
    data = _signal()
    out = (
        fx.Wave(data.clone(), fs=FS)
        | LoButterworth(4000, order=4)
        | Gain(2.0)
        | HiButterworth(200, order=4)
    ).ys

    f1 = LoButterworth(4000, order=4, fs=FS)
    f2 = HiButterworth(200, order=4, fs=FS)
    ref = f2(2.0 * f1(data.clone()))

    torch.testing.assert_close(out, ref)


def test_gain_at_start_and_end_accumulates():
    data = _signal()
    out = (fx.Wave(data.clone(), fs=FS) | Gain(2.0) | LoButterworth(4000, order=4) | Gain(0.25)).ys

    f = LoButterworth(4000, order=4, fs=FS)
    ref = 0.25 * f(2.0 * data.clone())  # == 0.5 * f(data)

    torch.testing.assert_close(out, ref)
    plan = fx.Wave._build_plan([Gain(2.0), LoButterworth(4000, order=4, fs=FS), Gain(0.25)])
    assert len(plan) == 1  # both gains folded into the single cascade


def test_db_gain_folds():
    data = _signal()
    out = (fx.Wave(data.clone(), fs=FS) | LoButterworth(4000, order=4) | Gain(6.0, "db")).ys

    f = LoButterworth(4000, order=4, fs=FS)
    ref = (10 ** (6.0 / 20)) * f(data.clone())

    torch.testing.assert_close(out, ref)
