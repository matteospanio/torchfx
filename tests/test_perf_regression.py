"""Performance-regression gates (issue #21).

CI runners are shared and noisy, so an absolute wall-time budget against a stored baseline
is flaky. These gates instead assert the *deterministic* invariants that the headline perf
wins depend on — the fusion planner must keep collapsing a depth-K IIR cascade (and a
static ``Gain`` folded between sections) into a **single** native dispatch — plus one
**relative** wall-time smoke (fused is not dramatically slower than the unfused path),
which is machine-speed-independent and has a large margin. If fusion silently regresses,
the dispatch count jumps and these fail in ordinary CI.

The dispatch count is measured by wrapping the ``torchfx._ops`` dispatch functions, the
same mechanism as ``tools/count_kernel_launches.py``.

"""

from __future__ import annotations

import time
from contextlib import contextmanager
from functools import reduce
from typing import Any

import pytest
import torch

from torchfx import Wave, _ops
from torchfx.effect import Gain
from torchfx.filter.iir import HiButterworth, LoButterworth

FS = 16000
N = 8192


@contextmanager
def count_native():
    """Count Python-level dispatches into the native extension (biquad / sos /
    delay)."""
    counts = {"biquad": 0, "sos": 0, "delay": 0}
    originals = (_ops.biquad_forward, _ops.parallel_iir_forward, _ops.delay_line_forward)

    def make(key: str, fn: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            counts[key] += 1
            return fn(*args, **kwargs)

        return wrapper

    _ops.biquad_forward = make("biquad", originals[0])
    _ops.parallel_iir_forward = make("sos", originals[1])
    _ops.delay_line_forward = make("delay", originals[2])
    try:
        yield counts
    finally:
        _ops.biquad_forward, _ops.parallel_iir_forward, _ops.delay_line_forward = originals


def _wave() -> Wave:
    gen = torch.Generator().manual_seed(0)
    return Wave(torch.randn(1, N, generator=gen, dtype=torch.float64), FS)


def _filters(k: int) -> list:
    out = []
    for i in range(k):
        if i % 2 == 0:
            out.append(LoButterworth(cutoff=2000 + 200 * i, order=2))
        else:
            out.append(HiButterworth(cutoff=100 + 10 * i, order=2))
    return out


# --------------------------------------------------------------------------- #
# Deterministic fusion-dispatch invariants (the perf-regression core)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k", [2, 5, 10])
def test_fused_chain_is_single_dispatch(k):
    """A depth-K IIR cascade must fuse to exactly one native SOS dispatch."""
    wave = _wave()
    chain = reduce(lambda a, b: a | b, _filters(k))
    with count_native() as counts:
        _ = (wave | chain).ys
    assert counts["sos"] == 1, f"depth-{k} cascade should fuse to 1 SOS dispatch, got {counts}"
    assert counts["biquad"] == 0


@pytest.mark.parametrize("k", [2, 5, 10])
def test_unfused_chain_is_k_dispatches(k):
    """The reference (each filter applied separately) issues K dispatches — the baseline
    the fused path must keep beating."""
    wave = _wave()
    filters = _filters(k)
    for f in filters:
        f.fs = FS
        f.compute_coefficients()
    data = wave.ys.clone()
    with count_native() as counts:
        for f in filters:
            data = f(data)
    assert counts["sos"] == k


def test_static_gain_between_iir_folds_to_single_dispatch():
    """A constant `Gain` between two IIR filters must fold into the cascade (0.6.0 win),
    so the chain still collapses to one dispatch."""
    wave = _wave()
    chain = LoButterworth(cutoff=4000, order=4) | Gain(2.0) | LoButterworth(cutoff=6000, order=4)
    with count_native() as counts:
        _ = (wave | chain).ys
    assert counts["sos"] == 1, f"a folded Gain must not add a dispatch, got {counts}"


# --------------------------------------------------------------------------- #
# Relative wall-time smoke (machine-independent, large margin -> not flaky)
# --------------------------------------------------------------------------- #
def _median_wall(fn, *, warmup=3, iters=15) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        samples.append(time.perf_counter_ns() - t0)
    samples.sort()
    return samples[len(samples) // 2]


def test_fused_forward_not_slower_than_unfused():
    """Compare *forward* cost only (both modules pre-built): the fused single-dispatch
    cascade must not be slower than K separate filter forwards.

    Normally it is several times faster, so the margin (fused < 1.1x unfused) keeps this
    robust on noisy CI.

    """
    k = 10
    data = _wave().ys.clone()

    # Fused module, built once via the planner.
    fused_filters = _filters(k)
    for f in fused_filters:
        f.fs = FS
    plan = Wave._build_plan(fused_filters)
    assert len(plan) == 1  # the whole chain collapsed
    fused_mod = plan[0]

    # Unfused: K prepared filters applied in sequence.
    sep = _filters(k)
    for f in sep:
        f.fs = FS
        f.compute_coefficients()

    def run_fused():
        return fused_mod(data)

    def run_unfused():
        d = data
        for f in sep:
            d = f(d)
        return d

    fused = _median_wall(run_fused)
    unfused = _median_wall(run_unfused)
    assert fused < unfused * 1.1, f"fused={fused}ns vs unfused={unfused}ns — fusion regressed"
