"""Tests for batched multi-signal processing (issue #19)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import pytest
import torch

import torchfx as fx
from torchfx import Wave, _ops, batch_process
from torchfx.effect import Gain
from torchfx.filter.iir import HiButterworth, LoButterworth

FS = 48000


@contextmanager
def count_sos():
    """Count native SOS dispatches (to assert the batch is a single launch)."""
    n = {"sos": 0}
    orig = _ops.parallel_iir_forward

    def wrap(*a: Any, **k: Any) -> Any:
        n["sos"] += 1
        return orig(*a, **k)

    _ops.parallel_iir_forward = wrap
    try:
        yield n
    finally:
        _ops.parallel_iir_forward = orig


def _wave(c: int, t: int, seed: int) -> Wave:
    gen = torch.Generator().manual_seed(seed)
    return Wave(torch.randn(c, t, generator=gen, dtype=torch.float64), FS)


# --------------------------------------------------------------------------- #
# Numerical equivalence to per-file processing
# --------------------------------------------------------------------------- #
def test_batch_equals_per_file_filter():
    specs = [(1, 4000), (2, 6000), (1, 5000), (2, 3000)]
    waves = [_wave(c, t, i) for i, (c, t) in enumerate(specs)]
    out = batch_process(waves, LoButterworth(4000, order=8))
    assert len(out) == len(waves)
    for w, o in zip(waves, out, strict=True):
        ref = (w | LoButterworth(4000, order=8)).ys
        assert o.ys.shape == w.ys.shape
        torch.testing.assert_close(o.ys, ref, rtol=1e-9, atol=1e-12)


def test_batch_equals_per_file_chain():
    waves = [_wave(2, 4000 + 500 * i, i) for i in range(5)]
    out = batch_process(waves, HiButterworth(120, order=4) | LoButterworth(8000, order=4))
    for w, o in zip(waves, out, strict=True):
        ref = (w | (HiButterworth(120, order=4) | LoButterworth(8000, order=4))).ys
        torch.testing.assert_close(o.ys, ref, rtol=1e-9, atol=1e-12)


def test_batch_equals_per_file_gain():
    waves = [_wave(1, 2000, i) for i in range(4)]
    out = batch_process(waves, Gain(0.5))
    for w, o in zip(waves, out, strict=True):
        torch.testing.assert_close(o.ys, w.ys * 0.5, rtol=1e-9, atol=1e-12)


# --------------------------------------------------------------------------- #
# Single dispatch (the whole point: one launch for the batch)
# --------------------------------------------------------------------------- #
def test_batch_is_single_dispatch():
    waves = [_wave(2, 4000, i) for i in range(16)]
    with count_sos() as n:
        batch_process(waves, LoButterworth(4000, order=8))
    assert n["sos"] == 1, f"batch of 16 should be one SOS dispatch, got {n['sos']}"


def test_per_file_is_n_dispatches():
    waves = [_wave(2, 4000, i) for i in range(16)]
    with count_sos() as n:
        for w in waves:
            _ = (w | LoButterworth(4000, order=8)).ys
    assert n["sos"] == 16


# --------------------------------------------------------------------------- #
# Shapes, metadata, validation
# --------------------------------------------------------------------------- #
def test_preserves_shapes_and_metadata():
    w0 = Wave(torch.randn(2, 7000, dtype=torch.float64), FS, metadata={"src": "a"})
    w1 = Wave(torch.randn(1, 3000, dtype=torch.float64), FS, metadata={"src": "b"})
    out = batch_process([w0, w1], Gain(1.0))
    assert out[0].ys.shape == (2, 7000)
    assert out[1].ys.shape == (1, 3000)
    assert out[0].metadata == {"src": "a"}
    assert out[1].metadata == {"src": "b"}


def test_single_wave_batch():
    w = _wave(2, 5000, 0)
    out = batch_process([w], LoButterworth(4000, order=6))
    torch.testing.assert_close(out[0].ys, (w | LoButterworth(4000, order=6)).ys)


def test_empty_raises():
    with pytest.raises(ValueError, match="at least one"):
        batch_process([], Gain(1.0))


def test_mixed_fs_raises():
    w0 = Wave(torch.randn(1, 1000, dtype=torch.float64), 48000)
    w1 = Wave(torch.randn(1, 1000, dtype=torch.float64), 44100)
    with pytest.raises(ValueError, match="sampling rate"):
        batch_process([w0, w1], Gain(1.0))


def test_exported_at_top_level():
    assert fx.batch_process is batch_process


# --------------------------------------------------------------------------- #
# GPU: batch == per-file == CPU, in one launch
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_batch_gpu_matches_cpu():
    specs = [(1, 4000), (2, 6000), (2, 5000)]
    waves_cpu = [_wave(c, t, i) for i, (c, t) in enumerate(specs)]
    waves_gpu = [Wave(w.ys, FS, device="cuda") for w in waves_cpu]
    out_cpu = batch_process(waves_cpu, LoButterworth(4000, order=8))
    out_gpu = batch_process(waves_gpu, LoButterworth(4000, order=8))
    for oc, og in zip(out_cpu, out_gpu, strict=True):
        assert og.ys.is_cuda
        torch.testing.assert_close(oc.ys, og.ys.cpu(), rtol=1e-9, atol=1e-10)
