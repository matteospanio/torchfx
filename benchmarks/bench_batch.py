"""Benchmark: batched multi-signal processing vs one-at-a-time (issue #19).

Processing many short signals one at a time underuses the GPU (few threads per launch,
one launch per file). ``fx.batch_process`` concatenates them on the channel dimension and
issues a single launch over all channels. This script reports the wall-clock of both for a
sweep of file counts, on CPU and (if available) CUDA.

Run::

    uv run python benchmarks/bench_batch.py
"""

from __future__ import annotations

import time

import torch

import torchfx as fx
from torchfx.filter.iir import LoButterworth

FS = 48000


def _design():
    return LoButterworth(4000, order=8)


def _bench(device: str, n_files: int, channels: int, samples: int, reps: int = 5):
    waves = [
        fx.Wave(torch.randn(channels, samples, dtype=torch.float32, device=device), FS)
        for _ in range(n_files)
    ]

    def one_by_one():
        return [(w | _design()).ys for w in waves]

    def batched():
        return fx.batch_process(waves, _design())

    def timeit(fn):
        fn()  # warmup
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        if device == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps * 1e3  # ms

    return timeit(one_by_one), timeit(batched)


def main() -> None:
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    channels, samples = 2, FS  # stereo, 1 s
    for dev in devices:
        print(f"\n# {dev} | stereo x 1s @ {FS} Hz | 8th-order Butterworth")
        print("# files | one-by-one |   batched |  speedup")
        for n in (8, 32, 128, 512):
            t_obo, t_bat = _bench(dev, n, channels, samples)
            print(f"  {n:5d} | {t_obo:8.2f}ms | {t_bat:7.2f}ms | {t_obo / t_bat:6.2f}x")


if __name__ == "__main__":
    main()
