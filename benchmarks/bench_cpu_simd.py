"""CPU cross-channel SIMD SOS throughput vs channel count (Roadmap Epic F1).

Times a multichannel IIR cascade on the CPU. The dispatch between the scalar OpenMP-
over-channels kernel and the cross-channel SIMD kernel is controlled by the
TORCHFX_SIMD_MIN_CHANNELS env var (read once per process), so run this twice to A/B:

TORCHFX_SIMD_MIN_CHANNELS=999999 python benchmarks/bench_cpu_simd.py   # force scalar
TORCHFX_SIMD_MIN_CHANNELS=1      python benchmarks/bench_cpu_simd.py   # force SIMD

The SIMD win grows with channels-per-core, so it shows most at high channel counts (and
on few-core edge devices like the Raspberry Pi 5).

"""

from __future__ import annotations

import os
import statistics
import time

import torch

from torchfx.filter import LoButterworth

FS, ORDER = 48000, 8  # 4 SOS sections


def bench(c: int, t_sec: float = 10.0, iters: int = 10, warmup: int = 3) -> float:
    n = int(t_sec * FS)
    x = torch.randn(c, n, dtype=torch.float32)
    f = LoButterworth(4000, order=ORDER, fs=FS)
    for _ in range(warmup):
        f(x)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        f(x)
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def main() -> None:
    torch.set_num_threads(torch.get_num_threads())
    mode = os.environ.get("TORCHFX_SIMD_MIN_CHANNELS", "(default 16)")
    print(f"CPU SOS cascade | {ORDER}th-order Butterworth (4 SOS) | 10s @ 48kHz | float32")
    print(f"torch threads = {torch.get_num_threads()} | TORCHFX_SIMD_MIN_CHANNELS = {mode}")
    print(f"{'channels':>9} | {'median ms':>10}")
    for c in (2, 4, 8, 16, 32, 64):
        print(f"{c:9d} | {bench(c):10.1f}")


if __name__ == "__main__":
    main()
