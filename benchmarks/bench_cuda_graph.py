"""CUDA Graph replay vs eager per-chunk latency for streaming (Roadmap Epic C1).

Streams a fixed-size chunk through a multi-section SOS cascade and compares the per-
chunk wall time of eager dispatch vs a captured-and-replayed CUDA graph. The launch
overhead the graph amortizes (~135 us for the parallel scan on an RTX 3070) dominates
short chunks, so the win is largest at small chunk sizes — the realtime regime.

python benchmarks/bench_cuda_graph.py

"""

from __future__ import annotations

import statistics

import torch

from torchfx.filter import HiButterworth, LoButterworth
from torchfx.filter.biquad import BiquadLPF
from torchfx.filter.fused import FusedSOSCascade
from torchfx.realtime.cuda_graph import CudaGraphRunner

FS = 48000


def _chain() -> FusedSOSCascade:
    return FusedSOSCascade(
        HiButterworth(80, order=2, fs=FS),
        LoButterworth(8000, order=4, fs=FS),
        BiquadLPF(2000, 0.707, fs=FS),
    )


def _time(fn, iters: int = 200, warmup: int = 30) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)  # us
    return statistics.median(ts)


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available — this benchmark requires a GPU.")
        return
    print(f"{torch.cuda.get_device_name(0)} | 4-section SOS cascade | per-chunk median us")
    print(f"{'chunk':>8} | {'eager':>9} | {'graph':>9} | {'speedup':>8}")
    for chunk in (128, 256, 512, 1024, 2048, 4096):
        x = torch.randn(2, chunk, dtype=torch.float32, device="cuda")

        eager = _chain()
        eager_us = _time(lambda m=eager, xx=x: m(xx))

        runner = CudaGraphRunner(_chain(), x.clone())
        graph_us = _time(lambda r=runner, xx=x: r.run(xx))

        print(f"{chunk:8d} | {eager_us:9.1f} | {graph_us:9.1f} | {eager_us / graph_us:7.2f}x")


if __name__ == "__main__":
    main()
