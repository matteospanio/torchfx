"""FP32-vs-FP64 CUDA throughput for the native SOS IIR path (Roadmap Epic B / B5).

Standalone CUDA-event benchmark: for an Nth-order Butterworth cascade, times the
FP64 and FP32 GPU kernels across (length, channels) and reports the FP32 speedup,
plus the multichannel "inversion" check (does FP32 let the consumer GPU beat its
own CPU again?). Run on a CUDA host:

    python benchmarks/bench_fp32_speedup.py

Measured on an RTX 3070 (1:32 FP32:FP64), 8th-order Butterworth @ 48 kHz:
~3.0-3.6x FP32 speedup; 60s/8ch GPU FP32 (~24 ms) beats CPU (~28 ms) where GPU
FP64 (~89 ms) lost.

"""

from __future__ import annotations

import statistics
import time

import torch

from torchfx.filter import LoButterworth

FS, ORDER = 48000, 8


def bench_gpu(t_sec: float, c: int, dtype: torch.dtype, iters: int = 30, warmup: int = 8) -> float:
    n = int(t_sec * FS)
    x = torch.randn(c, n, dtype=dtype, device="cuda")
    f = LoButterworth(4000, order=ORDER, fs=FS)
    for _ in range(warmup):
        f(x)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        s.record()
        f(x)
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def bench_cpu(t_sec: float, c: int, dtype: torch.dtype, iters: int = 15, warmup: int = 3) -> float:
    n = int(t_sec * FS)
    x = torch.randn(c, n, dtype=dtype)
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
    if not torch.cuda.is_available():
        print("CUDA not available — this benchmark requires a GPU.")
        return

    name = torch.cuda.get_device_name(0)
    print(f"{name} | {ORDER}th-order Butterworth ({ORDER // 2} SOS) @ {FS // 1000}kHz | median ms")
    print(f"{'workload':>14} | {'GPU f64':>9} | {'GPU f32':>9} | {'f32 speedup':>11}")
    cases = [(30, 1), (60, 1), (60, 2), (60, 4), (60, 8)]
    res: dict[tuple[int, int], tuple[float, float]] = {}
    for t, c in cases:
        g64 = bench_gpu(t, c, torch.float64)
        g32 = bench_gpu(t, c, torch.float32)
        res[(t, c)] = (g64, g32)
        print(f"{f'{t}s/{c}ch':>14} | {g64:9.3f} | {g32:9.3f} | {g64 / g32:10.2f}x")

    c64 = bench_cpu(60, 8, torch.float64)
    c32 = bench_cpu(60, 8, torch.float32)
    g64, g32 = res[(60, 8)]
    cpu_best = min(c64, c32)
    gpu64_verdict = "loses to" if g64 > cpu_best else "beats"
    gpu32_verdict = "beats" if g32 < cpu_best else "loses to"
    print()
    print("60s/8ch inversion check (where consumer-GPU FP64 lost to CPU):")
    print(f"  CPU f64 = {c64:.1f} ms | CPU f32 = {c32:.1f} ms")
    print(f"  GPU f64 = {g64:.1f} ms ({gpu64_verdict} CPU)")
    print(f"  GPU f32 = {g32:.1f} ms ({gpu32_verdict} CPU)")


if __name__ == "__main__":
    main()
