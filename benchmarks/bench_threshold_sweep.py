"""Dispatch-threshold crossover sweep (Roadmap Epic G2).

For a single biquad section, times the sequential CUDA kernel vs the Blelloch
parallel scan across (length, channels), using the force-routing hook added in
G2 (``threshold=0`` forces the parallel scan, a huge threshold forces sequential).
The crossover — the T at which the parallel scan overtakes the sequential kernel —
is what ``PARALLEL_SCAN_THRESHOLD`` should be set to, per device and dtype. Re-run
after any kernel change (the FP32 path of Epic B shifts the crossover vs FP64).

    python benchmarks/bench_threshold_sweep.py

"""

from __future__ import annotations

import statistics

import torch
from scipy.signal import butter

from torchfx._ops import PARALLEL_SCAN_THRESHOLD, parallel_iir_forward

FORCE_SEQ = 1 << 30  # T <= huge  -> always sequential
FORCE_PAR = 0  # T <= 0 never true -> always parallel scan

TS = [256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 4096, 8192]
CS = [1, 2, 4, 8]


def _time(
    t: int, c: int, threshold: int, dtype: torch.dtype, iters: int = 60, warmup: int = 15
) -> float:
    x = torch.randn(c, t, dtype=dtype, device="cuda")
    sos = torch.tensor(butter(2, 0.2, output="sos"), dtype=dtype, device="cuda")
    sos_cpu = sos.cpu()
    for _ in range(warmup):
        parallel_iir_forward(x, sos, None, None, sos_cpu=sos_cpu, threshold=threshold)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        s.record()
        parallel_iir_forward(x, sos, None, None, sos_cpu=sos_cpu, threshold=threshold)
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)  # us
    return statistics.median(ts)


def sweep(dtype: torch.dtype) -> None:
    print(f"\n=== {dtype} | current PARALLEL_SCAN_THRESHOLD={PARALLEL_SCAN_THRESHOLD} ===")
    for c in CS:
        crossover = None
        print(f"C={c}: (us)")
        for t in TS:
            seq = _time(t, c, FORCE_SEQ, dtype)
            par = _time(t, c, FORCE_PAR, dtype)
            win = "seq" if seq < par else "par"
            if crossover is None and par < seq:
                crossover = t
            print(f"  T={t:5d}: seq={seq:8.1f}  par={par:8.1f}  -> {win}")
        print(f"  crossover (parallel first wins) ~ T={crossover}")


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available — this sweep requires a GPU.")
        return
    print(torch.cuda.get_device_name(0))
    sweep(torch.float32)
    sweep(torch.float64)


if __name__ == "__main__":
    main()
