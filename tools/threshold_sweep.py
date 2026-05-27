"""PARALLEL_SCAN_THRESHOLD ablation for the IS² 2026 paper (C-7).

The native CUDA kernel switches between a *sequential* per-channel
recurrence and a *parallel-scan* (Blelloch) implementation at
``T <= 2048`` (see ``src/torchfx/_csrc/cuda/parallel_scan.cu`` and
``src/torchfx/_ops.py:34``). That threshold was set without a
documented ablation. This tool sweeps signal length ``T`` densely
around the crossover, times each point, and emits per-``(T, channels)``
medians as JSON so a plot can show:

* whether the crossover is actually at 2048 on the target GPU;
* how the optimal threshold varies with channel count.

Strategy
--------

We can't force the sequential vs. parallel branch from Python (the
decision is baked into the C++ binding). Instead we sweep ``T`` finely
and read off the dispatch boundary from the time curve:

* For ``T < 2048`` the sequential kernel runs.
* For ``T > 2048`` the parallel-scan kernel runs.

At ``T = 2048`` exactly one branch is taken; the kink in the curve
shows whether the threshold is well-chosen. If the parallel-scan
kernel becomes faster well below 2048 we should lower the threshold;
if it's slower well above, raise it.

Output is JSON of the form::

    [
      {"channels": 1, "T": 1024, "median_us": 12.4, "iter": 50, "device": "cuda"},
      {"channels": 1, "T": 1536, "median_us": 18.1, ...},
      ...
    ]

Usage
-----

.. code-block:: bash

    uv run python tools/threshold_sweep.py \\
        --device cuda \\
        --channels 1 2 8 32 \\
        --out IS22026/results/threshold-cuda-rtx3070.json

    # Re-run on a different GPU to compare:
    uv run python tools/threshold_sweep.py \\
        --device cuda \\
        --channels 1 2 8 32 \\
        --out IS22026/results/threshold-cuda-l40s.json

CPU mode is supported (``--device cpu``) for plotting infrastructure
validation, but the threshold story is GPU-only since the CPU path
has no parallel-scan branch.

"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass

import torch

from torchfx.filter import BiquadLPF
from torchfx.filter._design import design_butterworth_sos


@dataclass
class SweepPoint:
    channels: int
    T: int
    device: str
    sections: int
    median_us: float
    p95_us: float
    min_us: float
    samples: int
    # Useful context for the plot caption.
    fs: int
    parallel_threshold: int


# Range chosen to bracket the C++ threshold of 2048 by ~4× either side
# and to be dense around it.
DEFAULT_T_GRID = [
    256,
    384,
    512,
    768,
    1024,
    1280,
    1536,
    1792,
    2048,
    2304,
    2560,
    3072,
    4096,
    6144,
    8192,
    12288,
    16384,
    32768,
]


def _build_filter(sections: int, fs: int, device: str) -> torch.nn.Module:
    """Build a K=sections SOS cascade as a single FusedSOSCascade.

    SOS coefficients are pre-designed once and stamped in; we don't go through the lazy
    compute path because that would muddy the timing on the first call.

    """
    from torchfx.filter import FusedSOSCascade

    sos = design_butterworth_sos(order=sections * 2, cutoff_norm=0.4, btype="low")
    biquads = []
    for row in sos:
        b = BiquadLPF(cutoff=fs * 0.2, q=0.707, fs=fs)
        b._sos = torch.tensor([[row[0], row[1], row[2], 1.0, row[4], row[5]]], dtype=torch.float64)
        b._a1 = float(row[4])  # type: ignore[assignment]
        b._a2 = float(row[5])  # type: ignore[assignment]
        biquads.append(b)
    fused = FusedSOSCascade(*biquads)
    fused.to(device)
    return fused


def _measure(
    chain: torch.nn.Module, x: torch.Tensor, iters: int, device: str
) -> tuple[float, float, float, int]:
    """Run ``chain(x)`` ``iters`` times, return (median_us, p95_us, min_us, n)."""
    times_us: list[float] = []
    is_cuda = device == "cuda"
    # Warm-up
    for _ in range(5):
        with torch.no_grad():
            chain(x)
        if is_cuda:
            torch.cuda.synchronize()
    # Timed iterations
    for _ in range(iters):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        with torch.no_grad():
            chain(x)
        if is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        times_us.append((t1 - t0) / 1000.0)
    times_us.sort()
    med = statistics.median(times_us)
    p95_idx = min(len(times_us) - 1, int(0.95 * len(times_us)))
    return med, times_us[p95_idx], times_us[0], len(times_us)


def sweep(
    channels_list: list[int],
    T_grid: list[int],
    sections: int,
    iters: int,
    fs: int,
    device: str,
) -> list[SweepPoint]:
    from torchfx._ops import PARALLEL_SCAN_THRESHOLD

    out: list[SweepPoint] = []
    for C in channels_list:
        chain = _build_filter(sections, fs, device)
        # Prime any lazy state allocations with the largest T.
        T_max = max(T_grid)
        x_warm = torch.randn(C, T_max, dtype=torch.float32, device=device) * 0.1
        with torch.no_grad():
            chain(x_warm)
        if device == "cuda":
            torch.cuda.synchronize()

        for T in T_grid:
            x = torch.randn(C, T, dtype=torch.float32, device=device) * 0.1
            med, p95, mn, n = _measure(chain, x, iters, device)
            pt = SweepPoint(
                channels=C,
                T=T,
                device=device,
                sections=sections,
                median_us=med,
                p95_us=p95,
                min_us=mn,
                samples=n,
                fs=fs,
                parallel_threshold=PARALLEL_SCAN_THRESHOLD,
            )
            out.append(pt)
            print(
                f"C={C:>3d}  T={T:>6d}  median={med:9.2f}us  "
                f"p95={p95:9.2f}us  min={mn:9.2f}us  "
                f"(threshold={PARALLEL_SCAN_THRESHOLD})"
            )
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--channels", type=int, nargs="+", default=[1, 2, 8, 32])
    p.add_argument(
        "--T-grid",
        type=int,
        nargs="+",
        default=DEFAULT_T_GRID,
        help="Signal lengths to sweep. Default brackets 2048 ± 4×.",
    )
    p.add_argument(
        "--sections",
        type=int,
        default=1,
        help="K (SOS sections). The threshold story is clearest at K=1.",
    )
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--fs", type=int, default=48000)
    p.add_argument(
        "--out",
        type=str,
        default="-",
        help="Output JSON path; '-' for stdout.",
    )
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[error] CUDA requested but unavailable")
        return 2

    points = sweep(
        channels_list=args.channels,
        T_grid=args.T_grid,
        sections=args.sections,
        iters=args.iters,
        fs=args.fs,
        device=args.device,
    )

    payload = json.dumps([asdict(p) for p in points], indent=2)
    if args.out == "-":
        print(payload)
    else:
        import os

        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(payload)
        print(f"[ok] wrote {len(points)} sweep points to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
