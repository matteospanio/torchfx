"""Count native kernel calls and CUDA kernel launches for IIR cascades.

Substantiates claim C1 ("fusion reduces dispatches") in the IS² 2026
plan: for an IIR cascade of K second-order sections, we compare the
**fused** path (single ``parallel_iir_forward`` call through
``FusedSOSCascade``) against the **unfused** path (K individual calls
through ``nn.Sequential``). Two metrics are reported:

* ``native_calls`` — Python-level dispatches into the native extension
  (``biquad_forward`` / ``parallel_iir_forward``). Counted by wrapping
  the dispatch in ``torchfx._ops`` with a thin counting decorator
  installed only for the run.
* ``cuda_launches`` — number of ``cuda::launchKernel`` events captured
  by ``torch.profiler``. Reported as 0 on CPU-only runs.

Output is JSON to stdout (or ``--out PATH``) suitable for downstream
plotting. Sample::

    [
      {"depth": 5, "fused": false, "native_calls": 5, "cuda_launches": 0,
       "wall_us": 187.3},
      {"depth": 5, "fused": true,  "native_calls": 1, "cuda_launches": 0,
       "wall_us": 102.4},
      ...
    ]

Usage
-----

.. code-block:: bash

    uv run python tools/count_kernel_launches.py \\
        --depths 2 5 10 20 \\
        --duration 5.0 \\
        --channels 2 \\
        --fs 48000 \\
        --device cpu \\
        --out IS22026/results/launches.json

Notes
-----

* The unfused path is constructed via ``nn.Sequential(*biquads)`` which
  has K filters; this is what a user gets when they bypass ``Wave``'s
  auto-fusion (e.g. wrap the chain in ``Sequential`` themselves and call
  ``chain(x)`` directly). For the fused path, we materialise the same
  K filters through ``FusedSOSCascade``.
* The CPU path reports ``cuda_launches=0``. On CUDA, the same ``Wave``
  pipeline is used; we report both the native-call count (still 1
  for fused) and the GPU launches reported by the profiler.

"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from torchfx import _ops
from torchfx.filter import Biquad, BiquadLPF, FusedSOSCascade
from torchfx.filter._design import design_butterworth_sos


@dataclass
class RunResult:
    depth: int
    fused: bool
    device: str
    duration_s: float
    channels: int
    fs: int
    native_calls: int
    cuda_launches: int
    wall_us: float


@contextmanager
def _count_native_calls() -> Iterator[dict[str, int]]:
    """Install a counting wrapper around ``_ops`` dispatches.

    Yields a dict whose ``"biquad"``, ``"sos"``, and ``"delay"`` keys
    are incremented on every Python call into the native extension.

    """
    counts: dict[str, int] = {"biquad": 0, "sos": 0, "delay": 0}
    orig_biquad = _ops.biquad_forward
    orig_sos = _ops.parallel_iir_forward
    orig_delay = _ops.delay_line_forward

    def wrap_biquad(*args: Any, **kwargs: Any) -> Any:
        counts["biquad"] += 1
        return orig_biquad(*args, **kwargs)

    def wrap_sos(*args: Any, **kwargs: Any) -> Any:
        counts["sos"] += 1
        return orig_sos(*args, **kwargs)

    def wrap_delay(*args: Any, **kwargs: Any) -> Any:
        counts["delay"] += 1
        return orig_delay(*args, **kwargs)

    _ops.biquad_forward = wrap_biquad
    _ops.parallel_iir_forward = wrap_sos
    _ops.delay_line_forward = wrap_delay
    try:
        yield counts
    finally:
        _ops.biquad_forward = orig_biquad
        _ops.parallel_iir_forward = orig_sos
        _ops.delay_line_forward = orig_delay


def _build_cascade(order: int, fs: int) -> list[Biquad]:
    """Build K = order/2 biquads forming a Butterworth LPF cascade.

    Coefficients are designed once and stamped into each biquad so the
    fused and unfused paths are bit-for-bit equivalent. ``BiquadLPF`` is
    used as the concrete subclass (``Biquad`` itself is abstract); the
    SOS rows are stamped directly into ``_sos`` so the subclass's own
    ``compute_coefficients`` is effectively a no-op for this run.

    """
    # cutoff_norm is the cutoff normalised to Nyquist (fs/2). 0.4 here ≈ 0.2*fs.
    sos = design_butterworth_sos(order=order, cutoff_norm=0.4, btype="low")
    biquads: list[Biquad] = []
    for row in sos:
        # BiquadLPF needs a cutoff at construction; the actual value is
        # irrelevant since we overwrite _sos below.
        b = BiquadLPF(cutoff=fs * 0.2, q=0.707, fs=fs)
        b._sos = torch.tensor([[row[0], row[1], row[2], 1.0, row[4], row[5]]], dtype=torch.float64)
        # Cast through Any: nn.Module's __setattr__ treats _a1/_a2 as
        # potential Tensor parameters, but here we deliberately stash
        # Python floats for the native-kernel fast path.
        any_b: Any = b
        any_b._a1 = float(row[4])
        any_b._a2 = float(row[5])
        biquads.append(b)
    return biquads


def _measure(
    chain: nn.Module,
    x: torch.Tensor,
    n_iter: int,
    use_profiler: bool,
) -> tuple[int, int, float]:
    """Run ``chain(x)`` n times and return (native_calls, cuda_launches, mean_us)."""
    if use_profiler:
        from torch.profiler import ProfilerActivity, profile

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available() and x.is_cuda:
            activities.append(ProfilerActivity.CUDA)

        # Warm caches first so the profiler doesn't capture compilation.
        with torch.no_grad():
            chain(x)
            if x.is_cuda:
                torch.cuda.synchronize()

        with _count_native_calls() as counts:
            t0 = time.perf_counter()
            with profile(activities=activities, record_shapes=False) as prof:
                with torch.no_grad():
                    for _ in range(n_iter):
                        chain(x)
                if x.is_cuda:
                    torch.cuda.synchronize()
            elapsed_s = time.perf_counter() - t0
        # CUDA launches: count events that correspond to a kernel launch.
        cuda_launches = 0
        for ev in prof.key_averages():
            if "cuda" in ev.key.lower() or "::launchKernel" in ev.key:
                cuda_launches += int(ev.count)
        native_calls = sum(counts.values())
    else:
        with torch.no_grad():
            chain(x)  # warm
            if x.is_cuda:
                torch.cuda.synchronize()
        with _count_native_calls() as counts:
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(n_iter):
                    chain(x)
            if x.is_cuda:
                torch.cuda.synchronize()
            elapsed_s = time.perf_counter() - t0
        cuda_launches = 0
        native_calls = sum(counts.values())

    mean_us = (elapsed_s / n_iter) * 1e6
    # Per-iteration counts (counts above are totals over n_iter runs).
    native_per_iter = native_calls // n_iter if n_iter > 0 else 0
    cuda_per_iter = cuda_launches // n_iter if n_iter > 0 else 0
    return native_per_iter, cuda_per_iter, mean_us


def run_one(
    depth: int, channels: int, duration_s: float, fs: int, device: str, n_iter: int
) -> tuple[RunResult, RunResult]:
    """Run one fused/unfused pair at the given configuration."""
    n_samples = int(duration_s * fs)
    x = torch.randn(channels, n_samples, dtype=torch.float32, device=device) * 0.1

    # Unfused: nn.Sequential of K biquads. This bypasses Wave._materialize
    # so no fusion happens.
    unfused = nn.Sequential(*_build_cascade(depth * 2, fs)).to(device)
    # Force coefficient materialisation once before measurement.
    with torch.no_grad():
        unfused(x)
    if device == "cuda":
        torch.cuda.synchronize()

    nc_un, gl_un, us_un = _measure(unfused, x, n_iter, use_profiler=(device == "cuda"))
    unfused_result = RunResult(
        depth=depth,
        fused=False,
        device=device,
        duration_s=duration_s,
        channels=channels,
        fs=fs,
        native_calls=nc_un,
        cuda_launches=gl_un,
        wall_us=us_un,
    )

    # Fused: FusedSOSCascade collapses the K biquads into one SOS matrix.
    fused = FusedSOSCascade(*_build_cascade(depth * 2, fs)).to(device)
    with torch.no_grad():
        fused(x)
    if device == "cuda":
        torch.cuda.synchronize()

    nc_f, gl_f, us_f = _measure(fused, x, n_iter, use_profiler=(device == "cuda"))
    fused_result = RunResult(
        depth=depth,
        fused=True,
        device=device,
        duration_s=duration_s,
        channels=channels,
        fs=fs,
        native_calls=nc_f,
        cuda_launches=gl_f,
        wall_us=us_f,
    )

    return unfused_result, fused_result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--depths", type=int, nargs="+", default=[2, 5, 10, 20])
    p.add_argument("--duration", type=float, default=5.0, help="seconds")
    p.add_argument("--channels", type=int, default=2)
    p.add_argument("--fs", type=int, default=48000)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--iters", type=int, default=30, help="Iterations per measurement.")
    p.add_argument(
        "--out",
        type=str,
        default="-",
        help="Output path for JSON (default '-' = stdout).",
    )
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warning] CUDA requested but unavailable; falling back to CPU")
        args.device = "cpu"

    results: list[dict[str, Any]] = []
    for d in args.depths:
        u, f = run_one(
            depth=d,
            channels=args.channels,
            duration_s=args.duration,
            fs=args.fs,
            device=args.device,
            n_iter=args.iters,
        )
        for r in (u, f):
            results.append(asdict(r))
            label = "fused" if r.fused else "unfused"
            print(
                f"depth={r.depth:3d} {label:7s} "
                f"native_calls={r.native_calls:4d} "
                f"cuda_launches={r.cuda_launches:4d} "
                f"wall={r.wall_us:9.2f}us"
            )

    payload = json.dumps(results, indent=2)
    if args.out == "-":
        print(payload)
    else:
        import os

        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(payload)
        print(f"[ok] wrote {len(results)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
