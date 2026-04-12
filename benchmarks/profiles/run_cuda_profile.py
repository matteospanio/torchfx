"""Run torchfx scenarios under torch.profiler on CUDA.

This script is meant to run on the SLURM cluster (which has the GPU). It
produces:

1. A Chrome-trace JSON (viewable at chrome://tracing or https://ui.perfetto.dev)
2. A text table (top ops by CUDA time)
3. A torch.cuda memory-history pickle for the most interesting scenario

Usage on the cluster::

    PYTHONPATH=. python benchmarks/profiles/run_cuda_profile.py \\
        --out benchmarks/results/cuda-$(date +%Y%m%d-%H%M%S)

The ``benchmarks/slurm/run_profiles.sbatch`` job wraps this script so you do
not invoke it by hand normally.

"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, schedule

from benchmarks.profiles.scenarios import (
    ALL_SCENARIOS,
    Scenario,
    offline_batch_chain,
    offline_filter_chain,
    realtime_chunks_cpu,
    realtime_chunks_gpu,
)

PROFILE_ITERS: dict[str, int] = {
    offline_filter_chain.name: 5,
    offline_batch_chain.name: 5,
    realtime_chunks_cpu.name: 200,
    realtime_chunks_gpu.name: 200,
}


def _profile_one(scenario: Scenario, device: str, out_dir: Path, iters: int) -> None:
    module = scenario.build(device)
    x = scenario.make_input(device)

    # Warm up (triggers compile_coefficients + any JIT + cuBLAS workspace alloc).
    for _ in range(3):
        _ = module(x)
    if device == "cuda":
        torch.cuda.synchronize()

    # schedule: skip first 1 (wait), warm-up 1, record 3 iters
    sched = schedule(wait=1, warmup=1, active=3, repeat=1)
    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    trace_path = out_dir / f"{scenario.name}.json"
    table_path = out_dir / f"{scenario.name}.txt"

    with profile(
        activities=activities,
        schedule=sched,
        on_trace_ready=lambda p: p.export_chrome_trace(str(trace_path)),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in range(iters):
            _ = module(x)
            if device == "cuda":
                torch.cuda.synchronize()
            prof.step()

    sort_key = "cuda_time_total" if device == "cuda" else "cpu_time_total"
    table = prof.key_averages(group_by_input_shape=True).table(sort_by=sort_key, row_limit=40)
    table_path.write_text(table)
    print(f"    wrote {trace_path}")
    print(f"    wrote {table_path}")


def _memory_snapshot(scenario: Scenario, device: str, out_path: Path) -> None:
    if device != "cuda":
        return
    torch.cuda.empty_cache()
    torch.cuda.memory._record_memory_history(max_entries=100_000)
    module = scenario.build(device)
    x = scenario.make_input(device)
    for _ in range(5):
        _ = module(x)
    torch.cuda.synchronize()
    snapshot = torch.cuda.memory._snapshot()
    torch.cuda.memory._record_memory_history(enabled=None)
    with out_path.open("wb") as f:
        pickle.dump(snapshot, f)
    print(f"    wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(os.environ.get("TORCHFX_PROFILE_OUT", "benchmarks/results/cuda-profile")),
        help="output directory for traces and tables",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="torch device (default: cuda if available)",
    )
    parser.add_argument(
        "--scenario",
        choices=[s.name for s in ALL_SCENARIOS] + ["all"],
        default="all",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available; submit this job on a GPU node")

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"profiling device={args.device} → {args.out}")

    names = [s.name for s in ALL_SCENARIOS] if args.scenario == "all" else [args.scenario]
    for name in names:
        scenario = next(s for s in ALL_SCENARIOS if s.name == name)
        iters = PROFILE_ITERS[name]
        print(f">>> {name}  iters={iters}")
        _profile_one(scenario, args.device, args.out, iters)

    # One memory snapshot from the scenario most likely to stress the allocator.
    if args.device == "cuda":
        _memory_snapshot(offline_batch_chain, args.device, args.out / "memory_snapshot.pkl")


if __name__ == "__main__":
    main()
