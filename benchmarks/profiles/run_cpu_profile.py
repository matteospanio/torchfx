"""Run torchfx scenarios under scalene for CPU profiling.

Usage (locally, on this machine — no GPU required)::

    uv run scalene --outfile benchmarks/results/scalene-<scenario>.html \\
                   --html --cpu --memory \\
                   benchmarks/profiles/run_cpu_profile.py --scenario offline_filter_chain

    # Or time all scenarios in sequence without scalene:
    uv run python benchmarks/profiles/run_cpu_profile.py --all

The script loops each scenario's forward call ``--iters`` times (default 5 for
offline, 200 for realtime) so scalene has something to sample. Running under
``scalene`` captures CPU / memory / line-level hotspots; running under plain
``python`` just prints wall-clock timings so we can sanity-check before
invoking scalene.

"""

from __future__ import annotations

import argparse
import time

import torch

from benchmarks.profiles.scenarios import (
    ALL_SCENARIOS,
    Scenario,
    offline_batch_chain,
    offline_filter_chain,
    realtime_chunks_cpu,
    realtime_chunks_gpu,
)

# Default iteration counts per scenario. Offline scenarios process a long
# signal per call, so 5 iterations is enough; realtime processes a 512-sample
# chunk per call, so we loop many times to accumulate measurable work.
DEFAULT_ITERS: dict[str, int] = {
    offline_filter_chain.name: 5,
    offline_batch_chain.name: 3,
    realtime_chunks_cpu.name: 500,
    realtime_chunks_gpu.name: 500,
}


def _run_scenario(scenario: Scenario, device: str, iters: int) -> dict[str, float]:
    module = scenario.build(device)
    x = scenario.make_input(device)

    # Warm up (triggers coefficient computation on first forward + any JIT)
    _ = module(x)

    torch.manual_seed(0)
    start = time.perf_counter()
    for _ in range(iters):
        _ = module(x)
    elapsed = time.perf_counter() - start

    return {
        "iters": float(iters),
        "total_s": elapsed,
        "mean_ms": 1_000.0 * elapsed / iters,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=[s.name for s in ALL_SCENARIOS] + ["all"],
        default="all",
        help="scenario to run (default: all)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="torch device (default: cpu); the realtime_chunks_gpu scenario needs cuda",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=None,
        help="number of forward calls per scenario (default: per-scenario heuristic)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="shortcut for --scenario all",
    )
    args = parser.parse_args()

    names = (
        [s.name for s in ALL_SCENARIOS] if args.scenario == "all" or args.all else [args.scenario]
    )

    for name in names:
        scenario = next(s for s in ALL_SCENARIOS if s.name == name)
        if args.device == "cpu" and "gpu" in name:
            print(f"SKIP {name}  (cpu device on a gpu scenario)")
            continue
        iters = args.iters or DEFAULT_ITERS[name]
        print(f">>> {name}  ({scenario.description})  device={args.device}  iters={iters}")
        result = _run_scenario(scenario, args.device, iters)
        print(
            f"    total={result['total_s']:.3f}s  mean={result['mean_ms']:.3f}ms/call",
        )


if __name__ == "__main__":
    main()
