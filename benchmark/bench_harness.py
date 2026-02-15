"""Reusable benchmark framework for torchfx.

Provides utilities for timing GPU and CPU operations with proper warmup,
CUDA synchronization, and JSON output for regression tracking.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import torch


@dataclass
class BenchmarkResult:
    """Result of a single benchmark case."""

    name: str
    mean_ms: float
    std_ms: float = 0.0
    samples: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class BenchmarkSuite:
    """Reusable benchmark framework with warmup, timing, and reporting.

    Parameters
    ----------
    warmup_iters : int
        Number of warmup iterations before measurement.
    measure_iters : int
        Number of measurement iterations.
    """

    def __init__(self, warmup_iters: int = 5, measure_iters: int = 50) -> None:
        self.warmup_iters = warmup_iters
        self.measure_iters = measure_iters
        self.results: list[BenchmarkResult] = []

    def time_fn(
        self,
        fn: Callable,
        *args,
        sync_cuda: bool = True,
        **kwargs,
    ) -> BenchmarkResult:
        """Time a function with proper CUDA synchronization.

        Parameters
        ----------
        fn : Callable
            Function to benchmark.
        *args, **kwargs
            Arguments passed to the function.
        sync_cuda : bool
            Whether to synchronize CUDA before/after timing.

        Returns
        -------
        BenchmarkResult
            Timing statistics.
        """
        # Warmup
        for _ in range(self.warmup_iters):
            fn(*args, **kwargs)

        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

        samples = []
        for _ in range(self.measure_iters):
            if sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            fn(*args, **kwargs)
            if sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000  # ms
            samples.append(elapsed)

        mean_ms = sum(samples) / len(samples)
        variance = sum((s - mean_ms) ** 2 for s in samples) / len(samples)
        std_ms = variance**0.5

        return BenchmarkResult(
            name="",
            mean_ms=mean_ms,
            std_ms=std_ms,
            samples=samples,
        )

    def bench(
        self,
        name: str,
        fn: Callable,
        *args,
        sync_cuda: bool = True,
        metadata: dict | None = None,
        **kwargs,
    ) -> BenchmarkResult:
        """Run a named benchmark and store the result.

        Parameters
        ----------
        name : str
            Name of this benchmark case.
        fn : Callable
            Function to benchmark.
        *args, **kwargs
            Arguments passed to the function.
        sync_cuda : bool
            Whether to synchronize CUDA.
        metadata : dict or None
            Extra metadata to attach to the result.
        """
        result = self.time_fn(fn, *args, sync_cuda=sync_cuda, **kwargs)
        result.name = name
        result.metadata = metadata or {}
        self.results.append(result)
        return result

    def print_results(self) -> None:
        """Print results as a formatted table."""
        print(f"\n{'Name':<50} {'Mean (ms)':>12} {'Std (ms)':>12}")
        print("-" * 76)
        for r in self.results:
            print(f"{r.name:<50} {r.mean_ms:>12.3f} {r.std_ms:>12.3f}")
        print()

    def save_json(self, path: str) -> None:
        """Save results to a JSON file for regression tracking."""
        data = {
            "warmup_iters": self.warmup_iters,
            "measure_iters": self.measure_iters,
            "results": [asdict(r) for r in self.results],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def compare(self, baseline_path: str) -> None:
        """Compare current results against a baseline JSON file.

        Prints a warning for any benchmark that regressed by more than 10%.
        """
        with open(baseline_path) as f:
            baseline = json.load(f)

        baseline_map = {r["name"]: r["mean_ms"] for r in baseline["results"]}

        print(f"\n{'Name':<50} {'Current':>10} {'Baseline':>10} {'Change':>10}")
        print("-" * 82)
        for r in self.results:
            if r.name in baseline_map:
                base = baseline_map[r.name]
                change = (r.mean_ms - base) / base * 100
                flag = " REGRESSION" if change > 10 else ""
                print(f"{r.name:<50} {r.mean_ms:>10.3f} {base:>10.3f} {change:>+9.1f}%{flag}")
        print()
