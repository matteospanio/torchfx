"""CUDA performance benchmarks for torchfx.

Compares the performance of:
1. Python loop (pure PyTorch stateful path)
2. Native C++ CPU kernel
3. CUDA parallel prefix scan kernel

Usage:
    python benchmark/cuda_bench.py
    python benchmark/cuda_bench.py --save results.json
    python benchmark/cuda_bench.py --compare baseline.json
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn as nn

from bench_harness import BenchmarkSuite

sys.path.insert(0, "src")

from torchfx import Wave
from torchfx.filter import HiButterworth, LoButterworth
from torchfx.filter.biquad import BiquadLPF

SAMPLE_RATE = 44100


def create_signal(channels: int, duration_sec: float, device: str = "cpu") -> torch.Tensor:
    """Create a random test signal."""
    T = int(SAMPLE_RATE * duration_sec)
    x = torch.randn(channels, T, device=device, dtype=torch.float32)
    return x / x.abs().max()


def bench_biquad_stateful(suite: BenchmarkSuite) -> None:
    """Benchmark stateful biquad processing (the main bottleneck)."""
    print("\n=== Biquad Stateful Processing ===")

    durations = [0.1, 1.0, 5.0, 30.0]
    channels_list = [1, 2]

    for dur in durations:
        for ch in channels_list:
            T = int(SAMPLE_RATE * dur)
            label = f"T={T}, C={ch}"

            # CPU benchmark
            filt = BiquadLPF(cutoff=1000, q=0.707, fs=SAMPLE_RATE)
            filt.compute_coefficients()
            x_cpu = create_signal(ch, dur, "cpu")

            # Force stateful mode
            _ = filt(x_cpu)
            filt.reset_state()
            _ = filt(x_cpu)  # Now stateful

            filt.reset_state()
            _ = filt(x_cpu)

            suite.bench(
                f"biquad_cpu ({label})",
                lambda f=filt, x=x_cpu: f(x),
                sync_cuda=False,
                metadata={"T": T, "C": ch, "device": "cpu"},
            )

            # CUDA benchmark (if available)
            if torch.cuda.is_available():
                filt_gpu = BiquadLPF(cutoff=1000, q=0.707, fs=SAMPLE_RATE)
                filt_gpu.compute_coefficients()
                filt_gpu.move_coeff("cuda")
                x_gpu = create_signal(ch, dur, "cuda")

                _ = filt_gpu(x_gpu)
                filt_gpu.reset_state()
                _ = filt_gpu(x_gpu)

                filt_gpu.reset_state()
                _ = filt_gpu(x_gpu)

                suite.bench(
                    f"biquad_cuda ({label})",
                    lambda f=filt_gpu, x=x_gpu: f(x),
                    sync_cuda=True,
                    metadata={"T": T, "C": ch, "device": "cuda"},
                )


def bench_sos_cascade(suite: BenchmarkSuite) -> None:
    """Benchmark SOS cascade (higher-order IIR filters)."""
    print("\n=== SOS Cascade (Butterworth) ===")

    orders = [4, 8]
    dur = 5.0

    for order in orders:
        T = int(SAMPLE_RATE * dur)
        label = f"order={order}, T={T}"

        filt = LoButterworth(cutoff=2000, order=order, fs=SAMPLE_RATE)
        filt.compute_coefficients()
        x_cpu = create_signal(2, dur, "cpu")

        # Force stateful
        _ = filt(x_cpu)
        filt.reset_state()
        _ = filt(x_cpu)

        filt.reset_state()
        _ = filt(x_cpu)

        suite.bench(
            f"sos_cpu ({label})",
            lambda f=filt, x=x_cpu: f(x),
            sync_cuda=False,
            metadata={"order": order, "T": T, "device": "cpu"},
        )

        if torch.cuda.is_available():
            filt_gpu = LoButterworth(cutoff=2000, order=order, fs=SAMPLE_RATE)
            filt_gpu.compute_coefficients()
            filt_gpu.move_coeff("cuda")
            x_gpu = create_signal(2, dur, "cuda")

            _ = filt_gpu(x_gpu)
            filt_gpu.reset_state()
            _ = filt_gpu(x_gpu)

            filt_gpu.reset_state()
            _ = filt_gpu(x_gpu)

            suite.bench(
                f"sos_cuda ({label})",
                lambda f=filt_gpu, x=x_gpu: f(x),
                sync_cuda=True,
                metadata={"order": order, "T": T, "device": "cuda"},
            )


def bench_pipeline(suite: BenchmarkSuite) -> None:
    """Benchmark end-to-end pipeline: HiPass | LoPass."""
    print("\n=== End-to-End Pipeline ===")

    dur = 10.0
    T = int(SAMPLE_RATE * dur)

    chain = nn.Sequential(
        HiButterworth(cutoff=100, order=2, fs=SAMPLE_RATE),
        LoButterworth(cutoff=8000, order=4, fs=SAMPLE_RATE),
    )
    for f in chain:
        f.compute_coefficients()

    x_cpu = create_signal(2, dur, "cpu")

    suite.bench(
        f"pipeline_cpu (T={T})",
        lambda c=chain, x=x_cpu: c(x),
        sync_cuda=False,
        metadata={"T": T, "device": "cpu"},
    )

    if torch.cuda.is_available():
        chain_gpu = nn.Sequential(
            HiButterworth(cutoff=100, order=2, fs=SAMPLE_RATE),
            LoButterworth(cutoff=8000, order=4, fs=SAMPLE_RATE),
        )
        for f in chain_gpu:
            f.compute_coefficients()
            f.move_coeff("cuda")

        x_gpu = create_signal(2, dur, "cuda")

        suite.bench(
            f"pipeline_cuda (T={T})",
            lambda c=chain_gpu, x=x_gpu: c(x),
            sync_cuda=True,
            metadata={"T": T, "device": "cuda"},
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="TorchFX CUDA benchmarks")
    parser.add_argument("--save", type=str, help="Save results to JSON file")
    parser.add_argument("--compare", type=str, help="Compare against baseline JSON")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=20, help="Measurement iterations")
    args = parser.parse_args()

    print("TorchFX CUDA Benchmark Suite")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    suite = BenchmarkSuite(warmup_iters=args.warmup, measure_iters=args.iters)

    bench_biquad_stateful(suite)
    bench_sos_cascade(suite)
    bench_pipeline(suite)

    suite.print_results()

    if args.save:
        suite.save_json(args.save)
        print(f"Results saved to {args.save}")

    if args.compare:
        suite.compare(args.compare)


if __name__ == "__main__":
    main()
