"""Benchmark: FFT convolution vs direct conv1d.

Compares torchfx's FFT-based FIR filtering (adapted from Julius) against
direct ``torch.nn.functional.conv1d`` on both CPU and CUDA.

Usage::

    # Light mode (low memory, safe for laptops)
    uv run python benchmarks/bench_fftconv.py --light

    # Full mode (requires more RAM / GPU memory)
    uv run python benchmarks/bench_fftconv.py

Output is a markdown table suitable for pasting into docs or issues.
"""

from __future__ import annotations

import argparse
import gc
import time

import torch
import torch.nn.functional as F

from torchfx.filter._fftconv import fft_conv1d

LIGHT_DURATIONS = [1, 5, 30]
LIGHT_KERNELS = [64, 128, 256, 512, 1024]
FULL_DURATIONS = [1, 10, 60]
FULL_KERNELS = [64, 128, 256, 512, 1024]


def _bench(fn, warmup: int = 3, repeats: int = 10, sync_cuda: bool = False) -> float:
    """Return median time in milliseconds."""
    for _ in range(warmup):
        fn()
        if sync_cuda:
            torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(repeats):
        if sync_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if sync_cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def _run_suite(device: str, *, light: bool = False) -> list[dict]:
    sync = device == "cuda"
    fs = 44100
    durations = LIGHT_DURATIONS if light else FULL_DURATIONS
    kernel_sizes = LIGHT_KERNELS if light else FULL_KERNELS
    channels = 2
    rows: list[dict] = []

    for dur in durations:
        T = fs * dur
        for K in kernel_sizes:
            x = torch.randn(1, channels, T, device=device)
            w_grouped = torch.randn(channels, 1, K, device=device)
            w_fft = w_grouped[:1, :1, :]
            pad = K - 1

            direct_ms = _bench(
                lambda x=x, pad=pad, w=w_grouped: F.conv1d(F.pad(x, (pad, 0)), w, groups=channels),
                sync_cuda=sync,
            )
            fft_ms = _bench(
                lambda x=x, w=w_fft, pad=pad: fft_conv1d(x, w, padding=(pad, 0)),
                sync_cuda=sync,
            )
            speedup = direct_ms / fft_ms if fft_ms > 0 else float("inf")
            rows.append(
                {
                    "device": device,
                    "dur_s": dur,
                    "K": K,
                    "direct_ms": direct_ms,
                    "fft_ms": fft_ms,
                    "speedup": speedup,
                }
            )
            # Free tensors between iterations to limit peak memory
            del x, w_grouped, w_fft
            gc.collect()
    return rows


def _print_table(rows: list[dict], device: str) -> None:
    subset = [r for r in rows if r["device"] == device]
    if not subset:
        return
    print(f"\n### {device.upper()} ({subset[0]['dur_s']}s-{subset[-1]['dur_s']}s, 2ch stereo)\n")
    print("| Duration | K    | direct (ms) | FFT (ms) | Speedup |")
    print("|----------|------|-------------|----------|---------|")
    for r in subset:
        print(
            f"| {r['dur_s']:>6}s  | {r['K']:>4} | {r['direct_ms']:>11.2f} "
            f"| {r['fft_ms']:>8.2f} | {r['speedup']:>6.1f}x |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark FFT vs direct conv1d")
    parser.add_argument(
        "--light",
        action="store_true",
        help="Use smaller signal lengths and fewer kernel sizes (safe for laptops)",
    )
    args = parser.parse_args()
    mode = "light" if args.light else "full"
    print(f"# FFT conv1d vs direct conv1d benchmark ({mode})\n")
    all_rows: list[dict] = []

    print("Running CPU benchmarks...")
    all_rows.extend(_run_suite("cpu", light=args.light))
    _print_table(all_rows, "cpu")

    if torch.cuda.is_available():
        print("\nRunning CUDA benchmarks...")
        all_rows.extend(_run_suite("cuda", light=args.light))
        _print_table(all_rows, "cuda")
    else:
        print("\nCUDA not available, skipping GPU benchmarks.")


if __name__ == "__main__":
    main()
