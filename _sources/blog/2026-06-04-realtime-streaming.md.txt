---
blogpost: true
date: Jun 04, 2026
author: Matteo Spanio
category: features
tags: realtime, streaming, dispatch, correctness, performance
---

# A Hardened Realtime Path: Worker Threads, Allocation-Free Streaming, and Dtype-Aware Dispatch

The flashy 0.6.0 numbers are [FP32 on the GPU](2026-06-04-fp32-cuda.md) and [CUDA Graphs](2026-06-04-cuda-graphs.md). This post covers the quieter work that makes the streaming path actually dependable: the realtime architecture, the per-call allocations, the dispatch heuristic, and a handful of silent-correctness bugs.

## Realtime: DSP off the audio callback

Previously, `RealtimeProcessor` ran the entire effect chain *inside* the PortAudio callback. That is the classic realtime-audio mistake: any allocation, GC pause, or slow kernel on the callback thread is an audible glitch.

0.6.0 restructures it into a **producer/consumer split**:

- The **audio callback** does almost nothing — it moves samples between the backend buffers and a pair of lock-free `TensorRingBuffer`s (input + output), then returns.
- A dedicated **worker thread** (`torchfx-realtime-dsp`) pulls input chunks, runs the effect chain, and writes results back to the output ring.
- The output ring is primed with one chunk of silence at start, so the first callback never underflows.

It also ships **instrumentation** you can actually measure with: `latency_log_ns()`, `latency_stats_ms()` (count/min/median/mean/p95/p99/max), and granular xrun counters (`xrun_count`, `input_overflow_count`, `output_underflow_count`, …) written only from the audio thread.

On the workstation CPU, a 5-section Butterworth cascade at a 256-sample buffer / 48 kHz holds a **p99 per-callback time of ~0.038 ms against a 5.33 ms deadline — under 1% of budget — with zero xruns** over thousands of callbacks. (The live path remains CPU-only; GPU realtime is future work.)

## Allocation-free GPU streaming

Streaming means many small forwards, and the GPU SOS cascade was allocating on every one. For a `K`-section cascade it allocated, *per section*, a forcing buffer, an output buffer, and the scan's block-aggregate scratch — `O(K)` allocations per chunk, recurring forever.

0.6.0 makes `sos_forward_cuda` **allocation-free per forward**:

- One forcing buffer, two ping-pong output buffers, and one block-aggregate scratch are allocated **once per forward and reused across all sections** (`O(K)` → `O(1)`).
- The per-section DF1 state is updated **in place** in the persistent state buffers, instead of allocating `narrow`/`flip`/`cat` temporaries.

The payoff on the eager path is **~27%** (276 → 202 µs/chunk for a 4-section cascade on an RTX 3070) — and, as a bonus, this allocation discipline is exactly what made [CUDA Graph capture](2026-06-04-cuda-graphs.md) possible.

## A dispatch threshold that knows about dtype

TorchFX picks between a sequential CUDA kernel and the work-efficient parallel scan based on signal length. That boundary, `PARALLEL_SCAN_THRESHOLD`, used to be a hard-coded `2048` — and worse, it was *dead* in Python while the kernel hard-coded the same constant in two places.

0.6.0 threads it through as a real, tunable parameter and re-measured the crossover with a sweep (`benchmarks/bench_threshold_sweep.py`). The result is that **the crossover depends on dtype**:

- The parallel scan is essentially flat at ~135 µs (it's launch-overhead-bound).
- The sequential kernel grows ~2× faster in FP64 than FP32.

So the sequential kernel stays competitive up to ~2560 samples in FP32 but only ~1024 in FP64. A single `2048` default left **FP64 ~57% slower at T≈2048** (211 µs sequential vs 134 µs parallel). The new default is **dtype-aware: float32 → 2048, float64 → 1024**, and you can override it per call (`threshold=0` forces the parallel scan, a large value forces sequential — handy for benchmarking).

## Silent-correctness fixes

Performance is worthless if the output is quietly wrong. Three footguns closed in 0.6.0:

- **Stale coefficients on sample-rate change.** A filter materialised at 44.1 kHz, then reused on a 48 kHz signal, used to keep applying the 44.1 kHz coefficients — silently. Filters now track the `fs` their coefficients were designed for and recompute (and reset state) when it changes, on the direct `forward()` path as well as through the pipe operator.
- **State leak across `Wave` reuse.** Offline `Wave` materialisation now resets stateful modules before running, so piping one filter instance into two different `Wave`s can't bleed DF1 state from the first into the second. Streaming, which deliberately preserves state across chunks, is unaffected.
- **Silent half-precision upcast.** `float16` / `bfloat16` inputs to the native filters were silently promoted to `float64`. They now raise a clear `TypeError` — the IIR feedback recurrence is not numerically safe in half precision; cast to `float32` or `float64` yourself.

None of these are glamorous, but a DSP library that quietly returns the wrong audio is worse than a slow one. These are locked in by regression tests.

Back to the [0.6.0 release notes](2026-06-04-release-060.md).
