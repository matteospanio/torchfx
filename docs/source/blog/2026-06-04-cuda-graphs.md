---
blogpost: true
date: Jun 04, 2026
author: Matteo Spanio
category: features
tags: cuda, cuda-graphs, realtime, streaming, latency
---

# CUDA Graphs for Streaming: One Launch Instead of a Launch Storm

For offline batch processing, GPU kernel-launch overhead disappears into the noise. For **realtime streaming**, it *is* the cost. TorchFX 0.6.0 adds `torchfx.realtime.CudaGraphRunner`, which captures a fixed-shape filter forward into a CUDA Graph and replays it per chunk — up to **4× lower per-chunk latency**.

## Why launch overhead dominates short chunks

A `K`-section SOS cascade issues roughly `4·K` CUDA kernel launches per forward — a forcing pass plus the three Blelloch scan phases for each section. At 48 kHz with a 512-sample buffer you do ~94 of those per second, each re-issuing the whole launch sequence.

How much is overhead versus compute? Our dispatch-threshold sweep gave a clean answer: the parallel scan measures **~135 µs of essentially fixed launch/dispatch overhead regardless of chunk length** — a 256-sample chunk and an 8192-sample chunk cost almost the same. For a realtime chunk, you are paying for launches, not arithmetic.

CUDA Graphs are the textbook fix: record the kernel sequence once, then replay it as a single graph launch.

## The API

```python
import torch
from torchfx.filter import HiButterworth, LoButterworth
from torchfx.filter.fused import FusedSOSCascade
from torchfx.realtime import CudaGraphRunner

chain = FusedSOSCascade(
    HiButterworth(80, order=2, fs=48000),
    LoButterworth(8000, order=4, fs=48000),
)
example = torch.randn(2, 512, device="cuda")   # fixes shape + dtype
runner = CudaGraphRunner(chain, example)

for chunk in stream:                            # each chunk is [2, 512] cuda
    y = runner.run(chunk).clone()
```

`CudaGraphRunner` warms up on a side stream, captures the fixed-shape forward, and on each `run()` copies the chunk into the static input buffer and replays the graph. Streaming DF1 state is carried across replays — the captured kernels read and write the filter's persistent state buffers in place — so the result is identical to running the cascade eagerly chunk by chunk. Call `reset_state()` to start a fresh stream.

## The numbers

4-section SOS cascade, RTX 3070, per-chunk median (`benchmarks/bench_cuda_graph.py`):

| Chunk size | Eager | Graph replay | Speedup |
|---|---:|---:|---:|
| 128 | 209.9 µs | 52.2 µs | **4.02×** |
| 256 | 208.9 µs | 61.4 µs | **3.40×** |
| 512 | 208.9 µs | 79.9 µs | **2.62×** |
| 1024 | 207.9 µs | 116.7 µs | **1.78×** |
| 2048 | 240.6 µs | 190.5 µs | 1.26× |

The pattern is exactly what the theory predicts: the smaller the chunk, the more the fixed launch overhead dominates, and the bigger the graph win. At 128 samples — deep in the realtime regime — replay is **4× faster**. As chunks grow and real compute takes over, the multiplier shrinks toward 1.

## The bug that taught us something

Getting here took a detour worth sharing, because the root cause is a trap any CUDA extension author can fall into.

The first attempts *looked* like they captured correctly but **replayed as a no-op** — the output was frozen no matter what input we fed in. We chased several wrong theories (the returned tensor living in the graph's private memory pool; per-section scratch aliasing) and even shipped allocation-free kernels to rule them out.

The actual cause was one line, repeated across every kernel launch:

```cpp
forcing_kernel<scalar_t><<<blocks, threads>>>(...);   // default stream!
```

The hand-written kernels launched on the **default stream**. `torch.cuda.graph` capture runs on a *side* stream and **does not record default-stream work** — so the captured graph contained none of our kernels, and replay did nothing. (A trivial `y = x * 2` graph worked fine, because PyTorch's own aten ops correctly launch on the current stream. That contrast is what cracked it.)

The fix is to pass the current stream to every launch:

```cpp
const auto stream = c10::cuda::getCurrentCUDAStream();
forcing_kernel<scalar_t><<<blocks, threads, 0, stream>>>(...);
```

This is not just a graph fix — it is *more correct in eager mode too*, since the kernels now run on PyTorch's stream instead of relying on implicit default-stream synchronisation. The lesson, now written into our contributor notes: **a hand-written CUDA kernel must always launch on `getCurrentCUDAStream()`.**

## Caveats

- The graph is **shape- and dtype-fixed**: every `run()` input must match the captured example. Change buffer size and you re-capture.
- The captured filter must have **static coefficients** — the SOS taps are baked into the graph.
- Graph replay matches eager streaming to *float precision*, not bit-for-bit: the graphed warmup runs on a side stream + capture while an eager filter runs straight-line, so the post-warmup state carries float noise (~2e-11 in float64) that propagates but does not grow.
- The live `RealtimeProcessor` is still CPU-only; `CudaGraphRunner` targets GPU streaming (e.g. a GPU `StreamProcessor` or your own chunk loop). Wiring it into a GPU realtime path is future work.

```bash
python benchmarks/bench_cuda_graph.py
```

Back to the [0.6.0 release notes](2026-06-04-release-060.md).
