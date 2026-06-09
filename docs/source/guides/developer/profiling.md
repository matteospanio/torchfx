# Profiling pipelines

How to find where time goes in a TorchFX pipeline, and the usual fixes. For *comparative*
measurement against a baseline see {doc}`/guides/developer/benchmarking`; this guide is
about diagnosing a single pipeline.

## Time it correctly first

Three things will give you wrong numbers if you skip them:

- **Warm up.** Filters compute their coefficients lazily on the *first* `forward`, and the
  native extension's first call pays one-time setup. Run the pipeline a few times before
  timing.
- **Synchronise CUDA.** Kernel launches are asynchronous, so `perf_counter` around a GPU
  call measures only the *launch*, not the work. Call `torch.cuda.synchronize()` before
  reading the clock.
- **Use the median**, not a single sample — the OS scheduler adds multi-millisecond
  outliers (see the realtime tail discussion in the benchmarking guide).

```python
import time
import torch
import torchfx as fx

wave = fx.Wave(torch.randn(2, 480_000), 48_000)
chain = fx.filter.HiButterworth(80, order=4) | fx.filter.LoButterworth(12_000, order=8)

for _ in range(3):                      # warm up (lazy coeffs + first-call setup)
    _ = wave | chain

samples = []
for _ in range(20):
    torch.cuda.synchronize() if wave.ys.is_cuda else None
    t0 = time.perf_counter_ns()
    _ = wave | chain
    torch.cuda.synchronize() if wave.ys.is_cuda else None
    samples.append(time.perf_counter_ns() - t0)
samples.sort()
print(f"median {samples[len(samples)//2] / 1e6:.3f} ms")
```

## The kernel timeline with `torch.profiler`

To see *which* operations dominate (and, on CUDA, the launch vs. compute split):

```python
from torch.profiler import profile, ProfilerActivity

acts = [ProfilerActivity.CPU]
if wave.ys.is_cuda:
    acts.append(ProfilerActivity.CUDA)

with profile(activities=acts, record_shapes=True) as prof:
    for _ in range(10):
        _ = wave | chain

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))
# CPU-only: sort_by="self_cpu_time_total"
```

Look for the `torchfx_ext::` entries (`sos_forward`, `biquad_forward`, …) — those are the
native kernels. A long tail of many short native calls usually means the chain **didn't
fuse** (see below). Export a Chrome trace with `prof.export_chrome_trace("trace.json")`
and open it in `chrome://tracing` for a visual timeline.

## Counting native dispatches

The single most useful TorchFX-specific check: how many times does the pipeline cross into
the native extension? A fused IIR cascade should be **one** dispatch regardless of depth.

```bash
uv run python tools/count_kernel_launches.py --depths 2,5,10,20
```

It reports `native_calls` (Python→C++ dispatches) and `cuda_launches` for the fused vs.
unfused paths. In your own code, wrap the dispatch to count it (the mechanism
`tests/test_perf_regression.py` uses):

```python
from torchfx import _ops

orig = _ops.parallel_iir_forward
calls = 0
def counted(*a, **k):
    global calls; calls += 1
    return orig(*a, **k)
_ops.parallel_iir_forward = counted
_ = wave | chain
print("SOS dispatches:", calls)   # 1 if the cascade fused
_ops.parallel_iir_forward = orig
```

## Common bottlenecks and fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| Many short `sos_forward` calls | IIR filters applied one at a time | Build the chain with `\|` so the deferred planner **fuses** consecutive IIR/biquads into one `FusedSOSCascade`; avoid forcing materialisation between them. |
| GPU slower than expected on float32 in | Input upcast to float64 | Pass float32 tensors — the kernels dispatch on dtype and run the FP32 path (≈3× on consumer GPUs). |
| High per-chunk latency on short GPU chunks | Per-section launch overhead dominates | Use `torchfx.realtime.CudaGraphRunner` to replay a fixed-shape forward as one graph launch. |
| Throughput-bound over many short files | One launch per file, low occupancy | `torchfx.batch_process(waves, effect)` runs them in a single launch. |
| Sequential vs. parallel scan crossover | Signal length near `PARALLEL_SCAN_THRESHOLD` | Tune the `threshold=` argument (dtype-aware default); `benchmarks/bench_threshold_sweep.py`. |

## See also

- {doc}`/guides/developer/benchmarking` — comparative benchmarking and the suite layout.
- {doc}`/guides/developer/testing` — the deterministic perf-regression gates that keep
  fusion from silently breaking.
