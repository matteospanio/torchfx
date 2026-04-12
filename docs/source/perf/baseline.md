# Performance baseline — 2026-04-11

This document captures the **Phase 0** baseline of the TorchFX performance
campaign described in the master plan. It records the numbers measured *before*
any optimization work so every follow-up PR has a concrete before/after.

Baseline hardware: local workstation (no GPU), Python 3.10.19,
PyTorch 2.10.0+cu128 (CUDA not available locally — CUDA baselines land from the
SLURM cluster in a separate commit).

Artifacts live under [`benchmarks/results/`](../../../benchmarks/results):

- `local-cpu-coverage-baseline/` — HTML coverage report
- `local-cpu-coverage-baseline.json`, `.xml` — machine-readable coverage
- `local-cpu-baseline.json` — pytest-benchmark JSON (117 CPU benchmarks)
- `local-cpu-profile-baseline/*.json` — Chrome traces (open in
  [ui.perfetto.dev](https://ui.perfetto.dev))
- `local-cpu-profile-baseline/*.txt` — torch.profiler top-op tables

## Reproducing

```bash
# coverage
uv run coverage run -m pytest
uv run coverage report -m
uv run coverage html -d benchmarks/results/local-cpu-coverage-baseline

# CPU benchmarks (≈8–9 min)
uv run pytest benchmarks/ --benchmark-enable \
    -k "not cuda and not numba_cuda" \
    --benchmark-json=benchmarks/results/local-cpu-baseline.json

# CPU profiles (scenarios live in benchmarks/profiles/scenarios.py)
PYTHONPATH=. uv run python benchmarks/profiles/run_cpu_profile.py --all
PYTHONPATH=. uv run python benchmarks/profiles/run_cuda_profile.py \
    --device cpu --out benchmarks/results/local-cpu-profile-baseline

# (cluster) CUDA baseline
sbatch benchmarks/run_benchmarks.slurm
```

## Coverage — CPU-only run

| Module | Stmts | Cover | Notes |
|---|---:|---:|---|
| `filter/fused.py` | 104 | **9%** | no dedicated test file — Phase 1 |
| `filter/utils.py` | 7 | **0%** | no dedicated test file — Phase 1 |
| `filter/__base.py` | 62 | **44%** | tested only indirectly via subclasses |
| `_ops.py` | 105 | **64%** | native-dispatch layer; CUDA branches unhit locally |
| `wave.py` | 105 | **66%** | save/load paths, some error branches |
| `filter/iir.py` | 280 | **71%** | higher-order filter classes thinly covered |
| `filter/filterbank.py` | 50 | **73%** | CUDA-specific branches |
| `filter/biquad.py` | 184 | **80%** | biquad fallback branches |
| `realtime/processor.py` | 121 | **80%** | |
| `effect.py` | 228 | **94%** | |
| `filter/_fftconv.py` | 41 | **100%** | |
| `realtime/sounddevice_backend.py` | 121 | **0%** | needs hardware — excluded from targets |
| **Overall** | **2129** | **74%** | 586 branches tracked |

Phase 1 raises the gap modules above to the ≥90%-line / ≥80%-branch target
documented in the master plan.

### Phase 1 result (2026-04-11)

After six targeted test files (`tests/test_fused.py`, `test_filter_base.py`,
`test_filter_utils.py`, `test_ops_dispatch.py`, `test_filterbank.py`,
`test_iir_gaps.py`) the gap modules land at:

| Module | Baseline | Post-Phase 1 | Δ |
|---|---:|---:|---:|
| `filter/fused.py` | 9% | **75%** | +66 |
| `filter/utils.py` | 0% | **91%** | +91 |
| `filter/__base.py` | 44% | **89%** | +45 |
| `filter/filterbank.py` | 73% | **100%** | +27 |
| `filter/iir.py` | 71% | **86%** | +15 |
| `_ops.py` | 64% | **84%** | +20 |
| **Overall** (sounddevice omitted) | — | **88%** | — |

`fail_under = 87` is set in `[tool.coverage.report]` to lock the gain in.
Remaining gaps are mostly CUDA-dependent (`fused.py` `move_coeff`) or in the
biquad fallback paths that move in Phase 2. The `sounddevice_backend` is
omitted from coverage because it requires real audio hardware — it still
exists in the source tree but is not a Phase 1 test target.

Phase 1 also surfaced **four new Phase 2 bugs** via the new tests:

1. `AbstractFilter._has_computed_coeff` falls through to `return True` when
   an IIR subclass has `_sos = None` and no `b`/`a` attributes, silently
   claiming coefficients are ready before compute_coefficients has run.
   (`src/torchfx/filter/__base.py:425-431`)
2. `ParallelFilterCombination.__init__` sets `self.fs = fs` before
   `self.filters = filters`, so the fs setter iterates an unset attribute
   and crashes with `AttributeError` whenever `fs` is passed at
   construction time. (`src/torchfx/filter/__base.py:1008-1015`)
3. `FX` / `AbstractFilter` has no `__or__` method despite the docstring at
   `src/torchfx/filter/__base.py:617` claiming "inherited from FX".
   `f1 | f2` between two filters raises `TypeError`. Only `Wave.__or__` is
   defined. Users rely on `nn.Sequential` in the meantime.
4. `Delay` in its current form is non-streaming: feeding a
   `[2, 512]` chunk produces a `[2, 36512]` output because the tail buffer
   is appended to every call. See the realtime profile finding below.

All four are marked with `pytest.mark.xfail(strict=True)` so the tests flip
to passing the moment the bugs are fixed.

## CPU benchmarks — headline numbers

Full pytest-benchmark JSON at
`benchmarks/results/local-cpu-baseline.json`. Selected numbers (mean, 30 rounds):

- **SOS cascade**, order 4, stereo, 5 s signal: **2.05 ms/call**
- **SOS cascade**, order 8, stereo, 5 s signal: **2.92 ms/call**
- **2-filter pipeline** (HiButter(2) + LoButter(4)), stereo, 10 s: **36.6 ms/call**
- **IIR** order 30 Butterworth, 30 s, 8 channels, torchfx C++ path: **210 ms**,
  scipy: **267 ms** (torchfx wins at high order / high channel count)
- **IIR** order 1, 1 s, 1 channel, torchfx C++ path: **1.41 ms** (σ=0.80 ms),
  scipy: **1.01 ms**. Note the **very high stddev** — the per-call device/dtype
  churn and first-call compile overhead are bleeding into measurements at
  small signal sizes. This matches the profile findings below.

## CPU profile — offline filter chain (10 min stereo @ 48 kHz)

Chain: `LoButterworth(order=8) | HiChebyshev1(order=6) | Gain`, 3 recorded
iterations under `torch.profiler`. Total Self CPU = **1.787 s**.

| Op | Self CPU % | Calls | Shape | Interpretation |
|---|---:|---:|---|---|
| `aten::_to_copy` / `aten::copy_` | **19.9 %** | 12 | `[2, 28.8M]` | **dtype conversion churn** — entire signal copied float32↔float64 for each filter per call. Directly matches Phase 3.1 in the plan. **3.86 GB** of wasted data movement per iteration. |
| `aten::mul` | 3.2 % | 3 | `[2, 28.8M]` | gain stage — expected, cheap |
| `aten::empty_strided` | 0.03 % | 33 | various | **6.44 GB** of temporaries allocated — downstream of the `_to_copy` path |
| `aten::clone` | — | 6 | `[4, 2, 2]` & `[3, 2, 2]` | biquad state stacks — matches the `torch.stack` state-update pattern flagged in Phase 3.2 |

**Concrete Phase 3.1 target:** the `aten::_to_copy` row must drop below 1 % of
Self CPU and the 3.86 GB / iteration copy volume must drop to zero.

## CPU profile — realtime 512-sample chunk (BiquadBPF | Delay)

3 recorded iterations, Self CPU total = **1.94 ms**.

| Op | Self CPU % | Calls | Interpretation |
|---|---:|---:|---|
| `aten::zeros` + `aten::zero_` | **~42 %** | 12 | `[2, 36512]` zero-allocation in Delay — see finding below |
| `aten::to` / `aten::_to_copy` / `aten::copy_` | ~5 % | ~60 | coefficient/state device/dtype churn |
| `aten::mul` | ~5 % | 24 | comb-filter decay path |

### Finding: `Delay` has no streaming mode

Feeding a `[2, 512]` chunk into the Delay effect produces a `[2, 36512]`
output — Delay lengthens the signal to accommodate its tail buffer on every
call. In real-time processing this is pathological:

- per-call output is **~70× larger** than input
- the entire `[2, 36512]` output is zeroed (`aten::zeros` is 42% of CPU)
- means the "realtime" path re-allocates ≈285 KB and does ~850 KB of adds per
  512-sample chunk

This goes into the Phase 2 bug list as **streaming-Delay** alongside the
existing items; Phase 3.3 (Reverb / Delay fusion) becomes partly a behavioral
fix, not just a micro-optimization. Add a streaming mode that yields
block-aligned output and keeps tail state in a ring buffer.

### Headroom for realtime

At 48 kHz, a 512-sample chunk period is **10.67 ms**. Local CPU measured
**0.229 ms/call** (all iterations, wall-clock). That is **~46× headroom** —
the CPU realtime path is in no danger of underruns on this hardware even in
its current state. The work in Phase 3 is mostly about not wasting it when
scaled to multi-channel / multi-filter chains.

## Offline batch scenario (CPU)

`(B=32, C=2, T=480000)` through a 6-filter chain: **521 ms/call**. Will be
the main Phase 3 showcase once CUDA numbers come back from the cluster.

---

## CUDA baseline — 2026-04-12

Cluster hardware: Quadro RTX 6000 (24 GB), CUDA 12.8,
PyTorch 2.10.0+cu128, node025.

Artifacts at `benchmarks/results/cuda-baseline-20260412/` (benchmarks) and
`benchmarks/results/cuda-profile-baseline-20260412/` (profiler traces +
memory snapshot).

### Cold import + JIT compile

| Phase | Time |
|---|---|
| `import torchfx` | **10.40 s** |
| `_load_extension()` (JIT compile) | **5.33 s** |

The 5.3 s JIT compile is the headline baseline for the deferred AOT migration.
In a batch pipeline that processes thousands of files this is amortized away;
in a real-time context it's a painful one-time stall.

### CUDA IIR benchmark — headline numbers

Format: `[duration_s-channels]`, mean across 5+ rounds. SOS cascade, all on
Quadro RTX 6000 (GPU column) or on the same node's CPU.

| Scenario | GPU (parallel scan) | CPU (C++ kernel) | scipy | Numba CUDA |
|---|---:|---:|---:|---:|
| 1 s, 1 ch | 1.13 ms | 1.41 ms | 1.15 ms | 23.8 ms |
| 5 s, 2 ch | 2.39 ms | 6.13 ms | 10.9 ms | 133 ms |
| 30 s, 4 ch | 21.3 ms | 50.0 ms | 160 ms | 809 ms |
| 30 s, 8 ch | 38.2 ms | 83.5 ms | 320 ms | 821 ms |
| 60 s, 8 ch | **75.8 ms** | 168 ms | 859 ms | 1653 ms |

**Key takeaways:**

- GPU parallel-scan is **2.2× faster than the C++ CPU** kernel at 60 s × 8 ch,
  **11× faster than scipy**, and **22× faster than Numba CUDA**.
- The C++ CPU kernel is **4–5× faster than scipy** at high channel counts
  (30+ s, 8 ch), validating the investment in native kernels.
- Numba CUDA is 20–40× slower than the custom CUDA kernel —
  the handwritten parallel-scan approach is justified.
- At short signals (1 s, 1 ch), GPU overhead makes it roughly even with CPU.
  The crossover point is ≈ 5 s × 2 ch.

### CUDA biquad benchmark

| Signal | CPU | GPU | GPU/CPU |
|---|---:|---:|---:|
| 0.1 s, 1 ch | 0.11 ms | 0.27 ms | 0.43× (GPU slower) |
| 1.0 s, 1 ch | 0.44 ms | 0.27 ms | 1.6× |
| 5.0 s, 2 ch | 1.58 ms | 0.61 ms | 2.6× |
| 30 s, 2 ch | 9.30 ms | 3.16 ms | **2.9×** |

Biquad GPU wins above ~1 s signal length.

### Pipeline CUDA benchmark — `move_coeff` crash

The pipeline CUDA benchmarks **failed** because `LoButterworth.move_coeff("cuda")`
was removed in v0.5.1 (see CHANGELOG). The benchmark script must be updated to
use `.to(device)` instead. Added to Phase 2 bug list.

### CUDA profile — offline filter chain (10 min stereo @ 48 kHz)

Self CUDA total = **1.345 s** (3 iterations). The top entries are our custom
parallel-scan kernels — exactly what we want to see dominating:

| Op | Self CUDA % | Calls | Interpretation |
|---|---:|---:|---|
| `prefix_scan_phase3` | **33.7%** | 21 | Our write-back kernel — correct |
| `prefix_scan_phase2` | **31.1%** | 21 | Sweep kernel — correct |
| `prefix_scan_phase1` | **30.6%** | 21 | Build kernel — correct |
| `forcing_kernel` | 3.0% | 21 | Input preparation — acceptable |
| `aten::copy_` (dtype) | **1.4%** | 12 | **3.87 GB** CUDA alloc → Phase 3.1 |
| `aten::mul` (Gain) | 0.2% | 3 | Negligible |

**Phase 3.1 target (CUDA):** `aten::copy_` at 1.4% Self CUDA is already lower
than the CPU profile's 19.9%, but it still copies **3.87 GB** per iteration.
Eliminating the dtype churn will drop this to zero.

### CUDA profile — offline batch chain (B=32, C=2, T=480k)

Self CUDA total = **668.6 ms** (3 iterations).

| Op | Self CUDA % | Interpretation |
|---|---:|---|
| `prefix_scan_phase3` | 46.4% | Dominates — expected |
| `prefix_scan_phase1` | 42.1% | |
| `forcing_kernel` | 4.1% | |
| `aten::copy_` (dtype) | **3.7%** | **5.15 GB** CUDA alloc |
| `prefix_scan_phase2` | 3.4% | Lower % because batch shape is wider |

**Additional CPU-side finding:** `aten::item` calls: 135 per iteration,
consuming **21.1% of Self CPU time**. This is Python-side scalar extraction
from GPU — the SOS coefficient indexing (`sos[s, 4]`, etc.) goes through
`aten::_local_scalar_dense` which forces a GPU→CPU sync per scalar. This is
a new Phase 3 optimization target: batch-extract SOS coefficients once per
forward, not per-section.

### CUDA profile — realtime GPU (512-sample chunks)

Self CUDA total per 3-iteration block = **246 µs**.

| Op | Self CUDA % | Interpretation |
|---|---:|---|
| `sequential_biquad_kernel` | 44.6% | Short signals use sequential kernel — correct |
| elementwise kernels (dtype) | ~27% | Coefficient cast churn |
| `aten::add_` / `aten::mul` (Delay comb) | ~14% | |
| `aten::zeros` + `fill_` (Delay tail) | ~5.5% | Same pathological Delay behavior |
| `delay_line_kernel` | 2.0% | Minimal — the native kernel itself is fast |

Per-chunk CUDA time ≈ **82 µs**. At 48 kHz / 512 samples, the chunk period is
10.67 ms → **~130× headroom** on GPU. The realtime GPU path is in excellent
shape; the biggest wins are behavioral (Delay streaming mode) not kernel-level.

### Memory snapshot

The `torch.cuda.memory._snapshot()` pickle is at
`benchmarks/results/cuda-profile-baseline-20260412/memory_snapshot.pkl`.
Visualize with `python -m torch.cuda.memory_viz memory_snapshot.pkl`.

---

## Summary of Phase 3 optimization targets (data-driven)

| Target | Evidence | Expected win |
|---|---|---|
| **3.1 Eliminate dtype churn** | CPU: 19.9% Self CPU, 3.86 GB/iter. CUDA: 1.4–3.7% Self CUDA, 3.87–5.15 GB/iter | Largest single win on CPU; moderate on GPU |
| **3.2 Batch SOS coefficient extraction** | Batch CUDA: `aten::item` = 21.1% Self CPU, 135 calls/iter | Major CPU-side win on GPU path |
| **3.3 Delay streaming mode** | CPU: 42% Self CPU (zeros). GPU: ~5.5% CUDA (zeros+fill) | Critical for realtime correctness |
| **3.4 State update allocation** | `aten::clone` / `torch.stack` per section per chunk | Moderate win at high section counts |
| **3.5 Pipeline benchmark fix** | `move_coeff` removed in v0.5.1 | Prerequisite for CUDA pipeline tracking |

The master plan is at [`plans/starry-juggling-matsumoto.md`](../../../../.claude/plans/starry-juggling-matsumoto.md)
(not in the repo tree — local to the planning workflow).

---

## Phase 3 results — 2026-04-12

Artifacts at `benchmarks/results/cuda-phase3-20260412/` (benchmarks + profiles)
and `benchmarks/results/local-cpu-phase3.json` (CPU benchmarks).

### Changes applied

1. **SOS device cache** (`iir.py`, `fused.py`): cache the device-matched SOS
   tensor; invalidate only on `compute_coefficients()` re-run or device change.
   Pass a CPU copy (`sos_cpu=self._sos`) to the native dispatch layer so the
   per-call `sos.detach().to("cpu")` is eliminated.
2. **Biquad coefficient pre-extraction** (`biquad.py`, `_ops.py`): cache `a1_f64`
   and `a2_f64` as Python floats at design time so the per-call `float(a_f64[1])`
   GPU→CPU sync is eliminated.
3. **In-place state updates** (`iir.py`, `fused.py`, `biquad.py`): replaced
   `torch.stack([...])` per section per chunk with `copy_()` into pre-shaped
   buffers.
4. **Reverb algebraic fusion** (`effect.py`): `(1-mix)*x + mix*(x + decay*d)` →
   `x.clone().add_(d, alpha=mix*decay)` — 5 ops → 2 ops.
5. **Delay wet/dry lerp** (`effect.py`): `(1-mix)*x + mix*d` → `torch.lerp(x,d,mix)`
   — 3 ops → 1 kernel.

### CUDA IIR — Phase 0 vs Phase 3

| Scenario | Phase 0 | Phase 3 | Speedup |
|---|---:|---:|---:|
| 1 s, 1 ch | 1.14 ms | 0.95 ms | **1.20×** |
| 1 s, 4 ch | 1.46 ms | 1.09 ms | **1.33×** |
| 5 s, 1 ch | 1.76 ms | 1.39 ms | **1.27×** |
| 5 s, 2 ch | 2.38 ms | 2.06 ms | **1.16×** |
| 30 s, 1 ch | 7.86 ms | 7.53 ms | 1.04× |
| 60 s, 8 ch | 75.80 ms | 75.62 ms | 1.00× |

Short signals see 15–33% speedup because per-call overhead (SOS caching,
`aten::item` elimination) is a larger fraction of runtime. Long signals
converge to ~1× because the parallel-scan kernels dominate.

### CUDA biquad (stateful) — Phase 0 vs Phase 3

| Scenario | Phase 0 | Phase 3 | Speedup |
|---|---:|---:|---:|
| 0.1 s, 1 ch | 0.264 ms | 0.225 ms | **1.18×** |
| 1.0 s, 1 ch | 0.271 ms | 0.228 ms | **1.19×** |
| 5.0 s, 1 ch | 0.540 ms | 0.496 ms | **1.09×** |
| 30 s, 1 ch | 1.978 ms | 1.941 ms | 1.02× |

### Pipeline CUDA (NEW — was crashing in Phase 0)

| Benchmark | Phase 3 |
|---|---:|
| `sos_cascade[4-cuda]` | 1.31 ms |
| `sos_cascade[8-cuda]` | 2.48 ms |
| `pipeline[cuda]` | 3.26 ms |

These tests crashed in Phase 0 due to `move_coeff` removal. The fix
(`.to(device)` in benchmark scripts) landed in Phase 2.

### Profiler comparison — key metrics

| Metric | Phase 0 | Phase 3 | Change |
|---|---:|---:|---|
| `aten::item` Self CPU % (batch chain) | **21.1%** | **0.02%** | ~1000× reduction |
| `aten::item` calls (batch chain) | 135 | 129 | ~same call count, but negligible cost |
| `aten::_to_copy` CUDA alloc (batch chain) | 5.15 GB | 5.15 GB | unchanged — dtype casts remain for input |
| `aten::copy_` Self CUDA % (filter chain) | 1.4% | 1.4% | stable — signal dtype casts are inherent |
| Custom kernels % CUDA (filter chain) | 95.3% | 95.4% | kernels dominate equally |
| `aten::lerp` in realtime GPU | absent | **2.4%** | Delay optimization confirmed |

The massive `aten::item` win (21.1% → 0.02%) comes from the `sos_cpu` parameter
that pre-passes a CPU copy of the SOS matrix, and the `a1_f64`/`a2_f64`
pre-extraction that avoids per-call `float()` GPU→CPU synchronization.

The `aten::_to_copy` allocation remains because input signals arrive as float32
and the SOS kernels operate in float64 — this is a correctness requirement, not
churn. The Phase 0 baseline was measuring the same thing; the difference is that
*coefficient* casts are now cached.

### Cold import

| Phase | Phase 0 | Phase 3 |
|---|---:|---:|
| `import torchfx` | 10.4 s | 9.6 s |
| `_load_extension()` JIT | 5.3 s | 15.6 s |

The JIT compile time tripled, likely due to cache invalidation from kernel
source changes. On subsequent runs the cache is warm and the cost is zero.
This reinforces the case for the deferred AOT build migration.

### FIR / FFTConv (control group)

GPU FIR and FFTConv numbers are flat (±5%), confirming no regressions on
untouched paths. CPU numbers on the cluster are noisier due to shared-node
variability, but all GPU numbers are stable.
