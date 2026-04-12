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

## Outstanding — produced by the cluster

The following will be filled in from a SLURM run:

- [ ] CUDA equivalents of every benchmark in this table
- [ ] GPU cold-start time for `_load_extension()` (JIT compile baseline —
      informs the deferred AOT decision)
- [ ] Chrome trace for `offline_batch_chain` on CUDA
- [ ] Memory snapshot from `torch.cuda.memory._snapshot()` on the same scenario

Submit with:

```bash
sbatch benchmarks/run_benchmarks.slurm
# then pull benchmarks/results/<jobid>/ back to the repo
```

The master plan is at [`plans/starry-juggling-matsumoto.md`](../../../../.claude/plans/starry-juggling-matsumoto.md)
(not in the repo tree — local to the planning workflow).
