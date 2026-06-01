# TorchFX Optimization Roadmap

**Scope.** This document plans **library-level** performance, correctness, and usability work on
TorchFX — the runtime, the native kernels, the realtime path, and the developer/user experience.
It is **not** a paper plan. Where a task happens to also affect the IS² 2026 paper narrative, that
is a side effect noted in passing; the paper's own plan lives in
[IS22026/PLAN.md](IS22026/PLAN.md) and is updated separately as epics here land.

The three questions that motivated this roadmap, answered up front:

1. **Fusion stage** — today's fusion is *structural concatenation of SOS rows*: it collapses
   Python dispatch (`K → 1`) but CUDA launches still scale with `K`. The remaining wins are a
   real **single-kernel SOS cascade** (Epic D) and cheap **algebraic folding** (Epic E).
2. **Graph building / compilation** — go **CUDA Graphs** (Epic C), not `torch.compile` (our native
   ops are opaque to TorchInductor) and not a hand-rolled IR. Capturing the fused chain amortizes
   the per-section launch storm on the streaming/realtime path, which is where launches actually hurt.
3. **Torch vs. raw CUDA API** — **stay on `torch.cuda`.** It already gives us streams, events,
   pinned memory, and graph capture. The raw CUDA Driver API buys weeks of integration pain for ~0
   benefit. Every lever below lives inside the existing PyTorch C++/CUDA extension.

**The spine.** The CUDA kernels are double-only. On consumer cards (RTX 3070, A40: 1:32 FP32:FP64)
this is why the GPU *loses to its own CPU* at 8 channels. **Epic B (FP32 path) is the single
highest-leverage change** and a hard dependency for several others — build it first.

Grounded against `dev` @ `7590758`; current benchmark snapshot in
[Appendix A](#appendix-a--baseline-snapshot-regression-targets).

---

## How to work this roadmap

**Pick a task by its ID** (e.g. `B1`). Each task is self-contained: goal, rationale, files,
difficulty/effort/dependencies, implementation steps, and **verification & testability** criteria.
A task is *not done* until every checkbox in its verification block is green.

**Standard loop for any task:**

```bash
uv sync                                              # ensure the native ext is rebuilt
uv run pytest tests/                                 # full suite must stay green
uv run ruff check src/ tests/                        # lint
uv run black --check src/ tests/                     # format
uv run mypy --config-file pyproject.toml src/        # strict types
```

**Performance tasks additionally must:**

```bash
# Capture a BEFORE baseline on the touched path, then an AFTER, on the SAME machine:
uv run pytest benchmarks/test_iir_bench.py --benchmark-enable \
  --benchmark-json=/tmp/before.json
# ... make the change ...
uv run pytest benchmarks/test_iir_bench.py --benchmark-enable \
  --benchmark-json=/tmp/after.json
# Compare medians; record the delta in the task's PR description and the CHANGELOG.
```

**Conventions (enforced — see [CLAUDE.md](CLAUDE.md)):**

- Numerical equivalence is checked with `torch.testing.assert_close`; scalar checks with
  `pytest.approx`. The reference for filter correctness is `scipy.signal.sosfilt` /
  `sosfiltfilt` (see existing patterns in [tests/test_native_design.py](tests/test_native_design.py),
  [tests/test_cuda_kernels.py](tests/test_cuda_kernels.py)).
- Every user-visible change gets a [CHANGELOG](CHANGELOG) entry under `[0.6.0] - Unreleased`.
- No performance task may regress any *other* benchmark by more than noise (±3% median). If it
  does, it is not done — either fix the regression or gate the new path behind a dispatch condition.
- Custom CUDA kernels are templated on `scalar_t` (the CPU kernels already are — match that style).

**Difficulty legend:** `trivial` (≤0.5 d, mechanical) · `easy` (≤1.5 d) · `medium` (2–4 d) ·
`hard` (5–10 d, kernel-level or cross-cutting). Effort assumes one focused engineer/agent.

**Decisions that are settled (do not relitigate per-task):** stay on `torch.cuda`; do not adopt
`torch.compile` for the hot path; do not add fp16/bf16 *execution* (IIR feedback is unsafe) — only
guard against silent acceptance of those dtypes.

---

## Epic summary & recommended order

| Epic | Theme | Tasks | Net effort | Build order |
|---|---|---|---|---|
| **A** | Correctness & API hardening | A1–A4 | ~2.5 d | **1st (gates everything)** |
| **B** | FP32 CUDA execution path | B1–B5 | ~6–8 d | **2nd (the spine)** |
| **G** | Build & dispatch tuning | G1–G2 | ~1.5 d | with/after B |
| **C** | Realtime & streaming execution | C1–C4 | ~6–9 d | after B |
| **E** | Fusion-planner algebraic opts | E1–E2 | ~3.5–4 d | independent |
| **F** | CPU & edge performance | F1–F3 | ~5–6 d | after B (re-measure crossover) |
| **D** | Kernel-level fusion & scan | D1–D2 | ~13–14 d | last / opportunistic |

**Dependency graph (arrows = "must precede"):**

```
A (correctness)
   └──> B (FP32) ──> G2 (threshold retune, crossover moves once FP32 lands)
            ├──> C (graphs/streaming — capture is cleaner on the float32 path)
            ├──> F (CPU SIMD — re-measure CPU-vs-GPU crossover after FP32)
            └──> D2 (single-pass scan templated alongside FP32)
E (planner algebra) ── independent
D1 (mega-kernel)    ── independent but highest risk; do last
```

---

## Epic A — Correctness & API hardening

*Theme: a deferred-execution runtime cannot have silent wrong-output paths. These gate all
performance work and all benchmarking — a perf number measured on a buggy path is worthless.*

### A1 — Force coefficient recomputation when `fs` changes

- **Goal.** A filter materialized at one sample rate, then piped into a `Wave` of a different
  sample rate, must recompute its coefficients instead of silently reusing stale ones.
- **Why (library correctness).** Today `_has_computed_coeff` short-circuits recomputation
  ([filter/__base.py](src/torchfx/filter/__base.py)); reusing a filter across sample rates yields
  silently wrong audio. This is the worst class of bug — wrong, not loud.
- **Files.** [src/torchfx/filter/__base.py](src/torchfx/filter/__base.py),
  [src/torchfx/filter/iir.py](src/torchfx/filter/iir.py),
  [src/torchfx/wave.py](src/torchfx/wave.py).
- **Difficulty.** trivial · **Effort.** 0.5 d · **Depends on.** —
- **Implementation steps.**
  1. Track the `fs` that coefficients were last computed for (e.g. `_coeff_fs: int | None`).
  2. In the `fs` setter / `__update_config` path, if the new `fs` differs from `_coeff_fs`,
     clear `_has_computed_coeff`, drop `_sos`/`_sos_device_cache`, and reset DF1 state.
  3. Ensure `FusedSOSCascade` rebuilds (it is constructed fresh in `_materialize`, so confirm the
     planner re-runs when `fs` changed).
- **Verification & testability.**
  - [ ] New test in [tests/test_iir.py](tests/test_iir.py): design `LoButterworth(cutoff, fs=44100)`,
        run on a 44.1 kHz wave, then pipe the *same instance* into a 48 kHz wave; assert the second
        output matches a freshly-designed `fs=48000` filter via `assert_close`.
  - [ ] Test the fused path too (mix two IIRs, change `fs`) in
        [tests/test_chain_fusion.py](tests/test_chain_fusion.py).
  - [ ] Existing suite stays green.
- **Done when.** Reusing a filter across sample rates is provably correct and covered by a
  regression test that fails on the current `dev`.

### A2 — Define & enforce state-reset semantics across `Wave` reuse

- **Goal.** Make stateful-filter behaviour deterministic when the same filter instance is piped
  into two different `Wave`s (offline vs. streaming).
- **Why (usage).** Currently DF1 state persists across `Wave` instances if a filter is reused — an
  undocumented footgun that corrupts the second stream. The contract must be explicit.
- **Files.** [src/torchfx/wave.py](src/torchfx/wave.py),
  [src/torchfx/filter/__base.py](src/torchfx/filter/__base.py),
  [src/torchfx/filter/fused.py](src/torchfx/filter/fused.py) (`reset_state` already exists).
- **Difficulty.** trivial · **Effort.** 0.5 d · **Depends on.** —
- **Implementation steps.**
  1. Decide the contract: *offline `_materialize` resets per-filter state at the start of a
     materialize; streaming (`StreamProcessor`/`RealtimeProcessor`) preserves it across chunks.*
  2. Call `reset_state()` on stateful modules at the top of `Wave._materialize()` (offline path).
  3. Document the contract in the `Wave` and filter docstrings with a runnable example.
- **Verification & testability.**
  - [ ] Test: reuse one IIR across two distinct `Wave`s; assert the second wave's output equals a
        fresh filter's output (no state bleed) in offline mode.
  - [ ] Test: a `StreamProcessor` run preserves state across chunks (existing
        [tests/test_realtime.py](tests/test_realtime.py) coverage stays green).
  - [ ] Docstring example executes under doctest/pytest.
- **Done when.** The reuse contract is documented and both offline-reset and streaming-preserve
  behaviours are pinned by tests.

### A3 — Reject (don't silently upcast) fp16/bf16 inputs

- **Goal.** A `float16`/`bfloat16` input to a native filter raises a clear error instead of being
  silently promoted to `float64`.
- **Why (usage).** `_select_native_dtype` currently funnels any non-float64 CUDA input to float64,
  so an fp16 tensor produces 4× memory and surprising numerics with no warning. IIR feedback is
  not fp16-safe; failing loud is correct.
- **Files.** [src/torchfx/_ops.py](src/torchfx/_ops.py) (`_select_native_dtype`).
- **Difficulty.** trivial · **Effort.** 0.25 d · **Depends on.** —
- **Implementation steps.**
  1. In `_select_native_dtype`, if `x.dtype in (torch.float16, torch.bfloat16)`, raise
     `TypeError` with a message pointing the user to cast to float32/float64.
  2. Keep float32/float64 behaviour unchanged.
- **Verification & testability.**
  - [ ] Test in [tests/test_ops_dispatch.py](tests/test_ops_dispatch.py): `pytest.raises(TypeError)`
        for an fp16 and a bf16 input on both CPU and (if available) CUDA.
  - [ ] float32/float64 paths unchanged (existing dispatch tests green).
- **Done when.** Half-precision inputs fail fast with an actionable message.

### A4 — Lock in fused-vs-unfused state-equivalence tests

- **Goal.** Prove that a fused SOS cascade produces bit-for-bit-close output and state to the
  equivalent unfused `nn.Sequential`, in both batch and chunked-streaming modes.
- **Why (library correctness).** Fusion is only safe if it is observationally equivalent. There is
  no active test that compares fused vs. unfused *state* across chunk boundaries.
- **Files.** [tests/test_chain_fusion.py](tests/test_chain_fusion.py),
  [tests/test_fused.py](tests/test_fused.py).
- **Difficulty.** easy · **Effort.** 1 d · **Depends on.** A1.
- **Implementation steps.**
  1. Build a chain of 3–5 mixed IIR/Biquad sections.
  2. Compare `(fused output)` vs. `(sequential output)` on a single long signal via `assert_close`.
  3. Compare *chunked* execution: feed the same signal in N chunks through both, assert outputs and
     final DF1 state match (this is the streaming-correctness guarantee).
  4. Parametrize over channel counts `{1, 2, 8}` and dtypes `{float32, float64}`.
- **Verification & testability.**
  - [ ] Fused == unfused (batch) within `rtol=1e-5` (f32) / `1e-12` (f64).
  - [ ] Fused == unfused (chunked, state continuity) within the same tolerances.
  - [ ] Runs on CPU always; on CUDA when available (`@pytest.mark.skipif(not cuda)`).
- **Done when.** Fusion equivalence is a permanent regression gate for every later kernel change.

---

## Epic B — FP32 CUDA execution path *(the spine)* — ✅ DONE

*Status: landed and GPU-verified on an RTX 3070 (commit history on `dev`). Measured **3.0–3.6×**
(B1 templated kernels, B2 dtype dispatch, B3 precision harness; B4 evaluated as a no-op since the
cached device-dtype conversion already makes execution follow the input). B5 numbers in
[Appendix A](#appendix-a--baseline-snapshot-regression-targets).*

*Theme: lift the consumer-GPU FP64 penalty. Honest expected impact: **2–4× on RTX 3070 / A40**,
**~1.5–2× on L40S**. The headline outcome is that the GPU stops losing to its own CPU at higher
channel counts. The theoretical 32× FLOP ratio is not the realistic kernel speedup — the scan is
partly bandwidth/overhead-bound.*

### B1 — Template the CUDA kernels on `scalar_t`

- **Goal.** `forcing_kernel`, the three Blelloch phases, the sequential biquad kernel, `Mat3x3`,
  and `compute_forcing`/`parallel_biquad_scan`/`biquad_forward_cuda`/`sos_forward_cuda` all support
  `float` and `double` via a template parameter — mirroring the CPU kernel.
- **Why (realtime/throughput).** Float32 halves memory traffic and unlocks consumer-GPU FP32
  throughput; this is the difference between the GPU being usable and unusable for multichannel
  realtime on a 3070-class card.
- **Files.** [src/torchfx/_csrc/cuda/parallel_scan.cu](src/torchfx/_csrc/cuda/parallel_scan.cu),
  [src/torchfx/_csrc/cuda/biquad_forward.cu](src/torchfx/_csrc/cuda/biquad_forward.cu),
  [src/torchfx/_csrc/include/torchfx/](src/torchfx/_csrc/include/torchfx/) headers,
  [src/torchfx/_csrc/binding.cpp](src/torchfx/_csrc/binding.cpp).
- **Difficulty.** hard · **Effort.** 3–4 d · **Depends on.** A4 (equivalence gate).
- **Implementation steps.**
  1. Make `Mat3x3` a template `struct Mat3x3<T> { T m[6]; }`; template `mat_mul`, `mat_identity`,
     `extract_y`.
  2. Template the four kernels on `scalar_t`; replace hardcoded `double` literals with
     `scalar_t(...)`. Keep `BLOCK_SIZE`/shared-mem sizing in terms of `sizeof(Mat3x3<scalar_t>)`
     (FP32 halves shared-mem pressure — note the occupancy headroom for G2).
  3. Dispatch on dtype at the C++ boundary with `AT_DISPATCH_FLOATING_TYPES` (or an explicit
     `if (x.scalar_type()==kFloat) … else …`), matching `iir_cpu.cpp`'s structure.
  4. Update `binding.cpp` if the public signatures change (they should not — dtype is read from the
     tensor).
- **Verification & testability.**
  - [ ] [tests/test_cuda_kernels.py](tests/test_cuda_kernels.py): for float32 and float64 inputs,
        kernel output matches `scipy.signal.sosfilt` within dtype-appropriate tolerance.
  - [ ] A4 fused-vs-unfused equivalence passes on CUDA float32.
  - [ ] No build warnings; `nvcc` compiles both instantiations.
- **Done when.** The CUDA path runs natively in the input dtype with no float64 forced upcast in
  the kernel.

### B2 — dtype-aware CUDA dispatch

- **Goal.** `_select_native_dtype` returns `float32` for float32 CUDA inputs (today it forces
  float64), and the device-cache / state tensors follow suit.
- **Why.** Without this, B1's float32 kernels are never reached from Python.
- **Files.** [src/torchfx/_ops.py](src/torchfx/_ops.py),
  [src/torchfx/filter/iir.py](src/torchfx/filter/iir.py) (`_sos_cascade_forward` line ~115),
  [src/torchfx/filter/fused.py](src/torchfx/filter/fused.py).
- **Difficulty.** easy · **Effort.** 0.5–1 d · **Depends on.** B1.
- **Implementation steps.**
  1. Change `_select_native_dtype`: on CUDA, `return torch.float64 if x.dtype==torch.float64 else
     torch.float32` (same policy as CPU).
  2. Audit `_sos_cascade_forward`: `native_dtype` and the SOS device cache, state `[K,C,2]`, and the
     `sos_cpu` copy must all be the selected dtype.
  3. Decide the default-precision policy and document it: *dtype follows the input tensor; pass a
     float64 wave for maximum precision, float32 for speed.* Optionally expose
     `Wave.to(dtype=...)` ergonomics if not already present.
- **Verification & testability.**
  - [ ] [tests/test_ops_dispatch.py](tests/test_ops_dispatch.py): assert the dispatched dtype equals
        the input dtype on CUDA for float32 and float64.
  - [ ] End-to-end: a float32 `Wave | IIR` on CUDA stays float32 throughout (no hidden float64
        temporaries — check with a `dtype` assertion on intermediate via a hook or profiler).
- **Done when.** float32 in → float32 kernels → float32 out, with float64 still available on demand.

### B3 — FP32 numerical-validation harness

- **Goal.** A parametrized test suite that quantifies FP32-vs-FP64 error per `(filter_type, order)`
  against `scipy`, and classifies which designs are FP32-safe.
- **Why (usage + trust).** Users (and we) need to know when float32 is safe. This prevents shipping
  a "fast but silently wrong at order 16" path.
- **Files.** new `tests/test_fp32_precision.py`; reuse references from
  [tests/test_native_design.py](tests/test_native_design.py).
- **Difficulty.** easy · **Effort.** 1–1.5 d · **Depends on.** B1, B2.
- **Implementation steps.**
  1. Parametrize over filter families (Butterworth/Chebyshev1/Chebyshev2/Elliptic), orders
     `{2,4,8,16}`, and a long signal (≥60 s @ 48 kHz).
  2. Compute max L∞ and RMS error of float32-kernel output vs. float64-kernel output and vs.
     `scipy.signal.sosfilt(float64)`.
  3. Tabulate; assert errors are below a documented threshold for the "safe" set, and mark the
     "needs-float64" set with an xfail-or-warn so the boundary is explicit and tracked.
- **Verification & testability.**
  - [ ] Table of per-design errors emitted (e.g. to a JSON the test writes) for documentation.
  - [ ] Safe designs assert under threshold; unsafe designs are explicitly marked, not hidden.
- **Done when.** We can state, with evidence, which filters are float32-safe and the library docs
  reflect it.

### B4 — Remove the unconditional float64 cast in `FusedSOSCascade`

- **Goal.** Stop hard-casting fused SOS to float64 at construction.
- **Why.** [filter/fused.py:75](src/torchfx/filter/fused.py#L75) (`.to(dtype=torch.float64)`) and
  `move_coeff` (line ~111) force float64 on the canonical coefficients, fighting the dtype-aware
  dispatch and adding a per-forward downcast on the CPU float32 path. Small perf, but it's a code
  smell that actively contradicts B2.
- **Files.** [src/torchfx/filter/fused.py](src/torchfx/filter/fused.py).
- **Difficulty.** trivial · **Effort.** 0.25 d · **Depends on.** B2.
- **Implementation steps.**
  1. Keep the canonical SOS in float64 (it's the precision-safe reference for design), but let the
     device cache / execution dtype be chosen at dispatch (as `_sos_cascade_forward` already does
     for the non-fused path). Align `FusedSOSCascade.forward` with `_sos_cascade_forward`'s policy.
  2. Remove the float64 forcing in `move_coeff`.
- **Verification & testability.**
  - [ ] A4 equivalence still green (float32 + float64).
  - [ ] Micro-bench: CPU float32 fused cascade shows no per-forward float64 temporary (profiler:
        `aten::_to_copy` self-time does not grow with calls).
- **Done when.** Fused and non-fused paths share one dtype policy.

### B5 — FP32 throughput benchmark + regression guard

- **Goal.** Measure and lock in the FP32 win; prevent silent regressions.
- **Why.** The roadmap's central number must be tracked, not assumed.
- **Files.** [benchmarks/test_iir_bench.py](benchmarks/test_iir_bench.py),
  [benchmarks/test_hotpath_bench.py](benchmarks/test_hotpath_bench.py),
  [benchmarks/conftest.py](benchmarks/conftest.py).
- **Difficulty.** easy · **Effort.** 0.5–1 d · **Depends on.** B1–B2.
- **Implementation steps.**
  1. Add float32 parametrizations to the CUDA IIR benchmarks (today they run the float64 path).
  2. Run on the Alienware RTX 3070 (and L40S/A40 if available); capture medians into
     `benchmarks/results/`.
  3. Record the speedup vs. the float64 baseline in [Appendix A](#appendix-a--baseline-snapshot-regression-targets)
     and in the CHANGELOG.
- **Verification & testability.**
  - [ ] FP32 median ≥ 1.5× faster than FP64 on RTX 3070 for `60s/8ch` (the inversion case); if not,
        investigate occupancy (ties to G1/G2) before declaring done.
  - [ ] FP32 `60s/8ch` GPU median < CPU `60s/8ch` median (GPU re-wins multichannel).
  - [ ] No regression on the float64 path.
- **Done when.** The FP32 win is a checked-in, reproducible number with a guard against backsliding.

---

## Epic C — Realtime & streaming execution efficiency — 🟡 PARTIAL

*Status: **C4 done** (in-place DF1 state in `sos_forward_cuda` — removes the per-call
`[K,C,2]` clone and is the foundation for graph capture; full suite green). **C1
(CUDA-graph capture) attempted and reverted** — GPU testing confirmed the roadmap's
"C3 enables C1" dependency is hard: (1) the K=1 biquad path's `b.cpu()` coefficient
read is a device→host sync that invalidates stream capture, and (2) the K≥2 cascade
captures but replays ~39× wrong because the per-section `torch::empty` scratch
allocations alias in the capture memory pool. So **C1/C2 are deferred behind C3
(static scratch buffers) + a sync-free coefficient path** — these are the real
prerequisites, and the measured prize is large (the parallel scan is ~135 µs of pure
launch overhead, so graph replay would slash short-chunk/realtime latency).*

*Recommended sequencing for the GPU-streaming graph win: **C3 (static scratch) →
sync-free K=1 coeffs → C1 (capture) → C2 (pinned/async)**. C3 is the unlock.*

*Theme: the realtime path already has a worker-thread producer/consumer split and latency/xrun
instrumentation (shipped in 0.6.0). This epic attacks the per-chunk overheads that eat the audio
callback budget: launch storms, transfer stalls, and per-call allocation.*

### C1 — CUDA Graph capture of the fused chain (streaming)

- **Goal.** Capture the fixed per-chunk kernel sequence of a fused cascade into a
  `torch.cuda.CUDAGraph` and replay it each chunk, amortizing launch overhead.
- **Why (realtime efficacy).** A `K`-section cascade issues `~4K` launches *per chunk*; at 48 kHz /
  512-sample chunks that is ~94 chunk-calls/s each re-issuing the whole storm. Graph replay turns
  that into one launch per chunk. This is the concrete "deeper graph compilation" step, and — per
  our audit — it is **not blocked** by host-side syncs: coefficients are already read from a
  pre-supplied CPU copy ([biquad_forward.cu:67](src/torchfx/_csrc/cuda/biquad_forward.cu#L67)), so
  there is no device→host `.item()` sync to break capture.
- **Files.** [src/torchfx/realtime/processor.py](src/torchfx/realtime/processor.py),
  [src/torchfx/realtime/stream.py](src/torchfx/realtime/stream.py),
  [src/torchfx/filter/fused.py](src/torchfx/filter/fused.py); a small capture helper (new
  `src/torchfx/realtime/_cuda_graph.py`).
- **Difficulty.** medium · **Effort.** 2–3 d · **Depends on.** B1–B2 (capture is cleaner and the
  win larger once float32 is the streaming dtype).
- **Implementation steps.**
  1. Require static shapes (fixed chunk size, channel count, dtype, device) — already true for a
     running `StreamProcessor`/`RealtimeProcessor`.
  2. Warm up (a few eager iterations on a side stream), then capture the fused-cascade forward into
     a graph with static input/output tensors (copy chunk-in → static buffer, replay, copy-out).
  3. Fall back to eager if shapes change or capture fails; gate behind a `use_cuda_graph` flag,
     default on for CUDA streaming once validated.
  4. Confirm caching-allocator graph-pool usage so per-chunk `torch::empty` allocations are capture-safe
     (this couples with C3 — prefer preallocated scratch).
- **Verification & testability.**
  - [ ] Output of graph-replayed streaming equals eager streaming bit-for-bit-close (`assert_close`)
        over a multi-chunk signal, including final state.
  - [ ] Benchmark in [benchmarks/test_realtime_bench.py](benchmarks/test_realtime_bench.py): p99
        per-callback time on a `K≥10` CUDA chain drops vs. eager (only meaningful once a GPU realtime
        path exists; otherwise measure on `StreamProcessor` chunked-offline).
  - [ ] Eager fallback path covered by a test (force a shape change mid-stream).
- **Done when.** A captured fused chain replays correctly and measurably reduces per-chunk launch
  overhead on the streaming path.

### C2 — Pinned host buffers + async H2D/D2H for streaming

- **Goal.** Use pinned memory and a dedicated CUDA stream so per-chunk host↔device copies overlap
  with compute instead of serializing.
- **Why (realtime).** For GPU streaming, transfer latency is added to every chunk's deadline.
  Overlap halves the visible transfer cost (~0.04 ms → ~0.02 ms at 512 samples; larger at bigger
  buffers).
- **Files.** [src/torchfx/realtime/ring_buffer.py](src/torchfx/realtime/ring_buffer.py),
  [src/torchfx/realtime/processor.py](src/torchfx/realtime/processor.py).
- **Difficulty.** medium · **Effort.** 2–3 d · **Depends on.** B2; pairs with C1.
- **Implementation steps.**
  1. Allocate the host-side ring staging buffers as pinned (`pin_memory=True`).
  2. Issue H2D/compute/D2H on a non-default stream with events to order them; double-buffer so chunk
     `n+1`'s upload overlaps chunk `n`'s compute.
  3. Synchronize only at the chunk boundary the consumer needs.
- **Verification & testability.**
  - [ ] Streaming output unchanged vs. synchronous path (`assert_close`).
  - [ ] Benchmark shows reduced per-chunk wall time for large buffers (`≥2048`) on CUDA.
  - [ ] No deadlocks/races under a 60 s soak in [tests/test_realtime.py](tests/test_realtime.py).
- **Done when.** Transfers overlap compute and the streaming path's per-chunk budget shrinks.

### C3 — Preallocate & reuse scratch buffers across sections/chunks

- **Goal.** Stop allocating `forcing`, `y`, and `block_agg` per section per forward; reuse a
  persistent workspace.
- **Why (realtime).** On the streaming hot path, per-call `torch::empty` allocations add allocator
  pressure and (importantly) make CUDA-graph capture awkward. A workspace pattern removes both.
- **Files.** [src/torchfx/_csrc/cuda/parallel_scan.cu](src/torchfx/_csrc/cuda/parallel_scan.cu),
  [src/torchfx/_csrc/cuda/biquad_forward.cu](src/torchfx/_csrc/cuda/biquad_forward.cu),
  [src/torchfx/_ops.py](src/torchfx/_ops.py).
- **Difficulty.** medium · **Effort.** 2–3 d · **Depends on.** B1; enables C1.
- **Implementation steps.**
  1. Introduce a per-cascade workspace (sized for `[C,T]` and `[C,num_blocks]`) owned by the
     `FusedSOSCascade`/dispatch layer and resized only when shapes grow.
  2. Pass workspace pointers into the kernels instead of allocating inside `compute_forcing` /
     `parallel_biquad_scan`.
  3. Reuse `y` across sections via ping-pong instead of a fresh `empty` per section.
- **Verification & testability.**
  - [ ] Output unchanged (`assert_close`) across CPU/CUDA, float32/float64.
  - [ ] Profiler/`torch.cuda.memory_stats`: allocation count per forward drops from `O(K)` to `O(1)`.
  - [ ] Streaming benchmark: reduced GC/allocator self-time.
- **Done when.** A fused forward performs a constant number of allocations regardless of `K`.

### C4 — Drop redundant state clones in `sos_forward_cuda`

- **Goal.** Remove the `new_sx.clone()` / `new_sy.clone()` at the top of `sos_forward_cuda` where
  in-place update is safe.
- **Why.** Minor allocator pressure on the hot path; tidy the kernel for C1/C3.
- **Files.** [src/torchfx/_csrc/cuda/biquad_forward.cu](src/torchfx/_csrc/cuda/biquad_forward.cu).
- **Difficulty.** easy · **Effort.** 1 d · **Depends on.** A4 (equivalence gate).
- **Implementation steps.**
  1. Determine whether callers rely on the input state tensors being preserved (they should pass
     owned per-section state). If safe, write state updates in place; otherwise clone once outside
     the section loop, not per section.
- **Verification & testability.**
  - [ ] A4 equivalence + streaming state-continuity tests green.
  - [ ] No aliasing bug: a test that reuses the same state tensor across two forwards still matches
        the unfused reference.
- **Done when.** Per-forward clones are eliminated without changing observable state semantics.

---

## Epic D — Kernel-level fusion & scan efficiency

*Theme: make fusion real at the CUDA level and stop wasting scan work. Highest risk; do last or
opportunistically. D1 is the only item that turns "we concatenated SOS rows" into genuine
kernel-level fusion.*

### D1 — Single-kernel SOS-section fusion (mega-kernel)

- **Goal.** One CUDA kernel that processes all `K` sections of a cascade — sections serial within a
  thread block, time-blocks parallel — replacing the Python/C++ loop of `~4K` launches.
- **Why (library + realtime).** Collapses launch count ~85% (e.g. 280 → ~43 for `K=20`). Wall-time
  benefit is **moderate** (15–27% at `K≥5`; long signals are bandwidth- not launch-bound), but it
  removes the per-section launch tax that hurts most at small chunk sizes (realtime) and high `K`.
- **Files.** [src/torchfx/_csrc/cuda/parallel_scan.cu](src/torchfx/_csrc/cuda/parallel_scan.cu),
  [src/torchfx/_csrc/cuda/biquad_forward.cu](src/torchfx/_csrc/cuda/biquad_forward.cu) (`sos_forward_cuda`),
  headers, [src/torchfx/_ops.py](src/torchfx/_ops.py).
- **Difficulty.** hard · **Effort.** ~10 d · **Depends on.** B1 (template first so it's one dtype
  story), C3 (workspace).
- **Implementation steps.**
  1. Stage the cross-section dependency: section `s+1` consumes section `s`'s output. Within a
     block, hold the `K` sections' coefficients in shared/constant memory and apply them in sequence
     to each time-block's running state, composing the inter-block prefix per section.
  2. Carry per-section DF1 state across time-blocks via the existing block-aggregate mechanism,
     generalized to `K` sections.
  3. Keep the existing per-section kernels as a correctness oracle and a fallback for shapes the
     mega-kernel doesn't handle.
  4. Count launches with `torch.profiler` (`ProfilerActivity.CUDA`) before/after.
- **Verification & testability.**
  - [ ] Mega-kernel output == per-section-loop output == `scipy.sosfilt` reference (`assert_close`),
        across `K∈{2,5,10,20}`, `C∈{1,2,8}`, float32/float64.
  - [ ] `torch.profiler` confirms launch count drops from `~4K` to `O(1)` per forward.
  - [ ] Benchmark: ≥15% median speedup at `K≥10` on at least one device; no regression at small `K`.
  - [ ] Streaming state continuity (A4-style) holds for the mega-kernel.
- **Done when.** A fused cascade is genuinely one (or a small constant number of) kernel launches,
  with proven numerical equivalence.

### D2 — Single-pass scan (eliminate phase-1/phase-3 double-compute)

- **Goal.** Replace the Blelloch up/down-sweep + phase-3 recompute with a single-pass
  decoupled-look-back scan (CUB `DeviceScan` with a custom 3×3-matrix associative op, or a
  hand-rolled chained scan).
- **Why.** Phase 3 recomputes the intra-block scan for every block > 0 — roughly doubling scan work.
  Honest impact is small (scan is ~5% of runtime; this reclaims ~half of that → ~2% overall) but it
  also reduces launch count and simplifies the code for D1.
- **Files.** [src/torchfx/_csrc/cuda/parallel_scan.cu](src/torchfx/_csrc/cuda/parallel_scan.cu).
- **Difficulty.** medium–hard · **Effort.** 3–4 d · **Depends on.** B1.
- **Implementation steps.**
  1. Define an associative functor over `Mat3x3<scalar_t>` for CUB `DeviceScan::InclusiveScan`, or
     implement single-pass decoupled look-back with per-tile status flags.
  2. Remove phase-3; derive `y[n]` directly from the inclusive prefix and the initial state.
  3. Validate associativity/identity carefully (matrix mult is associative; identity is `mat_identity`).
- **Verification & testability.**
  - [ ] Output == current implementation == `scipy` reference (`assert_close`).
  - [ ] `torch.profiler`: kernel count for the scan drops (3 → 1–2).
  - [ ] Benchmark: no regression; small improvement on long signals.
- **Done when.** The scan computes each block once and the phase-3 kernel is gone.

---

## Epic E — Fusion-planner algebraic optimizations

*Theme: widen what the planner can fuse and improve numerical conditioning. Perf impact is small;
the real value is **composition ergonomics** (a `Gain` mid-chain no longer splits a fused run) and
**high-order stability**.*

### E1 — Fold `Gain`/`Normalize` scalars into the SOS numerator

- **Goal.** When a constant `Gain(g)` (or a resolved `Normalize` scalar) sits between IIR/Biquad
  stages, fold it into the next section's `b0,b1,b2 *= g` so the fused run is not broken.
- **Why (usage).** Today any non-IIR module flushes the fused run
  ([wave.py](src/torchfx/wave.py) `_materialize`), so `IIR | Gain | IIR` is two fused stages plus a
  separate gain op. Folding keeps it one stage and removes a kernel/dispatch. Mathematically exact
  for linear gain.
- **Files.** [src/torchfx/wave.py](src/torchfx/wave.py) (`_materialize` planner),
  [src/torchfx/filter/fused.py](src/torchfx/filter/fused.py),
  [src/torchfx/effect.py](src/torchfx/effect.py) (identify foldable `Gain`).
- **Difficulty.** medium · **Effort.** 2–2.5 d · **Depends on.** A4.
- **Implementation steps.**
  1. In the planner, recognize a `Gain` with a static scalar between two fusible runs; fold the
     scalar into the leading numerator of the following section (or trailing of the previous).
  2. Only fold *static* gains; a dynamic/streaming `Normalize` (per-chunk peak) is **not** foldable —
     leave it as a separate stage and document why (state coherence).
  3. Keep an explicit opt-out for users who want gain as a distinct stage.
- **Verification & testability.**
  - [ ] `IIR | Gain(g) | IIR` output == unfused equivalent (`assert_close`).
  - [ ] Planner test: folded chain produces one `FusedSOSCascade`, not two + a gain.
  - [ ] Dynamic `Normalize` is *not* folded (test asserts it stays separate).
- **Done when.** Static gains no longer fragment fused cascades, with equivalence proven.

### E2 — SOS section ordering for numerical conditioning

- **Goal.** Order sections within a fused cascade by a stability heuristic (e.g. poles by proximity
  to the unit circle, paired with nearest zeros) to reduce intermediate overflow/precision loss —
  especially relevant once float32 (Epic B) is in play.
- **Why (library quality).** High-order designs (16th-order elliptic, poles near the unit circle)
  can lose precision in a poorly-ordered cascade; ordering is the canonical DSP mitigation and
  matters more in float32.
- **Files.** [src/torchfx/filter/iir.py](src/torchfx/filter/iir.py) (coefficient computation),
  [src/torchfx/filter/fused.py](src/torchfx/filter/fused.py) (concatenation order),
  [src/torchfx/filter/_design.py](src/torchfx/filter/_design.py).
- **Difficulty.** medium · **Effort.** 1.5 d · **Depends on.** B3 (to measure the precision benefit).
- **Implementation steps.**
  1. Implement pole-zero pairing + ordering (mirror `scipy.signal.zpk2sos`'s `pairing='nearest'`
     semantics if not already inherited from the design path).
  2. Apply consistently when building `_sos` and when concatenating in `FusedSOSCascade`.
- **Verification & testability.**
  - [ ] B3 harness shows reduced float32 L∞ error for high-order designs after reordering.
  - [ ] Output still matches `scipy` (ordering changes intermediate values, not the final transfer
        function, within tolerance).
- **Done when.** Cascade ordering measurably improves float32 conditioning on high-order filters.

---

## Epic F — CPU & edge performance

*Theme: the CPU path already beats SciPy and (on consumer GPUs at high channel counts) beats the
GPU. This epic strengthens the CPU/edge story — most valuable on the Raspberry Pi 5 and for
many-channel realtime. Re-measure the CPU-vs-GPU crossover after Epic B, since FP32 shifts it.*

### F1 — SoA `[T,C]` layout + cross-channel SIMD

- **Goal.** Vectorize the one data-parallel axis (channels) by processing channels contiguously,
  enabling AVX2 (8-wide f32) / NEON (4-wide f32) FMA across channels.
- **Why (multichannel + edge).** The recurrence is serial in time and across sections but
  *independent across channels*. The current `[C,T]` row-major layout puts time contiguous, so the
  auto-vectorizer can't exploit channel parallelism — it falls back to OpenMP-over-channels only.
  Expected **1.6–1.8× on 8-channel float32** (transpose alone ~1.25×); larger at 32 channels — the
  ambisonic/edge case.
- **Files.** [src/torchfx/_csrc/cpu/iir_cpu.cpp](src/torchfx/_csrc/cpu/iir_cpu.cpp),
  possibly a layout shim in [src/torchfx/_ops.py](src/torchfx/_ops.py).
- **Difficulty.** medium · **Effort.** 3.5–4 d (2 d for the transpose alone) · **Depends on.** A4;
  re-measure after B5.
- **Implementation steps.**
  1. Add an internal channel-tiled inner loop: advance a vector of `W` channels' DF1 state per
     instruction, using explicit intrinsics or a vector-friendly SoA tile.
  2. Handle the transpose/gather cost; for low channel counts (1–2) keep the existing scalar path
     (the transpose can eat the win) — dispatch on `C`.
  3. Keep float64 correct (4-wide AVX2 / 2-wide NEON) as well.
- **Verification & testability.**
  - [ ] Output unchanged vs. current kernel (`assert_close`) for `C∈{1,2,8,32}`.
  - [ ] Benchmark: ≥1.5× on `60s/8ch` float32 CPU; no regression at `C∈{1,2}`.
  - [ ] Runs correctly on ARM (NEON) — at minimum cross-compiles; validate on Pi 5 when available.
- **Done when.** Multichannel CPU throughput improves measurably with bit-equivalent output.

### F2 — Runtime CPU feature dispatch (function multiversioning)

- **Goal.** Ship SIMD without `-march=native` breaking wheel portability, by compiling multiple
  versions and selecting at runtime (`__attribute__((target))` + `__builtin_cpu_supports`, à la
  NumPy/FFTW).
- **Why (deployment).** Lets F1's AVX2/AVX-512 code run on capable CPUs while a baseline build stays
  portable across the wheel matrix.
- **Files.** [src/torchfx/_csrc/cpu/iir_cpu.cpp](src/torchfx/_csrc/cpu/iir_cpu.cpp),
  [CMakeLists.txt](CMakeLists.txt).
- **Difficulty.** medium · **Effort.** 1.5 d · **Depends on.** F1.
- **Implementation steps.**
  1. Split the hot inner loop into target-specific clones (`default`, `avx2`, `avx512f`).
  2. Resolve at runtime; verify the dispatcher picks the right path under different `-mavx` envs.
- **Verification & testability.**
  - [ ] Same output on all code paths (force each via a test hook / env).
  - [ ] Wheels build cleanly across the existing CI matrix (no `-march=native` leakage).
- **Done when.** One wheel runs optimal SIMD per host without portability loss.

### F3 — Thread-local state buffer pool for `K > 16`

- **Goal.** Avoid the per-call `std::vector` heap allocation in `sos_forward_cpu_impl` when
  `K > STACK_MAX (16)`.
- **Why.** Minor, but removes a per-call allocation on deep cascades; cleaner and allocation-free on
  the hot path.
- **Files.** [src/torchfx/_csrc/cpu/iir_cpu.cpp](src/torchfx/_csrc/cpu/iir_cpu.cpp).
- **Difficulty.** trivial · **Effort.** 0.5 d · **Depends on.** —
- **Implementation steps.**
  1. Use a `thread_local` reusable buffer sized to the max `K` seen, instead of per-call `resize`.
- **Verification & testability.**
  - [ ] Output unchanged for `K∈{2,16,32,64}`.
  - [ ] No data race under OpenMP (thread-local is per-thread by construction); soak test passes.
- **Done when.** Deep cascades allocate no per-call heap memory.

---

## Epic G — Build, packaging & dispatch tuning — ✅ DONE

*Status: landed and GPU-verified. **G1** ships native SASS for `sm_75;80;86;89`
(verified via `cuobjdump --list-elf`), no first-call PTX JIT. **G2** made the
threshold the single source of truth, threaded into the kernels, with a `threshold=`
override (force-routing). The crossover sweep (`benchmarks/bench_threshold_sweep.py`,
RTX 3070) showed the crossover is **dtype-dependent** — the parallel scan is flat
~135 µs while the sequential kernel grows ~2× faster in FP64 — so the default is now
dtype-aware: **FP32 → 2048, FP64 → 1024** (a single 2048 left FP64 ~57% slower at
T≈2048). The crossover is essentially channel-independent (1–8 ch), so no
channel-axis tuning is needed.*

*Theme: cheap, high-certainty wins in how the extension is built and how it routes work.*

### G1 — Set explicit `CUDA_ARCHITECTURES`

- **Goal.** Compile SASS for the target compute capabilities (e.g. `75;80;86;89`) instead of relying
  on NVCC defaults / PTX-JIT.
- **Why (usage).** Without `-arch`, the first CUDA call may pay a 100–500 ms PTX-JIT stall and may
  run suboptimal SASS. Setting architectures removes the first-call freeze and improves occupancy.
- **Files.** [CMakeLists.txt](CMakeLists.txt).
- **Difficulty.** trivial · **Effort.** 0.25 d · **Depends on.** —
- **Implementation steps.**
  1. Set `CMAKE_CUDA_ARCHITECTURES` (cover the wheel targets + dev cards: RTX 3070 `86`, A40 `86`,
     L40S `89`, plus `75;80` for breadth). Make it overridable via env for niche targets.
- **Verification & testability.**
  - [ ] Build succeeds; `cuobjdump --list-elf` shows the expected SASS arches in the built `.so`.
  - [ ] First-call latency: time the first kernel invocation in a fresh process before/after; the
        JIT stall is gone.
- **Done when.** Shipped/dev builds contain native SASS for the supported cards with no first-call JIT.

### G2 — Unify, expose, and retune `PARALLEL_SCAN_THRESHOLD`

- **Goal.** Make the sequential-vs-parallel-scan threshold a single, configurable, per-device value
  and add a force-routing hook for characterization.
- **Why (library correctness + tuning).** The constant is duplicated in
  [_ops.py:34](src/torchfx/_ops.py#L34) (Python) and inside
  [parallel_scan.cu](src/torchfx/_csrc/cuda/parallel_scan.cu) (the `T<=2048` branch). They can
  disagree, and the current value spikes latency ~+30% near `T=2048` at `C=1` on every tested card;
  ~1024 is better there. After FP32 (Epic B) the crossover moves and must be re-measured.
- **Files.** [src/torchfx/_ops.py](src/torchfx/_ops.py),
  [src/torchfx/_csrc/cuda/parallel_scan.cu](src/torchfx/_csrc/cuda/parallel_scan.cu),
  [benchmarks/](benchmarks/).
- **Difficulty.** easy · **Effort.** 1–1.5 d · **Depends on.** B (re-measure after FP32).
- **Implementation steps.**
  1. Single source of truth for the threshold; pass it into the kernel rather than hardcoding `2048`
     in two places.
  2. Add a debug force-routing flag (`sequential` / `parallel`) in `parallel_iir_forward` to drive a
     `(T × C)` crossover sweep.
  3. Re-measure and set a sane default (and optionally a per-device map keyed on
     `torch.cuda.get_device_capability()`).
- **Verification & testability.**
  - [ ] Test: the Python threshold and kernel behaviour agree (no dead constant) — a routing test
        asserts which branch runs for representative `(T,C)`.
  - [ ] Crossover sweep JSON produced under `benchmarks/results/`; default eliminates the `+30%`
        spike at `C=1`.
  - [ ] No regression for `C≥4` (parallel branch should still win immediately there).
- **Done when.** One tunable threshold, justified by a checked-in sweep, with no Python/CUDA
  disagreement.

---

## Appendix A — Baseline snapshot (regression targets)

Captured from `benchmarks/results/` on **Alienware Aurora R11** (Intel i9-10900KF + RTX 3070 8 GB),
medians, signal length in seconds @ ~44.1–48 kHz. Use these as the *before* numbers; any perf task
must not regress them by more than ±3% on the same machine.

**CUDA IIR — Epic B FP32 path: MEASURED (RTX 3070, 8th-order Butterworth @ 48 kHz,
`benchmarks/bench_fp32_speedup.py`). ✅ Epic B complete.**

| Workload | GPU FP64 | GPU FP32 (Epic B) | Speedup |
|---|---|---|---|
| 30 s / 1 ch | 9.49 ms | 2.80 ms | 3.39× |
| 60 s / 1 ch | 18.31 ms | 6.00 ms | 3.05× |
| 60 s / 2 ch | 29.03 ms | 9.32 ms | 3.11× |
| 60 s / 4 ch | 49.22 ms | 14.41 ms | 3.42× |
| 60 s / 8 ch | 89.18 ms | **24.49 ms** | 3.64× |

Inversion resolved: at 60 s / 8 ch, GPU FP64 (89 ms) *lost* to CPU (FP32 28 ms / FP64 34 ms);
GPU FP32 (24 ms) now **beats** the CPU. The 3.0–3.6× win matches the predicted honest 2–4×
range (the scan is partly bandwidth/overhead-bound, not the theoretical 32× FLOP ratio).

**CPU (TorchFX fused vs. references) — keep these wins, don't regress:**

| Workload | TorchFX CPU | torchaudio | SciPy |
|---|---|---|---|
| single biquad (1 s / 1 ch) | 0.23 ms | 0.51 ms | — |
| 60 s / 8 ch IIR | 67.8 ms | — | 371.3 ms |

**Realtime (CPU path, shipped 0.6.0) — keep headroom:**

| Config | p99 / deadline |
|---|---|
| d=5, buffer 256 (5.333 ms deadline) | 0.0376 ms (0.71%) |
| d=10, buffer 256 | 0.0393 ms (0.74%) |
| xruns over 8192 callbacks | 0 |

**Kernel-launch fusion (current):** `K=50` cascade collapses Python dispatch 50→1, but CUDA
launches only 700→406 (1.72×). **Epic D1** target: launches → `O(1)` per forward.

---

## Cross-references

- Paper plan & narrative: [IS22026/PLAN.md](IS22026/PLAN.md) (updated to reference this roadmap).
- Style & filter-implementation conventions:
  [docs/source/guides/developer/style_guide.md](docs/source/guides/developer/style_guide.md).
- Architecture overview: [CLAUDE.md](CLAUDE.md).
