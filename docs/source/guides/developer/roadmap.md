# Roadmap to v1.0.0

**Current Version:** 0.6.0 (Beta Track)
**Target:** v1.0.0 Stable Release

This roadmap outlines the development path for TorchFX from the current beta state to a production-ready v1.0.0 release. The plan is organized into major epics, each containing specific deliverables and tasks.

## Vision

TorchFX v1.0.0 will be a production-ready, GPU-accelerated audio DSP library with:

- **Real-time processing** capabilities for live audio (microphone/instrument input)
- **Modern CLI tool** combining sox compatibility with GPU acceleration
- **Optimized performance** through custom CUDA kernels
- **Professional documentation** with comprehensive tutorials and API reference
- **>90% test coverage** with integration and audio quality tests
- **Semantic versioning** with backward compatibility guarantees

---

## Current State

### Strengths
- ✅ Solid core DSP architecture (~2000 LOC)
- ✅ GPU acceleration with custom CUDA kernels (parallel scan IIR, biquad, delay)
- ✅ **Build-time-compiled** C++/CUDA native extension (scikit-build-core), shipped as CPU + CUDA wheels with explicit `sm_75;80;86;89` SASS
- ✅ **FP32 CUDA execution path** — float32 runs natively, **3.0–3.6× faster** than the old FP64-only path on consumer GPUs (v0.6.0)
- ✅ **CUDA Graph capture** for fixed-shape streaming (`CudaGraphRunner`) — **up to 4× lower per-chunk latency** in the realtime regime (v0.6.0)
- ✅ Transparent IIR/biquad filter fusion via deferred pipeline, with static-gain folding
- ✅ 88% test coverage with coverage gate enforced in CI
- ✅ Published research paper (arXiv:2504.08624)
- ✅ Clean API with pipe operator support (`Wave | filter`, `filter | filter`)
- ✅ Professional Sphinx documentation with tutorials
- ✅ Real-time audio processing with a producer/consumer worker-thread split + latency/xrun instrumentation
- ✅ Full-featured CLI with sox compatibility
- ✅ Interactive REPL with live performance mode
- ✅ Complete validation and logging infrastructure
- ✅ API stability guarantees with deprecation system

### Gaps
- ❌ Limited ML integration examples
- ❌ Missing some advanced effects (compressor, phaser, pitch shift)
- ❌ No VST3 wrapper
- ❌ GPU realtime path (the live `RealtimeProcessor` is CPU-only; GPU is for offline/streaming)

**Estimated Completion:** ~92% ready for v1.0.0

---

## Epic 1: Core Library Stabilization

**Priority:** Critical (Foundation)
**Goal:** Complete essential features and stabilize the public API with semantic versioning guarantees.

### 1.1 Complete Missing Core Features

- [x] **Implement Wave.save() / to_file() method**
  - ✅ Support formats: WAV, FLAC (OGG/MP3/AAC require additional backend configuration)
  - ✅ High bit-depth: 32-bit float, 64-bit float (8, 16, 24, 32, 64 bits supported)
  - ✅ High sample rates: up to 192kHz+ (tested with 96kHz and 192kHz)
  - ✅ Metadata preservation (automatic extraction and storage via torchaudio.info)
  - Implementation details:
    - Uses torchaudio.save() as backend
    - Automatic parent directory creation
    - Format inference from file extension
    - CPU tensor conversion for compatibility
    - Comprehensive test suite (16 tests, 1 skipped for OGG)

- [x] **Complete LoShelving filter**
  - ✅ Implemented following HiShelving pattern
  - ✅ Uses Audio EQ Cookbook formulas
  - ✅ Supports both linear and dB gain scales
  - ✅ Full test coverage (7 tests)

- [x] **Add professional filters**
  - ✅ **Parametric EQ** (essential for music production)
    - Bell-shaped peaking filter with configurable Q and gain
    - Intuitive interface: frequency, Q, gain in dB
    - Perfect for surgical frequency adjustments
  - ✅ **Elliptic filters** (HiElliptic, LoElliptic)
    - Sharpest transition for given order
    - Configurable passband ripple and stopband attenuation
    - Optimal for applications where phase is not critical
  - [ ] State variable filters (TPT) - deferred to future version

### 1.2 API Stabilization

- [x] **Audit and freeze public API**
  - ✅ Marked all public classes in `__all__` exports
  - ✅ Created [api stability](../api_stability.md) with backward compatibility guarantees
  - ✅ Implemented deprecation warning system with decorators (`@deprecated`, `@deprecated_parameter`, `DeprecatedAlias`)
  - ✅ Full test coverage (9 tests) for deprecation utilities

- [x] **Implement semantic versioning policy**
  - ✅ Documented policy: No breaking changes in minor versions (1.x.x)
  - ✅ Deprecation warnings for at least one minor version before removal
  - ✅ Created [migration guide](../migration_guide.md) template with migration patterns
  - ✅ Added versioning examples and guidelines

- [x] **Parameter naming consistency**
  - ✅ Standardized naming conventions documented in [style guide](./style_guide.md):
    - `cutoff` for lowpass/highpass/shelving filters
    - `frequency` for ParametricEQ (center frequency)
    - `Q` (uppercase) for Peaking, Notch, AllPass (mathematical convention)
    - `q` (lowercase) for Shelving, ParametricEQ (industry convention)
    - `gain` with `gain_scale` for units ("linear" or "db")
    - `fs` for sampling frequency
  - ✅ Style guide includes naming, units, code organization, and documentation standards

### 1.3 Error Handling & Validation

- [x] **Input validation layer**
  - ✅ Validate sample rates, tensor shapes, parameter ranges
  - ✅ Custom exception hierarchy: `TorchFXError`, `InvalidParameterError`, `AudioProcessingError`
  - Implementation details:
    - New `torchfx.validation` subpackage with exceptions and validators
    - Exception hierarchy: `TorchFXError` (base), `InvalidParameterError`, `InvalidSampleRateError`, `InvalidRangeError`, `InvalidShapeError`, `InvalidTypeError`, `AudioProcessingError`, `CoefficientComputationError`, `FilterInstabilityError`
    - Validators: `validate_sample_rate`, `validate_positive`, `validate_range`, `validate_in_set`, `validate_tensor_ndim`, `validate_audio_tensor`, `validate_type`, `validate_cutoff_frequency`, `validate_filter_order`, `validate_q_factor`
    - Full test coverage (76 tests)

- [x] **Improved error messages**
  - ✅ Context-aware messages with actual vs. expected values
  - ✅ Suggestions for fixes
  - Implementation details:
    - Built into the validation exception classes (parameter_name, actual_value, expected, suggestion fields)
    - All exceptions format messages with full context automatically

- [x] **Logging infrastructure**
  - ✅ Structured logging with Python's `logging` module
  - ✅ Log levels: DEBUG, INFO, WARNING, ERROR
  - ✅ Performance logging (optional)
  - Implementation details:
    - New `torchfx.logging` subpackage
    - NullHandler by default (opt-in logging per Python guidelines)
    - Convenience functions: `enable_logging()`, `enable_debug_logging()`, `disable_logging()`, `get_logger()`
    - Performance utilities: `log_performance()` context manager, `LogPerformance` decorator
    - Hierarchical loggers: `torchfx`, `torchfx.performance`, `torchfx.<module>`
    - Full test coverage (25 tests)

---

## Epic 2: Real-Time Audio Processing

**Priority:** Critical (Major Feature)
**Goal:** Enable low-latency live audio processing with GPU acceleration.

### 2.1 Audio Backend Integration

- [x] **Abstract audio backend interface**
  - ✅ `AudioBackend` ABC with lifecycle methods (open, start, stop, close)
  - ✅ Support input, output, duplex streams via `StreamConfig` and `StreamDirection`
  - ✅ Callback-based and blocking APIs
  - Implementation details:
    - New `torchfx.realtime` subpackage
    - `StreamConfig` frozen dataclass with direction inference and latency calculation
    - `AudioCallback` type alias for `Callable[[Tensor, Tensor, int], None]`
    - Full test coverage (62 tests)

- [x] **PortAudio backend** (Priority 1)
  - ✅ `SoundDeviceBackend` using `sounddevice` library
  - ✅ Cross-platform support (Linux, macOS, Windows)
  - ✅ Buffer size: configurable (64-4096+ samples)
  - ✅ Optional dependency — core library works without sounddevice
  - Implementation details:
    - Lazy import via `_compat.py` module
    - Numpy-to-tensor zero-copy conversion in callback wrapper
    - Device enumeration and default device selection
    - `sounddevice` in optional dependency group `realtime`

- [ ] **PulseAudio/PipeWire backend** (Priority 2) - deferred to future version
  - Native Linux desktop integration

- [ ] **JACK backend** (Future) - deferred to future version
  - Professional Linux audio routing

### 2.2 Real-Time Processing Pipeline

- [x] **Ring buffer implementation**
  - ✅ Lock-free SPSC `TensorRingBuffer` on PyTorch tensors
  - ✅ GPU-compatible tensor buffers (configurable device)
  - ✅ Overlap-add support via `peek()` + `advance_read()`
  - Implementation details:
    - Power-of-2 capacity with bitwise modular arithmetic
    - Pre-allocated `(channels, capacity)` backing tensor
    - Separate read/write indices (SPSC model)
    - Wrap-around handling with split copy operations

- [x] **Real-time processor class**
  - ✅ `RealtimeProcessor` orchestrating backend + effect chain
  - ✅ `start()`, `stop()`, `set_parameter(name, value)` (thread-safe)
  - ✅ Automatic `fs` propagation and coefficient computation
  - ✅ `reset_state()` for clearing filter states
  - Implementation details:
    - Double-buffered parameter updates (lock only on swap)
    - Audio callback processes effects in sequence
    - Mono-to-stereo expansion for channel mismatch
    - Ring buffers for input/output queuing

- [x] **Latency optimization**
  - ✅ Target: <10ms total latency at 48kHz, 512 buffer (~10.7ms theoretical)
  - ✅ Pre-allocated tensor buffers via ring buffer
  - ✅ Lock-free audio path (parameters applied at buffer boundaries)

- [x] **Stream processing for large files**
  - ✅ `StreamProcessor` with chunk-based processing
  - ✅ `process_file()` for file-to-file processing
  - ✅ `process_chunks()` generator API for streaming pipelines
  - ✅ GPU acceleration support (device parameter)
  - Implementation details:
    - Uses `torchaudio.load(frame_offset, num_frames)` for efficient chunk reading
    - Uses `soundfile.SoundFile` for append-mode writing
    - Configurable overlap for overlap-add processing

### 2.3 Real-Time Effect Adaptations

- [x] **Stateful filter management**
  - ✅ `reset_state()` method on RealtimeProcessor
  - ✅ Ring buffer clear on state reset

- [x] **Thread-safe parameter updates**
  - ✅ Double-buffered parameter dict with lock-on-swap
  - ✅ Parameters applied atomically at buffer boundaries
  - ✅ Automatic coefficient recomputation for filter parameters

- [x] **CPU/GPU hybrid processing**
  - ✅ StreamProcessor supports configurable device ("cpu" or "cuda")
  - ✅ Automatic CPU↔GPU tensor transfers in stream processing
  - ✅ Real-time processor operates on CPU for low-latency callback

---

## Epic 3: CLI Application

**Priority:** High (Major Feature) ✅ **COMPLETED**
**Goal:** Modern, GPU-accelerated CLI tool with sox compatibility and unique features.

### 3.1 Core CLI Architecture ✅

- [x] **CLI framework with Typer**
  - ✅ Commands: `process`, `info`, `play`, `record`, `convert`, `trim`, `concat`, `stats`, `preset`, `interactive`, `watch`
  - ✅ Global options: `--device`, `--verbose`, `--config`, `--version`
  - ✅ Rich output with progress bars and tables
  - Implementation details:
    - Typer with Rich markup mode
    - Global state management via callback
    - Lazy imports for heavy dependencies (torch, sounddevice)
    - 71 CLI tests (356 total tests)

- [x] **Subcommand structure**
  ```bash
  torchfx process input.wav output.wav --effect reverb
  torchfx info audio.flac
  torchfx play audio.wav
  torchfx record output.wav --duration 10
  torchfx interactive  # REPL mode
  torchfx watch ./input/ ./output/ --effect normalize
  ```

- [x] **Configuration file support (TOML)**
  - ✅ Save/load effect chains from TOML files
  - ✅ Preset management: `~/.config/torchfx/presets/`
  - ✅ TOML-only (tomllib stdlib on 3.11+, tomli fallback for 3.10)
  - ✅ `[[effects]]` format compatible with presets

### 3.2 Pipeline Processing & Sox Compatibility ✅

- [x] **Unix pipe support**
  - ✅ Read from stdin: `cat audio.wav | torchfx process - - -e normalize`
  - ✅ Write to stdout: `torchfx process input.wav - -e normalize | aplay`
  - ✅ WAV and raw format support for pipes

- [x] **Batch processing**
  ```bash
  torchfx process "*.wav" --output-dir ./processed/ --effect normalize
  ```
  - ✅ Glob pattern matching
  - ✅ Rich progress bar with ETA
  - ✅ Error handling per file
  - ✅ GPU acceleration support

- [x] **Sox-compatible commands**
  - ✅ `convert` — format/rate/channel conversion
  - ✅ `trim` — extract time ranges
  - ✅ `concat` — join multiple files
  - ✅ `stats` — signal statistics (peak/RMS dBFS, crest factor, DC offset)

- [x] **GPU-accelerated batch processing**
  - ✅ Auto-batch via `--device cuda`
  - ✅ StreamProcessor with chunked processing
  - ✅ Progress bar with completion ETA

### 3.3 Interactive Mode (REPL) ✅

- [x] **Interactive shell**
  - ✅ prompt_toolkit with tab completion
  - ✅ Syntax highlighting and Rich formatting
  - ✅ Persistent command history (`~/.config/torchfx/repl_history`)
  - ✅ Effect name and command completion

- [x] **Live parameter tweaking**
  ```python
  torchfx> load song.wav
  torchfx> add reverb:decay=0.5
  torchfx> live
  ▶ Live playback started (2 ch, 44100 Hz, looping)
  torchfx> add normalize
  # ← Effect applies immediately during playback!
  torchfx> preset load mastering
  # ← Entire chain switches in real-time
  torchfx> live stop
  ```
  - ✅ Lock-free circular buffer pattern
  - ✅ Real-time effect hot-swapping
  - ✅ Audio loops continuously
  - ✅ Changes apply at buffer boundaries (~46ms latency)

- [x] **Commands**: `load`, `add`, `remove`, `list`, `effects`, `info`, `play`, `play raw`, `live`, `live stop`, `save`, `preset save/load/list`, `clear`, `help`, `exit`

- [ ] **Real-time visualization** — deferred to v1.1
  - Waveform display
  - Spectrum analyzer
  - VU meters

- [x] **Preset management**
  - ✅ Save/load/list/show/delete/apply presets
  - ✅ TOML format compatible with `--config`
  - ✅ Works in both CLI and REPL

### 3.4 Watch Mode & Automation ✅

- [x] **File system watcher**
  ```bash
  torchfx watch ./input/ --output ./processed/ --effect reverb
  torchfx watch ./bounces/ --preset mastering --recursive
  ```
  - ✅ Watchdog-based file monitoring
  - ✅ Auto-process new/modified audio files
  - ✅ Recursive directory watching
  - ✅ Process existing files on startup (--existing flag)
  - ✅ Preset and config file support

- [x] **DAW integration mode**
  - ✅ Monitor export folder
  - ✅ Auto-apply mastering chain from preset
  - ✅ Rich status display with live updates

---

## Epic 4: Performance Optimization & CUDA

**Priority:** Medium — **LARGELY COMPLETED** (v0.5.0–v0.5.2)
**Goal:** Maximize throughput with custom CUDA kernels.

### 4.1 CUDA Development Infrastructure ✅

- [x] **CUDA extension build system**
  - ✅ JIT-compiled C++/CUDA extension via `torch.utils.cpp_extension.load()`
  - ✅ Auto CUDA arch detection
  - ✅ Automatic fallback to pure-PyTorch if compilation fails
  - ✅ CPU-only C++ extension support (no CUDA toolkit required)
  - ✅ `TORCHFX_NO_CUDA=1` environment variable to force CPU-only
  - ✅ Cached in `~/.cache/torch_extensions/`

- [x] **Kernel development tools**
  - ✅ CUDA kernel unit tests and fallback behavior tests
  - ✅ pytest-benchmark harness with 5-backend comparison
  - ✅ SLURM harness for cluster GPU benchmarks (`benchmarks/slurm/`)
  - ✅ CPU + CUDA profile scenarios (`benchmarks/profiles/`)

### 4.2 IIR Filter CUDA Kernels ✅

- [x] **Parallel IIR implementation**
  - ✅ Blelloch parallel prefix scan — O(N) total work
  - ✅ `PARALLEL_SCAN_THRESHOLD` is now **dtype-aware and tunable** (float32 → 2048, float64 → 1024, threaded into the kernel; overridable per call) — a measured crossover sweep showed FP64 hits the parallel-scan overhead sooner
  - ✅ 4x faster than SciPy (single-channel), 11x faster (8-channel)

- [x] **FP32 CUDA execution path** (v0.6.0)
  - ✅ Kernels templated on `scalar_t` and dispatched on the input dtype: `float32` in → FP32 path, `float64` in → FP64. No more silent upcast.
  - ✅ **3.0–3.6× faster** for IIR cascades on an RTX 3070; resolves the multichannel inversion (60 s / 8-ch GPU now beats its own CPU)
  - ✅ FP32/FP64 precision validation harness (`tests/test_fp32_precision.py`) against `scipy.signal.sosfilt`

- [x] **Biquad cascade optimization**
  - ✅ Specialized biquad CUDA kernel — 128 channels batched per thread block
  - ✅ Scalar coefficient passing to eliminate GPU→CPU sync
  - ✅ Retained as fast path for K=1 in unified `_sos_cascade_forward`

- [x] **Stability guarantees**
  - ✅ SOS coefficients (v0.5.1) for numerical precision at high filter orders
  - ✅ `torch.testing.assert_close` validation against SciPy reference

### 4.3 Time-Domain Effects CUDA Kernels

- [x] **Optimized delay line**
  - ✅ CUDA delay forward kernel

- [x] **Reverb optimization** (#18) — `Reverb` reworked into a Freeverb-style network: 8
  parallel low-pass-feedback comb filters + 4 series all-pass diffusers per channel, in a
  fused native per-channel C++/CUDA kernel (ring buffers in per-channel scratch). Replaces
  the single feedforward comb; breaking API (`room_size`/`damping`/`mix`/`fs`).

### 4.4 Batch Processing Optimizations

- [x] **Operator fusion**
  - ✅ Deferred pipeline with auto-fusion — consecutive IIR/biquad filters merged
    into single `FusedSOSCascade` kernel call (~2.5x faster for IIR chains)
  - ✅ `FilterChain` and `FX.__or__` for composable filter chains
  - ✅ **Static-gain folding** (v0.6.0) — a constant linear `Gain` between filters is folded
    into the cascade numerator, so `IIR | Gain | IIR` stays one fused stage
  - [ ] Fuse a *dynamic* `Normalize` (per-chunk peak) — only the static, resolvable case is folded

- [x] **CUDA Graph capture for streaming** (v0.6.0)
  - ✅ `torchfx.realtime.CudaGraphRunner` captures a fixed-shape fused-cascade forward and
    replays it per chunk, collapsing the `~4·K` per-section launches into one
  - ✅ **Up to 4× lower per-chunk latency** at small (realtime) chunk sizes on an RTX 3070

- [x] **Memory optimization**
  - ✅ SOS coefficient device caching (eliminates per-forward `.to()`)
  - ✅ In-place state updates (`copy_()` instead of `torch.stack()`)
  - ✅ **Allocation-free GPU SOS forward** (v0.6.0) — per-section scratch reused once per
    forward (O(K) → O(1) allocations), cutting the eager GPU streaming path **~27%**
  - ✅ Reverb op fusion (5 tensor ops → 2)
  - ✅ Delay wet/dry mix via `torch.lerp` (3 ops → 1)

- [x] **Multi-file batch processing** (#19) — `torchfx.batch_process(waves, effect)` pads
  signals to a common length, concatenates them on the channel dimension, and runs the
  effect in a **single** native dispatch over all channels (causal per-channel kernels +
  trailing zero-pad ⇒ numerically identical to per-file). Measured **~2.5–7× on CPU and
  ~3–4× on GPU** for 8–512 stereo files (8th-order Butterworth). See
  `benchmarks/bench_batch.py`.

### 4.5 Performance Benchmarking ✅

- [x] **Comprehensive benchmark suite**
  - ✅ pytest-benchmark suite under `benchmarks/`
  - ✅ 5 backends: TorchFX GPU, TorchFX CPU, SciPy, Numba `@njit`, Numba `@cuda.jit`
  - ✅ IIR, biquad, pipeline, FIR, FFT convolution benchmarks
  - ✅ Signal durations 1–60s, 1–8 channels

- [x] **Performance baseline**
  - ✅ Phase 0 baseline documented (`docs/source/perf/baseline.md`)
  - ✅ CPU `torch.profiler` findings captured
  - ✅ Coverage gate `fail_under = 87` enforced in CI

- [x] **Performance regression testing** (#21) — `tests/test_perf_regression.py` runs in
  ordinary CI and gates on *deterministic* invariants (fusion collapses a depth-K cascade
  to one native dispatch; a static `Gain` folds without adding one) plus a same-machine
  relative wall-time smoke. Avoids flaky absolute-timing baselines on shared runners; see
  the developer testing guide. (Absolute wall-time budgets would need a dedicated runner.)

- [x] **Profiling guides** (#22) — developer guide on diagnosing a pipeline: correct
  warm-up/CUDA-sync timing, `torch.profiler` timelines, counting native dispatches to spot
  un-fused chains, and a bottleneck→fix table. See the developer profiling guide.

### 4.6 Advanced Kernel Optimizations (Future)

The 0.6.0 GPU work (FP32 path, CUDA graphs, allocation-free streaming, dispatch tuning)
landed the high-leverage wins. The remaining kernel-level optimizations — investigated,
prioritized, and (where attempted) measured during the 0.6.0 cycle — are tracked here.
Each item lists its difficulty and expected impact; none block a release.

- [x] **Single-pass scan (kernel-level fusion)** — *done, default-on (0.7.0).* Each cascade
  section's forcing pass + 3-phase Blelloch scan are folded into one **decoupled-look-back**
  kernel (Merrill–Garland / CUB style, atomic per-section tile dispenser for deadlock-free
  forward progress), and the per-section DF1 state update into one more — so a fused section
  is **2 CUDA launches** instead of ~8, with the phase-3 recompute eliminated. Measured on an
  RTX 3070 vs the 3-phase path: **5.9× fewer launches** (600 → 102 at K=50) and **~1.9×
  faster** cascades; fused launches now sit *below* the unfused baseline (102 vs 400). This
  is the item that makes fusion **kernel-level**, not just Python-dispatch-level. Now the
  **default**; set `TORCHFX_FUSED_SCAN=0` to force the legacy 3-phase oracle. Validated on
  sm_86 (RTX 3070, 46 SMs) and sm_89 (L40S, 142 SMs) — forward progress holds across the SM
  range. Source: `fused_scan_kernel` / `state_update_kernel` in
  [parallel_scan.cu](src/torchfx/_csrc/cuda/parallel_scan.cu).
- [ ] **Full all-sections mega-kernel** — *hard, future.* The single-pass scan above makes
  each section one kernel but the K sections are still K separate launches (cross-section
  dependency = a grid-wide barrier). A single kernel over all K sections (sections serial
  within a block, cooperative-groups grid sync) would take the launch count to `O(1)`. Lower
  marginal value now that per-section is already 2 launches; revisit if launch overhead
  resurfaces at very high K / tiny chunks.
- [ ] **Pinned host buffers + async H2D/D2H for GPU streaming** — *medium.* Overlap chunk
  transfers with compute. Only pays off once a **GPU realtime/streaming I/O path** exists
  (the live path is CPU-only today), so deferred until that lands.
- [x] **Cache-blocked cross-channel CPU SIMD** — *done (#24).* A naive full-transpose
  SIMD kernel (F1) regressed 4–8× on a 10-core desktop — the whole-`[C,T]` transpose
  moved as much memory as the streaming-light recurrence computes. The fix: transpose
  only small `[W=4, B=256]` tiles that stay in **L1**, auto-vectorise the per-time-step
  channel loop (NEON/SSE/AVX), and **gate the path to `C > num_threads`** so the
  scalar kernel stays for `C ≤ cores`. Validated on a **Raspberry Pi 5** (Cortex-A76
  ×4): **1.8× at C=8, 2.6× at C=16/32**; and on **x86 (12 cores)**: 1.3–2.4× for
  `C > cores`, **no regression** below. Correct vs scalar + `scipy.signal.sosfilt`
  (full suite green with `TORCHFX_FORCE_SIMD=1`). `TORCHFX_NO_SIMD=1` forces scalar.
  Source: `sos_forward_cpu_simd_impl` in
  [iir_cpu.cpp](src/torchfx/_csrc/cpu/iir_cpu.cpp).
- [ ] **CPU runtime feature dispatch (function multiversioning)** — *small.* Ship AVX2/AVX-512
  CPU paths without `-march=native` breaking wheel portability (paired with the SIMD work above).

---

## Epic 5: Comprehensive Documentation

**Priority:** Critical (Continuous)
**Goal:** Professional-grade documentation for v1.0.0 release.

### 5.1 API Reference Completion

- [ ] **Complete all docstrings**
  - Every public class, method, function
  - Parameters with types and ranges
  - Examples in docstrings
  - Mathematical formulas in LaTeX

- [ ] **Fix API documentation bugs**
  - Remove non-existent method references
  - Update all code examples
  - Validate examples run

- [ ] **Auto-generated API reference**
  - Sphinx autodoc with Napoleon
  - Type hints rendered
  - Cross-references

### 5.2 Tutorial & Guide Documentation

- [x] **Getting Started Tutorial** (expanded)
  - ✅ Installation
  - ✅ First pipeline
  - ✅ Wave class basics
  - ✅ Saving output

- [x] **CLI Guide** ✅ **NEW**
  - ✅ Complete CLI tutorial covering all commands
  - ✅ Effect specifications format
  - ✅ TOML configuration examples
  - ✅ Preset management workflows
  - ✅ Interactive REPL with live performance mode
  - ✅ Watch mode for DAW integration
  - ✅ Unix pipe examples

- [ ] **Advanced Tutorials**
  - Real-time audio processing (partially covered in CLI guide)
  - Custom filter design
  - GPU optimization
  - PyTorch model integration

- [ ] **How-To Guides**
  - Audio format conversion
  - Building EQ/filter bank
  - Mastering chain
  - Multi-channel processing
  - Guitar pedal simulator
  - ML model integration

### 5.3 Example Gallery

- [ ] **Expand examples**
  - Vocal processing chain
  - Mastering pipeline
  - Guitar effect pedal
  - Podcast cleanup
  - Music production effects
  - Real-time effects
  - ML model integration

- [ ] **Interactive examples**
  - Jupyter notebooks with audio playback
  - Parameter sliders

### 5.4 Project Documentation

- [ ] **Contributing Guide**
  - Code style and standards
  - Git workflow
  - Testing requirements

- [ ] **Architecture Documentation**
  - High-level overview
  - Design patterns
  - Extension points

- [ ] **Migration Guides**
  - Upgrading from 0.x to 1.0
  - API changes

- [ ] **FAQ & Troubleshooting**
  - Common errors
  - Performance issues
  - CUDA/GPU troubleshooting

---

## Epic 6: Testing & Quality Assurance

**Priority:** Critical (Parallel with Epic 1)
**Goal:** Achieve >90% test coverage with comprehensive testing.

### 6.1 Expand Unit Test Coverage

- [x] **Complete Wave class tests**
  - ✅ File I/O for all formats
  - ✅ Multi-channel audio
  - ✅ Sample rate conversion
  - ✅ Device transfers
  - ✅ Edge cases
  - ✅ 72 Wave tests

- [x] **Complete filter tests**
  - ✅ All filter types (IIR, FIR, Biquad)
  - ✅ Frequency/phase response validation
  - ✅ Filter composition
  - ✅ Edge cases
  - ✅ 85+ filter tests

- [x] **Complete effect tests**
  - ✅ All effects and parameters
  - ✅ Error handling
  - ✅ 43 effect tests

- [x] **CLI tests** ✅ **NEW**
  - ✅ All CLI commands (process, info, play, record, convert, trim, concat, stats)
  - ✅ Preset management (save, load, list, show, delete, apply)
  - ✅ REPL commands (add, remove, list, clear, load, save)
  - ✅ Watch mode (file monitoring)
  - ✅ 71 CLI tests
  - ✅ Total: **400+ tests** with 88% coverage

### 6.2 Integration Tests

- [x] **Complex pipeline tests**
  - ✅ Multi-stage effect chains
  - ✅ GPU end-to-end processing
  - ✅ File load → process → save

- [x] **Real-time processing tests**
  - ✅ Mock audio backend
  - ✅ Latency measurements
  - ✅ Parameter updates during processing
  - ✅ 62 realtime tests

- [x] **CLI integration tests**
  - ✅ All CLI commands
  - ✅ Pipe I/O
  - ✅ Batch processing
  - ✅ Config file loading
  - ✅ Preset workflows

### 6.3 Audio Quality Tests

- [ ] **Audio quality metrics**
  - SNR, THD, frequency response error
  - Compare against scipy/reference

- [ ] **Regression tests**
  - Golden output files
  - Detect quality degradation

- [ ] **Perceptual quality tests** (optional)
  - PESQ, PEAQ

### 6.4 Performance & Memory Tests

- [ ] **Memory leak detection**
  - Long-running tests
  - GPU memory monitoring

- [ ] **Performance benchmarks as tests**
  - Minimum speed requirements
  - Prevent regressions

### 6.5 CI/CD Improvements

- [x] **Coverage reporting**
  - ✅ HTML coverage CI job on Python 3.12
  - ✅ `fail_under = 87` coverage gate enforced
  - ✅ Codecov integration (#26) — CI uploads `coverage.xml` via `codecov-action`, README
    badge, and a `codecov.yml` with project/patch status targets (ignoring the
    CPU-unreachable sounddevice backend)
  - [ ] Coverage badge

- [ ] **Multi-platform testing**
  - Linux, macOS, Windows
  - Python 3.10-3.13
  - With/without CUDA

- [ ] **GPU CI runner**
  - Self-hosted or cloud GPU
  - CUDA tests and benchmarks

- [x] **Automated releases** (#29) — `release.yml` already auto-tags on a `pyproject.toml`
  version bump and the wheel workflows publish to PyPI / the CUDA index on the tag. Now it
  also publishes a **GitHub Release** whose notes are the `## [<version>]` CHANGELOG
  section (`tools/extract_changelog.py`). See the developer releasing guide.

---

## Epic 7: Additional Effects

**Priority:** Low (Can be v1.1+)
**Goal:** Expand effect library for common production needs.

### 7.1 Dynamics Processing

- [x] Compressor (threshold, ratio, attack, release, knee) — `Compressor` (#30), native
  per-channel C++/CUDA detector, peak/RMS, soft knee, `ratio=inf` limiter mode.
- [x] Limiter (brickwall, look-ahead) — `Limiter` (#31): look-ahead windowed peak
  (vectorised max-pool) + attack/release gain smoothing + a per-sample clamp that
  guarantees `|y| <= threshold`; native CPU + CUDA. (True-peak / oversampled detection
  is a possible follow-up.)
- [x] Expander / Gate — `Expander` (downward expansion below threshold) + `Gate`
  (infinite-ratio convenience) (#32), mirroring the compressor detector with an
  optional `floor` (range); native CPU + CUDA.

### 7.2 Modulation Effects

- [ ] Chorus (multi-tap delay with LFO)
- [ ] Flanger (short delay with feedback)
- [ ] Phaser (all-pass cascade with LFO)
- [ ] Tremolo / Vibrato

### 7.3 Distortion & Saturation

- [ ] Overdrive / Distortion (soft/hard clipping)
- [ ] Waveshaping (custom transfer functions)
- [ ] Bitcrusher (bit depth/sample rate reduction)

### 7.4 Pitch & Time Manipulation

- [ ] Pitch Shifting (phase vocoder)
- [ ] Time Stretching (tempo change)
- [ ] Formant Shifting

### 7.5 Spatial Audio

- [ ] Stereo Widening (mid-side, Haas effect)
- [ ] Panning (constant power, 3D)
- [ ] Binaural Audio (HRTF)

---

## Implementation Phases

### Phase 1: Foundation ✅ **COMPLETED**
**Priority:** Critical

- **Epic 1: Core Library Stabilization** ✅
  - ✅ Complete missing features
  - ✅ API stabilization
  - ✅ Error handling
- **Epic 6: Testing Infrastructure** ✅
  - ✅ Expand unit tests (393 tests, >90% coverage)
  - ✅ CI improvements

### Phase 2: Major Features ✅ **COMPLETED**
**Priority:** Critical

- **Epic 2: Real-Time Audio Processing** ✅
  - ✅ Audio backends (SoundDevice)
  - ✅ Real-time pipeline with circular buffers
  - ✅ Thread-safe parameter updates
  - ✅ Stream processor for large files
- **Epic 3: CLI Application** ✅
  - ✅ Core CLI with 11 commands
  - ✅ Pipeline processing (batch, pipes, watch)
  - ✅ Interactive mode with live performance
  - ✅ Preset management
  - ✅ Sox-compatible commands
- **Epic 5: Documentation** ✅
  - ✅ Complete API reference
  - ✅ CLI guide
  - ✅ Tutorials and examples
  - ✅ Migration guide and API stability docs

### Phase 3: CUDA Kernels & Native Extension (v0.5.0) ✅ **COMPLETED**
**Priority:** Medium

- **Epic 4: CUDA Kernels** ✅
  - ✅ JIT-compiled C++/CUDA native extension with automatic fallback
  - ✅ Blelloch parallel prefix scan for IIR filters (O(N) total work)
  - ✅ Specialized biquad CUDA kernel (128-channel batching)
  - ✅ CUDA delay forward kernel
  - ✅ CPU-only C++ extension (~2400x faster than pure-Python for stateful IIR)
  - ✅ FFT-based FIR convolution (up to 10x faster for kernel sizes ≥ 64)
  - ✅ LogFilterBank for logarithmic frequency band decomposition
  - ✅ pytest-benchmark suite with 5-backend comparison
  - ✅ SLURM harness for cluster GPU benchmarks

### Phase 4: Numerical Stability & SOS Migration (v0.5.1) ✅ **COMPLETED**
**Priority:** High

- ✅ IIR filters migrated to SOS-only coefficients (no more `ba` intermediate)
- ✅ Fixed `BadCoefficients` scipy warning on high-order filters
- ✅ Fixed `LinkwitzRiley` order parameter bug
- ✅ Removed dead code: `_compute_ba_from_sos()`, `move_coeff()`, `_bootstrap_state()`, `a`/`b` attributes
- ✅ `LinkwitzRiley` cascades via `np.vstack` SOS sections instead of `ba` polynomial convolution

### Phase 5: Transparent Filter Fusion & Code Unification (v0.5.2) ✅ **COMPLETED**
**Priority:** High

- **Deferred pipeline with auto-fusion** ✅
  - ✅ `Wave.__or__` accumulates filters in lazy pipeline, materializes on `.ys` access
  - ✅ Consecutive IIR/biquad filters auto-fused into single `FusedSOSCascade` (~2.5x faster)
  - ✅ All three syntaxes benefit: `wave | f1 | f2`, `wave | (f1 | f2)`, `wave | nn.Sequential(...)`
  - ✅ Non-fusible effects break chain naturally, independent runs fused separately
- **`FilterChain` and pipe operator** ✅
  - ✅ `FX.__or__` enables `f1 | f2 → FilterChain` between any filters/effects
  - ✅ Auto-flattening `nn.Sequential` subclass — no nested containers
  - ✅ Exported from top-level `torchfx` package
- **Unified Biquad/IIR forward path** ✅
  - ✅ Biquad stores coefficients as `[1, 6]` SOS tensor
  - ✅ Delegates to shared `_sos_cascade_forward` helper (~150 lines of duplication removed)
  - ✅ Mixed Biquad+IIR chains auto-fuse in deferred pipeline
  - ✅ Specialized CUDA biquad kernel retained as fast path for K=1
  - ✅ Backward-compatible read-only `b`/`a` properties
- **Performance caching** ✅
  - ✅ Device-matched SOS tensor cached between forward calls
  - ✅ In-place state updates (`copy_()` instead of `torch.stack()`)
  - ✅ Reverb op fusion (5 ops → 2), delay wet/dry via `torch.lerp` (3 ops → 1)
  - ✅ Biquad feedback coefficients pre-extracted as Python floats
- **Test coverage** ✅
  - ✅ 74% → 88% coverage, `fail_under = 87` gate in CI
  - ✅ 7 new test files covering fusion, dispatch, filter base, filterbank, utilities

### Phase 6: Build-Time Native Extension Compilation (v0.5.3) ✅ **COMPLETED**
**Priority:** High

The native extension is now **compiled at build time** with
[scikit-build-core](https://scikit-build-core.readthedocs.io/) + CMake, not JIT-compiled
at first import. Shipped:

- [x] **scikit-build-core migration** — CMake-based build (`CMakeLists.txt`) compiles
  `iir_cpu.cpp` / `delay_cpu.cpp` at `pip install` / `uv sync` time. The extension is
  **required** (no pure-PyTorch fallback); `TORCHFX_NO_CUDA=1` forces a CPU-only build.
- [x] **CUDA kernel packaging** — `biquad_forward.cu` / `parallel_scan.cu` /
  `delay_forward.cu` built at wheel time with explicit `CUDA_ARCHITECTURES`
  (`sm_75;80;86;89` by default, override via `TORCHFX_CUDA_ARCHITECTURES`) so wheels ship
  native SASS with no first-call PTX→SASS JIT.
- [x] **CI/CD wheel pipeline** — cibuildwheel across Linux x86_64 / macOS / Windows,
  Python 3.10–3.14; CPU wheels on PyPI and CUDA wheels (cu124/cu128) on a PEP-503 index.
- [x] **`is_native_available()`** unchanged; `TORCHFX_NO_CUDA=1` still forces CPU-only.

### Phase 8: GPU Performance & Streaming (v0.6.0) ✅ **COMPLETED**
**Priority:** High

The 0.6.0 cycle returned to the GPU side (the follow-up promised in 0.5.4) and the
realtime path:

- [x] **FP32 CUDA path** — templated `scalar_t` kernels + dtype-aware dispatch; **3.0–3.6×**
  for IIR cascades on an RTX 3070, resolving the multichannel GPU-vs-CPU inversion.
- [x] **CUDA Graph capture** (`CudaGraphRunner`) — **up to 4×** lower per-chunk latency for
  fixed-shape streaming; root-caused to launching kernels on the current CUDA stream.
- [x] **Allocation-free GPU SOS forward** — O(K) → O(1) per-forward allocations (eager
  streaming **~27%** faster).
- [x] **Dtype-aware, tunable dispatch threshold** + explicit CUDA architectures.
- [x] **Static-gain folding** in the fusion planner.
- [x] **Realtime producer/consumer split** + latency/xrun instrumentation; **correctness
  hardening** (stale-coefficient recompute on `fs` change, offline state-reset semantics,
  half-precision rejection).
- See {doc}`/blog/index` for the detailed release and feature posts, and Epic 4.6
  for the remaining (deferred) kernel optimizations.

### Phase 7: Polish & v1.0 Release
**Priority:** Medium

- **Epic 5: Documentation** (remaining items)
  - Advanced tutorials (custom filter design, GPU optimization, ML integration)
  - How-to guides and example gallery
- **Epic 6: Quality Assurance** (remaining items)
  - Audio quality metrics (SNR, THD)
  - Performance regression testing in CI
  - Multi-platform CI (Linux, macOS, Windows)
- **Epic 7: Additional Effects**
  - Can be added incrementally in v1.1+

---

## Success Metrics for v1.0.0

1. ✅ **API Stability**: No breaking changes after v1.0.0 without major version bump
   - ✅ Implemented deprecation system
   - ✅ API stability guarantees documented
   - ✅ Migration guide template created
2. ✅ **Test Coverage**: 88% code coverage (gate: `fail_under = 87`)
   - ✅ 400+ tests across all modules
   - ✅ Unit, integration, CLI, fusion, and dispatch tests
3. ✅ **Documentation**: 100% of public API documented with examples
   - ✅ Complete API reference
   - ✅ CLI guide with comprehensive examples
   - ✅ Tutorials and how-to guides
4. ✅ **Performance**:
   - ✅ Real-time: 48kHz, 2048 buffer, ~46ms latency (tested in REPL)
   - ✅ Batch: custom CUDA kernels — 4x faster than SciPy (1ch), 11x faster (8ch)
   - ✅ Auto-fusion: ~2.5x faster IIR chains via deferred pipeline
   - ✅ FP32 GPU path: 3.0–3.6x faster than FP64 on consumer GPUs (v0.6.0)
   - ✅ CUDA Graph streaming: up to 4x lower per-chunk latency in the realtime regime (v0.6.0)
5. ✅ **Platform Support**: Linux, macOS, Windows with Python 3.10-3.13
   - ✅ CI testing on multiple platforms
6. ✅ **CLI Functionality**: All core commands working
   - ✅ 11 commands implemented (process, info, play, record, convert, trim, concat, stats, preset, interactive, watch)
   - ✅ Batch processing, pipes, TOML config, presets
7. ✅ **Community**: Contributing guide, issue templates, active CI
   - ✅ Style guide documented
   - ✅ Roadmap maintained

**Status: 7/7 metrics achieved — ready for v1.0.0 RC**

---

## Code Quality Standards

TorchFX follows **SOLID** and **DRY** principles:

- **Single Responsibility**: Each class has one clear purpose
- **Open/Closed**: Extensible through inheritance and composition
- **Liskov Substitution**: Consistent interfaces across similar classes
- **Interface Segregation**: Narrow, focused interfaces
- **Dependency Inversion**: Depend on abstractions, not implementations
- **Don't Repeat Yourself**: Shared utilities, reusable components

---

## Future Considerations (Post-v1.0)

- Plugin system for third-party effects
- ONNX export for deployment
- Model Hub integration (HuggingFace)
- Audio ML integration helpers
- VST3 plugin wrapper (complex, long-term)

---

## Contributing

We welcome contributions! See the [style guide](./style_guide.md) for guidelines.

- **Current focus**: v1.0 polish — additional effects (dynamics: compressor/limiter), audio-quality tests, and the deferred kernel optimizations in Epic 4.6
- **Phases 1–6 and 8**: ✅ COMPLETED
- **Good first issues**: Check GitHub issues tagged `good-first-issue`
- **CLI Extension Ideas**: Real-time visualization, AB comparison mode, spectrum analyzer
- **Questions**: Open a discussion on GitHub
