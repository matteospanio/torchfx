# Roadmap to v1.0.0

**Current Version:** 0.2.1 (Alpha)
**Target:** v1.0.0 Stable Release

This roadmap outlines the development path for TorchFX from the current alpha state to a production-ready v1.0.0 release. The plan is organized into major epics, each containing specific deliverables and tasks.

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
- ✅ GPU acceleration working
- ✅ 95 tests with good coverage of existing features
- ✅ Published research paper (arXiv:2504.08624)
- ✅ Clean API with pipe operator support
- ✅ Basic Sphinx documentation

### Gaps
- ❌ No real-time audio input/output
- ❌ Wave class can't save files
- ❌ CLI is placeholder only
- ❌ Missing essential filters (LoShelving, parametric EQ)
- ❌ No custom CUDA kernels
- ❌ Documentation incomplete
- ❌ API not stabilized for v1.0

**Estimated Completion:** ~75% ready for v1.0.0

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

- [ ] **Input validation layer**
  - Validate sample rates, tensor shapes, parameter ranges
  - Custom exception hierarchy: `TorchFXError`, `InvalidParameterError`, `AudioProcessingError`

- [ ] **Improved error messages**
  - Context-aware messages with actual vs. expected values
  - Suggestions for fixes

- [ ] **Logging infrastructure**
  - Structured logging with Python's `logging` module
  - Log levels: DEBUG, INFO, WARNING, ERROR
  - Performance logging (optional)

---

## Epic 2: Real-Time Audio Processing

**Priority:** Critical (Major Feature)
**Goal:** Enable low-latency live audio processing with GPU acceleration.

### 2.1 Audio Backend Integration

- [ ] **Abstract audio backend interface**
  - Create `AudioBackend` base class
  - Support input, output, duplex streams
  - Callback-based and blocking APIs

- [ ] **PortAudio backend** (Priority 1)
  - Use `sounddevice` library
  - Cross-platform support
  - ASIO support on Windows
  - Buffer size: 64-2048 samples

- [ ] **PulseAudio/PipeWire backend** (Priority 2)
  - Native Linux desktop integration

- [ ] **JACK backend** (Future)
  - Professional Linux audio routing

### 2.2 Real-Time Processing Pipeline

- [ ] **Ring buffer implementation**
  - Lock-free SPSC ring buffer
  - GPU-compatible tensor buffers
  - Overlap-add support

- [ ] **Real-time processor class**
  ```python
  class RealtimeProcessor:
      def __init__(self, effect_chain, buffer_size, device)
      def start()
      def stop()
      def set_parameter(name, value)  # Thread-safe
  ```

- [ ] **Latency optimization**
  - Target: <10ms total latency at 48kHz, 512 buffer
  - GPU stream optimization
  - Pre-allocated tensor pools

- [ ] **Stream processing for large files**
  - Chunk-based processing without loading entire file

### 2.3 Real-Time Effect Adaptations

- [ ] **Stateful filter management**
  - IIR state maintenance
  - `reset_state()` method

- [ ] **Thread-safe parameter updates**
  - Lock-free parameter smoothing
  - Atomic swaps

- [ ] **CPU/GPU hybrid processing**
  - Small buffers on CPU for ultra-low latency
  - Large batches on GPU for throughput

---

## Epic 3: CLI Application

**Priority:** High (Major Feature)
**Goal:** Modern, GPU-accelerated CLI tool with sox compatibility and unique features.

### 3.1 Core CLI Architecture

- [ ] **CLI framework with Typer**
  - Commands: `process`, `info`, `play`, `record`, `interactive`
  - Global options: `--device`, `--verbose`, `--config`
  - Rich output with progress bars

- [ ] **Subcommand structure**
  ```bash
  torchfx process input.wav output.wav --effect reverb
  torchfx info audio.flac
  torchfx play audio.wav
  torchfx record output.wav --duration 10
  torchfx interactive  # REPL mode
  ```

- [ ] **Configuration file support (YAML/TOML)**
  - Save/load effect chains
  - Preset management: `~/.config/torchfx/presets/`

### 3.2 Pipeline Processing & Sox Compatibility

- [ ] **Unix pipe support**
  - Read from stdin: `cat audio.wav | torchfx process -`
  - Write to stdout: `torchfx process input.wav - | aplay`
  - Chain commands

- [ ] **Batch processing**
  ```bash
  torchfx process "*.wav" --output-dir ./processed/ --effect normalize
  ```

- [ ] **Sox-compatible commands** (subset)
  - `convert`, `trim`, `concat`, `stats`

- [ ] **GPU-accelerated batch processing**
  - Auto-batch multiple files
  - Progress bar with ETA

### 3.3 Interactive Mode (REPL)

- [ ] **Interactive shell**
  - Tab completion, syntax highlighting
  - Command history

- [ ] **Live parameter tweaking**
  ```python
  >>> load("audio.wav")
  >>> add_effect("reverb", room_size=0.5)
  >>> play()
  >>> set_param("reverb.room_size", 0.8)
  >>> ab_compare()
  ```

- [ ] **Real-time visualization**
  - Waveform display
  - Spectrum analyzer
  - VU meters

- [ ] **Preset management**
  - Save/load/list presets in REPL

### 3.4 Watch Mode & Automation

- [ ] **File system watcher**
  ```bash
  torchfx watch ./input/ --output ./processed/ --effect reverb
  ```

- [ ] **DAW integration mode**
  - Monitor export folder
  - Auto-apply mastering chain

---

## Epic 4: Performance Optimization & CUDA

**Priority:** Medium (Can be v1.1)
**Goal:** Maximize throughput with custom CUDA kernels.

### 4.1 CUDA Development Infrastructure

- [ ] **CUDA extension build system**
  - PyTorch C++ extension API
  - Auto CUDA arch detection
  - Fallback to PyTorch if CUDA unavailable

- [ ] **Kernel development tools**
  - CUDA profiling integration (nvprof, Nsight)
  - Unit tests for CUDA kernels
  - Benchmarking harness

### 4.2 IIR Filter CUDA Kernels (Priority 1)

- [ ] **Parallel IIR implementation**
  - Parallel prefix scan for state propagation
  - Target: 2-3x speedup for batch processing

- [ ] **Biquad cascade optimization**
  - Fuse multiple biquad sections
  - Reduce memory traffic

- [ ] **Stability guarantees**
  - Match scipy/PyTorch numerical behavior

### 4.3 Time-Domain Effects CUDA Kernels (Priority 2)

- [ ] **Optimized delay line**
  - Circular buffer with shared memory
  - Interpolation for fractional delays

- [ ] **Reverb optimization**
  - Parallel all-pass filters
  - Fused feedback delay network

### 4.4 Batch Processing Optimizations (Priority 3)

- [ ] **Multi-file batch processing**
  - Process multiple files in single kernel launch
  - Maximize GPU occupancy

- [ ] **Operator fusion**
  - Fuse multiple effects: `gain + filter + normalize` → single kernel

- [ ] **Memory optimization**
  - Tensor memory pooling
  - In-place operations

### 4.5 Performance Benchmarking

- [ ] **Comprehensive benchmark suite**
  - PyTorch vs. CUDA vs. CPU comparison
  - Report throughput and latency

- [ ] **Performance regression testing**
  - Automated benchmarks in CI
  - Alert on >5% regression

- [ ] **Profiling guides**
  - Documentation for profiling pipelines

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

- [ ] **Getting Started Tutorial** (expand)
  - Installation
  - First pipeline
  - Wave class basics
  - Saving output

- [ ] **Advanced Tutorials**
  - Real-time audio processing
  - Custom filter design
  - GPU optimization
  - CLI tool mastery

- [ ] **How-To Guides**
  - Audio format conversion
  - Building EQ/filter bank
  - Mastering chain
  - Multi-channel processing
  - Guitar pedal simulator
  - PyTorch model integration

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

- [ ] **Complete Wave class tests**
  - File I/O for all formats
  - Multi-channel audio
  - Sample rate conversion
  - Device transfers
  - Edge cases

- [ ] **Complete filter tests**
  - All filter types
  - Frequency/phase response validation
  - Filter composition
  - Edge cases

- [ ] **Complete effect tests**
  - All effects and parameters
  - Error handling

### 6.2 Integration Tests

- [ ] **Complex pipeline tests**
  - Multi-stage effect chains
  - GPU end-to-end processing
  - File load → process → save

- [ ] **Real-time processing tests**
  - Mock audio backend
  - Latency measurements
  - Parameter updates during processing

- [ ] **CLI integration tests**
  - All CLI commands
  - Pipe I/O
  - Batch processing

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

- [ ] **Coverage reporting**
  - Codecov integration
  - >90% coverage enforcement
  - Coverage badge

- [ ] **Multi-platform testing**
  - Linux, macOS, Windows
  - Python 3.10-3.13
  - With/without CUDA

- [ ] **GPU CI runner**
  - Self-hosted or cloud GPU
  - CUDA tests and benchmarks

- [ ] **Automated releases**
  - PyPI publishing on tag
  - Changelog generation

---

## Epic 7: Additional Effects

**Priority:** Low (Can be v1.1+)
**Goal:** Expand effect library for common production needs.

### 7.1 Dynamics Processing

- [ ] Compressor (threshold, ratio, attack, release, knee)
- [ ] Limiter (brickwall, true peak, look-ahead)
- [ ] Expander / Gate

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

### Phase 1: Foundation (Required for Beta)
**Priority:** Critical

- **Epic 1: Core Library Stabilization**
  - Complete missing features
  - API stabilization
  - Error handling
- **Epic 6: Testing Infrastructure** (parallel)
  - Expand unit tests
  - CI improvements

### Phase 2: Major Features (Required for v1.0)
**Priority:** Critical

- **Epic 2: Real-Time Audio Processing**
  - Audio backends
  - Real-time pipeline
- **Epic 3: CLI Application**
  - Core CLI
  - Pipeline processing
  - Interactive mode
- **Epic 5: Documentation** (continuous)
  - Complete before release

### Phase 3: Optimization & Polish (v1.0 or v1.1)
**Priority:** Medium

- **Epic 4: CUDA Kernels** (can start early)
  - IIR kernels (priority)
  - Effect kernels
- **Epic 7: Additional Effects**
  - Can be added incrementally in v1.1+

---

## Success Metrics for v1.0.0

1. ✅ **API Stability**: No breaking changes after v1.0.0 without major version bump
2. ✅ **Test Coverage**: >90% code coverage
3. ✅ **Documentation**: 100% of public API documented with examples
4. ✅ **Performance**:
   - Real-time: 48kHz, 512 buffer, <10ms latency on GPU
   - Batch: >100x real-time on modern GPU
5. ✅ **Platform Support**: Linux, macOS, Windows with Python 3.10-3.13
6. ✅ **CLI Functionality**: All core commands working
7. ✅ **Community**: Contributing guide, issue templates, active CI

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

- **Current focus**: Phase 1 (Core Stabilization)
- **Good first issues**: Check GitHub issues tagged `good-first-issue`
- **Questions**: Open a discussion on GitHub
