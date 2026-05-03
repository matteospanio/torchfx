# Migration Guide

This guide helps you migrate your code between major and minor versions of TorchFX.

## Semantic Versioning Policy

TorchFX follows [Semantic Versioning](https://semver.org/):

- **Major version (X.0.0)**: Breaking changes, API incompatible with previous version
- **Minor version (0.X.0)**: New features, backward compatible
- **Patch version (0.0.X)**: Bug fixes, backward compatible

### Compatibility Guarantees

- **No breaking changes in minor versions (1.x.x)**: Your code will continue to work
- **Deprecation warnings**: APIs will be deprecated for at least one minor version before removal
- **Migration period**: Deprecated APIs will remain functional for at least one minor version
- **Clear communication**: Breaking changes will be clearly documented with migration instructions

---

## Version 0.5.x → 1.0.0

> **Status**: 1.0.0 is not yet planned. This section will be updated as the API surface stabilizes toward a 1.0 release.

The 0.5.x series consolidates the public API (Wave, FX, FilterChain, the filter and effect modules) on top of a precompiled native extension. Most code written against 0.5.x is expected to keep working in 1.0; the [API Stability](api_stability.md) page tracks which surfaces are considered stable.

### Naming conventions (current as of 0.5.3)

- **Q factor**: lowercase `q` everywhere (`Peaking`, `ParametricEQ`, `Shelving`, biquads). Uppercase `Q` was retired as a breaking change in 0.3.0.
- **Frequency**: `cutoff` for lowpass / highpass / shelving filters, `frequency` for `ParametricEQ` (center frequency).
- **Gain**: parameter is always called `gain` (no units in the name). For `Gain` and `Peaking`, a sibling parameter (`gain_type` / `gain_scale`) selects between `"amplitude"`, `"db"`, `"linear"`, etc.
- **Sample rate**: always `fs: int | None = None` (never `sample_rate`).

---

## Deprecation Policy

### How Deprecations Work

1. **Deprecation Warning**: When an API is deprecated, using it will trigger a `DeprecationWarning`
2. **Migration Period**: The deprecated API remains functional for at least one minor version
3. **Removal**: The API is removed in the next major version

### Example

```python
# Version 0.3.0 - New API introduced, old API still works
wave.old_method()  # DeprecationWarning: old_method is deprecated, use new_method instead

# Version 0.4.0 - Old API still works with warning
wave.old_method()  # DeprecationWarning: old_method is deprecated, use new_method instead

# Version 1.0.0 - Old API removed
wave.old_method()  # AttributeError: Wave has no attribute 'old_method'
wave.new_method()  # Use this instead
```

### How to Handle Deprecation Warnings

#### 1. Update Your Code

When you see a deprecation warning, update your code to use the recommended alternative:

```python
# Before (deprecated)
filter = OldFilterName(cutoff=1000, fs=44100)

# After (recommended)
filter = NewFilterName(cutoff=1000, fs=44100)
```

#### 2. Suppress Warnings (Not Recommended)

If you can't update immediately, you can suppress deprecation warnings:

```python
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
```

**Warning**: This is not recommended as you'll miss important migration information.

---

## Breaking Changes History

### Version 1.0.0 (Future)

> **Release Date**: TBD

#### Breaking Changes

*To be determined. All changes will be documented here before release.*

#### Migration Instructions

*Detailed migration instructions will be provided for each breaking change.*

---

### Version 0.5.3 (Current)

#### Changes

- **Build system**: `hatchling` + runtime JIT compilation replaced with `scikit-build-core` + CMake. The C++/CUDA extension now compiles at install time and is bundled into the wheel.
- **Removed pure-PyTorch fallback**: `torchfx_ext` is required. The slow Python fallback paths for stateful IIR and delay are gone.
- **`setuptools` dropped from runtime dependencies** (was only needed for the JIT loader).
- **CPU delay kernel**: `delay_line_forward` now dispatches to a C++ kernel on CPU as well as CUDA.

#### Migration Instructions

No source-code changes are required. To verify the install picks up the precompiled extension:

```python
import torchfx
assert torchfx.is_native_available()
```

If `is_native_available()` returns `False` after upgrading, the install is broken --- see the installation {ref}`troubleshooting` section. The native extension is required and there is no Python fallback path.

---

### Version 0.5.2

#### Changes

- **`FilterChain`** auto-flattening container, exported from the top-level `torchfx` package: `(f1 | f2) | f3` now produces a flat `FilterChain(f1, f2, f3)`.
- **Pipe operator between filters**: `f1 | f2` now works (previously only `wave | f` was supported).
- **Deferred pipeline with auto-fusion**: `Wave.__or__` accumulates filters lazily and fuses adjacent IIR/biquad sections into a single SOS cascade on materialization. Transparent --- numerical results are identical.

#### Migration Instructions

No code changes required; this is a transparent performance optimization.

---

### Version 0.5.1

#### Breaking Changes

- **IIR filters now use SOS coefficients exclusively.** Removed `IIR.b`, `IIR.a`, and the dead `move_coeff()`, `_compute_ba_from_sos()`, `_bootstrap_state()`, and `_stateful` flag.
- **Removed `a` / `b` constructor parameters** from `Butterworth` and `Chebyshev1`.

#### Migration Instructions

```python
# Before 0.5.1: explicit coefficient transfer
fchain.to("cuda")
for f in fchain:
    f.move_coeff("cuda")   # AttributeError in 0.5.1+

# After 0.5.1: standard nn.Module.to() handles everything
fchain.to("cuda")
```

If you read coefficients off an IIR filter, switch to its SOS representation:

```python
# Before
b, a = my_butter.b, my_butter.a   # AttributeError

# After
sos = my_butter._sos              # [num_sections, 6] tensor
```

`Biquad` retains read-only `b` / `a` properties (since 0.5.2) for compatibility with code that inspected biquad coefficients.

---

### Version 0.5.0

#### Changes

- **JIT-compiled C++/CUDA native extension** introduced (`torchfx._ops`). At this point compilation happened on first import; this was replaced by install-time compilation in 0.5.3.
- **CUDA kernels** for biquad, SOS cascade, and delay line.
- **`LogFilterBank`** for logarithmically-spaced frequency band decomposition.
- **FFT-based 1D convolution** for FIR filters via overlap-save.

#### Migration Instructions

No source-code changes required. (Note: if you pinned `setuptools` for the JIT loader, you can drop it in 0.5.3+.)

---

### Version 0.4.0

#### Changes

- **CLI** introduced: `torchfx process`, `torchfx play`, `torchfx record`, `torchfx info`. Install with the `cli` dependency group.
- **`torchfx.validation`** module: validators and exception hierarchy.
- **`torchfx.logging`** module: structured logging and performance timing utilities.
- **`torchfx.realtime`** module: streaming processors and audio backends.
- **Biquad filters**: LPF, HPF, BPF, peak, notch, all-pass.

#### Migration Instructions

No source-code changes required.

---

### Version 0.3.0

#### Breaking Changes

- **`Q` → `q`**: all filter Q parameters renamed to lowercase.

#### Migration Instructions

```python
# Before
f = Peaking(cutoff=1000, Q=2.0, gain=3, gain_scale="db", fs=44100)

# After
f = Peaking(cutoff=1000, q=2.0, gain=3, gain_scale="db", fs=44100)
```

#### Other changes

- Added `Wave.save()`, `LoShelving`, `ParametricEQ`, Elliptic filters.

---

### Version 0.2.0

#### Breaking Changes

- **Module rename**: `torchfx.effects` → `torchfx.effect` (singular, to match `torchfx.filter`).

#### Migration Instructions

```python
# Before
from torchfx.effects import Gain

# After
from torchfx.effect import Gain
```

---

## Common Migration Patterns

### Pattern 1: Renaming Parameters

If a parameter is renamed:

```python
# Old code (deprecated in 0.3.0, removed in 1.0.0)
filter = SomeFilter(old_param=value)

# New code (recommended from 0.3.0+)
filter = SomeFilter(new_param=value)
```

### Pattern 2: Class Renaming

If a class is renamed:

```python
# Old code (deprecated in 0.3.0, removed in 1.0.0)
from torchfx.filter import OldClassName
filter = OldClassName(...)

# New code (recommended from 0.3.0+)
from torchfx.filter import NewClassName
filter = NewClassName(...)
```

### Pattern 3: Method Signature Changes

If a method signature changes:

```python
# Old code (deprecated in 0.3.0, removed in 1.0.0)
result = obj.method(old_arg1, old_arg2)

# New code (recommended from 0.3.0+)
result = obj.method(new_arg1, new_arg2)
```

## Version-Specific Guides

### Checking Your Version

```python
from importlib.metadata import version
print(version("torchfx"))
```

`torchfx` does not expose a `__version__` attribute --- query it via `importlib.metadata` (standard library since Python 3.8).

### Upgrading Safely

1. **Read this guide** for your current version → target version
2. **Run your tests** with warnings enabled: `python -W all::DeprecationWarning`
3. **Fix deprecation warnings** one by one
4. **Upgrade** to the new version
5. **Run tests again** to ensure everything works
