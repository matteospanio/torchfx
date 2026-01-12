# API Stability and Backward Compatibility

This document describes TorchFX's commitment to API stability and backward compatibility.

## Versioning Policy

TorchFX follows [Semantic Versioning 2.0.0](https://semver.org/):

```
MAJOR.MINOR.PATCH (e.g., 1.2.3)
```

- **MAJOR**: Incompatible API changes
- **MINOR**: Backward-compatible new features
- **PATCH**: Backward-compatible bug fixes

## Stability Guarantees

### What We Guarantee

#### 1. No Breaking Changes in Minor Versions

✅ **Your code will continue to work** when upgrading between minor versions (e.g., 1.0.0 → 1.1.0 → 1.2.0)

```python
# Code written for 1.0.0 will work in 1.1.0 and 1.2.0
filter = LoButterworth(cutoff=1000, order=5, fs=44100)
wave = Wave.from_file("audio.wav")
filtered = wave | filter
```

#### 2. Deprecation Period

✅ **Minimum one minor version** before removal

APIs will be marked as deprecated for at least one full minor version before removal in the next major version.

```
Version 1.0.0: Feature X introduced
Version 1.1.0: Feature X deprecated, Feature Y introduced
Version 1.2.0: Feature X still works (with warning)
Version 2.0.0: Feature X removed, use Feature Y
```

#### 3. Clear Communication

✅ **Deprecation warnings** will be clear and actionable

```python
DeprecationWarning: old_method() is deprecated since version 1.1.0.
Use new_method() instead. It will be removed in version 2.0.0.
```

#### 4. Migration Support

✅ **Documentation and migration guides** for all breaking changes

- Migration guide with step-by-step instructions
- Code examples showing before/after
- Rationale for changes

### What We Don't Guarantee

❌ **Private APIs** (prefixed with `_`) may change without notice

```python
# Public API - Stable
from torchfx.filter import LoButterworth

# Private API - May change
from torchfx._internal_utils import some_function  # Not guaranteed
```

❌ **Experimental features** may change until stabilized

Features marked as "experimental" in the documentation are subject to change.

❌ **Bug behavior** - Bug fixes may change observable behavior

If current behavior is a bug, fixing it is not considered a breaking change.

## Public API Surface

### Stable APIs (v0.3.0+)

These APIs are considered stable and will follow the versioning policy:

#### Core Classes
- `Wave` - Audio waveform representation
- `FX` - Base effect class

#### Filters (torchfx.filter)
- IIR Filters:
  - `Butterworth`, `HiButterworth`, `LoButterworth`
  - `Chebyshev1`, `HiChebyshev1`, `LoChebyshev1`
  - `Chebyshev2`, `HiChebyshev2`, `LoChebyshev2`
  - `Elliptic`, `HiElliptic`, `LoElliptic`
  - `LinkwitzRiley`, `HiLinkwitzRiley`, `LoLinkwitzRiley`
  - `HiShelving`, `LoShelving`
  - `ParametricEQ`
  - `Notch`, `AllPass`, `Peaking`

- FIR Filters:
  - `FIR`, `DesignableFIR`

#### Effects (torchfx.effect)
- `Reverb`
- `Delay`

### Parameter Stability

#### Stable Parameters

These parameter names will not change:

- `fs` - Sampling frequency (Hz)
- `cutoff` - Cutoff frequency for filters (Hz)
- `frequency` - Center frequency for parametric EQ (Hz)
- `order` - Filter order
- `gain` - Gain value
- `gain_scale` - Gain units ("linear" or "db")
- `Q` - Quality factor (uppercase for Peaking, Notch, AllPass)
- `q` - Quality factor (lowercase for Shelving, ParametricEQ)

### Type Signature Stability

Type signatures in stable APIs will not change in backward-incompatible ways:

```python
# Stable signature
def forward(self, x: Tensor) -> Tensor:
    pass

# Future additions are OK (backward compatible)
def forward(self, x: Tensor, *, mode: str = "default") -> Tensor:
    pass

# This would be breaking (requires major version bump)
def forward(self, x: Tensor, required_param: int) -> Tensor:  # NOT OK
    pass
```

## How to Check API Stability

### 1. Check Documentation

All stable APIs are marked in the documentation:

```python
class StableClass:
    """Stable since version 1.0.0.

    This API is considered stable and will follow semantic versioning.
    """
```

### 2. Check for Deprecation Warnings

Run your code with warnings enabled:

```bash
python -W all::DeprecationWarning your_script.py
```

### 3. Review CHANGELOG

Check [CHANGELOG.md](https://github.com/matteospanio/torchfx/CHANGELOG) for:
- New features
- Deprecations
- Breaking changes

### 4. Follow Migration Guide

See [migration guide](./migration_guide.md) for version-specific migration instructions.

## Handling Breaking Changes

### For Users

When a breaking change is necessary (major version bump):

1. **Read the migration guide**: [migration guide](./migration_guide.md)
2. **Update incrementally**: Update to latest minor version first
3. **Fix deprecation warnings**: Address all warnings before upgrading
4. **Test thoroughly**: Run your test suite
5. **Upgrade**: Install the new major version

### For Contributors

When proposing a breaking change:

1. **Justify the change**: Why is it necessary?
2. **Propose alternatives**: Can it be done without breaking compatibility?
3. **Provide migration path**: How will users migrate?
4. **Update documentation**: Migration guide, CHANGELOG, etc.
5. **Deprecate first**: Add deprecation warnings in current version

## Examples

### Example 1: Adding Optional Parameters (OK)

```python
# Version 1.0.0
class Filter:
    def __init__(self, cutoff: float, fs: int):
        pass

# Version 1.1.0 - Backward compatible
class Filter:
    def __init__(self, cutoff: float, fs: int, order: int = 4):
        pass

# Old code still works
filter = Filter(cutoff=1000, fs=44100)  # ✅ Works in both versions
```

### Example 2: Renaming Parameters (Breaking)

```python
# Version 1.0.0
class Filter:
    def __init__(self, freq: float, fs: int):
        pass

# Version 2.0.0 - Breaking change (requires major bump)
class Filter:
    def __init__(self, cutoff: float, fs: int):
        pass

# Migration path with deprecation in 1.x
# Version 1.1.0 - Add deprecation warning
class Filter:
    @deprecated_parameter("freq", version="1.1.0", alternative="cutoff")
    def __init__(self, cutoff: float = None, freq: float = None, fs: int = None):
        if freq is not None:
            cutoff = freq  # Support old parameter
        self.cutoff = cutoff
```

### Example 3: Adding Required Parameters (Breaking)

```python
# Version 1.0.0
class Filter:
    def __init__(self, cutoff: float):
        pass

# Version 2.0.0 - Breaking change
class Filter:
    def __init__(self, cutoff: float, fs: int):  # fs now required
        pass

# Migration: Make it optional with deprecation in 1.x
# Version 1.1.0
class Filter:
    def __init__(self, cutoff: float, fs: int = None):
        if fs is None:
            warnings.warn("fs will be required in 2.0.0", DeprecationWarning)
            fs = 44100  # Temporary default
```

## Contact

Questions about API stability?

- Open a [GitHub Issue](https://github.com/matteospanio/torchfx/issues) with label `api-stability`
- Start a [GitHub Discussion](https://github.com/matteospanio/torchfx/discussions)
- Check [CHANGELOG.md](https://github.com/matteospanio/torchfx/CHANGELOG) for latest changes
