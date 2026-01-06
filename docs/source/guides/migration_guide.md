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

## Version 0.3.x → 1.0.0

> **Status**: In development
> **Expected release**: TBD

### Summary of Changes

This section will be updated as we approach the 1.0.0 release.

### New Features

- ✅ Wave.save() method for saving audio files (multiple formats)
- ✅ Complete LoShelving filter implementation
- ✅ Professional filters: ParametricEQ, Elliptic filters
- ✅ Metadata support in Wave class

### API Stabilization

#### Parameter Naming Consistency

The following naming conventions are now standardized:

- **Q factor**: Use `Q` (uppercase) for `Peaking`, `Notch`, `AllPass`
- **Q factor**: Use `q` (lowercase) for `Shelving`, `ParametricEQ`
- **Frequency**: Use `cutoff` for lowpass/highpass/shelving filters
- **Frequency**: Use `frequency` for `ParametricEQ` (center frequency)
- **Gain**: Always accept `gain` parameter with `gain_scale` to specify units

#### No Breaking Changes

All existing code will continue to work without modifications.

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

### Version 0.3.0 (Current Development)

#### Changes

- **Wave class**: Added `metadata` parameter to `__init__` (backward compatible, optional)
- **Wave class**: Added `save()` method (new feature, no breaking changes)
- **Filter naming**: Standardized parameter names (no breaking changes, recommendations only)

#### Migration Instructions

No migration required. All changes are backward compatible.

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
import torchfx
print(torchfx.__version__)
```

### Upgrading Safely

1. **Read this guide** for your current version → target version
2. **Run your tests** with warnings enabled: `python -W all::DeprecationWarning`
3. **Fix deprecation warnings** one by one
4. **Upgrade** to the new version
5. **Run tests again** to ensure everything works
