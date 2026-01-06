# Style Guide

This document defines the coding standards, naming conventions, and best practices for TorchFX development.

## Naming Conventions

### Classes

#### Filter Classes

**Pattern**: `<Type><FilterName>`

- Use `Hi` prefix for high-pass filters: `HiButterworth`, `HiElliptic`
- Use `Lo` prefix for low-pass filters: `LoButterworth`, `LoElliptic`
- Base classes have no prefix: `Butterworth`, `Elliptic`

```python
# Good
class HiButterworth(Butterworth):
    """High-pass Butterworth filter."""
    pass

class LoButterworth(Butterworth):
    """Low-pass Butterworth filter."""
    pass

# Bad
class HighPassButterworth(Butterworth):  # Don't use full words
    pass

class ButterworthHP(Butterworth):  # Don't use suffix
    pass
```

#### Effect Classes

**Pattern**: `<EffectName>`

- Use descriptive noun names: `Reverb`, `Delay`, `Chorus`
- No prefix or suffix needed

```python
# Good
class Reverb(FX):
    """Reverb effect."""
    pass

# Bad
class ReverbEffect(FX):  # Don't add "Effect" suffix
    pass
```

### Parameters

#### Frequency Parameters

- **`cutoff`**: For lowpass/highpass/shelving filters (cutoff frequency)
- **`frequency`**: For parametric EQ and peaking filters (center frequency)

```python
# Good - Lowpass filter
class LoButterworth:
    def __init__(self, cutoff: float, order: int = 5, fs: int | None = None):
        self.cutoff = cutoff

# Good - Parametric EQ
class ParametricEQ:
    def __init__(self, frequency: float, q: float, gain: float, fs: int | None = None):
        self.frequency = frequency

# Bad - Mixed usage
class LoButterworth:
    def __init__(self, frequency: float, ...):  # Use cutoff for filters
        pass
```

#### Gain Parameters

- **`gain`**: Always use `gain` as the parameter name
- **`gain_scale`**: Specify units with `"linear"` or `"db"`

```python
# Good
class HiShelving:
    def __init__(
        self,
        cutoff: float,
        q: float,
        gain: float,
        gain_scale: FilterOrderScale = "linear",
        fs: int | None = None,
    ):
        self.gain = gain if gain_scale == "linear" else 10 ** (gain / 20)

# Bad - Ambiguous units
class HiShelving:
    def __init__(self, cutoff: float, q: float, gain_db: float, ...):  # Don't encode units in name
        pass
```

#### Order Parameters

- **`order`**: Filter order (always an integer)
- **`order_scale`**: Use `"linear"` (default) or `"octave"` (order per octave)

```python
# Good
class Butterworth:
    def __init__(
        self,
        btype: str,
        cutoff: float,
        order: int = 4,
        order_scale: FilterOrderScale = "linear",
        fs: int | None = None,
    ):
        self.order = order if order_scale == "linear" else order // 6
```

#### Sample Rate Parameter

- **`fs`**: Always use `fs` for sampling frequency
- Always type as `int | None` to support lazy initialization

```python
# Good
def __init__(self, cutoff: float, fs: int | None = None):
    self.fs = fs

# Bad
def __init__(self, cutoff: float, sample_rate: int = 44100):  # Use fs, make optional
    pass
```

### Methods

- Use descriptive verb names: `compute_coefficients()`, `reset_state()`
- Don't abbreviate unless the abbreviation is standard: `fft()` OK, `comp_coef()` NOT OK

---

## Unit Conventions

### Frequency Units

- **Hz**: All frequency parameters are in Hz (not normalized)
- **Nyquist normalization**: Handled internally by filters

```python
# Good - User specifies in Hz
filter = LoButterworth(cutoff=1000, fs=44100)

# Bad - User must normalize
filter = LoButterworth(cutoff=1000/22050, fs=44100)  # Don't do this
```

### Gain Units

Two units are supported with explicit specification:

- **Linear gain**: Amplitude multiplier (1.0 = unity, 2.0 = 6dB)
- **dB**: Decibel scale (0dB = unity, 6dB = 2x amplitude)

```python
# Good - Explicit unit specification
filter = HiShelving(cutoff=1000, q=0.707, gain=6, gain_scale="db", fs=44100)
filter = HiShelving(cutoff=1000, q=0.707, gain=2.0, gain_scale="linear", fs=44100)

# Bad - Ambiguous units
filter = HiShelving(cutoff=1000, q=0.707, gain=6, fs=44100)  # Is this dB or linear?
```

### Amplitude/Level Units

- **Range [0, 1]** for normalized audio: `wave.ys` is typically in [-1, 1]
- **dB**: For levels and gains
- **Linear**: For simple multipliers

---

## Code Organization Patterns

### Filter Implementation Pattern

All IIR filters should follow this structure:

```python
class NewFilter(IIR):
    """One-line description.

    Detailed description of what the filter does, when to use it,
    and any special characteristics.

    Parameters
    ----------
    cutoff : float
        The cutoff frequency in Hz.
    order : int
        The filter order. Default is 4.
    fs : int | None
        The sampling frequency in Hz.

    Examples
    --------
    >>> filter = NewFilter(cutoff=1000, order=4, fs=44100)
    >>> filtered = filter.forward(signal)

    Notes
    -----
    Additional technical details about the filter implementation,
    references to papers, or usage warnings.

    """

    def __init__(
        self,
        cutoff: float,
        order: int = 4,
        fs: int | None = None,
    ) -> None:
        super().__init__(fs)
        self.cutoff = cutoff
        self.order = order
        self.a = None
        self.b = None

    @override
    def compute_coefficients(self) -> None:
        """Compute the filter coefficients."""
        assert self.fs is not None

        # Use scipy or implement custom coefficients
        b, a = some_filter_design(self.order, self.cutoff / (0.5 * self.fs))
        self.b = b
        self.a = a
```

### Effect Implementation Pattern

All effects should follow this structure:

```python
class NewEffect(FX):
    """One-line description.

    Detailed description.

    Parameters
    ----------
    param1 : float
        Description with units.
    param2 : int
        Description with units.
    fs : int | None
        The sampling frequency in Hz.

    Examples
    --------
    >>> effect = NewEffect(param1=0.5, fs=44100)
    >>> processed = effect.forward(signal)

    """

    def __init__(
        self,
        param1: float,
        param2: int = 100,
        fs: int | None = None,
    ) -> None:
        super().__init__(fs)
        self.param1 = param1
        self.param2 = param2

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Apply the effect to the input tensor."""
        # Implementation
        return processed_signal
```

### Module Organization

```
torchfx/
├── __init__.py          # Public API exports
├── wave.py              # Wave class
├── effect.py            # Base FX class and effects
├── typing.py            # Type aliases
├── _deprecation.py      # Internal utilities (prefix with _)
└── filter/
    ├── __init__.py      # Filter exports
    ├── __base.py        # Base filter classes
    ├── iir.py           # IIR filters
    └── fir.py           # FIR filters
```

---

## Documentation Standards

### Docstring Format

Use NumPy-style docstrings:

```python
def function_name(param1: int, param2: float = 1.0) -> float:
    """One-line summary (imperative mood).

    Longer description if needed. Can span multiple paragraphs.

    Parameters
    ----------
    param1 : int
        Description of param1 with units if applicable.
    param2 : float, optional
        Description of param2. Default is 1.0.

    Returns
    -------
    float
        Description of return value with units.

    Raises
    ------
    ValueError
        If param1 is negative.

    Examples
    --------
    >>> result = function_name(5, 2.0)
    >>> print(result)
    10.0

    Notes
    -----
    Additional technical details, references, or warnings.

    See Also
    --------
    related_function : Brief description.

    """
    pass
```

### Examples Requirements

Every public class and method must have:
1. At least one basic usage example
2. Example should be runnable (or clearly marked if not)
3. Use realistic parameter values

```python
# Good example
"""
Examples
--------
Create a lowpass filter and apply it:

>>> filter = LoButterworth(cutoff=1000, order=5, fs=44100)
>>> wave = Wave.from_file("audio.wav")
>>> filtered = wave | filter
"""

# Bad example - too abstract
"""
Examples
--------
>>> filter = LoButterworth(x, y, z)
>>> result = filter(data)
"""
```

### Type Hints

Always use type hints for:
- Function parameters
- Return values
- Class attributes (when clear)

```python
# Good
def process_audio(
    signal: Tensor,
    fs: int,
    gain: float = 1.0,
) -> Tensor:
    pass

# Bad - no type hints
def process_audio(signal, fs, gain=1.0):
    pass
```

---

## Testing Standards

### Test Organization

```python
class TestFilterName:
    """Test suite for FilterName."""

    @pytest.fixture
    def sample_signal(self):
        """Create a test signal."""
        return torch.randn(1, 1000)

    def test_coefficient_computation(self):
        """Test that coefficients are computed correctly."""
        filter = FilterName(cutoff=1000, fs=44100)
        filter.compute_coefficients()
        assert filter.a is not None
        assert filter.b is not None

    def test_forward_pass(self, sample_signal):
        """Test that forward pass works."""
        filter = FilterName(cutoff=1000, fs=44100)
        output = filter.forward(sample_signal)
        assert output.shape == sample_signal.shape
```

### Test Coverage Requirements

- Every public method must have at least one test
- Edge cases must be tested
- Error handling must be tested

### Test Naming

- Use descriptive test names: `test_lowpass_filters_high_frequencies`
- Start with `test_`
- Use underscores, not camelCase

---

## Code Quality Tools

### Linters

- **ruff**: Python linter (configured in `pyproject.toml`)
- **black**: Code formatter

### Pre-commit Hooks

Configure git hooks for automatic formatting:

```bash
# Run before committing
ruff check src/ tests/
black src/ tests/
pytest tests/
```

---

## Deprecation Guidelines

When deprecating an API:

1. Use the `@deprecated` decorator
2. Provide clear alternative
3. Set removal version (at least +1 minor version)
4. Update documentation
5. Add migration guide entry

```python
from torchfx._deprecation import deprecated

@deprecated(
    version="0.3.0",
    reason="Renamed for consistency",
    alternative="new_method()",
    removal_version="1.0.0"
)
def old_method(self):
    return self.new_method()
```

---

## Questions?

If you have questions about the style guide:
1. Check existing code for examples
2. Open a GitHub discussion
3. Ask in code review
