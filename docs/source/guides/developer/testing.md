# Testing

Comprehensive guide to the TorchFX testing infrastructure, including test organization, execution, patterns, and coverage requirements.

## Overview

The TorchFX testing infrastructure uses pytest as the test runner with comprehensive coverage reporting. Tests validate the correctness of audio effects, filters, and the core Wave class through unit and integration tests. The test suite emphasizes strategy pattern testing, multi-channel processing, and edge case validation.

```{seealso}
{doc}`/guides/developer/project-structure` - Project structure
{doc}`/guides/developer/benchmarking` - Performance testing
```

## Test Infrastructure Configuration

### Pytest Configuration

The pytest configuration is defined in `pyproject.toml`:

| Configuration | Value | Purpose |
|---------------|-------|---------|
| `minversion` | `"7.0"` | Minimum pytest version required |
| `addopts` | `"--strict-markers --tb=short"` | Strict marker validation and short tracebacks |
| `testpaths` | `["tests"]` | Directory containing test files |
| `pythonpath` | `["src"]` | Python path for importing torchfx modules |

**Configuration details**:

- `--strict-markers`: Ensures only registered markers are used, preventing typos
- `--tb=short`: Provides concise traceback output for faster debugging
- `pythonpath = ["src"]`: Allows tests to import `torchfx` directly without installation

### Coverage Configuration

Coverage reporting is configured to track code execution:

| Configuration | Value | Purpose |
|---------------|-------|---------|
| `source` | `["src/torchfx"]` | Source directory to measure coverage |
| `branch` | `true` | Enable branch coverage analysis |

**Branch coverage** ensures that both `True` and `False` branches of conditional statements are tested, providing more thorough coverage metrics than simple line coverage.

## Test Organization

```{mermaid}
graph TB
    subgraph "Test Directory Structure"
        TestDir["tests/"]
        TestEffects["test_effects.py<br/>Primary test file"]
    end

    subgraph "Test Categories in test_effects.py"
        GainTests["Gain Tests<br/>test_gain_*"]
        NormalizeTests["Normalize Tests<br/>test_normalize_*"]
        StrategyTests["Strategy Tests<br/>test_*_strategy"]
        ReverbTests["Reverb Tests<br/>test_reverb_*"]
        DelayTests["Delay Tests<br/>test_delay_*"]
        MusicalTimeTests["MusicalTime Tests<br/>test_musical_time_*"]
    end

    subgraph "Source Code Under Test"
        EffectModule["src/torchfx/effect.py<br/>FX, Gain, Normalize<br/>Reverb, Delay"]
        TypingModule["src/torchfx/typing.py<br/>MusicalTime"]
    end

    TestDir --> TestEffects

    TestEffects --> GainTests
    TestEffects --> NormalizeTests
    TestEffects --> StrategyTests
    TestEffects --> ReverbTests
    TestEffects --> DelayTests
    TestEffects --> MusicalTimeTests

    GainTests -.->|tests| EffectModule
    NormalizeTests -.->|tests| EffectModule
    StrategyTests -.->|tests| EffectModule
    ReverbTests -.->|tests| EffectModule
    DelayTests -.->|tests| EffectModule
    MusicalTimeTests -.->|tests| TypingModule
```

### Test File Structure

The primary test file is `tests/test_effects.py`, containing all tests for the effect system:

| Test Group | Component Tested | Test Count |
|------------|------------------|------------|
| Gain tests | {class}`~torchfx.effect.Gain` effect | 5 tests |
| Normalize tests | {class}`~torchfx.effect.Normalize` effect and strategy pattern | 15 tests |
| Reverb tests | {class}`~torchfx.effect.Reverb` effect | 7 tests |
| Delay tests | {class}`~torchfx.effect.Delay` effect including BPM sync | 31 tests |
| MusicalTime tests | {class}`~torchfx.typing.MusicalTime` parsing | 8 tests |

### Naming Conventions

Test functions follow consistent naming patterns:

- `test_<component>_<scenario>`: Tests a specific scenario
- `test_<component>_invalid_<parameter>`: Tests validation and error handling
- `test_<strategy>_strategy`: Tests strategy pattern implementations

**Examples**:

```python
def test_gain_amplitude():
    """Tests gain in amplitude mode."""
    pass

def test_normalize_invalid_peak():
    """Tests peak validation in Normalize."""
    pass

def test_peak_normalization_strategy():
    """Tests PeakNormalizationStrategy."""
    pass
```

## Test Execution Flow

```{mermaid}
sequenceDiagram
    participant Dev as Developer
    participant Pytest as pytest Runner
    participant Test as Test Function
    participant SUT as System Under Test
    participant Assert as Assertion

    Dev->>Pytest: pytest tests/
    Pytest->>Pytest: Discover tests in tests/
    Pytest->>Pytest: Apply pythonpath = ["src"]

    loop For each test function
        Pytest->>Test: Execute test_*()

        alt Setup required
            Test->>Test: Create fixtures/mocks
        end

        Test->>SUT: Instantiate component
        Test->>SUT: Call forward() or __call__()
        SUT->>Test: Return processed tensor

        Test->>Assert: torch.testing.assert_close()

        alt Assertion passes
            Assert->>Pytest: Test passed
        else Assertion fails
            Assert->>Pytest: Test failed (short traceback)
        end
    end

    Pytest->>Dev: Test results + coverage report
```

## Running Tests

### Basic Execution

Run all tests:

```bash
pytest tests/
```

Run tests with coverage:

```bash
pytest tests/ --cov=src/torchfx --cov-report=html
```

Run a specific test file:

```bash
pytest tests/test_effects.py
```

Run a specific test function:

```bash
pytest tests/test_effects.py::test_gain_amplitude
```

### Verbose Output

For detailed output showing each test name:

```bash
pytest tests/ -v
```

For even more detailed output including print statements:

```bash
pytest tests/ -vv -s
```

### Filtering Tests

Run tests matching a pattern:

```bash
# Run all tests with "gain" in the name
pytest tests/ -k "gain"

# Run all tests with "delay" but not "lazy"
pytest tests/ -k "delay and not lazy"
```

Run tests with specific markers (requires marker registration):

```bash
pytest tests/ -m "slow"  # Run only slow tests
pytest tests/ -m "not slow"  # Skip slow tests
```

### Coverage Reports

Generate terminal coverage report:

```bash
pytest tests/ --cov=src/torchfx --cov-report=term-missing
```

Generate HTML coverage report:

```bash
pytest tests/ --cov=src/torchfx --cov-report=html
# Open htmlcov/index.html in browser
```

Generate XML coverage report (for CI):

```bash
pytest tests/ --cov=src/torchfx --cov-report=xml
```

## Test Types and Patterns

### Unit Tests

Unit tests validate individual components in isolation with known inputs and expected outputs.

**Example: Basic Gain Test**

```python
def test_gain_amplitude():
    """Test that Gain with amplitude mode multiplies waveform correctly."""
    waveform = torch.tensor([0.1, -0.2, 0.3])
    gain = Gain(gain=2.0, gain_type="amplitude")
    out = gain(waveform)
    torch.testing.assert_close(out, waveform * 2.0)
```

**Key characteristics**:
- Tests single component
- Predictable inputs and outputs
- No external dependencies
- Fast execution

### Integration Tests

Integration tests validate interaction between multiple components, such as pipeline chaining or multi-channel processing.

**Example: Wave Pipeline Integration**

```python
def test_delay_lazy_fs_inference_with_wave():
    """Test that Delay automatically infers fs when used with Wave."""
    from torchfx import Wave

    delay = Delay(bpm=120, delay_time="1/8", feedback=0.3, mix=0.2)
    assert delay.fs is None  # Not yet configured

    wave = Wave(torch.randn(2, 44100), fs=44100)
    _ = wave | delay  # Pipeline operator triggers configuration

    assert delay.fs == 44100  # Automatically configured
    assert delay.delay_samples == 11025  # Calculated from BPM
```

**Key characteristics**:
- Tests component interactions
- Validates pipeline behavior
- Tests automatic configuration
- May be slower than unit tests

### Parametrized Tests

Parametrized tests run the same test logic with multiple input values, reducing code duplication.

**Example: Invalid Parameter Tests**

```python
import pytest

@pytest.mark.parametrize("delay", [0, -1])
def test_reverb_invalid_delay(delay):
    """Test that Reverb rejects invalid delay values."""
    with pytest.raises(AssertionError):
        Reverb(delay=delay, decay=0.5, mix=0.5)
```

**Benefits**:
- Reduces code duplication
- Tests multiple edge cases
- Clear test output showing which parameter failed
- Easy to add new test cases

### Strategy Pattern Tests

The test suite extensively validates the strategy pattern used in effects. These tests verify that custom strategies can be injected and built-in strategies behave correctly.

**Example: Custom Strategy Injection**

```python
from torchfx.effect import Normalize, NormalizationStrategy

class DummyStrategy(NormalizationStrategy):
    """Custom strategy that sets all values to peak."""
    def __call__(self, waveform, peak):
        return waveform * 0 + peak

def test_normalize_custom_strategy():
    """Test that custom normalization strategies work."""
    waveform = torch.tensor([0.2, -0.5, 0.4])
    norm = Normalize(peak=2.0, strategy=DummyStrategy())
    out = norm(waveform)
    torch.testing.assert_close(out, torch.full_like(waveform, 2.0))
```

**Key characteristics**:
- Tests extensibility points
- Validates strategy interface
- Ensures custom implementations work
- Documents strategy usage

### Mocking with Monkeypatch

Some tests use pytest's `monkeypatch` fixture to replace external dependencies.

**Example: Mocking torchaudio.functional.gain**

```python
def test_gain_db(monkeypatch):
    """Test that Gain calls torchaudio.functional.gain with correct params."""
    waveform = torch.tensor([0.1, -0.2, 0.3])
    called = {}

    def fake_gain(waveform, gain):
        called["args"] = (waveform, gain)
        return waveform + gain

    monkeypatch.setattr("torchaudio.functional.gain", fake_gain)

    gain = Gain(gain=6.0, gain_type="db")
    out = gain(waveform)

    assert torch.allclose(out, waveform + 6.0)
    assert called["args"][1] == 6.0  # Verify gain parameter
```

**Benefits**:
- Tests internal logic without side effects
- Validates parameter passing
- Avoids external dependencies
- Enables testing of hard-to-test code

## Assertion Patterns

### torch.testing.assert_close

The primary assertion method for tensor comparisons:

```python
torch.testing.assert_close(actual, expected)
```

**Features**:
- Element-wise closeness checking
- Appropriate tolerances for floating-point comparisons
- Clear error messages showing differences

**Example**:

```python
def test_peak_normalization_strategy():
    """Test peak normalization strategy."""
    waveform = torch.tensor([0.2, -0.5, 0.4])
    strat = PeakNormalizationStrategy()
    out = strat(waveform, 2.0)
    torch.testing.assert_close(out, waveform / 0.5 * 2.0)
```

### pytest.approx

For scalar comparisons with tolerance:

```python
assert value.item() == pytest.approx(expected, abs=1e-5)
```

**Example**:

```python
def test_delay_basic():
    """Test basic delay functionality."""
    waveform = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
    delay = Delay(delay_samples=2, feedback=0.0, mix=1.0, taps=1)
    out = delay(waveform)
    assert out[2].item() == pytest.approx(1.0, abs=1e-5)
```

### torch.allclose

For boolean comparisons with tolerance:

```python
assert torch.allclose(actual, expected, atol=1e-5)
```

**Example**:

```python
def test_delay_mix_zero():
    """Test that mix=0 produces only dry signal."""
    waveform = torch.randn(10)
    delay = Delay(delay_samples=3, feedback=0.5, mix=0.0)
    out = delay(waveform)
    # Output beyond original length should be zeros
    assert torch.allclose(out[10:], torch.zeros(out.size(0) - 10), atol=1e-5)
```

### pytest.raises

For validating exception handling:

```python
with pytest.raises(ExceptionType):
    # Code that should raise exception
```

**Example**:

```python
def test_gain_invalid_gain_type():
    """Test that invalid gain raises ValueError."""
    with pytest.raises(ValueError):
        Gain(gain=-1.0, gain_type="amplitude")
```

**Advanced usage with match**:

```python
def test_delay_lazy_fs_inference_error():
    """Test that missing fs raises helpful error message."""
    delay = Delay(bpm=120, delay_time="1/8", feedback=0.3, mix=0.2)
    waveform = torch.randn(2, 44100)

    with pytest.raises(AssertionError, match="Sample rate \\(fs\\) is required"):
        delay(waveform)
```

## Writing New Tests

### Test Structure Template

Follow this template for new tests:

```python
def test_<component>_<scenario>():
    """Brief description of what this test validates.

    Include any important details about edge cases or expected behavior.
    """
    # Arrange: Set up test data
    waveform = torch.randn(2, 44100)
    component = ComponentClass(param1=value1, param2=value2)

    # Act: Execute the operation
    result = component(waveform)

    # Assert: Verify results
    torch.testing.assert_close(result, expected_result)
```

### Common Test Patterns

#### Testing Effects

```python
def test_my_effect_basic():
    """Test basic functionality of MyEffect."""
    # Create test waveform
    waveform = torch.tensor([1.0, 0.5, -0.5, -1.0])

    # Create effect
    effect = MyEffect(param=value)

    # Apply effect
    output = effect(waveform)

    # Verify output
    assert output.shape == waveform.shape
    torch.testing.assert_close(output, expected)
```

#### Testing Filters

```python
def test_my_filter_frequency_response():
    """Test filter frequency response characteristics."""
    # Create filter
    filt = MyFilter(cutoff=1000, fs=44100)

    # Create test signal (impulse or sine wave)
    impulse = torch.zeros(1000)
    impulse[0] = 1.0

    # Apply filter
    output = filt(impulse)

    # Verify characteristics
    # (e.g., DC component, high-frequency attenuation)
    assert output[0] > 0  # Filter passes signal
```

#### Testing Multi-Channel Processing

```python
def test_effect_multichannel():
    """Test that effect handles multi-channel audio correctly."""
    # Create multi-channel waveform (2 channels)
    waveform = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],  # Left channel
        [0.5, 1.5, 2.5, 3.5]   # Right channel
    ])

    effect = MyEffect(param=value)
    output = effect(waveform)

    # Verify shape preserved
    assert output.shape == waveform.shape

    # Verify independent channel processing (if applicable)
    # Or verify cross-channel effects (if applicable)
```

#### Testing Error Handling

```python
def test_effect_invalid_parameter():
    """Test that invalid parameters raise appropriate errors."""
    # Test invalid value
    with pytest.raises(ValueError, match="Parameter must be positive"):
        MyEffect(param=-1.0)

    # Test invalid type
    with pytest.raises(TypeError):
        MyEffect(param="invalid")

    # Test assertion errors
    with pytest.raises(AssertionError):
        MyEffect(param=0.0)  # Zero not allowed
```

### Test Fixtures

Use pytest fixtures for reusable test data:

```python
import pytest

@pytest.fixture
def stereo_waveform():
    """Fixture providing a standard stereo waveform."""
    return torch.randn(2, 44100)

@pytest.fixture
def mono_waveform():
    """Fixture providing a standard mono waveform."""
    return torch.randn(1, 44100)

def test_with_fixture(stereo_waveform):
    """Test using stereo waveform fixture."""
    effect = MyEffect()
    output = effect(stereo_waveform)
    assert output.shape == stereo_waveform.shape
```

## Coverage Requirements

### Coverage Targets

The TorchFX project aims for comprehensive test coverage:

- **Line coverage**: >90% for all modules
- **Branch coverage**: >85% for control flow
- **Strategy coverage**: 100% for all built-in strategies

### Coverage Analysis

Generate coverage reports to identify untested code:

```bash
# Terminal report with missing lines
pytest tests/ --cov=src/torchfx --cov-report=term-missing

# HTML report for detailed analysis
pytest tests/ --cov=src/torchfx --cov-report=html
```

**Interpreting coverage reports**:

```
Name                           Stmts   Miss Branch BrPart  Cover   Missing
--------------------------------------------------------------------------
src/torchfx/__init__.py            8      0      0      0   100%
src/torchfx/effect.py            250     15     60      8    92%   125-130, 245
src/torchfx/filter/__base.py      80      5     20      2    91%   45-47
src/torchfx/wave.py              120      8     30      3    90%   88-92
--------------------------------------------------------------------------
TOTAL                            458     28    110     13    91%
```

- **Stmts**: Total number of statements
- **Miss**: Number of statements not executed
- **Branch**: Total number of branches
- **BrPart**: Number of partially covered branches
- **Cover**: Overall coverage percentage
- **Missing**: Line numbers not covered

## Common Test Patterns

### Testing BPM Synchronization

```python
def test_delay_bpm_synced():
    """Test BPM-synchronized delay calculation."""
    waveform = torch.randn(2, 44100)
    delay = Delay(bpm=120, delay_time="1/8", fs=44100, feedback=0.3, mix=0.2)

    # 120 BPM = 0.5 seconds per beat
    # 1/8 note = 0.25 seconds = 11025 samples at 44.1kHz
    assert delay.delay_samples == 11025
```

### Testing Strategy Pattern Extensibility

```python
def test_custom_strategy():
    """Test that custom strategies can be implemented."""
    class CustomStrategy(BaseStrategy):
        def apply(self, waveform, **kwargs):
            return waveform * 2

    effect = MyEffect(strategy=CustomStrategy())
    waveform = torch.ones(5)
    output = effect(waveform)

    assert torch.allclose(output, torch.ones(5) * 2)
```

### Testing Lazy Initialization

```python
def test_lazy_fs_inference():
    """Test lazy sample rate inference with Wave."""
    effect = MyEffect(param=value)  # No fs provided
    assert effect.fs is None

    wave = Wave(torch.randn(2, 44100), fs=44100)
    result = wave | effect

    assert effect.fs == 44100  # Automatically configured
```

## CI Testing

Tests run automatically in continuous integration via GitHub Actions:

```yaml
# .github/workflows/ci.yml
- name: Run tests
  run: |
    pytest tests/ --cov=src/torchfx --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

**CI test matrix**:
- Python versions: 3.10, 3.11, 3.12, 3.13
- Operating systems: Ubuntu, macOS, Windows
- PyTorch versions: Latest stable

## Best Practices

### Write Descriptive Test Names

```python
# ✅ GOOD: Descriptive, explains what is tested
def test_gain_amplitude_mode_doubles_waveform():
    pass

# ❌ BAD: Vague, unclear what is tested
def test_gain():
    pass
```

### Use Meaningful Assertions

```python
# ✅ GOOD: Clear assertion with context
def test_normalize_peak():
    waveform = torch.tensor([0.5, -1.0, 0.75])
    norm = Normalize(peak=0.8)
    out = norm(waveform)
    # Max absolute value should equal target peak
    assert torch.max(torch.abs(out)).item() == pytest.approx(0.8)

# ❌ BAD: Unclear what is being tested
def test_normalize():
    out = Normalize(peak=0.8)(torch.randn(10))
    assert out is not None
```

### Test Edge Cases

```python
def test_delay_zero_feedback():
    """Test delay with zero feedback produces single echo."""
    delay = Delay(delay_samples=100, feedback=0.0, taps=5)
    # Should only produce first tap, rest should be silent
    pass

def test_delay_max_feedback():
    """Test delay with maximum allowed feedback."""
    delay = Delay(delay_samples=100, feedback=0.95)  # Max allowed
    # Should produce long decay
    pass

def test_normalize_already_normalized():
    """Test normalizing already-normalized audio."""
    waveform = torch.tensor([0.5, -0.5])  # Already peak=0.5
    norm = Normalize(peak=0.5)
    out = norm(waveform)
    torch.testing.assert_close(out, waveform)  # Should be unchanged
```

### Keep Tests Independent

```python
# ✅ GOOD: Each test is self-contained
def test_gain_1():
    waveform = torch.ones(5)
    gain = Gain(gain=2.0)
    assert torch.allclose(gain(waveform), torch.ones(5) * 2.0)

def test_gain_2():
    waveform = torch.zeros(5)
    gain = Gain(gain=3.0)
    assert torch.allclose(gain(waveform), torch.zeros(5))

# ❌ BAD: Tests depend on order
global_waveform = None

def test_setup():
    global global_waveform
    global_waveform = torch.ones(5)

def test_gain():  # Depends on test_setup
    gain = Gain(gain=2.0)
    assert torch.allclose(gain(global_waveform), torch.ones(5) * 2.0)
```

## Related Resources

- {doc}`/guides/developer/project-structure` - Project organization
- {doc}`/guides/developer/benchmarking` - Performance benchmarking
- {doc}`/guides/developer/style_guide` - Coding standards
- [pytest documentation](https://docs.pytest.org/) - Official pytest docs
- [Coverage.py documentation](https://coverage.readthedocs.io/) - Coverage tool docs
