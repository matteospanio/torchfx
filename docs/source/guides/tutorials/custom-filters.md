# Creating Custom Filters

Learn how to build your own digital filters by extending TorchFX's {class}`~torchfx.filter.AbstractFilter` base class. This tutorial covers filter design patterns, coefficient computation, and integration with the TorchFX pipeline system.

## Overview

Custom filters in TorchFX inherit from {class}`~torchfx.filter.AbstractFilter`, which provides the foundation for:

- **Pipeline integration**: Automatic support for the `|` operator
- **Parallel combination**: Built-in `+` operator for filter banks
- **PyTorch compatibility**: Inherits from {class}`torch.nn.Module`
- **Device management**: Automatic GPU/CPU handling
- **Sample rate configuration**: Automatic `fs` propagation from {class}`~torchfx.Wave`

```{mermaid}
classDiagram
    class Module {
        <<PyTorch>>
        +forward(x)*
        +to(device)
        +parameters()
    }

    class FX {
        <<abstract>>
        +forward(x) Tensor*
    }

    class AbstractFilter {
        <<abstract>>
        +compute_coefficients()*
        +__add__(other) ParallelFilterCombination
        +_has_computed_coeff bool
    }

    class IIR {
        <<abstract>>
        +fs int|None
        +a Tensor|None
        +b Tensor|None
        +move_coeff(device, dtype)
        +forward(x) Tensor
    }

    class CustomFilter {
        +__init__(...)
        +compute_coefficients()
        +forward(x) Tensor
    }

    Module <|-- FX
    FX <|-- AbstractFilter
    AbstractFilter <|-- IIR
    AbstractFilter <|-- CustomFilter

    note for Module "PyTorch base class"
    note for AbstractFilter "Adds filter-specific features"
    note for CustomFilter "Your implementation"
```

## Required Methods

Every custom filter must implement these three methods:

### 1. `__init__` - Initialization

Initialize filter parameters and call parent constructor:

```python
from torchfx.filter import AbstractFilter

class CustomFilter(AbstractFilter):
    def __init__(self, param1: float, param2: int, fs: int | None = None):
        super().__init__()  # REQUIRED: Initialize parent class

        # Store filter parameters
        self.param1 = param1
        self.param2 = param2
        self.fs = fs

        # Initialize coefficient storage
        self.a = None
        self.b = None
```

**Key points**:
- Always call `super().__init__()` first
- Accept `fs` parameter (can be `None`)
- Initialize coefficient attributes to `None`
- Validate parameters if needed

### 2. `compute_coefficients` - Filter Design

Compute filter coefficients based on parameters:

```python
def compute_coefficients(self) -> None:
    """Compute filter coefficients."""
    # Verify fs is set
    if self.fs is None:
        raise ValueError("Sample rate must be set before computing coefficients")

    # Design filter (example using scipy)
    from scipy.signal import butter

    # Normalize frequency to Nyquist
    nyquist = 0.5 * self.fs
    normalized_freq = self.cutoff / nyquist

    # Compute coefficients
    self.b, self.a = butter(self.order, normalized_freq, btype='low')
```

**Key points**:
- Check that `fs` is not `None`
- Use SciPy or custom formulas
- Set `self.a` and `self.b` attributes
- Only computed once (cached automatically)

### 3. `forward` - Apply Filter

Process audio through the filter:

```python
import torch
from torch import Tensor
from torchaudio.functional import lfilter

@torch.no_grad()
def forward(self, x: Tensor) -> Tensor:
    """Apply filter to audio tensor."""
    # Compute coefficients if not already done
    if self.a is None or self.b is None:
        self.compute_coefficients()

    # Convert to tensors if needed (move to correct device)
    if not isinstance(self.a, Tensor):
        self.a = torch.as_tensor(self.a, device=x.device, dtype=x.dtype)
        self.b = torch.as_tensor(self.b, device=x.device, dtype=x.dtype)

    # Apply IIR filter
    return lfilter(x, self.a, self.b)
```

**Key points**:
- Use `@torch.no_grad()` for efficiency
- Lazy coefficient computation
- Handle device/dtype conversion
- Use {func}`torchaudio.functional.lfilter` for IIR filters

## Complete Example: Custom Bandpass Filter

Here's a complete, working example of a custom bandpass filter:

```python
import numpy as np
import torch
from torch import Tensor
from scipy.signal import butter
from torchaudio.functional import lfilter

from torchfx.filter import AbstractFilter

class CustomBandpass(AbstractFilter):
    """Custom bandpass filter using Butterworth design.

    Parameters
    ----------
    low_cutoff : float
        Lower cutoff frequency in Hz
    high_cutoff : float
        Upper cutoff frequency in Hz
    order : int, optional
        Filter order (default: 4)
    fs : int, optional
        Sample rate in Hz (can be set later)

    Examples
    --------
    >>> import torchfx as fx
    >>> wave = fx.Wave.from_file("audio.wav")
    >>> bandpass = CustomBandpass(low_cutoff=200, high_cutoff=2000, order=4)
    >>> filtered = wave | bandpass
    """

    def __init__(
        self,
        low_cutoff: float,
        high_cutoff: float,
        order: int = 4,
        fs: int | None = None,
    ):
        super().__init__()

        # Validate parameters
        assert low_cutoff > 0, "Low cutoff must be positive"
        assert high_cutoff > low_cutoff, "High cutoff must be > low cutoff"
        assert order > 0, "Order must be positive"

        # Store parameters
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        self.order = order
        self.fs = fs

        # Initialize coefficients
        self.a = None
        self.b = None

    def compute_coefficients(self) -> None:
        """Compute Butterworth bandpass coefficients."""
        if self.fs is None:
            raise ValueError("Sample rate must be set before computing coefficients")

        # Normalize frequencies to Nyquist frequency
        nyquist = 0.5 * self.fs
        low_norm = self.low_cutoff / nyquist
        high_norm = self.high_cutoff / nyquist

        # Validate normalized frequencies
        if not (0 < low_norm < 1 and 0 < high_norm < 1):
            raise ValueError(
                f"Cutoff frequencies must be between 0 and Nyquist ({nyquist} Hz)"
            )

        # Design Butterworth bandpass filter
        self.b, self.a = butter(
            self.order,
            [low_norm, high_norm],
            btype='bandpass'
        )

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """Apply bandpass filter to input tensor.

        Parameters
        ----------
        x : Tensor
            Input audio tensor of shape (channels, samples)

        Returns
        -------
        Tensor
            Filtered audio tensor
        """
        # Lazy coefficient computation
        if self.a is None or self.b is None:
            self.compute_coefficients()

        # Convert coefficients to tensors (match input device/dtype)
        if not isinstance(self.a, Tensor):
            self.a = torch.as_tensor(self.a, device=x.device, dtype=x.dtype)
            self.b = torch.as_tensor(self.b, device=x.device, dtype=x.dtype)

        # Apply filter
        return lfilter(x, self.a, self.b)
```

### Using the Custom Filter

```python
import torchfx as fx

# Load audio
wave = fx.Wave.from_file("audio.wav")

# Create and apply filter (fs auto-configured from wave)
bandpass = CustomBandpass(low_cutoff=200, high_cutoff=2000, order=4)
filtered = wave | bandpass

# Save result
filtered.save("bandpass_filtered.wav")

# Can also chain with other filters
from torchfx.filter import iir

processed = (
    wave
    | iir.HiButterworth(cutoff=80, order=2)  # Remove rumble
    | bandpass                                # Bandpass 200-2000 Hz
    | fx.effect.Normalize()                   # Normalize
)
```

## Filter Design Patterns

### Pattern 1: SciPy-Based Filters

Use SciPy's signal processing functions for coefficient design:

```python
from scipy.signal import butter, cheby1, cheby2, ellip, iirnotch, iirpeak

class CustomLowpass(AbstractFilter):
    def compute_coefficients(self) -> None:
        nyquist = 0.5 * self.fs
        norm_freq = self.cutoff / nyquist

        # Choose a design function
        self.b, self.a = butter(self.order, norm_freq, btype='low')
        # self.b, self.a = cheby1(self.order, self.ripple, norm_freq, btype='low')
        # self.b, self.a = ellip(self.order, self.ripple, self.atten, norm_freq, btype='low')
```

**Available SciPy filters**:
- {func}`scipy.signal.butter` - Butterworth (maximally flat passband)
- {func}`scipy.signal.cheby1` - Chebyshev Type I (ripple in passband)
- {func}`scipy.signal.cheby2` - Chebyshev Type II (ripple in stopband)
- {func}`scipy.signal.ellip` - Elliptic (ripple in both bands)
- {func}`scipy.signal.iirpeak` - Peaking filter
- {func}`scipy.signal.iirnotch` - Notch filter

```{seealso}
[SciPy Signal Processing](https://docs.scipy.org/doc/scipy/reference/signal.html) - Full SciPy signal module documentation
```

### Pattern 2: Biquad Formulas

Use direct biquad formulas for second-order sections:

```python
class CustomPeakingEQ(AbstractFilter):
    """Peaking EQ filter using biquad formulas."""

    def __init__(self, freq: float, gain_db: float, q: float = 1.0, fs: int | None = None):
        super().__init__()
        self.freq = freq
        self.gain_db = gain_db
        self.q = q
        self.fs = fs
        self.a = None
        self.b = None

    @property
    def _omega(self) -> float:
        """Angular frequency."""
        return 2 * np.pi * self.freq / self.fs

    @property
    def _alpha(self) -> float:
        """Biquad alpha parameter."""
        return np.sin(self._omega) / (2 * self.q)

    def compute_coefficients(self) -> None:
        if self.fs is None:
            raise ValueError("Sample rate must be set")

        A = 10 ** (self.gain_db / 40)  # Linear gain
        omega = self._omega
        alpha = self._alpha
        cos_omega = np.cos(omega)

        # Biquad coefficients for peaking EQ
        b0 = 1 + alpha * A
        b1 = -2 * cos_omega
        b2 = 1 - alpha * A

        a0 = 1 + alpha / A
        a1 = -2 * cos_omega
        a2 = 1 - alpha / A

        # Normalize by a0
        self.b = [b0 / a0, b1 / a0, b2 / a0]
        self.a = [1.0, a1 / a0, a2 / a0]
```

**Biquad transfer function**:

$$
H(z) = \frac{b_0 + b_1 z^{-1} + b_2 z^{-2}}{a_0 + a_1 z^{-1} + a_2 z^{-2}}
$$

Normalized form (dividing by $a_0$):

$$
H(z) = \frac{b_0' + b_1' z^{-1} + b_2' z^{-2}}{1 + a_1' z^{-1} + a_2' z^{-2}}
$$

```{seealso}
[Digital Biquad Filter on Wikipedia](https://en.wikipedia.org/wiki/Digital_biquad_filter) - Biquad filter theory
```

### Pattern 3: Cascaded Filters

Chain multiple filter stages by convolving coefficients:

```python
class LinkwitzRiley(AbstractFilter):
    """Linkwitz-Riley crossover filter (cascaded Butterworth)."""

    def __init__(self, cutoff: float, order: int = 4, btype: str = 'low', fs: int | None = None):
        super().__init__()
        assert order % 2 == 0, "Linkwitz-Riley order must be even"

        self.cutoff = cutoff
        self.order = order
        self.btype = btype
        self.fs = fs
        self.a = None
        self.b = None

    def compute_coefficients(self) -> None:
        if self.fs is None:
            raise ValueError("Sample rate must be set")

        # Linkwitz-Riley is two cascaded Butterworth filters
        butter_order = self.order // 2

        # Get base Butterworth coefficients
        b_butter, a_butter = butter(
            butter_order,
            self.cutoff / (0.5 * self.fs),
            btype=self.btype
        )

        # Cascade by convolving coefficients
        self.b = np.convolve(b_butter, b_butter)
        self.a = np.convolve(a_butter, a_butter)
```

**Cascading filters**: Convolving filter coefficients is equivalent to cascading filters in series.

### Pattern 4: Custom Coefficient Computation

Implement your own filter design algorithms:

```python
class CustomResonator(AbstractFilter):
    """Resonant filter with custom coefficient computation."""

    def __init__(self, freq: float, resonance: float = 0.5, fs: int | None = None):
        super().__init__()
        self.freq = freq
        self.resonance = np.clip(resonance, 0.0, 0.99)  # Stability constraint
        self.fs = fs
        self.a = None
        self.b = None

    def compute_coefficients(self) -> None:
        if self.fs is None:
            raise ValueError("Sample rate must be set")

        # Normalized frequency
        omega = 2.0 * np.pi * self.freq / self.fs

        # Quality factor from resonance parameter
        Q = 1.0 / (1.0 - self.resonance)

        # Compute coefficients using resonator formulas
        alpha = np.sin(omega) / (2.0 * Q)
        cos_omega = np.cos(omega)

        # Resonant lowpass coefficients
        b0 = (1.0 - cos_omega) / 2.0
        b1 = 1.0 - cos_omega
        b2 = (1.0 - cos_omega) / 2.0

        a0 = 1.0 + alpha
        a1 = -2.0 * cos_omega
        a2 = 1.0 - alpha

        # Normalize by a0
        self.b = [b0 / a0, b1 / a0, b2 / a0]
        self.a = [1.0, a1 / a0, a2 / a0]
```

## Pipeline Integration

Custom filters automatically work with TorchFX's pipeline system:

### Series Combination (Pipe Operator)

```python
import torchfx as fx

wave = fx.Wave.from_file("audio.wav")

# Chain custom filters in series
custom_bp = CustomBandpass(200, 2000, order=4)
custom_res = CustomResonator(1000, resonance=0.7)

processed = wave | custom_bp | custom_res
```

```{mermaid}
sequenceDiagram
    participant Wave
    participant CustomBandpass
    participant CustomResonator

    Wave->>CustomBandpass: wave | custom_bp
    Note over CustomBandpass: fs auto-configured
    CustomBandpass->>CustomBandpass: compute_coefficients()
    CustomBandpass->>CustomBandpass: forward(wave.ys)
    CustomBandpass->>Wave: Return new Wave

    Wave->>CustomResonator: result | custom_res
    Note over CustomResonator: fs auto-configured
    CustomResonator->>CustomResonator: compute_coefficients()
    CustomResonator->>CustomResonator: forward(result.ys)
    CustomResonator->>Wave: Return final Wave
```

### Parallel Combination (Addition Operator)

```python
# Parallel combination (sum outputs)
bandpass1 = CustomBandpass(200, 500)
bandpass2 = CustomBandpass(1000, 2000)

parallel = bandpass1 + bandpass2  # Creates ParallelFilterCombination
processed = wave | parallel
```

The `+` operator is inherited from {class}`~torchfx.filter.AbstractFilter` and automatically creates a {class}`~torchfx.filter.ParallelFilterCombination`.

```{seealso}
{doc}`series-parallel-filters` - Detailed guide on combining filters
```

## Device and Dtype Management

### Automatic Device Handling

Filters automatically handle CPU/GPU transfer:

```python
import torchfx as fx
import torch

wave = fx.Wave.from_file("audio.wav")

# Move to GPU
if torch.cuda.is_available():
    wave = wave.to("cuda")

    # Filter runs on GPU automatically
    filtered = wave | CustomBandpass(200, 2000)

    # Move back to CPU for saving
    filtered.to("cpu").save("output.wav")
```

### Helper Method for Coefficient Transfer

Include a helper method for moving coefficients:

```python
class CustomFilter(AbstractFilter):
    def move_coeff(self, device: torch.device, dtype: torch.dtype) -> None:
        """Move filter coefficients to specified device and dtype."""
        self.a = torch.as_tensor(self.a, device=device, dtype=dtype)
        self.b = torch.as_tensor(self.b, device=device, dtype=dtype)

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        if self.a is None or self.b is None:
            self.compute_coefficients()

        # Use helper method for device transfer
        if not isinstance(self.a, Tensor):
            self.move_coeff(x.device, x.dtype)

        return lfilter(x, self.a, self.b)
```

## Testing and Validation

### Coefficient Stability Check

For IIR filters, ensure poles are inside the unit circle:

```python
def compute_coefficients(self) -> None:
    # ... compute coefficients ...

    # Validate stability
    roots = np.roots(self.a)
    if np.any(np.abs(roots) >= 1.0):
        raise ValueError(
            f"Filter is unstable! Poles outside unit circle: {roots[np.abs(roots) >= 1.0]}"
        )
```

### Frequency Response Visualization

Plot the filter's frequency response:

```python
def plot_response(self) -> None:
    """Plot magnitude and phase response."""
    from scipy.signal import freqz
    import matplotlib.pyplot as plt

    if self.a is None or self.b is None:
        self.compute_coefficients()

    # Compute frequency response
    w, h = freqz(self.b, self.a, worN=2000, fs=self.fs)

    # Plot magnitude
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(w, 20 * np.log10(np.abs(h)))
    plt.title('Frequency Response')
    plt.ylabel('Magnitude [dB]')
    plt.xlabel('Frequency [Hz]')
    plt.grid(True)

    # Plot phase
    plt.subplot(2, 1, 2)
    plt.plot(w, np.angle(h))
    plt.ylabel('Phase [radians]')
    plt.xlabel('Frequency [Hz]')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Usage
bandpass = CustomBandpass(200, 2000, fs=44100)
bandpass.plot_response()
```

### Unit Test Example

```python
def test_custom_bandpass():
    """Test custom bandpass filter."""
    import torch
    import torchfx as fx

    # Create test signal (440 Hz + 880 Hz tones)
    fs = 44100
    duration = 1.0
    t = torch.linspace(0, duration, int(fs * duration))
    signal = (
        torch.sin(2 * torch.pi * 440 * t) +
        torch.sin(2 * torch.pi * 880 * t)
    )

    wave = fx.Wave(signal.unsqueeze(0), fs)

    # Create and apply filter (passes 440 Hz, attenuates 880 Hz)
    bandpass = CustomBandpass(low_cutoff=300, high_cutoff=600, fs=fs)
    filtered = wave | bandpass

    # Verify output shape
    assert filtered.ys.shape == wave.ys.shape

    # Verify filtering occurred
    assert not torch.allclose(filtered.ys, wave.ys)

    # Verify coefficients computed
    assert bandpass.a is not None
    assert bandpass.b is not None
    assert isinstance(bandpass.a, Tensor)
    assert isinstance(bandpass.b, Tensor)

    print("✓ All tests passed!")
```

## Best Practices

### Parameter Validation

Validate parameters in `__init__`:

```python
def __init__(self, cutoff: float, q: float = 1.0, fs: int | None = None):
    super().__init__()

    # Validate parameters
    assert cutoff > 0, "Cutoff frequency must be positive"
    assert q > 0, "Q factor must be positive"
    if fs is not None:
        assert fs > 0, "Sample rate must be positive"
        assert cutoff < fs / 2, "Cutoff must be below Nyquist frequency"

    self.cutoff = cutoff
    self.q = q
    self.fs = fs
```

### Use Properties for Computed Values

```python
@property
def _omega(self) -> float:
    """Normalized angular frequency."""
    return 2 * np.pi * self.cutoff / self.fs

@property
def _q_factor(self) -> float:
    """Quality factor from bandwidth."""
    return self.cutoff / self.bandwidth
```

### Document Thoroughly

Include comprehensive docstrings:

```python
class CustomFilter(AbstractFilter):
    """One-line summary.

    Longer description explaining what the filter does,
    its characteristics, and when to use it.

    Parameters
    ----------
    param1 : type
        Description of param1
    param2 : type, optional
        Description of param2 (default: value)
    fs : int, optional
        Sample rate in Hz (default: None, auto-configured from Wave)

    Attributes
    ----------
    a : Tensor | None
        Denominator coefficients
    b : Tensor | None
        Numerator coefficients

    Examples
    --------
    >>> import torchfx as fx
    >>> wave = fx.Wave.from_file("audio.wav")
    >>> filt = CustomFilter(param1=value, param2=value)
    >>> filtered = wave | filt

    Notes
    -----
    Additional technical notes, references, or mathematical formulations.

    References
    ----------
    .. [1] Author, "Title," Journal, Year.

    See Also
    --------
    RelatedFilter : Related filter implementation
    """
```

### Handle Edge Cases

```python
@torch.no_grad()
def forward(self, x: Tensor) -> Tensor:
    # Handle empty input
    if x.numel() == 0:
        return x

    # Handle very short signals
    min_length = self.order * 3
    if x.shape[-1] < min_length:
        import warnings
        warnings.warn(
            f"Signal length ({x.shape[-1]}) is shorter than recommended "
            f"({min_length} samples). Results may be unreliable."
        )

    # Normal processing
    # ...
```

## Advanced Topics

### State-ful Filters

For real-time processing, maintain filter state:

```python
class StatefulFilter(AbstractFilter):
    """Filter that maintains state for real-time processing."""

    def __init__(self, cutoff: float, fs: int | None = None):
        super().__init__()
        self.cutoff = cutoff
        self.fs = fs
        self.a = None
        self.b = None
        self.zi = None  # Initial conditions

    def reset_state(self) -> None:
        """Reset filter state to zero."""
        if self.a is not None and self.b is not None:
            from scipy.signal import lfilter_zi
            self.zi = lfilter_zi(self.b, self.a)

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        # ... coefficient computation ...

        # Use lfilter with initial conditions
        from torchaudio.functional import lfilter
        result = lfilter(x, self.a, self.b, zi=self.zi)

        # Update state for next call
        # (zi updated by lfilter in-place)

        return result
```

### Differentiable Filters

Enable gradients for learned filter parameters:

```python
class LearnableFilter(AbstractFilter):
    """Filter with learnable cutoff frequency."""

    def __init__(self, initial_cutoff: float = 1000, fs: int | None = None):
        super().__init__()
        self.fs = fs

        # Learnable parameter (log-scale for numerical stability)
        self.log_cutoff = torch.nn.Parameter(
            torch.log(torch.tensor([initial_cutoff]))
        )

    @property
    def cutoff(self) -> float:
        """Current cutoff frequency."""
        return torch.exp(self.log_cutoff).item()

    def forward(self, x: Tensor) -> Tensor:
        # Recompute coefficients each forward pass (cutoff may have changed)
        self.compute_coefficients()

        # DON'T use @torch.no_grad() - we need gradients!
        return lfilter(x, self.a, self.b)
```

## Related Concepts

- {doc}`/guides/core-concepts/fx` - FX base class architecture
- {doc}`series-parallel-filters` - Combining filters
- {doc}`/guides/advanced/pytorch-integration` - PyTorch integration
- {doc}`filters-design` - Filter design theory

## External Resources

- [SciPy Signal Processing](https://docs.scipy.org/doc/scipy/reference/signal.html) - SciPy filter design functions
- [Digital Filter Design on Wikipedia](https://en.wikipedia.org/wiki/Digital_filter) - Filter theory
- [IIR Filter Design](https://ccrma.stanford.edu/~jos/filters/) - Julius O. Smith's filter design book
- [Audio EQ Cookbook](https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html) - Biquad filter formulas

## References

```{bibliography}
:filter: docname in docnames
:style: alpha
```
