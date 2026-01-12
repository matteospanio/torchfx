"""Base classes for filter implementations in torchfx.

This module provides the foundational infrastructure for all frequency-domain filters
in torchfx. It includes the abstract base class (`AbstractFilter`) that all filters
must inherit from, and the `ParallelFilterCombination` class for combining multiple
filters in parallel configurations.

Filter Architecture
-------------------
The filter system in torchfx supports two main categories of digital filters:

- **IIR (Infinite Impulse Response)**: Recursive filters with feedback that provide
  efficient frequency response shaping using relatively few coefficients. Examples
  include Butterworth, Chebyshev, Shelving, and Peaking filters.

- **FIR (Finite Impulse Response)**: Non-recursive filters that are always stable
  and can provide linear phase response. Implemented using efficient convolution
  operations via PyTorch's `conv1d`.

All filters inherit from `AbstractFilter`, which itself inherits from the `FX` base
class, making them compatible with PyTorch's `nn.Module` system and enabling
seamless integration with the torchfx pipeline operator.

Key Features
------------
- **Lazy coefficient computation**: Coefficients are computed on first use rather than
  at initialization, allowing filters to be constructed without a sampling rate. This
  enables flexible pipeline construction where the sampling rate is inferred from the
  `Wave` object.

- **Automatic sampling frequency propagation**: When used in a pipeline with a `Wave`
  object, the sampling rate is automatically configured via the pipeline's `fs`
  attribute propagation mechanism.

- **Operator overloading for filter composition**:
  - The `+` operator creates parallel combinations (outputs summed)
  - The `|` operator (via `FX` base class) creates series chains (sequential)
  - Combinations can be arbitrarily nested: `(f1 + f2) | f3 | (f4 + f5)`

- **Multi-channel processing**: All filters support mono `[T]`, stereo/multi-channel
  `[C, T]`, and batched `[B, C, T]` audio tensors.

- **GPU compatibility**: Filters automatically adapt to the device and dtype of their
  input tensors, seamlessly working on both CPU and CUDA devices.

Filter Composition Patterns
----------------------------
**Series (Sequential) Filtering**: Use the pipe operator (`|`) to chain filters.
Each filter processes the output of the previous filter:

    wave | filter1 | filter2 | filter3

**Parallel Filtering**: Use the addition operator (`+`) to combine filters. All
filters process the same input, and their outputs are summed:

    wave | (filter1 + filter2 + filter3)

**Mixed Topology**: Combine series and parallel patterns to create complex signal
processing graphs:

    wave | (lowpass + highpass) | peaking | (notch1 + notch2)

Coefficient Computation Flow
-----------------------------
The coefficient computation follows a lazy evaluation pattern:

1. **Filter instantiation**: Parameters are stored, coefficients set to None, fs may
   be None or provided explicitly
2. **Pipeline application**: When `wave | filter` is evaluated, the Wave object sets
   filter.fs if it is None
3. **First forward pass**: The forward() method checks `_has_computed_coeff` property
4. **Coefficient computation**: If coefficients are not computed, the abstract
   `compute_coefficients()` method is called, which uses SciPy design functions
   (e.g., `butter`, `cheby1`, `firwin`) or custom algorithms
5. **Device management**: Coefficients are converted to PyTorch tensors and moved to
   match the input tensor's device and dtype
6. **Filter application**: Coefficients are applied using `torchaudio.functional.lfilter`
   for IIR filters or `torch.nn.functional.conv1d` for FIR filters

Inheritance Pattern for Custom Filters
---------------------------------------
To create a custom filter, subclass `AbstractFilter` and implement:

1. `__init__(*args, **kwargs)`: Call `super().__init__()` and initialize parameters
2. `compute_coefficients()`: Compute filter coefficients (a, b) and store them
3. `forward(x)`: Apply the filter to the input tensor (typically inherited from IIR/FIR)

See Also
--------
torchfx.filter.iir : IIR filter implementations (Butterworth, Chebyshev, Shelving, etc.)
torchfx.filter.fir : FIR filter implementations
torchfx.effect : Base FX class and effect system
ParallelFilterCombination : Class for parallel filter combination

Notes
-----
For a comprehensive guide on creating custom filters, see the "Creating Custom Filters"
tutorial (wiki/4.4 Creating Custom Filters.md).

For detailed information on parallel filter combination patterns and use cases, see
the "Parallel Filter Combination" tutorial (wiki/4.3 Parallel Filter Combination.md).

For an overview of the filter system architecture and available filter types, see
the "Filters" documentation (wiki/4 Filters.md).

For API reference of all filter classes, see wiki/8.3 torchfx.filter.md.

Examples
--------
Basic filter usage with automatic sampling rate configuration:

>>> import torchfx as fx
>>>
>>> # Create filters without specifying sampling rate
>>> lowpass = fx.filter.LoButterworth(1000, order=4)
>>> highpass = fx.filter.HiButterworth(200, order=4)
>>>
>>> # Load audio - the Wave object has an fs attribute
>>> wave = fx.Wave.from_file("audio.wav")
>>>
>>> # Apply filter using pipeline operator - fs is set automatically
>>> filtered = wave | lowpass
>>> print(filtered.fs)  # Same as wave.fs

Parallel filter combination using the + operator:

>>> # Combine two filters in parallel - outputs are summed
>>> parallel = lowpass + highpass
>>> result = wave | parallel
>>>
>>> # This is equivalent to manually summing filter outputs:
>>> manual_result = (wave | lowpass).ys + (wave | highpass).ys

Series and parallel combinations for complex topologies:

>>> # Create individual filters
>>> lo1 = fx.filter.LoButterworth(1000, order=4)
>>> hi1 = fx.filter.HiButterworth(200, order=4)
>>> peak = fx.filter.Peaking(500, Q=2.0, gain=6.0, gain_scale="octave")
>>>
>>> # Apply: (lo1 + hi1) in parallel, then peak in series
>>> # Signal flow: input -> [lo1, hi1] -> sum -> peak -> output
>>> result = wave | (lo1 + hi1) | peak

Creating a multi-band processor with parallel filters:

>>> # Design a three-band frequency splitter
>>> low_band = fx.filter.LoButterworth(200, order=4, fs=44100)
>>> mid_band = (fx.filter.HiButterworth(200, order=4, fs=44100) +
...             fx.filter.LoButterworth(2000, order=4, fs=44100))
>>> high_band = fx.filter.HiButterworth(2000, order=4, fs=44100)
>>>
>>> # Process each band independently
>>> wave = fx.Wave.from_file("audio.wav")
>>> low = wave | low_band
>>> mid = wave | mid_band
>>> high = wave | high_band

Explicit sampling frequency specification:

>>> # Filters can also be created with explicit fs
>>> notch = fx.filter.Notch(60, Q=30, fs=48000)  # Remove 60 Hz hum
>>> wave = fx.Wave.from_file("audio_48k.wav")
>>> clean = wave | notch

"""

import abc
from collections.abc import Sequence

import torch
from torch import Tensor
from typing_extensions import override

from torchfx.effect import FX


class AbstractFilter(FX, abc.ABC):
    """Base class for all filter implementations in torchfx.

    `AbstractFilter` provides the foundational interface and common infrastructure for
    implementing both IIR (Infinite Impulse Response) and FIR (Finite Impulse Response)
    filters. It inherits from `FX` (which itself inherits from `torch.nn.Module`),
    making all filters compatible with PyTorch's neural network ecosystem, device
    management, and the torchfx pipeline operator system.

    All concrete filter classes must inherit from `AbstractFilter` and implement the
    two required abstract methods: `__init__` and `compute_coefficients`.

    Key Concepts
    ------------
    **Lazy Coefficient Computation**
        Filters store their parameters (cutoff frequencies, filter order, Q factors, etc.)
        at initialization but defer the actual coefficient computation until the first
        forward pass. This allows filters to be constructed without specifying a sampling
        rate, which can be provided later through the pipeline system or set explicitly.

        The computation is triggered automatically when the filter's `forward()` method
        is called for the first time and the `_has_computed_coeff` property returns False.

    **Automatic Sampling Frequency Configuration**
        When a filter is used in a pipeline with a `Wave` object (via the `|` operator),
        the sampling frequency is automatically propagated from the `Wave` object to the
        filter if the filter's `fs` attribute is `None`. This enables a streamlined API
        where users don't need to manually specify the sampling rate:

            wave = fx.Wave.from_file("audio.wav")  # wave.fs = 44100
            filtered = wave | fx.filter.LoButterworth(1000, order=4)  # fs set automatically

    **Parallel Combination with + Operator**
        The `+` operator is overloaded to create parallel filter combinations. When two
        filters are added using `+`, a `ParallelFilterCombination` instance is created
        that applies both filters independently to the same input signal and sums their
        outputs. This enables intuitive filter topology construction:

            parallel = filter1 + filter2 + filter3
            result = wave | parallel  # All three filters process wave, outputs summed

    **Series Combination with | Operator**
        The `|` operator (inherited from `FX` base class) creates series filter chains
        where each filter processes the output of the previous filter sequentially:

            result = wave | filter1 | filter2 | filter3

    **Device and Dtype Adaptation**
        Filters automatically adapt to the device (CPU/CUDA) and dtype of their input
        tensors. Coefficients are converted and moved to match the input during the
        forward pass.

    Attributes
    ----------
    _has_computed_coeff : bool, property (read-only)
        Returns `True` if filter coefficients have been computed, `False` otherwise.
        For most filters, this checks that both `a` and `b` attributes exist and are
        not `None`. Subclasses can override this property for custom coefficient storage.

    Abstract Methods
    ----------------
    Subclasses must implement the following abstract methods:

    __init__(*args, **kwargs) : None
        Constructor that must call `super().__init__()` and initialize:
        - Filter-specific parameters (e.g., cutoff, order, Q, gain)
        - Sampling frequency attribute (typically `self.fs`)
        - Coefficient storage attributes (typically `self.a` and `self.b` set to None)

    compute_coefficients() : None
        Computes the filter's transfer function coefficients and stores them in instance
        attributes (typically `self.a` for denominator and `self.b` for numerator).

        This method is called automatically by the filter's `forward()` method when
        coefficients have not yet been computed. It typically uses SciPy signal
        processing functions (e.g., `scipy.signal.butter`, `scipy.signal.cheby1`,
        `scipy.signal.firwin`) or custom coefficient calculation algorithms.

        The method should verify that required parameters (especially `self.fs`) are
        not None before attempting coefficient computation.

    Implementing Subclasses
    ------------------------
    In addition to implementing the abstract methods, subclasses typically need to:

    1. **Define a forward() method** that:
       - Checks if coefficients have been computed (via `_has_computed_coeff`)
       - Calls `compute_coefficients()` if needed
       - Moves coefficients to the input tensor's device and dtype
       - Applies the filter using appropriate method (lfilter for IIR, conv1d for FIR)

    2. **Handle coefficient storage** using attributes like `self.a` and `self.b`

    3. **Implement device management** for moving coefficients to GPU/CPU as needed

    See Also
    --------
    ParallelFilterCombination : Combines multiple filters in parallel (outputs summed)
    torchfx.filter.iir.IIR : Base class for IIR filters with common IIR functionality
    torchfx.filter.fir.FIR : Base class for FIR filters with convolution-based filtering
    torchfx.effect.FX : Base class for all effects and filters

    Notes
    -----
    For a comprehensive guide on creating custom filters with complete working examples,
    see the "Creating Custom Filters" tutorial in wiki/4.4 Creating Custom Filters.md.

    For details on the coefficient computation pattern and the filter system architecture,
    see the "Filters" overview in wiki/4 Filters.md.

    For information on parallel filter combination patterns, including series-parallel
    mixed topologies, see wiki/4.3 Parallel Filter Combination.md.

    For complete API reference of all built-in filter classes, see the API documentation
    in wiki/8.3 torchfx.filter.md.

    Examples
    --------
    Creating a custom IIR bandpass filter by subclassing `AbstractFilter`:

    >>> import torch
    >>> from torch import Tensor
    >>> from torchfx.filter.__base import AbstractFilter
    >>> from scipy.signal import butter
    >>> from torchaudio.functional import lfilter
    >>>
    >>> class CustomBandpass(AbstractFilter):
    ...     '''Custom bandpass filter using Butterworth design.'''
    ...
    ...     def __init__(self, low_cutoff: float, high_cutoff: float,
    ...                  order: int = 4, fs: int | None = None):
    ...         super().__init__()
    ...         # Store filter parameters
    ...         self.low_cutoff = low_cutoff
    ...         self.high_cutoff = high_cutoff
    ...         self.order = order
    ...         self.fs = fs
    ...         # Initialize coefficient storage
    ...         self.a = None
    ...         self.b = None
    ...
    ...     def compute_coefficients(self):
    ...         '''Compute Butterworth bandpass coefficients.'''
    ...         if self.fs is None:
    ...             raise ValueError("Sampling frequency must be set")
    ...         # Normalize frequencies to Nyquist
    ...         nyquist = 0.5 * self.fs
    ...         low_norm = self.low_cutoff / nyquist
    ...         high_norm = self.high_cutoff / nyquist
    ...         # Use scipy to design filter
    ...         self.b, self.a = butter(self.order, [low_norm, high_norm],
    ...                                 btype='bandpass')
    ...
    ...     @torch.no_grad()
    ...     def forward(self, x: Tensor) -> Tensor:
    ...         '''Apply bandpass filter to input tensor.'''
    ...         # Compute coefficients if not already computed
    ...         if self.a is None or self.b is None:
    ...             self.compute_coefficients()
    ...         # Move coefficients to input device/dtype if needed
    ...         if not isinstance(self.a, Tensor):
    ...             self.a = torch.as_tensor(self.a, device=x.device, dtype=x.dtype)
    ...             self.b = torch.as_tensor(self.b, device=x.device, dtype=x.dtype)
    ...         # Apply IIR filter
    ...         return lfilter(x, self.a, self.b)

    Using the custom filter in a pipeline with automatic fs configuration:

    >>> import torchfx as fx
    >>> wave = fx.Wave.from_file("audio.wav")
    >>> # Create filter without fs - it will be set automatically
    >>> bandpass = CustomBandpass(100, 1000)
    >>> filtered = wave | bandpass
    >>> print(bandpass.fs)  # Now set to wave.fs

    Using the custom filter with explicit fs:

    >>> # Create filter with explicit sampling frequency
    >>> bandpass = CustomBandpass(100, 1000, fs=44100)
    >>> wave = fx.Wave.from_file("audio.wav")
    >>> filtered = wave | bandpass

    Parallel combination using the + operator:

    >>> # Create two bandpass filters
    >>> lowpass = CustomBandpass(100, 1000, fs=44100)
    >>> highpass = CustomBandpass(5000, 10000, fs=44100)
    >>>
    >>> # Combine in parallel using + operator
    >>> parallel = lowpass + highpass  # Creates ParallelFilterCombination
    >>> print(type(parallel).__name__)  # ParallelFilterCombination
    >>>
    >>> # Apply to audio
    >>> wave = fx.Wave.from_file("audio.wav")
    >>> result = wave | parallel

    Series combination using the | operator:

    >>> # Chain filters sequentially
    >>> wave = fx.Wave.from_file("audio.wav")
    >>> highpass = CustomBandpass(5000, 10000, fs=wave.fs)
    >>> lowpass = CustomBandpass(100, 1000, fs=wave.fs)
    >>> # Apply highpass, then lowpass
    >>> result = wave | highpass | lowpass

    Mixed series and parallel combinations:

    >>> # Create complex filter topology
    >>> bp1 = CustomBandpass(100, 500, fs=44100)
    >>> bp2 = CustomBandpass(500, 2000, fs=44100)
    >>> bp3 = CustomBandpass(2000, 8000, fs=44100)
    >>>
    >>> # Apply bp1 in series, then (bp2 + bp3) in parallel
    >>> wave = fx.Wave.from_file("audio.wav")
    >>> result = wave | bp1 | (bp2 + bp3)

    The coefficient computation pattern (execution flow):

    >>> # Step 1: Create filter with fs=None
    >>> filt = CustomBandpass(100, 1000)
    >>> print(filt.a, filt.b)  # None, None
    >>>
    >>> # Step 2: Apply filter via pipeline
    >>> wave = fx.Wave.from_file("audio.wav")  # wave.fs = 44100
    >>> # The pipeline sets filt.fs = wave.fs automatically
    >>>
    >>> # Step 3: First forward() call triggers coefficient computation
    >>> result = wave | filt
    >>> # - forward() checks _has_computed_coeff -> False
    >>> # - Calls compute_coefficients()
    >>> # - SciPy butter() designs filter, returns (b, a) coefficients
    >>> # - Coefficients stored in filt.a and filt.b
    >>> # - Coefficients moved to input device/dtype
    >>> # - lfilter(x, a, b) applies filter
    >>>
    >>> # Step 4: Subsequent forward() calls reuse computed coefficients
    >>> print(filt.a is not None, filt.b is not None)  # True, True

    Understanding _has_computed_coeff property:

    >>> filt = CustomBandpass(100, 1000, fs=44100)
    >>> print(filt._has_computed_coeff)  # False (a=None, b=None)
    >>> filt.compute_coefficients()
    >>> print(filt._has_computed_coeff)  # True (a and b are set)

    """

    @property
    def _has_computed_coeff(self) -> bool:
        if hasattr(self, "b") and hasattr(self, "a"):
            return self.b is not None and self.a is not None
        return True

    @abc.abstractmethod
    def __init__(self, *args, **kwargs) -> None:  # type: ignore
        super().__init__(*args, **kwargs)

    @abc.abstractmethod
    def compute_coefficients(self) -> None:
        """Compute the filter's transfer function coefficients.

        This abstract method must be implemented by all `AbstractFilter` subclasses.
        It is responsible for computing the filter's transfer function coefficients
        (typically the numerator `b` and denominator `a` coefficients) and storing
        them in instance attributes.

        The method is called automatically during the first `forward()` pass when
        the `_has_computed_coeff` property returns `False`. It should not be called
        directly by users in typical usage scenarios.

        Implementation Requirements
        ---------------------------
        Implementations of this method should:

        1. **Verify sampling frequency**: Check that `self.fs` is not `None` before
           attempting to compute coefficients. Raise a `ValueError` if it is `None`.

        2. **Design filter coefficients**: Use SciPy signal processing functions
           (e.g., `scipy.signal.butter`, `scipy.signal.cheby1`, `scipy.signal.firwin`)
           or custom algorithms to compute filter coefficients based on the filter's
           parameters (cutoff frequency, order, Q factor, etc.).

        3. **Store coefficients**: Save the computed coefficients in instance attributes
           (typically `self.a` for denominator and `self.b` for numerator).

        4. **Normalize frequencies**: For filters that use normalized frequencies
           (most SciPy functions), normalize cutoff frequencies by the Nyquist
           frequency (fs/2).

        Notes
        -----
        - For IIR filters, both `a` (denominator) and `b` (numerator) coefficients
          are typically stored as sequences or arrays.
        - For FIR filters, only `b` (numerator) coefficients are needed, with `a`
          set to `[1.0]`.
        - Coefficients are initially stored as Python sequences or NumPy arrays and
          are later converted to PyTorch tensors during the `forward()` pass.
        - This method should be deterministic - calling it multiple times with the
          same parameters should produce identical coefficients.

        Raises
        ------
        ValueError
            If required parameters (especially `fs`) are not set before coefficient
            computation is attempted.

        Examples
        --------
        Implementing `compute_coefficients` for a Butterworth lowpass filter:

        >>> from scipy.signal import butter
        >>>
        >>> def compute_coefficients(self):
        ...     '''Compute Butterworth lowpass coefficients.'''
        ...     if self.fs is None:
        ...         raise ValueError("Sampling frequency must be set")
        ...     # Normalize cutoff to Nyquist frequency
        ...     nyquist = 0.5 * self.fs
        ...     normalized_cutoff = self.cutoff / nyquist
        ...     # Design filter using SciPy
        ...     self.b, self.a = butter(self.order, normalized_cutoff, btype='low')

        Implementing `compute_coefficients` for a custom shelving filter:

        >>> import numpy as np
        >>>
        >>> def compute_coefficients(self):
        ...     '''Compute high-shelving filter coefficients using biquad formulas.'''
        ...     if self.fs is None:
        ...         raise ValueError("Sampling frequency must be set")
        ...     # Calculate angular frequency
        ...     omega = 2 * np.pi * self.cutoff / self.fs
        ...     alpha = np.sin(omega) / (2 * self.q)
        ...     A = self.gain  # Linear gain
        ...     # Compute biquad coefficients
        ...     b0 = A * ((A + 1) + (A - 1) * np.cos(omega) + 2 * np.sqrt(A) * alpha)
        ...     b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(omega))
        ...     b2 = A * ((A + 1) + (A - 1) * np.cos(omega) - 2 * np.sqrt(A) * alpha)
        ...     a0 = (A + 1) - (A - 1) * np.cos(omega) + 2 * np.sqrt(A) * alpha
        ...     a1 = 2 * ((A - 1) - (A + 1) * np.cos(omega))
        ...     a2 = (A + 1) - (A - 1) * np.cos(omega) - 2 * np.sqrt(A) * alpha
        ...     # Normalize by a0 and store
        ...     self.b = [b0/a0, b1/a0, b2/a0]
        ...     self.a = [1.0, a1/a0, a2/a0]

        Implementing `compute_coefficients` for an FIR filter:

        >>> from scipy.signal import firwin
        >>>
        >>> def compute_coefficients(self):
        ...     '''Compute FIR filter coefficients using window method.'''
        ...     if self.fs is None:
        ...         raise ValueError("Sampling frequency must be set")
        ...     # Use scipy firwin to design FIR filter
        ...     self.b = firwin(self.num_taps, self.cutoff, fs=self.fs,
        ...                     pass_zero=self.pass_zero, window=self.window)
        ...     self.a = [1.0]  # FIR filters have trivial denominator

        """
        pass

    def __add__(self, other: "AbstractFilter") -> "ParallelFilterCombination":
        """Create a parallel filter combination using the + operator.

        The `+` operator enables intuitive parallel filter combination syntax. When two
        filters are added, a `ParallelFilterCombination` instance is created that applies
        both filters independently to the same input signal and sums their outputs
        element-wise.

        This is a key feature of the torchfx filter system, allowing complex signal
        processing topologies to be constructed using simple mathematical operators.

        Parameters
        ----------
        other : AbstractFilter
            Another filter to combine in parallel with this filter. Must be an instance
            of `AbstractFilter` or a subclass.

        Returns
        -------
        ParallelFilterCombination
            A new filter combination that applies both `self` and `other` in parallel
            and sums their outputs.

        Raises
        ------
        AssertionError
            If `other` is not an instance of `AbstractFilter`.

        Notes
        -----
        **Signal Flow in Parallel Combination**:

        For two filters f1 and f2 combined as `parallel = f1 + f2`, the output is:

            output = f1(input) + f2(input)

        Both filters process the same input tensor independently, and their outputs
        are summed element-wise. This is equivalent to summing the impulse responses
        of both filters in the time domain.

        **Topology Diagram**::

                        ┌──────┐
                   ┌───→│  f1  │───┐
                   │    └──────┘   │
            input──┤               ├──→ sum ──→ output
                   │    ┌──────┐   │
                   └───→│  f2  │───┘
                        └──────┘

        **Chaining Multiple Filters**:

        The `+` operator can be chained to combine more than two filters:

            combined = f1 + f2 + f3 + f4

        This creates a nested structure of `ParallelFilterCombination` instances.

        **Mixing Series and Parallel**:

        The `+` operator can be mixed with the `|` (pipe) operator for series-parallel
        topologies:

            result = wave | (f1 + f2) | f3 | (f4 + f5)

        This applies f1 and f2 in parallel, then f3 in series, then f4 and f5 in parallel.

        **Sampling Frequency Propagation**:

        When filters are combined with `+`, the `ParallelFilterCombination` automatically
        propagates the sampling frequency to all child filters that have `fs=None`.

        See Also
        --------
        ParallelFilterCombination : The class created by this operator
        __radd__ : Right-hand addition operator (enables `filter1 + filter2`)
        __or__ : Pipe operator for series combination (inherited from FX)

        Examples
        --------
        Basic parallel combination of two filters:

        >>> import torchfx as fx
        >>>
        >>> # Create two filters
        >>> lowpass = fx.filter.LoButterworth(1000, order=4)
        >>> highpass = fx.filter.HiButterworth(200, order=4)
        >>>
        >>> # Combine in parallel using +
        >>> parallel = lowpass + highpass
        >>> print(type(parallel).__name__)  # ParallelFilterCombination
        >>>
        >>> # Apply to audio
        >>> wave = fx.Wave.from_file("audio.wav")
        >>> result = wave | parallel

        Comparing parallel combination to manual summation:

        >>> import torch
        >>>
        >>> # Parallel combination
        >>> parallel = lowpass + highpass
        >>> result1 = wave | parallel
        >>>
        >>> # Manual summation (equivalent)
        >>> result2_low = wave | lowpass
        >>> result2_high = wave | highpass
        >>> result2 = fx.Wave(result2_low.ys + result2_high.ys, wave.fs)
        >>>
        >>> # Both produce the same output
        >>> torch.allclose(result1.ys, result2.ys)  # True

        Chaining multiple filters in parallel:

        >>> # Create three bandpass filters
        >>> bp1 = fx.filter.LoButterworth(500, order=4, fs=44100)
        >>> bp2 = fx.filter.LoButterworth(2000, order=4, fs=44100)
        >>> bp3 = fx.filter.HiButterworth(2000, order=4, fs=44100)
        >>>
        >>> # Combine all three in parallel
        >>> parallel = bp1 + bp2 + bp3
        >>>
        >>> # Apply to wave
        >>> wave = fx.Wave.from_file("audio.wav")
        >>> result = wave | parallel

        Mixed series and parallel topology:

        >>> # Create filters
        >>> pre = fx.filter.HiButterworth(50, order=2, fs=44100)  # Remove DC
        >>> lo = fx.filter.LoButterworth(1000, order=4, fs=44100)
        >>> hi = fx.filter.HiButterworth(1000, order=4, fs=44100)
        >>> post = fx.filter.Peaking(500, Q=2.0, gain=6.0, gain_scale="octave", fs=44100)
        >>>
        >>> # Complex topology: pre | (lo + hi) | post
        >>> wave = fx.Wave.from_file("audio.wav")
        >>> result = wave | pre | (lo + hi) | post
        >>>
        >>> # Signal flow:
        >>> # wave -> pre (series) -> [lo, hi] (parallel) -> sum -> post (series) -> result

        Creating a crossover filter for multi-band processing:

        >>> # Design a 3-way crossover
        >>> low_band = fx.filter.LoButterworth(200, order=4, fs=44100)
        >>> mid_band = (fx.filter.HiButterworth(200, order=4, fs=44100) +
        ...             fx.filter.LoButterworth(2000, order=4, fs=44100))
        >>> high_band = fx.filter.HiButterworth(2000, order=4, fs=44100)
        >>>
        >>> # Process each band
        >>> wave = fx.Wave.from_file("audio.wav")
        >>> low = wave | low_band
        >>> mid = wave | mid_band
        >>> high = wave | high_band

        Automatic sampling frequency propagation:

        >>> # Create filters without fs
        >>> f1 = fx.filter.LoButterworth(1000, order=4)
        >>> f2 = fx.filter.HiButterworth(200, order=4)
        >>> print(f1.fs, f2.fs)  # None, None
        >>>
        >>> # Combine in parallel
        >>> parallel = f1 + f2
        >>>
        >>> # When applied in pipeline, fs is set automatically
        >>> wave = fx.Wave.from_file("audio.wav")  # wave.fs = 44100
        >>> result = wave | parallel
        >>> print(f1.fs, f2.fs)  # 44100, 44100

        """
        assert isinstance(other, AbstractFilter), "Can only add AbstractFilter instances"
        return ParallelFilterCombination(self, other)

    def __radd__(self, other: "AbstractFilter") -> "ParallelFilterCombination":
        """Create a parallel filter combination using right-hand addition.

        This method enables the `+` operator to work when the left operand is an
        `AbstractFilter` instance and the right operand is `self`. It is automatically
        called by Python when `other + self` is evaluated.

        Parameters
        ----------
        other : AbstractFilter
            Another filter to combine in parallel with this filter.

        Returns
        -------
        ParallelFilterCombination
            A new filter combination that applies both filters in parallel.

        Raises
        ------
        AssertionError
            If `other` is not an instance of `AbstractFilter`.

        See Also
        --------
        __add__ : Left-hand addition operator (main documentation)
        ParallelFilterCombination : The class created by this operator

        Notes
        -----
        This method exists to support the commutative property of filter addition:
        `f1 + f2` should produce the same result as `f2 + f1` (though the order of
        filters in the ParallelFilterCombination may differ).

        """
        assert isinstance(other, AbstractFilter), "Can only add AbstractFilter instances"
        return ParallelFilterCombination(other, self)


class ParallelFilterCombination(AbstractFilter):
    """Combine multiple filters in parallel with summed outputs.

    `ParallelFilterCombination` is a concrete implementation of `AbstractFilter` that
    manages multiple child filters and combines their outputs through element-wise
    summation. It is typically created automatically via the `+` operator rather than
    being instantiated directly.

    When a `ParallelFilterCombination` processes an input signal, each child filter
    receives the same input tensor and processes it independently. The final output is
    the element-wise sum of all individual filter outputs. This is equivalent to
    summing the impulse responses of all filters in the time domain.

    Signal Processing Topology
    ---------------------------
    For N filters combined in parallel, the signal flow is::

                    ┌──────────┐
               ┌───→│ Filter 1 │───┐
               │    └──────────┘   │
               │    ┌──────────┐   │
        input──┼───→│ Filter 2 │───┼──→ sum ──→ output
               │    └──────────┘   │
               │         ...        │
               │    ┌──────────┐   │
               └───→│ Filter N │───┘
                    └──────────┘

    Mathematically: `output = Σ(filter_i(input))` for i=1 to N

    Parameters
    ----------
    *filters : AbstractFilter
        Variable number of filters to combine in parallel. All filters receive the
        same input and their outputs are summed. Must be instances of `AbstractFilter`
        or its subclasses.
    fs : int or None, optional
        The sampling frequency in Hz. If provided, it is propagated to all child
        filters that have an `fs` attribute set to `None`. Default is None.

    Attributes
    ----------
    filters : Sequence[AbstractFilter]
        Tuple of child filters being combined. Stored as an immutable sequence.
    fs : int or None
        Sampling frequency. When set, automatically propagates to all child filters
        that have `fs=None`, ensuring consistent sampling rate across all filters.

    Properties
    ----------
    _has_computed_coeff : bool
        Returns `True` only if all child filters have computed their coefficients.
        Overrides the base `AbstractFilter` property to aggregate the state of all
        child filters.

    Methods
    -------
    compute_coefficients()
        Triggers coefficient computation for all child filters by calling their
        individual `compute_coefficients()` methods sequentially.
    forward(x)
        Applies all child filters to the input tensor `x` in parallel and returns
        the element-wise sum of their outputs.

    Notes
    -----
    **Automatic Sampling Frequency Propagation**:

    When `fs` is set on a `ParallelFilterCombination` (either at initialization or
    later via assignment), it is automatically propagated to all child filters that
    have an `fs` attribute and whose current `fs` value is `None`. This ensures that
    all filters use consistent sampling rates without requiring manual configuration.

    **Coefficient Computation**:

    The `compute_coefficients()` method delegates to each child filter's
    `compute_coefficients()` method. Coefficients are computed lazily on first use,
    following the standard AbstractFilter pattern.

    **Device and Dtype Handling**:

    Each child filter independently manages its own coefficient device placement and
    dtype conversion. The `ParallelFilterCombination` itself doesn't store coefficients,
    so it relies on child filters to handle device management correctly.

    **Performance Considerations**:

    - All filters process the input independently (no shared state between filters)
    - Outputs are collected in a list before summation, which may use extra memory
      for large numbers of filters or long signals
    - The implementation uses `torch.zeros_like()` and in-place addition for the
      final summation, which is memory efficient

    See Also
    --------
    AbstractFilter : Base class with `__add__` operator that creates this class
    AbstractFilter.__add__ : The `+` operator that creates ParallelFilterCombination
    torchfx.filter.iir.IIR : Base class for IIR filters
    torchfx.filter.fir.FIR : Base class for FIR filters

    References
    ----------
    For detailed information on parallel filter combination patterns and design
    strategies, see wiki/4.3 Parallel Filter Combination.md.

    For the general filter system architecture and coefficient computation flow,
    see wiki/4 Filters.md.

    Examples
    --------
    Creating a parallel combination using the + operator (recommended):

    >>> import torchfx as fx
    >>>
    >>> # Create two filters
    >>> lowpass = fx.filter.LoButterworth(1000, order=2)
    >>> highpass = fx.filter.HiButterworth(200, order=2)
    >>>
    >>> # Combine using + operator
    >>> combined = lowpass + highpass
    >>> print(type(combined).__name__)  # ParallelFilterCombination
    >>> print(len(combined.filters))  # 2
    >>>
    >>> # Apply to audio
    >>> wave = fx.Wave.from_file("audio.wav")
    >>> result = wave | combined

    Creating a parallel combination with explicit constructor:

    >>> # Direct instantiation (less common)
    >>> from torchfx.filter import ParallelFilterCombination
    >>> lowpass = fx.filter.LoButterworth(1000, order=2)
    >>> highpass = fx.filter.HiButterworth(200, order=2)
    >>> combined = ParallelFilterCombination(lowpass, highpass, fs=44100)

    Combining more than two filters:

    >>> # Create three bandpass filters for different frequency ranges
    >>> low = fx.filter.LoButterworth(500, order=4, fs=44100)
    >>> mid = fx.filter.LoButterworth(2000, order=4, fs=44100)
    >>> high = fx.filter.HiButterworth(2000, order=4, fs=44100)
    >>>
    >>> # Combine all three
    >>> multiband = low + mid + high
    >>> print(len(multiband.filters))  # This creates nested ParallelFilterCombination
    >>>
    >>> # Apply to audio
    >>> wave = fx.Wave.from_file("audio.wav")
    >>> result = wave | multiband

    Automatic sampling frequency propagation:

    >>> # Create filters without fs
    >>> f1 = fx.filter.LoButterworth(1000, order=4)
    >>> f2 = fx.filter.HiButterworth(200, order=4)
    >>> f3 = fx.filter.Peaking(500, Q=2.0, gain=6.0, gain_scale="octave")
    >>> print(f1.fs, f2.fs, f3.fs)  # None, None, None
    >>>
    >>> # Combine them
    >>> parallel = f1 + f2 + f3
    >>>
    >>> # Load audio and apply - fs is set automatically
    >>> wave = fx.Wave.from_file("audio.wav")  # wave.fs = 44100
    >>> result = wave | parallel
    >>> print(f1.fs, f2.fs, f3.fs)  # 44100, 44100, 44100

    Demonstrating output summation behavior:

    >>> import torch
    >>>
    >>> # Create parallel combination
    >>> lowpass = fx.filter.LoButterworth(1000, order=4, fs=44100)
    >>> highpass = fx.filter.HiButterworth(1000, order=4, fs=44100)
    >>> parallel = lowpass + highpass
    >>>
    >>> # Apply parallel combination
    >>> wave = fx.Wave.from_file("audio.wav")
    >>> result_parallel = wave | parallel
    >>>
    >>> # Manual summation (equivalent)
    >>> result_low = wave | lowpass
    >>> result_high = wave | highpass
    >>> result_manual = fx.Wave(result_low.ys + result_high.ys, wave.fs)
    >>>
    >>> # Verify they're equivalent
    >>> torch.allclose(result_parallel.ys, result_manual.ys)  # True

    Series-parallel mixed topology:

    >>> # Create filter chain with mixed series and parallel stages
    >>> # Topology: input -> highpass -> (low + mid + high) -> peaking -> output
    >>> highpass = fx.filter.HiButterworth(50, order=2, fs=44100)
    >>> low = fx.filter.LoButterworth(500, order=4, fs=44100)
    >>> mid = fx.filter.LoButterworth(2000, order=4, fs=44100)
    >>> high = fx.filter.HiButterworth(2000, order=4, fs=44100)
    >>> peaking = fx.filter.Peaking(1000, Q=2.0, gain=3.0, gain_scale="octave", fs=44100)
    >>>
    >>> # Construct the topology
    >>> wave = fx.Wave.from_file("audio.wav")
    >>> result = wave | highpass | (low + mid + high) | peaking

    Creating a crossover network:

    >>> # Design a 2-way crossover at 2000 Hz using Linkwitz-Riley filters
    >>> crossover_freq = 2000
    >>> fs = 44100
    >>>
    >>> # Create complementary filters
    >>> woofer = fx.filter.LoLinkwitzRiley(crossover_freq, order=4, fs=fs)
    >>> tweeter = fx.filter.HiLinkwitzRiley(crossover_freq, order=4, fs=fs)
    >>>
    >>> # Combine to verify flat response
    >>> crossover = woofer + tweeter
    >>>
    >>> # Apply to audio
    >>> wave = fx.Wave.from_file("audio.wav")
    >>> result = wave | crossover  # Should have flat magnitude response

    Accessing individual filters in a combination:

    >>> # Create combination
    >>> f1 = fx.filter.LoButterworth(1000, order=4, fs=44100)
    >>> f2 = fx.filter.HiButterworth(200, order=4, fs=44100)
    >>> parallel = f1 + f2
    >>>
    >>> # Access individual filters
    >>> print(parallel.filters[0] is f1)  # True
    >>> print(parallel.filters[1] is f2)  # True
    >>> print(len(parallel.filters))  # 2

    Checking coefficient computation status:

    >>> # Create filters
    >>> f1 = fx.filter.LoButterworth(1000, order=4, fs=44100)
    >>> f2 = fx.filter.HiButterworth(200, order=4, fs=44100)
    >>> parallel = f1 + f2
    >>>
    >>> # Initially, coefficients are not computed
    >>> print(parallel._has_computed_coeff)  # False
    >>>
    >>> # Manually compute coefficients
    >>> parallel.compute_coefficients()
    >>> print(parallel._has_computed_coeff)  # True
    >>> print(f1._has_computed_coeff)  # True
    >>> print(f2._has_computed_coeff)  # True

    """

    filters: Sequence[AbstractFilter]

    @property
    @override
    def _has_computed_coeff(self) -> bool:
        return all(f._has_computed_coeff for f in self.filters)

    def __init__(self, *filters: AbstractFilter, fs: int | None = None) -> None:
        super().__init__()
        self.fs = fs
        self.filters = filters
        if fs is not None:
            for f in self.filters:
                if hasattr(f, "fs") and f.fs is None:
                    f.fs = fs

    @property
    def fs(self) -> int | None:
        return self._fs

    @fs.setter
    def fs(self, value: int | None) -> None:
        self._fs = value
        if value is not None:
            for f in self.filters:
                if hasattr(f, "fs") and f.fs is None:
                    f.fs = value

    @override
    def compute_coefficients(self) -> None:
        for f in self.filters:
            f.compute_coefficients()

    @override
    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        outputs = [f.forward(x) for f in self.filters]
        results = torch.zeros_like(x)
        for t in outputs:
            results += t
        return results
