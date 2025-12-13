import abc
from collections.abc import Sequence

import torch
from torch import Tensor
from typing_extensions import override

from torchfx.effect import FX


class AbstractFilter(FX, abc.ABC):
    """Base class for filters.

    This class provides the basic structure for implementing filters. It inherits from
    `FX`. It provides the method `compute_coefficients` to compute the filter coefficients.

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
        """Compute the filter coefficients."""
        pass

    def __add__(self, other: "AbstractFilter") -> "ParallelFilterCombination":
        assert isinstance(other, AbstractFilter), "Can only add AbstractFilter instances"
        return ParallelFilterCombination(self, other)

    def __radd__(self, other: "AbstractFilter") -> "ParallelFilterCombination":
        assert isinstance(other, AbstractFilter), "Can only add AbstractFilter instances"
        return ParallelFilterCombination(other, self)


class ParallelFilterCombination(AbstractFilter):
    """Combine multiple filters in parallel.

    The output is the sum of the outputs of each filter.

    Parameters
    ----------
    *filters : AbstractFilter
        The filters to combine in parallel.
    fs : int, optional
        The sampling frequency of the input signal. If provided, it will be set to each
        filter that has an `fs` attribute.

    Examples
    --------
    >>> from torchfx import filter as fx_filter
    >>> lowpass = fx_filter.LoButterworth(1000, order=2)
    >>> highpass = fx_filter.HiButterworth(200, order=2)
    >>> combined_filter = lowpass + highpass
    >>> wave = fx.Wave.from_file("path/to/file.wav")
    >>> result = wave | combined_filter

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
