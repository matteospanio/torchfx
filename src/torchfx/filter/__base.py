import abc

from torch import conv1d
from typing_extensions import Self

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
        if hasattr(self, "b"):
            return self.b is not None
        return True

    @abc.abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abc.abstractmethod
    def compute_coefficients(self) -> None:
        """Compute the filter coefficients."""
        pass

    def combine(self, other: Self) -> Self:
        if not isinstance(other, AbstractFilter):
            raise TypeError(
                f"Unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'"
            )
        if not self._has_computed_coeff:
            self.compute_coefficients()
        if not other._has_computed_coeff:
            other.compute_coefficients()
        b = conv1d(self.b, other.b)
        a = conv1d(self.a, other.a)
        new_filter = self.__class__(b=b, a=a)
        return new_filter
