import abc

from torchfx.effects import FX


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
