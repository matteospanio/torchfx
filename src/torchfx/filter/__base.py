import abc

from torchfx.effects import FX


class AbstractFilter(FX, abc.ABC):
    """Base class for filters."""

    @abc.abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
