import abc

from torch import Tensor, nn
from typing_extensions import override


class AbstractFilter(nn.Module, abc.ABC):
    """Base class for filters."""

    @abc.abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @override
    @abc.abstractmethod
    def forward(self, x: Tensor) -> Tensor: ...
