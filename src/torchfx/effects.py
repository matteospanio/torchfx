"""Base class for all effects."""

import abc
from torch import nn, Tensor
from typing_extensions import override


class FX(nn.Module, abc.ABC):
    """Abstract base class for all effects.
    This class defines the interface for all effects in the library. It inherits from
    `torch.nn.Module` and provides the basic structure for implementing effects.
    """
    @abc.abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @override
    @abc.abstractmethod
    def forward(self, x: Tensor) -> Tensor: ...
