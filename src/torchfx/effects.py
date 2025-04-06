import abc
from torch import nn, Tensor
from typing_extensions import override


class FX(nn.Module, abc.ABC):
    @abc.abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @override
    @abc.abstractmethod
    def forward(self, x: Tensor) -> Tensor: ...
