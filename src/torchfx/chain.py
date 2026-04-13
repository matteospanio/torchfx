"""Filter/effect chain with automatic flattening."""

from __future__ import annotations

from torch import nn


class FilterChain(nn.Sequential):
    """Flat sequence of effects/filters built by the ``|`` operator.

    Automatically flattens nested ``FilterChain`` instances so that
    ``(f1 | f2) | f3`` produces ``FilterChain(f1, f2, f3)`` rather than
    ``FilterChain(FilterChain(f1, f2), f3)``.

    When passed to ``Wave.__or__``, the chain is flattened into individual
    steps, and consecutive IIR filters are automatically fused via
    ``FusedSOSCascade`` for better performance.

    Examples
    --------
    >>> from torchfx.filter.iir import LoButterworth, HiButterworth
    >>> chain = LoButterworth(1000, order=4) | HiButterworth(100, order=2)
    >>> result = wave | chain  # consecutive IIR filters are auto-fused

    """

    def __init__(self, *modules: nn.Module) -> None:
        flat: list[nn.Module] = []
        for m in modules:
            if isinstance(m, FilterChain):
                flat.extend(m.children())
            else:
                flat.append(m)
        super().__init__(*flat)

    def __or__(self, other: nn.Module) -> FilterChain:
        if not isinstance(other, nn.Module):
            return NotImplemented
        return FilterChain(*list(self.children()), other)

    def __ror__(self, other: object) -> FilterChain:
        return NotImplemented
