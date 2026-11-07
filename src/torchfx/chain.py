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

    .. versionadded:: 0.5.2

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

    def compile(self, fs: int) -> FilterChain:  # type: ignore[override]
        """Eagerly design every node's coefficients for ``fs`` (see ``FX.compile``).

        Overrides ``nn.Module.compile`` with coefficient-design semantics; use the
        ``torch.compile(chain)`` function for graph compilation. Returns ``self``.

        """
        from torchfx._config import apply_fs

        apply_fs(self, fs)
        return self

    def freeze(self, fs: int | None = None) -> FilterChain:
        """Freeze coefficients for export/deployment (see ``FX.freeze``).

        Returns ``self``.

        """
        from torchfx._config import freeze_fx

        freeze_fx(self, fs)
        return self

    def summary(self) -> str:
        """Return a human-readable view of the chain topology and SOS fusion.

        Lists the series nodes (with any parallel sub-combinations) and, when the
        chain has been :meth:`compile`-d, the fused execution plan that
        materialization/realtime would run.

        """
        from torchfx.filter.fused import build_fused_plan

        nodes = list(self.children())
        lines = [f"FilterChain ({len(nodes)} node(s)):"]
        for i, m in enumerate(nodes):
            lines.append(f"  [{i}] {type(m).__name__}")
            for j, f in enumerate(getattr(m, "filters", []) or []):
                lines.append(f"        + ({j}) {type(f).__name__}")
        try:
            plan = build_fused_plan(nodes)
        except Exception:
            lines.append("fused plan: unavailable (call .compile(fs) first)")
            return "\n".join(lines)
        lines.append(f"fused plan ({len(plan)} stage(s)):")
        for i, m in enumerate(plan):
            name = type(m).__name__
            sections = getattr(m, "_num_sections", None)
            if sections is not None:
                name += f" [{sections} SOS section(s)]"
            lines.append(f"  [{i}] {name}")
        return "\n".join(lines)
