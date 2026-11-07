"""Trainable effects (DDSP).

Are learnable filters applicable to learnable *effects*? Partly — and the distinction
is worth stating precisely:

* **Pointwise / FIR-like effects** (gain, mix, static EQ) are differentiable by
  construction. Making them trainable just means holding their parameters as
  ``nn.Parameter`` and running a plain (not ``@torch.no_grad()``) forward. No custom
  backward is needed. :class:`LearnableGain` is the minimal example.

* **Filter-like effects** (anything expressible as an SOS cascade) should subclass
  :class:`torchfx.ddsp.LearnableFilter` and reuse the shared differentiable cascade.

* **Recursive / native-kernel effects** (``Delay``, ``Reverb``, the dynamics
  processors) run forward-only native kernels, exactly like the deterministic
  filters did. To train *those* you need their own analytic-backward
  ``torch.autograd.Function`` — the same pattern as
  :class:`torchfx.ddsp.BiquadFunction` — or a pure-torch differentiable
  reimplementation. They are **not** covered automatically by the filter base; each
  is a separate piece of work (see the DDSP / DiffAPF literature in PROJECT.md §13).

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

if TYPE_CHECKING:
    from torchfx.effect import Gain


class LearnableGain(nn.Module):
    """A trainable scalar gain (amplitude). Differentiable by construction.

    The simplest learnable *effect*: it multiplies the signal by a learnable scalar,
    so gradients flow with plain autograd — no custom backward required. Mirrors the
    filter side's :meth:`~torchfx.ddsp.LearnableFilter.freeze` bridge: once trained,
    :meth:`freeze` returns a deterministic :class:`torchfx.effect.Gain`.

    Parameters
    ----------
    gain : float
        Initial linear (amplitude) gain.

    Examples
    --------
    >>> import torch
    >>> from torchfx.ddsp import LearnableGain
    >>> g = LearnableGain(0.5)
    >>> g(torch.randn(1, 1024, requires_grad=True)).sum().backward()
    >>> det = g.freeze()  # deterministic torchfx.effect.Gain

    """

    def __init__(self, gain: float = 1.0) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(float(gain)))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gain

    @torch.no_grad()
    def freeze(self) -> Gain:
        """Bake the trained gain into a deterministic :class:`torchfx.effect.Gain`."""
        from torchfx.effect import Gain

        return Gain(float(self.gain), gain_type="amplitude")
