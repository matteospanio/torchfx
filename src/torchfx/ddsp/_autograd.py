"""Differentiable SOS cascade: a hand-derived VJP over the native biquad kernel.

The deterministic filters run under ``@torch.no_grad()`` for speed. To train filter
coefficients (DDSP) we need gradients, but naively unrolling the IIR recursion under
autograd is ~30x slower and memory-heavy. Instead we wrap a *single* biquad in a
``torch.autograd.Function`` whose backward is the analytic adjoint — reusing the same
fast native forward kernel — and let autograd compose a cascade from K of them.

Backward (adjoint method). For ``y[n] = b0 x[n] + b1 x[n-1] + b2 x[n-2]
- a1 y[n-1] - a2 y[n-2]`` (a0 normalised to 1, so the kernel ignores ``a[0]``):

* costate ``mu`` solves the time-reversed all-pole recursion driven by ``grad_y``,
  i.e. ``mu = flip(allpole(a; flip(grad_y)))`` — one native call with ``b = [1,0,0]``;
* ``grad_x = flip(fir(b; flip(mu)))`` — one native call with ``a = [1,0,0]``;
* ``grad_b[j] = sum_n mu[n] x[n-j]``  (j = 0,1,2);
* ``grad_a = [0, -sum_n mu[n] y[n-1], -sum_n mu[n] y[n-2]]``  (a0 has no effect).

See PROJECT.md §13: RBJ cookbook, torchlpc, and "Differentiable All-Pole Filters".
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn.functional import pad

from torchfx._ops import biquad_forward


def _shift(s: Tensor, k: int) -> Tensor:
    """Return ``s[n - k]`` (causal shift, zero-padded at the front)."""
    return pad(s, (k, 0))[..., : s.shape[-1]]


class BiquadFunction(torch.autograd.Function):
    """Differentiable single biquad: native forward, hand-derived analytic backward.

    ``apply(x, b, a)`` filters ``x`` (``[..., T]``) by the biquad with numerator
    ``b = [b0, b1, b2]`` and denominator ``a = [a0, a1, a2]`` (``a0`` ignored, as in
    the native kernel). Gradients flow to ``x``, ``b`` and ``a[1:]``.
    """

    @staticmethod
    def forward(ctx: Any, x: Tensor, b: Tensor, a: Tensor) -> Tensor:
        x2 = x if x.dim() >= 2 else x.unsqueeze(0)
        with torch.no_grad():
            y2, _, _ = biquad_forward(x2, b, a, None, None)
        ctx.save_for_backward(x2, y2, b, a)
        ctx.x_ndim = x.dim()
        return y2 if x.dim() >= 2 else y2.squeeze(0)

    @staticmethod
    def backward(ctx: Any, grad_y: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x2, y2, b, a = ctx.saved_tensors
        gy2 = grad_y if grad_y.dim() >= 2 else grad_y.unsqueeze(0)
        ones = torch.tensor([1.0, 0.0, 0.0], dtype=b.dtype, device=b.device)
        with torch.no_grad():
            mu, _, _ = biquad_forward(gy2.flip(-1), ones, a, None, None)  # all-pole adjoint
            mu = mu.flip(-1)
            gx, _, _ = biquad_forward(mu.flip(-1), b, ones, None, None)  # FIR adjoint -> grad_x
            gx = gx.flip(-1)

        grad_b = torch.stack([(mu * _shift(x2, j)).sum() for j in range(3)]).to(b.dtype)
        zero = torch.zeros((), dtype=a.dtype, device=a.device)
        grad_a = torch.stack(
            [
                zero,  # a0 has no effect on the kernel output
                -(mu * _shift(y2, 1)).sum().to(a.dtype),
                -(mu * _shift(y2, 2)).sum().to(a.dtype),
            ]
        )
        grad_x = gx if ctx.x_ndim >= 2 else gx.squeeze(0)
        return grad_x.to(x2.dtype), grad_b, grad_a


def differentiable_sos_cascade(x: Tensor, sos: Tensor) -> Tensor:
    """Filter ``x`` through a ``[K, 6]`` SOS cascade, differentiably.

    Each section is a :class:`BiquadFunction`; autograd chains their per-section VJPs
    into the full cascade gradient, w.r.t. both ``x`` and the coefficients in ``sos``.

    Parameters
    ----------
    x : Tensor
        Input of shape ``[..., T]`` (e.g. ``[C, T]`` or ``[B, C, T]``).
    sos : Tensor
        Second-order sections, shape ``[K, 6]`` (``[b0, b1, b2, a0, a1, a2]`` rows).

    Returns
    -------
    Tensor
        Filtered signal, same shape as ``x``.

    Examples
    --------
    >>> import torch
    >>> from torchfx.ddsp import differentiable_sos_cascade
    >>> sos = torch.tensor([[0.5, 0.5, 0.0, 1.0, 0.0, 0.0]], requires_grad=True)
    >>> x = torch.randn(1, 64, dtype=torch.float64, requires_grad=True)
    >>> y = differentiable_sos_cascade(x, sos.double())
    >>> y.sum().backward()

    """
    y = x
    for k in range(sos.shape[0]):
        y = BiquadFunction.apply(y, sos[k, :3], sos[k, 3:])  # type: ignore[no-untyped-call]
    return y
