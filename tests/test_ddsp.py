"""Tests for :mod:`torchfx.ddsp` — the opt-in differentiable filter path.

Validates the hand-derived analytic VJP of the SOS cascade against
``torch.autograd.gradcheck`` and a naive autograd-through-the-recursion reference,
checks forward parity with ``scipy.signal.sosfilt``, and exercises the learnable
filters + the ``freeze()`` bridge back to the deterministic native path.

"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.signal as sps
import torch

from torchfx.ddsp import (
    BiquadFunction,
    LearnableFilter,
    LearnableGain,
    LearnableLowpass,
    LearnableParametricEQ,
    LearnablePeaking,
    differentiable_sos_cascade,
)
from torchfx.ddsp.coeffs import rbj_lowpass_sos
from torchfx.effect import Gain
from torchfx.filter import SOSFilter


def _naive_biquad(x: torch.Tensor, b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Pure-PyTorch DF1 biquad (autograd-through-the-recursion) — the grad oracle."""
    c = x.shape[0]
    y = torch.zeros_like(x)
    xm1 = x.new_zeros(c)
    xm2 = x.new_zeros(c)
    ym1 = x.new_zeros(c)
    ym2 = x.new_zeros(c)
    out = []
    for n in range(x.shape[-1]):
        xn = x[:, n]
        yn = b[0] * xn + b[1] * xm1 + b[2] * xm2 - a[1] * ym1 - a[2] * ym2
        out.append(yn)
        xm2, xm1 = xm1, xn
        ym2, ym1 = ym1, yn
    y = torch.stack(out, dim=-1)
    return y


class TestVJP:
    def test_gradcheck_biquad(self):
        torch.manual_seed(0)
        x = torch.randn(2, 48, dtype=torch.float64, requires_grad=True)
        b = torch.tensor([0.5, 0.3, 0.1], dtype=torch.float64, requires_grad=True)
        a = torch.tensor([1.0, -0.4, 0.2], dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(BiquadFunction.apply, (x, b, a), atol=1e-6, rtol=1e-4)

    def test_gradcheck_cascade(self):
        torch.manual_seed(1)
        x = torch.randn(1, 60, dtype=torch.float64, requires_grad=True)
        sos = torch.tensor(
            [
                [0.5, 0.3, 0.1, 1.0, -0.4, 0.2],
                [0.8, -0.2, 0.05, 1.0, -0.3, 0.15],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )
        assert torch.autograd.gradcheck(differentiable_sos_cascade, (x, sos), atol=1e-6, rtol=1e-4)

    def test_a0_has_no_gradient(self):
        """The kernel normalises a0 to 1, so the output is independent of a[0]."""
        x = torch.randn(1, 32, dtype=torch.float64)
        b = torch.tensor([0.5, 0.3, 0.1], dtype=torch.float64)
        a = torch.tensor([1.0, -0.4, 0.2], dtype=torch.float64, requires_grad=True)
        y = BiquadFunction.apply(x, b, a)
        y.sum().backward()
        assert a.grad is not None
        assert a.grad[0].item() == 0.0

    def test_matches_naive_autograd_reference(self):
        """Hand-derived VJP matches autograd-through-the-recursion."""
        torch.manual_seed(2)
        b = torch.tensor([0.6, -0.2, 0.05], dtype=torch.float64)
        a = torch.tensor([1.0, -0.5, 0.25], dtype=torch.float64)

        x1 = torch.randn(2, 80, dtype=torch.float64, requires_grad=True)
        b1 = b.clone().requires_grad_(True)
        a1 = a.clone().requires_grad_(True)
        BiquadFunction.apply(x1, b1, a1).pow(2).sum().backward()

        x2 = x1.detach().clone().requires_grad_(True)
        b2 = b.clone().requires_grad_(True)
        a2 = a.clone().requires_grad_(True)
        _naive_biquad(x2, b2, a2).pow(2).sum().backward()

        torch.testing.assert_close(x1.grad, x2.grad, atol=1e-9, rtol=1e-6)
        torch.testing.assert_close(b1.grad, b2.grad, atol=1e-9, rtol=1e-6)
        # a[0] differs by construction (kernel ignores it); compare a[1:].
        torch.testing.assert_close(a1.grad[1:], a2.grad[1:], atol=1e-9, rtol=1e-6)


class TestForwardParity:
    def test_cascade_matches_scipy(self):
        torch.manual_seed(3)
        x = torch.randn(1, 1024, dtype=torch.float64)
        sos_np = sps.butter(4, 2000 / (0.5 * 44100), btype="lowpass", output="sos")
        y = differentiable_sos_cascade(x, torch.tensor(sos_np, dtype=torch.float64))
        ref = sps.sosfilt(sos_np, x.numpy(), axis=-1)
        np.testing.assert_allclose(y.numpy(), ref, atol=1e-9, rtol=1e-6)


class TestLearnableFilter:
    def test_training_reduces_loss(self):
        torch.manual_seed(4)
        fs = 48_000
        target = LearnableLowpass(cutoff=3000.0, fs=fs)
        model = LearnableLowpass(cutoff=800.0, fs=fs)
        opt = torch.optim.Adam(model.parameters(), lr=0.05)
        x = torch.randn(1, 8000)
        with torch.no_grad():
            y_t = target(x)

        first = None
        loss = torch.tensor(0.0)
        for i in range(60):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model(x), y_t)
            loss.backward()
            opt.step()
            if i == 0:
                first = loss.item()
        assert first is not None
        assert loss.item() < first
        assert model.cutoff.item() > 1500.0  # moved toward the 3 kHz target

    def test_freeze_matches_differentiable_forward(self):
        torch.manual_seed(5)
        model = LearnableLowpass(cutoff=1500.0, fs=48_000)
        x = torch.randn(1, 4000)
        det = model.freeze()
        assert isinstance(det, SOSFilter)
        with torch.no_grad():
            torch.testing.assert_close(det(x), model(x), atol=1e-5, rtol=1e-4)

    def test_concrete_subclasses_are_differentiable(self):
        x = torch.randn(1, 2048, requires_grad=True)
        for filt in (
            LearnableLowpass(1000.0, fs=48_000),
            LearnablePeaking(2000.0, q=2.0, gain_db=6.0, fs=48_000),
        ):
            y = filt(x)
            assert y.shape == x.shape
            grads = torch.autograd.grad(y.sum(), list(filt.parameters()), retain_graph=True)
            assert all(g is not None and torch.isfinite(g).all() for g in grads)

    def test_eq_forward_and_freeze(self):
        torch.manual_seed(6)
        eq = LearnableParametricEQ(n_bands=6, fs=48_000)
        y = eq(torch.randn(1, 4096))
        assert y.shape == (1, 4096)
        assert isinstance(eq.freeze(), SOSFilter)

    def test_abstract_base_cannot_instantiate(self):
        with pytest.raises(TypeError):
            LearnableFilter(fs=48_000)  # type: ignore[abstract]

    def test_custom_user_subclass(self):
        """A user can extend LearnableFilter by implementing sos()."""

        class TwoPole(LearnableFilter):
            def __init__(self, cutoff: float, fs: int):
                super().__init__(fs)
                self.log_cutoff = torch.nn.Parameter(torch.tensor(float(cutoff)).log())

            def sos(self) -> torch.Tensor:
                q = torch.tensor(0.707, dtype=self.log_cutoff.dtype)
                return rbj_lowpass_sos(self.log_cutoff.exp(), q, self._require_fs())

        f = TwoPole(1000.0, fs=48_000)
        x = torch.randn(1, 1024, requires_grad=True)
        f(x).sum().backward()
        assert f.log_cutoff.grad is not None
        assert isinstance(f.freeze(), SOSFilter)


class TestLearnableEffect:
    def test_gain_trains_and_freezes(self):
        torch.manual_seed(7)
        g = LearnableGain(0.1)
        x = torch.randn(1, 2000)
        target = 0.7 * x
        opt = torch.optim.Adam(g.parameters(), lr=0.1)
        first = None
        loss = torch.tensor(0.0)
        for i in range(100):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(g(x), target)
            loss.backward()
            opt.step()
            if i == 0:
                first = loss.item()
        assert first is not None and loss.item() < first
        assert g.gain.item() == pytest.approx(0.7, abs=0.02)

        det = g.freeze()
        assert isinstance(det, Gain)
        with torch.no_grad():
            torch.testing.assert_close(det(x), g(x), atol=1e-6, rtol=1e-5)
