"""Differentiable DSP (DDSP) — opt-in trainable filters for TorchFX.

The deterministic filters in :mod:`torchfx.filter` stay fast and ``@torch.no_grad()``.
This package is the *opt-in* differentiable counterpart: train a
:class:`LearnableFilter` / :class:`LearnableParametricEQ` with gradient descent, then
:meth:`~LearnableFilter.freeze` it into a deterministic
:class:`~torchfx.filter.SOSFilter` for fast inference.

The differentiable cascade (:func:`differentiable_sos_cascade`) reuses the native
forward kernel and a hand-derived analytic backward, so it avoids the ~30x slowdown
of unrolling the IIR recursion under autograd.

"""

from ._autograd import BiquadFunction, differentiable_sos_cascade
from .coeffs import (
    rbj_highpass_sos,
    rbj_lowpass_sos,
    rbj_peaking_sos,
    sos_freq_response,
)
from .effect import LearnableGain
from .filter import (
    LearnableFilter,
    LearnableHighpass,
    LearnableLowpass,
    LearnableParametricEQ,
    LearnablePeaking,
)
from .losses import multires_stft_loss

__all__ = [
    "BiquadFunction",
    "LearnableFilter",
    "LearnableGain",
    "LearnableHighpass",
    "LearnableLowpass",
    "LearnableParametricEQ",
    "LearnablePeaking",
    "differentiable_sos_cascade",
    "multires_stft_loss",
    "rbj_highpass_sos",
    "rbj_lowpass_sos",
    "rbj_peaking_sos",
    "sos_freq_response",
]
