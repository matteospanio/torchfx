#!/usr/bin/env python3
"""Differentiable DSP: a learnable parametric EQ built on ``torchfx.ddsp``.

This example trains a cascade of RBJ peaking-EQ biquads — the *same* second-order
sections the native TorchFX kernels execute — by gradient descent. Filtering happens
in the **time domain** through :func:`torchfx.ddsp.differentiable_sos_cascade`, which
runs the fast native forward kernel and a hand-derived analytic backward (no autograd
unrolling of the IIR recursion, and the deterministic ``@torch.no_grad()`` filters are
left untouched).

Task: invert an unknown coloration filter. A synthetic "speaker" response (a few
fixed resonances) colors a noise signal; the EQ is trained with a multi-resolution
STFT loss so that ``eq(colored)`` matches the original signal. Gradients flow
``loss -> STFT -> waveform -> SOS cascade -> RBJ coefficients -> (fc, Q, gain)``.

Run (CPU is fine; CUDA used when available)::

    python examples/ddsp_graphic_eq.py            # train + save figure
    python examples/ddsp_graphic_eq.py --steps 0  # just plot the untrained EQ

After training, ``eq.freeze()`` bakes the learned coefficients into a deterministic
``torchfx.filter.SOSFilter`` for fast no-grad inference.
"""

from __future__ import annotations

import argparse
import math

import torch
from torch import Tensor

from torchfx.ddsp import (
    LearnableParametricEQ,
    multires_stft_loss,
    rbj_peaking_sos,
    sos_freq_response,
)
from torchfx.filter import SOSFilter


def make_coloration(fs: float) -> Tensor:
    """A synthetic 'speaker' coloration: three resonances + one deep notch."""
    fc = torch.tensor([120.0, 900.0, 3200.0, 7800.0])
    q = torch.tensor([1.2, 2.5, 1.8, 3.0])
    g = torch.tensor([+9.0, -12.0, +7.0, -10.0])
    return rbj_peaking_sos(fc, q, g, fs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--bands", type=int, default=10)
    parser.add_argument("--fs", type=int, default=48_000)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--out", type=str, default="examples/ddsp_eq_result.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    # Training signal: pink-ish noise (white noise shaped by 1/sqrt(f)).
    n_samples = int(args.fs * args.seconds)
    n_fft = 2 ** math.ceil(math.log2(n_samples))
    white = torch.randn(1, n_samples, device=device)
    freqs = torch.fft.rfftfreq(n_fft, 1 / args.fs, device=device)
    shape = 1.0 / torch.sqrt(torch.clamp(freqs, min=20.0))
    x = torch.fft.irfft(torch.fft.rfft(white, n=n_fft) * shape, n=n_fft)[..., :n_samples]
    x = x / x.abs().max()

    # Color the signal with the unknown response (deterministic native path, no grad).
    color_sos = make_coloration(args.fs).to(device)
    colorizer = SOSFilter(color_sos, fs=args.fs).to(device)
    with torch.no_grad():
        colored = colorizer(x)

    # The learnable EQ must invert the coloration: eq(colored) ~= x.
    eq = LearnableParametricEQ(n_bands=args.bands, fs=args.fs).to(device)
    opt = torch.optim.Adam(eq.parameters(), lr=args.lr)

    losses: list[float] = []
    for step in range(args.steps):
        opt.zero_grad()
        y = eq(colored)
        loss = multires_stft_loss(y.squeeze(0), x.squeeze(0))
        loss.backward()
        opt.step()
        losses.append(loss.detach().item())
        if step % 50 == 0 or step == args.steps - 1:
            print(f"step {step:4d}  loss {losses[-1]:.4f}")

    # ----- report ----------------------------------------------------------- #
    with torch.no_grad():
        h_eq = sos_freq_response(eq.sos().to(torch.float64), n_fft)
        h_col = sos_freq_response(color_sos.to(torch.float64), n_fft)
        f = torch.fft.rfftfreq(n_fft, 1 / args.fs)
        eq_db = 20 * torch.log10(h_eq.abs() + 1e-9).cpu()
        col_db = 20 * torch.log10(h_col.abs() + 1e-9).cpu()
        residual_db = eq_db + col_db  # perfect inversion -> 0 dB everywhere

        band = (f > 60) & (f < 12_000)
        flatness = residual_db[band].abs().mean()
        print(f"\nresidual |coloration + EQ| over 60 Hz..12 kHz: {flatness:.2f} dB mean abs")
        print("learned bands (fc Hz / Q / gain dB):")
        for fc_i, q_i, g_i in zip(eq.fc.cpu(), eq.q.cpu(), eq.gain_db.cpu(), strict=True):
            print(f"  {fc_i:8.1f}  {q_i:5.2f}  {g_i:+6.2f}")

    # The trained EQ can be frozen into a fast deterministic filter for inference.
    frozen = eq.freeze()
    print(f"frozen to {type(frozen).__name__} for fast no-grad inference")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
        ax1.plot(losses)
        ax1.set_xlabel("step")
        ax1.set_ylabel("multi-res STFT loss")
        ax1.set_title("Training loss")
        ax2.semilogx(f, col_db, label="coloration (unknown)", lw=1.2)
        ax2.semilogx(f, eq_db, label="learned EQ", lw=1.2)
        ax2.semilogx(f, residual_db, label="residual (sum)", lw=1.2, ls="--")
        ax2.set_xlim(20, args.fs / 2)
        ax2.set_ylim(-24, 24)
        ax2.set_xlabel("frequency (Hz)")
        ax2.set_ylabel("magnitude (dB)")
        ax2.legend(fontsize=8)
        ax2.set_title("Learned inverse response")
        fig.tight_layout()
        fig.savefig(args.out, dpi=150)
        print(f"figure saved to {args.out}")
    except ImportError:
        print("matplotlib not available; skipping figure")


if __name__ == "__main__":
    main()
