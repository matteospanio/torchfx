#!/usr/bin/env python3
"""Differentiable DSP proof-of-concept: a learnable parametric EQ in TorchFX.

This example demonstrates that TorchFX's SOS filter representation is usable in a
fully differentiable, gradient-trained pipeline. A cascade of RBJ peaking-EQ
biquads — the same second-order sections the native TorchFX kernels execute — is
parameterised by learnable per-band ``(center frequency, Q, gain)`` and applied in
the **frequency domain** (FFT x H), where plain autograd differentiates everything:
no custom backward pass, no modification of the fast ``@torch.no_grad()`` inference
kernels.

Task: invert an unknown coloration filter. A synthetic "speaker" response (a few
fixed resonances + a low shelf) colors a noise signal; the EQ is trained with a
multi-resolution STFT loss so that ``eq(colored)`` matches the original signal.
Gradients flow  loss -> STFT -> waveform -> H(e^jw) -> RBJ coefficients ->
(fc, Q, gain)  end to end.

Run (CPU is fine; CUDA used when available)::

    python examples/ddsp_graphic_eq.py            # train + save figure/audio
    python examples/ddsp_graphic_eq.py --steps 0  # just plot the untrained EQ

The trained band parameters can be transferred 1:1 onto native TorchFX biquads for
fast no-grad inference, since both share the RBJ SOS parameterisation.

"""

from __future__ import annotations

import argparse
import math

import torch
from torch import Tensor, nn

# --------------------------------------------------------------------------- #
# Differentiable RBJ peaking-EQ cascade (frequency-domain application).        #
# --------------------------------------------------------------------------- #


def rbj_peaking_sos(fc: Tensor, q: Tensor, gain_db: Tensor, fs: float) -> Tensor:
    """RBJ-cookbook peaking-EQ section as a differentiable ``[K, 6]`` SOS stack.

    Identical math to ``torchfx.filter.biquad.BiquadPeak`` but in batched torch ops
    so the coefficients stay on the autograd tape: gradients flow back to ``fc``,
    ``q`` and ``gain_db``.

    """
    A = torch.pow(10.0, gain_db / 40.0)
    w0 = 2.0 * math.pi * fc / fs
    cos_w0 = torch.cos(w0)
    alpha = torch.sin(w0) / (2.0 * q)

    b0 = 1.0 + alpha * A
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / A

    sos = torch.stack([b0 / a0, b1 / a0, b2 / a0, torch.ones_like(a0), a1 / a0, a2 / a0], dim=-1)
    return sos


def sos_freq_response(sos: Tensor, n_fft: int) -> Tensor:
    """Complex frequency response of an SOS cascade on the ``rfft`` grid.

    ``H(z) = prod_k (b0 + b1 z^-1 + b2 z^-2) / (1 + a1 z^-1 + a2 z^-2)`` evaluated at
    ``z = e^{jw}`` for the ``n_fft//2 + 1`` non-negative frequency bins. All ops are
    complex-differentiable, so this is the autograd path from coefficients to audio.

    """
    w = torch.linspace(0.0, math.pi, n_fft // 2 + 1, device=sos.device, dtype=sos.dtype)
    z1 = torch.exp(torch.complex(torch.zeros_like(w), -w))  # z^-1
    z2 = z1 * z1
    b = sos[:, 0:1] + sos[:, 1:2] * z1 + sos[:, 2:3] * z2  # [K, F]
    a = sos[:, 3:4] + sos[:, 4:5] * z1 + sos[:, 5:6] * z2
    return torch.prod(b / a, dim=0)  # [F]


class LearnableParametricEQ(nn.Module):
    """A cascade of peaking-EQ bands with learnable (fc, Q, gain) per band.

    Raw parameters are unconstrained; sigmoid/exp maps (FLAMO-style) keep the
    effective values in stable, audible ranges so training cannot push a pole
    outside the unit circle.

    """

    def __init__(
        self,
        n_bands: int = 10,
        fs: float = 48_000,
        f_lo: float = 40.0,
        f_hi: float = 16_000.0,
        max_gain_db: float = 18.0,
    ) -> None:
        super().__init__()
        self.fs = fs
        self.f_lo, self.f_hi = f_lo, f_hi
        self.max_gain_db = max_gain_db
        # Initialise band centers log-spaced over [f_lo, f_hi]; the sigmoid map
        # below is centered so raw zeros land back on this grid.
        grid = torch.linspace(0.0, 1.0, n_bands + 2)[1:-1]
        self._fc_raw = nn.Parameter(torch.logit(grid))
        self._q_raw = nn.Parameter(torch.zeros(n_bands))  # exp map: q0 * e^raw
        self.gain_db_raw = nn.Parameter(torch.zeros(n_bands))

    @property
    def fc(self) -> Tensor:
        """Band centers, log-spaced sigmoid map into [f_lo, f_hi]."""
        t = torch.sigmoid(self._fc_raw)
        return self.f_lo * (self.f_hi / self.f_lo) ** t

    @property
    def q(self) -> Tensor:
        """Band Q, exp map around 1.0, clamped to [0.3, 8]."""
        return torch.clamp(math.sqrt(2.0) * torch.exp(self._q_raw), 0.3, 8.0)

    @property
    def gain_db(self) -> Tensor:
        """Band gain in dB, tanh-bounded to +/- max_gain_db."""
        return self.max_gain_db * torch.tanh(self.gain_db_raw)

    def sos(self) -> Tensor:
        """The cascade as a differentiable ``[K, 6]`` SOS stack (TorchFX layout)."""
        return rbj_peaking_sos(self.fc, self.q, self.gain_db, self.fs)

    def forward(self, x: Tensor) -> Tensor:
        """Filter ``x`` (``[C, T]``) through the cascade in the frequency domain."""
        n_fft = 2 ** math.ceil(math.log2(x.shape[-1]))
        H = sos_freq_response(self.sos().to(torch.float64), n_fft).to(torch.complex64)
        X = torch.fft.rfft(x, n=n_fft)
        y = torch.fft.irfft(X * H, n=n_fft)
        return y[..., : x.shape[-1]]


# --------------------------------------------------------------------------- #
# Multi-resolution STFT loss (the standard DDSP reconstruction objective).     #
# --------------------------------------------------------------------------- #


def multires_stft_loss(x: Tensor, y: Tensor, sizes: tuple[int, ...] = (512, 1024, 2048)) -> Tensor:
    """Sum of spectral-convergence + log-magnitude L1 over several STFT resolutions."""
    loss = x.new_zeros(())
    for n_fft in sizes:
        win = torch.hann_window(n_fft, device=x.device)
        X = torch.stft(x, n_fft, n_fft // 4, window=win, return_complex=True).abs()
        Y = torch.stft(y, n_fft, n_fft // 4, window=win, return_complex=True).abs()
        loss = loss + torch.norm(X - Y) / (torch.norm(Y) + 1e-8)
        loss = loss + torch.nn.functional.l1_loss(torch.log(X + 1e-5), torch.log(Y + 1e-5))
    return loss


# --------------------------------------------------------------------------- #
# The unknown coloration to invert (fixed, non-learnable).                     #
# --------------------------------------------------------------------------- #


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
    T = int(args.fs * args.seconds)
    n_fft = 2 ** math.ceil(math.log2(T))
    white = torch.randn(1, T, device=device)
    freqs = torch.fft.rfftfreq(n_fft, 1 / args.fs, device=device)
    shape = 1.0 / torch.sqrt(torch.clamp(freqs, min=20.0))
    x = torch.fft.irfft(torch.fft.rfft(white, n=n_fft) * shape, n=n_fft)[..., :T]
    x = x / x.abs().max()

    # Color the signal with the unknown response (no grad — it is the plant).
    color_sos = make_coloration(args.fs).to(device)
    with torch.no_grad():
        Hc = sos_freq_response(color_sos.to(torch.float64), n_fft).to(torch.complex64)
        colored = torch.fft.irfft(torch.fft.rfft(x, n=n_fft) * Hc, n=n_fft)[..., :T]

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
        H_eq = sos_freq_response(eq.sos().to(torch.float64), n_fft)
        H_col = sos_freq_response(color_sos.to(torch.float64), n_fft)
        f = torch.fft.rfftfreq(n_fft, 1 / args.fs)
        eq_db = 20 * torch.log10(H_eq.abs() + 1e-9).cpu()
        col_db = 20 * torch.log10(H_col.abs() + 1e-9).cpu()
        residual_db = eq_db + col_db  # perfect inversion -> 0 dB everywhere

        band = (f > 60) & (f < 12_000)
        flatness = residual_db[band].abs().mean()
        print(f"\nresidual |coloration + EQ| over 60 Hz..12 kHz: {flatness:.2f} dB mean abs")
        print("learned bands (fc Hz / Q / gain dB):")
        for fc_i, q_i, g_i in zip(eq.fc.cpu(), eq.q.cpu(), eq.gain_db.cpu(), strict=True):
            print(f"  {fc_i:8.1f}  {q_i:5.2f}  {g_i:+6.2f}")

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
