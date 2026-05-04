"""Native filter-design helpers (no scipy/numpy dependency).

This module implements the coefficient-design algorithms used by torchfx's IIR
and FIR filters with the goal of matching ``scipy.signal.{butter, cheby1,
cheby2, ellip, firwin}`` numerically while removing scipy as a runtime
dependency.

All public design functions return canonical CPU ``float64`` tensors:

- IIR designs return ``Tensor[K, 6]`` second-order-section matrices in the same
  format as ``scipy.signal.iirfilter(..., output='sos')`` — each row is
  ``[b0, b1, b2, 1, a1, a2]``.
- FIR designs return ``Tensor[N]`` impulse-response vectors in the same format
  as ``scipy.signal.firwin``.

Performance note: filter design works on at most ~16 poles/zeros. PyTorch's
per-op dispatch overhead dominates over the actual arithmetic at this scale,
so the IIR pipeline (analog prototype -> band transform -> bilinear -> SOS)
operates on Python lists of ``complex`` and only converts to a single
``Tensor[K, 6]`` at the very end. The window-method FIR design stays in torch
because it works on length-N (taps, ~31..1024) arrays where vectorization wins.

"""

from __future__ import annotations

import cmath
import math
from collections.abc import Sequence

import torch
from torch import Tensor

# --------------------------------------------------------------------------- #
# Bilinear-transform plumbing (analog prototype -> digital SOS).              #
# Operates on Python lists of complex for speed on small problems (N <= 16).  #
# --------------------------------------------------------------------------- #


def _prewarp(wn_norm: float) -> float:
    """Bilinear-transform frequency prewarp.

    Maps a digital cutoff (normalized to Nyquist, ``wn_norm in (0, 1)``) to the
    analog frequency that the bilinear transform will map back to ``wn_norm``.
    Uses the scipy convention ``fs = 2`` so the bilinear formula is
    ``z = (2 + s) / (2 - s)``.

    """
    return 2.0 * math.tan(math.pi * wn_norm * 0.5)


def _lp_to_lp(
    z: list[complex], p: list[complex], k: complex, wn: float
) -> tuple[list[complex], list[complex], complex]:
    """Analog lowpass-prototype -> analog lowpass at cutoff ``wn``.

    Substitution ``s -> s / wn`` shifts every pole/zero by ``wn``; the gain
    rescales by ``wn ** (n_poles - n_zeros)``.

    """
    degree = len(p) - len(z)
    z_lp = [zi * wn for zi in z]
    p_lp = [pi * wn for pi in p]
    k_lp = k * (wn**degree)
    return z_lp, p_lp, k_lp


def _lp_to_hp(
    z: list[complex], p: list[complex], k: complex, wn: float
) -> tuple[list[complex], list[complex], complex]:
    """Analog lowpass-prototype -> analog highpass at cutoff ``wn``.

    Substitution ``s -> wn / s`` reflects every finite root through the unit
    circle scaled by ``wn``; each analog zero at infinity becomes a digital
    zero at the origin; the gain picks up the leading-coefficient ratio.

    """
    degree = len(p) - len(z)
    z_hp = [wn / zi for zi in z]
    p_hp = [wn / pi for pi in p]
    if degree > 0:
        z_hp.extend([0.0 + 0.0j] * degree)
    z_prod = _prod(-zi for zi in z) if z else 1.0 + 0j
    p_prod = _prod(-pi for pi in p)
    k_hp = k * complex((z_prod / p_prod).real, 0.0)
    return z_hp, p_hp, k_hp


def _bilinear_zpk(
    z: list[complex], p: list[complex], k: complex, fs2: float = 2.0
) -> tuple[list[complex], list[complex], complex]:
    """Analog ZPK -> digital ZPK via the bilinear transform.

    ``z_digital = (fs2 + s) / (fs2 - s)``. Analog zeros at infinity become
    digital zeros at ``z = -1``. ``fs2 = 2`` matches the scipy convention used
    after :func:`_prewarp`.

    """
    z_d = [(fs2 + zi) / (fs2 - zi) for zi in z]
    p_d = [(fs2 + pi) / (fs2 - pi) for pi in p]
    n_extra = len(p) - len(z)
    if n_extra > 0:
        z_d.extend([complex(-1.0, 0.0)] * n_extra)
    num_factor = _prod(fs2 - zi for zi in z) if z else 1.0 + 0j
    den_factor = _prod(fs2 - pi for pi in p)
    k_d = k * complex((num_factor / den_factor).real, 0.0)
    return z_d, p_d, k_d


def _prod(iterable: object) -> complex:
    """Stable cumulative product for an iterable of complex numbers."""
    result: complex = 1.0 + 0j
    for x in iterable:  # type: ignore[attr-defined]
        result = result * x
    return result


def _cplxpair(roots: list[complex], tol: float = 1e-8) -> list[complex]:
    """Reorder ``roots`` so complex conjugates are adjacent and reals come last."""
    if not roots:
        return roots
    reals: list[complex] = []
    pos: list[complex] = []
    for r in roots:
        scale = max(abs(r), 1.0)
        if abs(r.imag) <= tol * scale:
            reals.append(complex(r.real, 0.0))
        elif r.imag > 0:
            pos.append(r)
    pos.sort(key=lambda c: c.real)
    paired: list[complex] = []
    for r in pos:
        paired.append(r)
        paired.append(r.conjugate())
    return paired + reals


def _zpk_to_sos(z: list[complex], p: list[complex], k: complex) -> Tensor:
    """Convert digital ZPK to a stack of second-order sections (Tensor[K, 6])."""
    z = _cplxpair(z)
    p = _cplxpair(p)
    n_z, n_p = len(z), len(p)
    n_sections = max((n_p + 1) // 2, (n_z + 1) // 2, 1)

    # Pad with zeros at the origin (benign — adds a redundant z factor).
    target = 2 * n_sections
    z = z + [complex(0.0, 0.0)] * (target - n_z)
    p = p + [complex(0.0, 0.0)] * (target - n_p)

    # Reshape into pairs and reorder so the section closest to the unit circle
    # is last (matches scipy's cascade-stability ordering).
    pole_pairs = [(p[2 * i], p[2 * i + 1]) for i in range(n_sections)]
    zero_pairs = [(z[2 * i], z[2 * i + 1]) for i in range(n_sections)]
    distances = [min(abs(abs(p1) - 1.0), abs(abs(p2) - 1.0)) for p1, p2 in pole_pairs]
    order = sorted(range(n_sections), key=lambda i: -distances[i])

    rows: list[list[float]] = []
    k_real = k.real
    for idx, i in enumerate(order):
        z1, z2 = zero_pairs[i]
        p1, p2 = pole_pairs[i]
        # b = (s - z1) * (s - z2) = s^2 - (z1+z2)*s + z1*z2
        b0, b1, b2 = 1.0, -(z1 + z2).real, (z1 * z2).real
        a0, a1, a2 = 1.0, -(p1 + p2).real, (p1 * p2).real
        if idx == 0:
            b0 *= k_real
            b1 *= k_real
            b2 *= k_real
        rows.append([b0, b1, b2, a0, a1, a2])
    return torch.tensor(rows, dtype=torch.float64)


# --------------------------------------------------------------------------- #
# Butterworth.                                                                #
# --------------------------------------------------------------------------- #


def _butter_analog_prototype(order: int) -> tuple[list[complex], list[complex], complex]:
    """Analog Butterworth lowpass prototype (cutoff = 1 rad/s)."""
    poles: list[complex] = []
    inv_2N = math.pi / (2.0 * order)
    for k in range(1, order + 1):
        angle = (2.0 * k + order - 1.0) * inv_2N
        poles.append(complex(math.cos(angle), math.sin(angle)))
    return [], poles, 1.0 + 0j


def _apply_btype(
    z: list[complex], p: list[complex], k: complex, btype: str, cutoff_norm: float
) -> tuple[list[complex], list[complex], complex]:
    """Apply lowpass→target transform after prewarping the cutoff."""
    wn = _prewarp(cutoff_norm)
    btype_normalized = {"low": "lowpass", "high": "highpass"}.get(btype, btype)
    if btype_normalized == "lowpass":
        return _lp_to_lp(z, p, k, wn)
    if btype_normalized == "highpass":
        return _lp_to_hp(z, p, k, wn)
    raise ValueError(f"Unsupported btype {btype!r}; expected 'lowpass'/'low' or 'highpass'/'high'.")


def design_butterworth_sos(order: int, cutoff_norm: float, btype: str) -> Tensor:
    """Native Butterworth SOS design.

    Parameters
    ----------
    order : int
        Filter order.
    cutoff_norm : float
        Cutoff frequency normalized to Nyquist, in ``(0, 1)``.
    btype : {"lowpass"/"low", "highpass"/"high"}
        Filter type.

    """
    z, p, k = _butter_analog_prototype(order)
    z, p, k = _apply_btype(z, p, k, btype, cutoff_norm)
    z, p, k = _bilinear_zpk(z, p, k, fs2=2.0)
    return _zpk_to_sos(z, p, k)


# --------------------------------------------------------------------------- #
# Chebyshev I (passband ripple).                                              #
# --------------------------------------------------------------------------- #


def _cheby1_analog_prototype(
    order: int, rp_db: float
) -> tuple[list[complex], list[complex], complex]:
    """Analog Chebyshev-I lowpass prototype (cutoff = 1 rad/s, equiripple passband)."""
    eps = math.sqrt(math.expm1(math.log(10) * 0.1 * rp_db))
    mu = math.asinh(1.0 / eps) / order
    sinh_mu = math.sinh(mu)
    cosh_mu = math.cosh(mu)
    inv_2N = math.pi / (2.0 * order)
    poles: list[complex] = []
    for j in range(-order + 1, order, 2):
        theta = j * inv_2N
        # p = -sinh(mu + j*theta) = -(sinh(mu)*cos(theta) + j*cosh(mu)*sin(theta))
        poles.append(complex(-sinh_mu * math.cos(theta), -cosh_mu * math.sin(theta)))
    k_real = _prod(-pi for pi in poles).real
    if order % 2 == 0:
        k_real /= math.sqrt(1.0 + eps * eps)
    return [], poles, complex(k_real, 0.0)


def design_cheby1_sos(order: int, rp_db: float, cutoff_norm: float, btype: str) -> Tensor:
    """Native Chebyshev-I SOS design."""
    z, p, k = _cheby1_analog_prototype(order, rp_db)
    z, p, k = _apply_btype(z, p, k, btype, cutoff_norm)
    z, p, k = _bilinear_zpk(z, p, k, fs2=2.0)
    return _zpk_to_sos(z, p, k)


# --------------------------------------------------------------------------- #
# Chebyshev II (stopband ripple, also called inverse Chebyshev).              #
# --------------------------------------------------------------------------- #


def _cheby2_analog_prototype(
    order: int, rs_db: float
) -> tuple[list[complex], list[complex], complex]:
    """Analog Chebyshev-II lowpass prototype (stopband edge = 1 rad/s)."""
    de = 1.0 / math.sqrt(math.expm1(math.log(10) * 0.1 * rs_db))
    mu = math.asinh(1.0 / de) / order
    sinh_mu = math.sinh(mu)
    cosh_mu = math.cosh(mu)
    inv_2N = math.pi / (2.0 * order)

    # Poles: p = -1 / sinh(mu + j*theta), equivalent to scipy's
    # negation-of-warped-circle convention.
    poles: list[complex] = []
    for j in range(-order + 1, order, 2):
        theta = j * inv_2N
        denom = complex(sinh_mu * math.cos(theta), cosh_mu * math.sin(theta))
        poles.append(-1.0 / denom)

    # Zeros: 1j / sin(theta_m). Skip the m=0 zero (at infinity) for odd order.
    if order % 2 == 1:
        m_z = list(range(-order + 1, 0, 2)) + list(range(2, order, 2))
    else:
        m_z = list(range(-order + 1, order, 2))
    zeros: list[complex] = []
    for j in m_z:
        sj = math.sin(j * inv_2N)
        zeros.append(complex(0.0, 1.0 / sj))

    z_prod = _prod(-zi for zi in zeros) if zeros else 1.0 + 0j
    k_real = (_prod(-pi for pi in poles) / z_prod).real
    return zeros, poles, complex(k_real, 0.0)


def design_cheby2_sos(order: int, rs_db: float, cutoff_norm: float, btype: str) -> Tensor:
    """Native Chebyshev-II SOS design.

    ``cutoff_norm`` is the *stopband-edge* frequency.

    """
    z, p, k = _cheby2_analog_prototype(order, rs_db)
    z, p, k = _apply_btype(z, p, k, btype, cutoff_norm)
    z, p, k = _bilinear_zpk(z, p, k, fs2=2.0)
    return _zpk_to_sos(z, p, k)


# --------------------------------------------------------------------------- #
# Elliptic (Cauer) — analog prototype.                                        #
# Reference: Orfanidis, "Lecture Notes on Elliptic Filter Design".            #
# --------------------------------------------------------------------------- #


_AGM_EPS = 1e-15
_LANDEN_MAXITER = 32
_ELLIPDEG_MMAX = 7  # match scipy.signal._filter_design._ELLIPDEG_MMAX


def _ellipk(m: float) -> float:
    """Complete elliptic integral of the first kind, K(m), via AGM."""
    if m == 1.0:
        return math.inf
    if m == 0.0:
        return 0.5 * math.pi
    a = 1.0
    b = math.sqrt(1.0 - m)
    for _ in range(_LANDEN_MAXITER):
        if abs(a - b) <= _AGM_EPS * a:
            break
        a, b = 0.5 * (a + b), math.sqrt(a * b)
    return 0.5 * math.pi / a


def _ellipkm1(m: float) -> float:
    """``ellipk(1 - m)``, matches scipy.special.ellipkm1."""
    return _ellipk(1.0 - m)


def _ellipj_real(u: float, m: float) -> tuple[float, float, float]:
    """Jacobi elliptic functions ``(sn, cn, dn)`` at real ``u``, real ``m`` in ``[0,
    1)``."""
    if m == 0.0:
        return math.sin(u), math.cos(u), 1.0
    if m == 1.0:
        ch = math.cosh(u)
        return math.tanh(u), 1.0 / ch, 1.0 / ch
    a, b, c = 1.0, math.sqrt(1.0 - m), math.sqrt(m)
    a_seq = [a]
    c_seq = [c]
    for _ in range(_LANDEN_MAXITER):
        a_new = 0.5 * (a + b)
        b_new = math.sqrt(a * b)
        c_new = 0.5 * (a - b)
        a, b, c = a_new, b_new, c_new
        a_seq.append(a)
        c_seq.append(c)
        if abs(c) <= _AGM_EPS * a:
            break
    n = len(a_seq) - 1
    phi = (2**n) * a * u
    for i in range(n, 0, -1):
        phi = 0.5 * (phi + math.asin((c_seq[i] / a_seq[i]) * math.sin(phi)))
    sn = math.sin(phi)
    cn = math.cos(phi)
    dn = math.sqrt(max(0.0, 1.0 - m * sn * sn))
    return sn, cn, dn


def _ellipdeg(n: int, m1: float) -> float:
    """Solve the elliptic-filter degree equation for ``m`` (Orfanidis Eq.

    49).

    """
    K1 = _ellipk(m1)
    K1p = _ellipkm1(m1)
    q1 = math.exp(-math.pi * K1p / K1)
    q: float = q1 ** (1.0 / n)
    num: float = sum(q ** (k * (k + 1)) for k in range(_ELLIPDEG_MMAX + 1))
    den: float = 1.0 + 2.0 * sum(q ** (k * k) for k in range(1, _ELLIPDEG_MMAX + 2))
    result: float = 16.0 * q * (num / den) ** 4
    return result


def _arc_jac_sn_complex(w: complex, m: float) -> complex:
    """Inverse Jacobi sn for complex ``w``, real ``m`` (descending-Landen, Orfanidis Eq.

    56).

    """

    def comp(kx: float) -> float:
        return math.sqrt((1.0 - kx) * (1.0 + kx))

    k = math.sqrt(m)
    if k > 1.0:
        return complex(math.nan, math.nan)
    if k == 1.0:
        return 0.5 * (cmath.log(1 + w) - cmath.log(1 - w))

    ks = [k]
    while ks[-1] != 0.0:
        k_ = ks[-1]
        k_p = comp(k_)
        ks.append((1.0 - k_p) / (1.0 + k_p))
        if len(ks) > _LANDEN_MAXITER:
            raise ValueError("Landen transformation not converging in _arc_jac_sn_complex")

    K = math.pi / 2.0
    for kn in ks[1:]:
        K *= 1.0 + kn

    wns: list[complex] = [w]
    for kn, knext in zip(ks[:-1], ks[1:], strict=True):
        wn = wns[-1]
        denom_inner = ((1.0 - kn * wn) * (1.0 + kn * wn)) ** 0.5
        wnext = (2.0 * wn) / ((1.0 + knext) * (1.0 + denom_inner))
        wns.append(wnext)

    u = (2.0 / math.pi) * cmath.asin(wns[-1])
    return K * u


def _arc_jac_sc1(w: float, m: float) -> float:
    """Real inverse Jacobian sc with complementary modulus."""
    z = _arc_jac_sn_complex(complex(0.0, w), m)
    if abs(z.real) > 1e-12:
        raise ValueError("Inverse sc1 produced non-imaginary result")
    return z.imag


def _ellipap(
    order: int, rp_db: float, rs_db: float
) -> tuple[list[complex], list[complex], complex]:
    """Analog elliptic lowpass prototype (cutoff = 1 rad/s).

    Mirrors :func:`scipy.signal.ellipap`.

    """
    if order < 0:
        raise ValueError("Filter order must be a nonnegative integer")
    if order == 0:
        return [], [], complex(10 ** (-rp_db / 20.0), 0.0)
    if order == 1:
        eps_sq = math.expm1(math.log(10) * 0.1 * rp_db)
        p_val = -math.sqrt(1.0 / eps_sq)
        return [], [complex(p_val, 0.0)], complex(-p_val, 0.0)

    eps_sq = math.expm1(math.log(10) * 0.1 * rp_db)
    eps = math.sqrt(eps_sq)
    rs_eps_sq = math.expm1(math.log(10) * 0.1 * rs_db)
    if rs_eps_sq == 0.0:
        raise ValueError("Cannot design elliptic filter with given rp / rs.")
    ck1_sq = eps_sq / rs_eps_sq

    K1 = _ellipk(ck1_sq)
    m = _ellipdeg(order, ck1_sq)
    capk = _ellipk(m)

    # j-grid: 1 - N%2, 3, 5, ..., N-1 (matches scipy).
    j_start = 1 - (order % 2)
    j_list = list(range(j_start, order, 2))

    s_vals: list[float] = []
    c_vals: list[float] = []
    d_vals: list[float] = []
    inv_N = 1.0 / order
    for j in j_list:
        s, c, d = _ellipj_real(j * capk * inv_N, m)
        s_vals.append(s)
        c_vals.append(c)
        d_vals.append(d)

    sqrt_m = math.sqrt(m)
    z_list: list[complex] = []
    for s in s_vals:
        if abs(s) > _AGM_EPS:
            z_list.append(complex(0.0, 1.0 / (sqrt_m * s)))
    z_full = z_list + [z.conjugate() for z in z_list]

    r = _arc_jac_sc1(1.0 / eps, ck1_sq)
    v0 = capk * r / (order * K1)
    sv, cv, dv = _ellipj_real(v0, 1.0 - m)

    p_half: list[complex] = []
    dv_sq = dv  # readability alias unused — keep dv direct
    for s, c, d in zip(s_vals, c_vals, d_vals, strict=True):
        denom = 1.0 - (d * sv) ** 2
        re = -(c * d * sv * cv) / denom
        im = -s * dv / denom
        p_half.append(complex(re, im))
    del dv_sq  # silence accidental shadow

    if order % 2 == 1:
        norm = math.sqrt(sum(abs(p) ** 2 for p in p_half))
        threshold = _AGM_EPS * norm
        new_p = [p for p in p_half if abs(p.imag) > threshold]
        p_full = p_half + [p.conjugate() for p in new_p]
    else:
        p_full = p_half + [p.conjugate() for p in p_half]

    z_prod = _prod(-zi for zi in z_full) if z_full else 1.0 + 0j
    k_real = (_prod(-pi for pi in p_full) / z_prod).real
    if order % 2 == 0:
        k_real /= math.sqrt(1.0 + eps_sq)
    return z_full, p_full, complex(k_real, 0.0)


def design_ellip_sos(
    order: int, rp_db: float, rs_db: float, cutoff_norm: float, btype: str
) -> Tensor:
    """Native elliptic (Cauer) SOS design."""
    z, p, k = _ellipap(order, rp_db, rs_db)
    z, p, k = _apply_btype(z, p, k, btype, cutoff_norm)
    z, p, k = _bilinear_zpk(z, p, k, fs2=2.0)
    return _zpk_to_sos(z, p, k)


# --------------------------------------------------------------------------- #
# FIR window-method design (firwin).                                          #
# Stays in torch — windows are length-N (~31..1024) where vectorization wins. #
# --------------------------------------------------------------------------- #


def _window(window: str, n: int) -> Tensor:
    """Symmetric window of length ``n`` matching scipy.signal.get_window(...,
    fftbins=False)."""
    if n == 1:
        return torch.ones(1, dtype=torch.float64)
    k = torch.arange(n, dtype=torch.float64)
    n_minus_1 = n - 1
    two_pi_k = 2.0 * math.pi * k / n_minus_1
    four_pi_k = 4.0 * math.pi * k / n_minus_1
    six_pi_k = 6.0 * math.pi * k / n_minus_1
    eight_pi_k = 8.0 * math.pi * k / n_minus_1

    if window == "boxcar":
        return torch.ones(n, dtype=torch.float64)
    if window == "hann":
        return 0.5 - 0.5 * torch.cos(two_pi_k)
    if window == "hamming":
        return 0.54 - 0.46 * torch.cos(two_pi_k)
    if window == "blackman":
        return 0.42 - 0.5 * torch.cos(two_pi_k) + 0.08 * torch.cos(four_pi_k)
    if window == "bartlett":
        m = (n - 1) / 2.0
        return 1.0 - torch.abs(k - m) / m
    if window == "barthann":
        x = k / n_minus_1
        return 0.62 - 0.48 * torch.abs(x - 0.5) + 0.38 * torch.cos(2.0 * math.pi * (x - 0.5))
    if window == "nuttall":
        return (
            0.3635819
            - 0.4891775 * torch.cos(two_pi_k)
            + 0.1365995 * torch.cos(four_pi_k)
            - 0.0106411 * torch.cos(six_pi_k)
        )
    if window == "flattop":
        return (
            0.21557895
            - 0.41663158 * torch.cos(two_pi_k)
            + 0.277263158 * torch.cos(four_pi_k)
            - 0.083578947 * torch.cos(six_pi_k)
            + 0.006947368 * torch.cos(eight_pi_k)
        )
    if window == "parzen":
        m = (n - 1) / 2.0
        x = (k - m) / (n / 2.0)
        ax = torch.abs(x)
        return torch.where(ax <= 0.5, 1.0 - 6.0 * ax**2 + 6.0 * ax**3, 2.0 * (1.0 - ax) ** 3)
    if window == "bohman":
        m = (n - 1) / 2.0
        x = torch.abs((k - m) / m)
        return (1.0 - x) * torch.cos(math.pi * x) + (1.0 / math.pi) * torch.sin(math.pi * x)
    if window == "kaiser":
        beta = 0.5
        m = (n - 1) / 2.0
        arg = beta * torch.sqrt(1.0 - ((k - m) / m) ** 2)
        result: Tensor = torch.special.i0(arg) / torch.special.i0(
            torch.tensor(beta, dtype=torch.float64)
        )
        return result

    raise ValueError(f"Unsupported window type: {window!r}")


def design_firwin(
    num_taps: int,
    cutoff: float | Sequence[float],
    fs: float,
    pass_zero: bool,
    window: str | tuple[str, float],
) -> Tensor:
    """Native FIR window-method design (lowpass, highpass, bandpass, bandstop).

    Mirrors ``scipy.signal.firwin(num_taps, cutoff, fs=fs, pass_zero=pass_zero,
    window=window, scale=True)`` across single- and multi-cutoff cases.

    Parameters
    ----------
    num_taps : int
        Filter length (number of coefficients).
    cutoff : float | Sequence[float]
        Cutoff frequency in Hz, or sorted list of band-edge frequencies for
        multi-band designs.
    fs : float
        Sample rate in Hz.
    pass_zero : bool
        If True, the resulting filter passes DC.
    window : str | (str, float)
        Window name, or ``(name, param)`` tuple where the parameter is the
        Kaiser ``beta``.

    """
    cutoff_list = (
        [float(cutoff)] if isinstance(cutoff, (int, float)) else [float(c) for c in cutoff]
    )
    if not cutoff_list:
        raise ValueError("Cutoff sequence must not be empty.")

    nyq = 0.5 * fs
    cutoff_norm = [c / nyq for c in cutoff_list]
    for wc in cutoff_norm:
        if not 0.0 < wc < 1.0:
            raise ValueError(f"Cutoff frequency must be in (0, fs/2); got {wc * nyq} with fs={fs}.")

    pass_nyquist = bool(pass_zero) ^ (len(cutoff_norm) % 2 == 1)
    if pass_nyquist and num_taps % 2 == 0:
        raise ValueError(
            "A filter with an even number of coefficients must have zero "
            "response at the Nyquist frequency. Reduce numtaps by 1, or "
            "switch pass_zero so the design does not pass Nyquist."
        )
    edges: list[float] = []
    if pass_zero:
        edges.append(0.0)
    edges.extend(cutoff_norm)
    if pass_nyquist:
        edges.append(1.0)
    if len(edges) % 2 != 0:
        raise ValueError("Cutoff/pass_zero combination produced unbalanced bands.")
    bands = list(zip(edges[0::2], edges[1::2], strict=True))

    alpha = 0.5 * (num_taps - 1)
    n = torch.arange(num_taps, dtype=torch.float64)
    m = n - alpha

    h = torch.zeros(num_taps, dtype=torch.float64)
    for left, right in bands:
        h = h + right * torch.special.sinc(right * m)
        h = h - left * torch.special.sinc(left * m)

    if isinstance(window, tuple):
        win_name, win_param = window
        if win_name == "kaiser":
            arg = win_param * torch.sqrt(1.0 - ((n - alpha) / alpha) ** 2)
            w = torch.special.i0(arg) / torch.special.i0(
                torch.tensor(win_param, dtype=torch.float64)
            )
        else:
            raise ValueError(f"Tuple windows only supported for 'kaiser'; got {win_name!r}.")
    else:
        w = _window(window, num_taps)
    h = h * w

    if pass_zero:
        scale_freq = 0.0
    elif len(bands) == 1 and bands[0][1] == 1.0:
        scale_freq = 1.0
    else:
        scale_freq = 0.5 * (bands[0][0] + bands[0][1])
    c = torch.cos(math.pi * m * scale_freq)
    h_scaled: Tensor = h / torch.sum(h * c)
    return h_scaled
