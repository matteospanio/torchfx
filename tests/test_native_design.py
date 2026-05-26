"""Numerical-equivalence tests: native filter design vs scipy reference.

These tests guard the native designs in :mod:`torchfx.filter._design` against
``scipy.signal.{butter, cheby1, cheby2, ellip, firwin}``. scipy is a dev/test
dependency only — it is not installed at runtime.

Comparison strategy:

- IIR designs: compare the filter's complex frequency response
  (``scipy.signal.sosfreqz`` over 512 log-spaced bins). Section ordering and
  pole-pair grouping may differ from scipy, but the cascaded transfer function
  is invariant.
- FIR designs: compare coefficients element-wise (``firwin`` is a deterministic
  closed-form sum of sinc + window).
"""

from __future__ import annotations

import pytest
import torch
from scipy.signal import butter, cheby1, cheby2, ellip, firwin, sosfreqz

from torchfx.filter._design import (
    design_butterworth_sos,
    design_cheby1_sos,
    design_cheby2_sos,
    design_ellip_sos,
    design_firwin,
)


def _max_freq_response_error(sos_native, sos_scipy, n_bins: int = 512) -> float:
    _, h_n = sosfreqz(sos_native, worN=n_bins)
    _, h_s = sosfreqz(sos_scipy, worN=n_bins)
    return float(abs(h_n - h_s).max())


# --------------------------------------------------------------------------- #
# Butterworth                                                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("order", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("btype", ["lowpass", "highpass"])
@pytest.mark.parametrize("wn", [0.05, 0.1, 0.25, 0.4, 0.7])
def test_butterworth_matches_scipy(order, btype, wn):
    ours = design_butterworth_sos(order, wn, btype).numpy()
    ref = butter(order, wn, btype=btype, output="sos")
    err = _max_freq_response_error(ours, ref)
    assert err < 1e-10, f"Butter N={order} {btype} Wn={wn}: max|ΔH|={err:.2e}"


# --------------------------------------------------------------------------- #
# Chebyshev I                                                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("order", [2, 4, 8, 16])
@pytest.mark.parametrize("rp", [0.01, 0.1, 1.0])
@pytest.mark.parametrize("btype", ["lowpass", "highpass"])
@pytest.mark.parametrize("wn", [0.05, 0.1, 0.25, 0.4])
def test_cheby1_matches_scipy(order, rp, btype, wn):
    ours = design_cheby1_sos(order, rp, wn, btype).numpy()
    ref = cheby1(order, rp, wn, btype=btype, output="sos")
    err = _max_freq_response_error(ours, ref)
    assert err < 1e-9, f"Cheby1 N={order} Rp={rp} {btype} Wn={wn}: max|ΔH|={err:.2e}"


# --------------------------------------------------------------------------- #
# Chebyshev II                                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("order", [2, 4, 8, 16])
@pytest.mark.parametrize("rs", [40, 60, 80])
@pytest.mark.parametrize("btype", ["lowpass", "highpass"])
@pytest.mark.parametrize("wn", [0.05, 0.1, 0.25, 0.4, 0.7])
def test_cheby2_matches_scipy(order, rs, btype, wn):
    ours = design_cheby2_sos(order, rs, wn, btype).numpy()
    ref = cheby2(order, rs, wn, btype=btype, output="sos")
    err = _max_freq_response_error(ours, ref)
    assert err < 1e-9, f"Cheby2 N={order} Rs={rs} {btype} Wn={wn}: max|ΔH|={err:.2e}"


# --------------------------------------------------------------------------- #
# Elliptic                                                                    #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("order", [2, 3, 4, 5, 8])
@pytest.mark.parametrize("rp", [0.1, 1.0])
@pytest.mark.parametrize("rs", [40, 60, 80])
@pytest.mark.parametrize("btype", ["lowpass", "highpass"])
@pytest.mark.parametrize("wn", [0.1, 0.25, 0.4])
def test_elliptic_matches_scipy_low_order(order, rp, rs, btype, wn):
    ours = design_ellip_sos(order, rp, rs, wn, btype).numpy()
    ref = ellip(order, rp, rs, wn, btype=btype, output="sos")
    err = _max_freq_response_error(ours, ref)
    assert err < 1e-7, f"Elliptic N={order} Rp={rp} Rs={rs} {btype} Wn={wn}: max|ΔH|={err:.2e}"


@pytest.mark.parametrize("order", [12, 16])
@pytest.mark.parametrize("rs", [60, 80])
@pytest.mark.parametrize("btype", ["lowpass", "highpass"])
@pytest.mark.parametrize("wn", [0.1, 0.25, 0.4])
def test_elliptic_matches_scipy_high_order(order, rs, btype, wn):
    """High-order elliptic loosens to ``rtol=1e-5`` per the plan."""
    ours = design_ellip_sos(order, 0.1, rs, wn, btype).numpy()
    ref = ellip(order, 0.1, rs, wn, btype=btype, output="sos")
    err = _max_freq_response_error(ours, ref)
    assert err < 1e-5, f"Elliptic N={order} Rs={rs} {btype} Wn={wn}: max|ΔH|={err:.2e}"


# --------------------------------------------------------------------------- #
# FIR firwin                                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("num_taps", [15, 31, 63, 127])
@pytest.mark.parametrize("window", ["hann", "hamming", "blackman", "bartlett"])
@pytest.mark.parametrize("pass_zero", [True, False])
def test_firwin_matches_scipy(num_taps, window, pass_zero):
    if not pass_zero and num_taps % 2 == 0:
        pytest.skip("firwin requires odd numtaps for highpass")
    ours = design_firwin(num_taps, 1000.0, fs=8000.0, pass_zero=pass_zero, window=window)
    ref = torch.from_numpy(firwin(num_taps, 1000.0, fs=8000.0, pass_zero=pass_zero, window=window))
    torch.testing.assert_close(ours, ref, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("num_taps", [15, 31, 63])
@pytest.mark.parametrize("pass_zero", [True, False])
def test_firwin_kaiser_matches_scipy(num_taps, pass_zero):
    if not pass_zero and num_taps % 2 == 0:
        pytest.skip("firwin requires odd numtaps for highpass")
    beta = 8.6
    ours = design_firwin(num_taps, 1000.0, fs=8000.0, pass_zero=pass_zero, window=("kaiser", beta))
    ref = torch.from_numpy(
        firwin(num_taps, 1000.0, fs=8000.0, pass_zero=pass_zero, window=("kaiser", beta))
    )
    torch.testing.assert_close(ours, ref, rtol=1e-8, atol=1e-10)


# --------------------------------------------------------------------------- #
# Round-trip: native-designed filter applied to a real signal matches scipy.  #
# --------------------------------------------------------------------------- #


def test_butterworth_audio_roundtrip():
    """Process an audio buffer through native vs scipy Butterworth → equal output."""
    from scipy.signal import sosfilt

    fs = 48000
    audio = torch.randn(fs).numpy()
    sos_n = design_butterworth_sos(4, 1000.0 / (0.5 * fs), "lowpass").numpy()
    sos_s = butter(4, 1000.0 / (0.5 * fs), btype="lowpass", output="sos")
    y_n = sosfilt(sos_n, audio)
    y_s = sosfilt(sos_s, audio)
    torch.testing.assert_close(torch.from_numpy(y_n), torch.from_numpy(y_s), rtol=1e-10, atol=1e-12)
