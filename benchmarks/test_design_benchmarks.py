"""Design-time benchmarks: native filter design vs scipy reference.

Measures wall-clock cost of computing filter coefficients side-by-side. Design
runs once per filter at first ``forward()``, so absolute numbers in the
single-millisecond range are fine — the goal is visibility, not gating.

Soft expectations (per the plan):

- Native within 3× of scipy → fine.
- Native > 5× of scipy → investigate (avoidable Python loop, .item() sync).
- Native > 50 ms wall-clock for a single design → block (would be visible to
  users automating filter parameters block-by-block).

Run with::

    uv run pytest benchmarks/test_design_benchmarks.py --benchmark-enable \\
        --benchmark-columns=mean,stddev,rounds --benchmark-group-by=group
"""

from __future__ import annotations

import pytest
from scipy.signal import butter as scipy_butter
from scipy.signal import cheby1 as scipy_cheby1
from scipy.signal import cheby2 as scipy_cheby2
from scipy.signal import ellip as scipy_ellip
from scipy.signal import firwin as scipy_firwin

from torchfx.filter._design import (
    design_butterworth_sos,
    design_cheby1_sos,
    design_cheby2_sos,
    design_ellip_sos,
    design_firwin,
)

# Common parameter axes. Keep these short — the goal is to expose cost trends,
# not exhaustively cover the parameter space (the equivalence tests do that).
ORDERS = [2, 4, 8, 16]
BTYPES = ["lowpass", "highpass"]
WN = 1000.0 / 24000.0  # 1 kHz at 48 kHz sample rate, normalized to Nyquist
WIN_TAPS = [31, 127, 511]
FIR_WINDOWS = ["hann", "hamming", "blackman", "kaiser"]


# --------------------------------------------------------------------------- #
# Butterworth                                                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.benchmark(group="butterworth")
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("btype", BTYPES)
def test_native_butterworth(benchmark, order, btype):
    benchmark(design_butterworth_sos, order, WN, btype)


@pytest.mark.benchmark(group="butterworth")
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("btype", BTYPES)
def test_scipy_butterworth(benchmark, order, btype):
    benchmark(scipy_butter, order, WN, btype=btype, output="sos")


# --------------------------------------------------------------------------- #
# Chebyshev I                                                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.benchmark(group="cheby1")
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("rp", [0.1, 1.0])
def test_native_cheby1(benchmark, order, rp):
    benchmark(design_cheby1_sos, order, rp, WN, "lowpass")


@pytest.mark.benchmark(group="cheby1")
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("rp", [0.1, 1.0])
def test_scipy_cheby1(benchmark, order, rp):
    benchmark(scipy_cheby1, order, rp, WN, btype="lowpass", output="sos")


# --------------------------------------------------------------------------- #
# Chebyshev II                                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.benchmark(group="cheby2")
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("rs", [40, 80])
def test_native_cheby2(benchmark, order, rs):
    benchmark(design_cheby2_sos, order, rs, WN, "lowpass")


@pytest.mark.benchmark(group="cheby2")
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("rs", [40, 80])
def test_scipy_cheby2(benchmark, order, rs):
    benchmark(scipy_cheby2, order, rs, WN, btype="lowpass", output="sos")


# --------------------------------------------------------------------------- #
# Elliptic                                                                    #
# --------------------------------------------------------------------------- #


@pytest.mark.benchmark(group="elliptic")
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("rp_rs", [(0.1, 60), (1.0, 80)])
def test_native_elliptic(benchmark, order, rp_rs):
    rp, rs = rp_rs
    benchmark(design_ellip_sos, order, rp, rs, WN, "lowpass")


@pytest.mark.benchmark(group="elliptic")
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("rp_rs", [(0.1, 60), (1.0, 80)])
def test_scipy_elliptic(benchmark, order, rp_rs):
    rp, rs = rp_rs
    benchmark(scipy_ellip, order, rp, rs, WN, btype="lowpass", output="sos")


# --------------------------------------------------------------------------- #
# FIR firwin                                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.benchmark(group="firwin")
@pytest.mark.parametrize("num_taps", WIN_TAPS)
@pytest.mark.parametrize("window", FIR_WINDOWS)
def test_native_firwin(benchmark, num_taps, window):
    win = ("kaiser", 8.6) if window == "kaiser" else window
    benchmark(design_firwin, num_taps, 1000.0, 48000.0, True, win)


@pytest.mark.benchmark(group="firwin")
@pytest.mark.parametrize("num_taps", WIN_TAPS)
@pytest.mark.parametrize("window", FIR_WINDOWS)
def test_scipy_firwin(benchmark, num_taps, window):
    win = ("kaiser", 8.6) if window == "kaiser" else window
    benchmark(scipy_firwin, num_taps, 1000.0, fs=48000.0, pass_zero=True, window=win)
