# ruff: noqa: A001

import pytest
import torch
from scipy.signal import butter, cheby1, cheby2

from torchfx.filter import (
    Butterworth,
    Chebyshev1,
    Chebyshev2,
    HiButterworth,
    HiChebyshev2,
    LoChebyshev1,
)


@pytest.fixture
def sample_signal():
    # Create a sample signal for testing
    return torch.sin(torch.linspace(0, 2 * torch.pi, 2000))  # 1 second of a sine wave


def test_butterworth_coefficients():
    fs = 1000
    cutoff = 100
    filter = Butterworth(btype="low", cutoff=cutoff, order=4, fs=fs)
    filter.compute_coefficients()

    b, a = butter(4, cutoff / (0.5 * fs), btype="low")
    assert filter.b == pytest.approx(b, rel=1e-3)
    assert filter.a == pytest.approx(a, rel=1e-3)


def test_chebyshev1_coefficients():
    fs = 1000
    cutoff = 100
    ripple = 0.1
    filter = Chebyshev1(btype="low", cutoff=cutoff, order=4, ripple=ripple, fs=fs)
    filter.compute_coefficients()

    b, a = cheby1(4, ripple, cutoff / (0.5 * fs), btype="low")
    assert filter.b == pytest.approx(b, rel=1e-3)
    assert filter.a == pytest.approx(a, rel=1e-3)


def test_chebyshev2_coefficients():
    fs = 1000
    cutoff = 100
    ripple = 0.1
    filter = Chebyshev2(btype="low", cutoff=cutoff, order=4, ripple=ripple, fs=fs)
    filter.compute_coefficients()

    b, a = cheby2(4, ripple, cutoff / (0.5 * fs), btype="low")
    assert filter.b == pytest.approx(b, rel=1e-3)
    assert filter.a == pytest.approx(a, rel=1e-3)


def test_highpass_butterworth(sample_signal):
    fs = 1000
    cutoff = 100
    filter = HiButterworth(cutoff=cutoff, order=5, fs=fs)
    filter.compute_coefficients()

    # Ensure the filter can process the signal
    filtered_signal = filter.forward(sample_signal)
    assert filtered_signal.shape == sample_signal.shape


def test_lowpass_chebyshev1(sample_signal):
    fs = 1000
    cutoff = 100
    filter = LoChebyshev1(cutoff=cutoff, order=4, fs=fs)
    filter.compute_coefficients()

    # Ensure the filter can process the signal
    filtered_signal = filter.forward(sample_signal)
    assert filtered_signal.shape == sample_signal.shape


def test_highpass_chebyshev2(sample_signal):
    fs = 1000
    cutoff = 100
    filter = HiChebyshev2(cutoff=cutoff, order=4, fs=fs)
    filter.compute_coefficients()

    # Ensure the filter can process the signal
    filtered_signal = filter.forward(sample_signal)
    assert filtered_signal.shape == sample_signal.shape
