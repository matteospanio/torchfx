import pytest
import torch
from scipy.signal import firwin

from torchfx.filter import FIR, DesignableFIR


@pytest.fixture
def sample_signal():
    # Create a sample signal for testing
    return torch.linspace(0, 1, 44100).unsqueeze(0)  # Shape: [1, 44100]


def test_fir_initialization():
    # Test initialization of FIR with known coefficients
    b = [0.1, 0.15, 0.5, 0.15, 0.1]
    fir_filter = FIR(b)

    # Check if kernel is correctly registered
    assert torch.allclose(fir_filter.kernel[0, 0], torch.tensor(b[::-1], dtype=torch.float32))


def test_fir_forward(sample_signal):
    # Test forward pass of FIR filter
    b = [0.1, 0.15, 0.5, 0.15, 0.1]
    fir_filter = FIR(b)

    # Apply filter
    filtered_signal = fir_filter.forward(sample_signal)

    # Check if filtered signal has same shape as input
    assert filtered_signal.shape == sample_signal.shape


def test_designable_fir_coefficients():
    # Test coefficient computation of DesignableFIR
    num_taps = 5
    cutoff = 0.2
    fs = 44100
    designable_fir = DesignableFIR(cutoff=cutoff, num_taps=num_taps, fs=fs)

    # Compute expected coefficients using scipy
    expected_b = firwin(num_taps, cutoff, fs=fs, pass_zero=True, window="hamming", scale=True)

    # Check if computed coefficients match expected coefficients
    assert designable_fir.b is not None
    assert designable_fir.b == pytest.approx(expected_b, rel=1e-3)


def test_designable_fir_forward(sample_signal):
    # Test forward pass of DesignableFIR
    num_taps = 5
    cutoff = 0.2
    fs = 44100
    designable_fir = DesignableFIR(cutoff=cutoff, num_taps=num_taps, fs=fs)

    # Apply filter
    filtered_signal = designable_fir.forward(sample_signal)

    # Check if filtered signal has same shape as input
    assert filtered_signal.shape == sample_signal.shape
