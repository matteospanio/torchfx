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


def test_designable_fir_deferred_init_is_valid_module():
    fir = DesignableFIR(cutoff=5000, num_taps=101, fs=None)

    state = fir.state_dict()
    assert "kernel" in state
    assert fir.b is None


def test_designable_fir_deferred_init_compute_after_fs_set():
    fir = DesignableFIR(cutoff=5000, num_taps=101, fs=None)
    fir.fs = 44100
    fir.compute_coefficients()

    signal = torch.randn(44100)
    result = fir(signal)
    assert result.shape == signal.shape


# ---------------------------------------------------------------------------
# FFT convolution tests
# ---------------------------------------------------------------------------


def test_default_conv_mode_is_fft():
    fir = FIR([0.2, 0.2, 0.2, 0.2, 0.2])
    assert fir._conv_mode == "fft"


def test_conv_mode_validation():
    with pytest.raises(ValueError, match="conv_mode"):
        FIR([0.2, 0.2, 0.2], conv_mode="invalid")


@pytest.mark.parametrize("num_taps", [5, 32, 64, 128, 256, 512, 1024])
def test_fft_matches_direct(num_taps):
    """FFT and direct convolution produce numerically equivalent output."""
    b = firwin(num_taps, 5000, fs=44100, window="hamming")
    fir_fft = FIR(b, conv_mode="fft")
    fir_direct = FIR(b, conv_mode="direct")

    signal = torch.randn(2, 44100)
    out_fft = fir_fft(signal)
    out_direct = fir_direct(signal)

    assert torch.allclose(out_fft, out_direct, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "shape",
    [(44100,), (2, 44100), (4, 2, 44100)],
    ids=["mono", "stereo", "batch"],
)
def test_fft_conv_shapes(shape):
    b = [0.1, 0.15, 0.5, 0.15, 0.1]
    fir = FIR(b, conv_mode="fft")
    signal = torch.randn(*shape)
    result = fir(signal)
    assert result.shape == signal.shape


def test_designable_fir_conv_mode():
    fir = DesignableFIR(cutoff=5000, num_taps=101, fs=44100, conv_mode="fft")
    assert fir._conv_mode == "fft"
    signal = torch.randn(44100)
    result = fir(signal)
    assert result.shape == signal.shape


def test_designable_fir_direct_mode():
    fir = DesignableFIR(cutoff=5000, num_taps=101, fs=44100, conv_mode="direct")
    assert fir._conv_mode == "direct"
    signal = torch.randn(44100)
    result = fir(signal)
    assert result.shape == signal.shape


def test_fft_conv_short_signal():
    """FFT path works when signal is short but >= kernel after padding."""
    b = firwin(32, 5000, fs=44100, window="hamming")
    fir_fft = FIR(b, conv_mode="fft")
    fir_direct = FIR(b, conv_mode="direct")
    signal = torch.randn(1, 100)
    out_fft = fir_fft(signal)
    out_direct = fir_direct(signal)
    assert out_fft.shape == signal.shape
    assert torch.allclose(out_fft, out_direct, atol=1e-4, rtol=1e-4)
