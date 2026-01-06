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
    HiElliptic,
    HiShelving,
    LoChebyshev1,
    LoElliptic,
    LoShelving,
    ParametricEQ,
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


class TestShelvingFilters:
    """Test suite for shelving filters."""

    @pytest.fixture
    def sample_signal_2d(self):
        """Create a 2D signal for testing."""
        return torch.sin(torch.linspace(0, 2 * torch.pi, 2000)).unsqueeze(0)

    def test_hishelving_coefficients(self):
        """Test HiShelving filter coefficient computation."""
        fs = 44100
        cutoff = 1000
        q = 0.707
        gain = 6.0  # dB

        filter = HiShelving(cutoff=cutoff, q=q, gain=gain, gain_scale="db", fs=fs)
        filter.compute_coefficients()

        # Verify coefficients exist
        assert filter.a is not None
        assert filter.b is not None
        assert len(filter.a) == 3
        assert len(filter.b) == 3

        # Verify first coefficient of denominator is normalized to 1
        assert filter.a[0] == pytest.approx(1.0, abs=1e-6)

    def test_loshelving_coefficients(self):
        """Test LoShelving filter coefficient computation."""
        fs = 44100
        cutoff = 1000
        q = 0.707
        gain = 6.0  # dB

        filter = LoShelving(cutoff=cutoff, q=q, gain=gain, gain_scale="db", fs=fs)
        filter.compute_coefficients()

        # Verify coefficients exist
        assert filter.a is not None
        assert filter.b is not None
        assert len(filter.a) == 3
        assert len(filter.b) == 3

        # Verify first coefficient of denominator is normalized to 1
        assert filter.a[0] == pytest.approx(1.0, abs=1e-6)

    def test_hishelving_forward(self, sample_signal_2d):
        """Test HiShelving filter forward pass."""
        fs = 2000
        cutoff = 400
        q = 0.707
        gain = 3.0  # dB

        filter = HiShelving(cutoff=cutoff, q=q, gain=gain, gain_scale="db", fs=fs)
        filtered_signal = filter.forward(sample_signal_2d)

        # Ensure output shape matches input
        assert filtered_signal.shape == sample_signal_2d.shape

        # Ensure output is not identical to input (filter has effect)
        assert not torch.allclose(filtered_signal, sample_signal_2d, atol=1e-3)

    def test_loshelving_forward(self, sample_signal_2d):
        """Test LoShelving filter forward pass."""
        fs = 2000
        cutoff = 400
        q = 0.707
        gain = 3.0  # dB

        filter = LoShelving(cutoff=cutoff, q=q, gain=gain, gain_scale="db", fs=fs)
        filtered_signal = filter.forward(sample_signal_2d)

        # Ensure output shape matches input
        assert filtered_signal.shape == sample_signal_2d.shape

        # Ensure output is not identical to input (filter has effect)
        assert not torch.allclose(filtered_signal, sample_signal_2d, atol=1e-3)

    def test_shelving_linear_gain_scale(self):
        """Test shelving filters with linear gain scale."""
        fs = 44100
        cutoff = 1000
        q = 0.707
        gain_linear = 2.0

        hi_filter = HiShelving(cutoff=cutoff, q=q, gain=gain_linear, gain_scale="linear", fs=fs)
        lo_filter = LoShelving(cutoff=cutoff, q=q, gain=gain_linear, gain_scale="linear", fs=fs)

        assert hi_filter.gain == gain_linear
        assert lo_filter.gain == gain_linear

    def test_shelving_db_gain_scale(self):
        """Test shelving filters with dB gain scale."""
        fs = 44100
        cutoff = 1000
        q = 0.707
        gain_db = 6.0

        hi_filter = HiShelving(cutoff=cutoff, q=q, gain=gain_db, gain_scale="db", fs=fs)
        lo_filter = LoShelving(cutoff=cutoff, q=q, gain=gain_db, gain_scale="db", fs=fs)

        expected_linear_gain = 10 ** (gain_db / 20)
        assert hi_filter.gain == pytest.approx(expected_linear_gain, rel=1e-6)
        assert lo_filter.gain == pytest.approx(expected_linear_gain, rel=1e-6)

    def test_shelving_positive_gain(self, sample_signal_2d):
        """Test shelving filters with positive gain boost the signal."""
        fs = 2000
        cutoff = 400
        q = 0.707
        gain = 6.0  # dB

        hi_filter = HiShelving(cutoff=cutoff, q=q, gain=gain, gain_scale="db", fs=fs)
        lo_filter = LoShelving(cutoff=cutoff, q=q, gain=gain, gain_scale="db", fs=fs)

        hi_filtered = hi_filter.forward(sample_signal_2d)
        lo_filtered = lo_filter.forward(sample_signal_2d)

        # With positive gain, output energy should be higher than input
        input_energy = torch.sum(sample_signal_2d**2)
        hi_energy = torch.sum(hi_filtered**2)
        lo_energy = torch.sum(lo_filtered**2)

        assert hi_energy > input_energy
        assert lo_energy > input_energy


class TestParametricEQ:
    """Test suite for Parametric EQ filter."""

    @pytest.fixture
    def sample_signal_2d(self):
        """Create a 2D signal for testing."""
        return torch.sin(torch.linspace(0, 2 * torch.pi, 2000)).unsqueeze(0)

    def test_parametric_eq_coefficients(self):
        """Test ParametricEQ coefficient computation."""
        fs = 44100
        frequency = 1000
        q = 0.707
        gain = 6.0  # dB

        eq = ParametricEQ(frequency=frequency, q=q, gain=gain, fs=fs)
        eq.compute_coefficients()

        # Verify coefficients exist
        assert eq.a is not None
        assert eq.b is not None
        assert len(eq.a) == 3
        assert len(eq.b) == 3

        # Verify first coefficient of denominator is normalized to 1
        assert eq.a[0] == pytest.approx(1.0, abs=1e-6)

    def test_parametric_eq_forward_boost(self, sample_signal_2d):
        """Test ParametricEQ with boost."""
        fs = 2000
        frequency = 500
        q = 1.0
        gain = 6.0  # dB

        eq = ParametricEQ(frequency=frequency, q=q, gain=gain, fs=fs)
        filtered = eq.forward(sample_signal_2d)

        # Ensure output shape matches input
        assert filtered.shape == sample_signal_2d.shape

        # With positive gain, output energy should be higher
        assert torch.sum(filtered**2) > torch.sum(sample_signal_2d**2)

    def test_parametric_eq_forward_cut(self, sample_signal_2d):
        """Test ParametricEQ with cut."""
        fs = 2000
        frequency = 500
        q = 1.0
        gain = -6.0  # dB

        eq = ParametricEQ(frequency=frequency, q=q, gain=gain, fs=fs)
        filtered = eq.forward(sample_signal_2d)

        # Ensure output shape matches input
        assert filtered.shape == sample_signal_2d.shape

        # With negative gain, output energy should be lower
        assert torch.sum(filtered**2) < torch.sum(sample_signal_2d**2)

    def test_parametric_eq_different_q_values(self):
        """Test ParametricEQ with different Q values."""
        fs = 44100
        frequency = 1000
        gain = 6.0

        for q in [0.5, 0.707, 1.0, 2.0, 5.0]:
            eq = ParametricEQ(frequency=frequency, q=q, gain=gain, fs=fs)
            eq.compute_coefficients()

            assert eq.a is not None
            assert eq.b is not None


class TestEllipticFilters:
    """Test suite for Elliptic filters."""

    @pytest.fixture
    def sample_signal_2d(self):
        """Create a 2D signal for testing."""
        return torch.sin(torch.linspace(0, 2 * torch.pi, 2000)).unsqueeze(0)

    def test_elliptic_lowpass_coefficients(self):
        """Test low-pass Elliptic filter coefficient computation."""
        fs = 44100
        cutoff = 1000
        order = 4

        filter = LoElliptic(cutoff=cutoff, order=order, fs=fs)
        filter.compute_coefficients()

        # Verify coefficients exist
        assert filter.a is not None
        assert filter.b is not None

    def test_elliptic_highpass_coefficients(self):
        """Test high-pass Elliptic filter coefficient computation."""
        fs = 44100
        cutoff = 200
        order = 4

        filter = HiElliptic(cutoff=cutoff, order=order, fs=fs)
        filter.compute_coefficients()

        # Verify coefficients exist
        assert filter.a is not None
        assert filter.b is not None

    def test_elliptic_lowpass_forward(self, sample_signal_2d):
        """Test low-pass Elliptic filter forward pass."""
        fs = 2000
        cutoff = 400
        order = 4

        filter = LoElliptic(cutoff=cutoff, order=order, fs=fs)
        filtered = filter.forward(sample_signal_2d)

        # Ensure output shape matches input
        assert filtered.shape == sample_signal_2d.shape

        # Ensure output is not identical to input (filter has effect)
        assert not torch.allclose(filtered, sample_signal_2d, atol=1e-3)

    def test_elliptic_highpass_forward(self, sample_signal_2d):
        """Test high-pass Elliptic filter forward pass."""
        fs = 2000
        cutoff = 400
        order = 4

        filter = HiElliptic(cutoff=cutoff, order=order, fs=fs)
        filtered = filter.forward(sample_signal_2d)

        # Ensure output shape matches input
        assert filtered.shape == sample_signal_2d.shape

        # Ensure output is not identical to input (filter has effect)
        assert not torch.allclose(filtered, sample_signal_2d, atol=1e-3)

    def test_elliptic_custom_parameters(self):
        """Test Elliptic filter with custom passband/stopband parameters."""
        fs = 44100
        cutoff = 1000
        order = 6
        passband_ripple = 0.5
        stopband_attenuation = 60

        filter = LoElliptic(
            cutoff=cutoff,
            order=order,
            passband_ripple=passband_ripple,
            stopband_attenuation=stopband_attenuation,
            fs=fs,
        )
        filter.compute_coefficients()

        assert filter.a is not None
        assert filter.b is not None
        assert filter.passband_ripple == passband_ripple
        assert filter.stopband_attenuation == stopband_attenuation
