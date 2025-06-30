import pytest
import torch

from torchfx import Wave  # Replace with the actual module name
from torchfx.filter import (
    HiButterworth,
    LoButterworth,
)


@pytest.fixture
def sample_wave():
    # Create a sample wave for testing
    signal = torch.sin(torch.linspace(0, 2 * torch.pi, 1000)).unsqueeze(0)  # [1, 1000]
    fs = 1000
    return Wave(signal, fs)


@pytest.fixture
def lowpass_filter():
    # Create a low pass filter instance
    return LoButterworth(cutoff=200, fs=1000)


@pytest.fixture
def highpass_filter():
    # Create a high pass filter instance
    return HiButterworth(cutoff=50, fs=1000)


def test_wave_initialization():
    # Test initialization of Wave
    signal = torch.tensor([[0.0, 1.0, 0.0]])
    fs = 1000
    wave = Wave(signal, fs)

    assert wave.fs == fs
    assert torch.equal(wave.ys, signal)


def test_wave_pipe_operator(sample_wave, lowpass_filter, highpass_filter):
    # Test applying filters using the pipe operator
    filtered_wave = sample_wave | lowpass_filter | highpass_filter

    # Check if the filtered wave is still a Wave object
    assert isinstance(filtered_wave, Wave)

    # Check if the sampling frequency is maintained
    assert filtered_wave.fs == sample_wave.fs

    # Check if the filtered signal has the same shape as the input
    assert filtered_wave.ys.shape == sample_wave.ys.shape

    # Optionally, you can add more specific checks for the filtered signal
    # For example, verifying that certain frequencies are attenuated


def test_wave_transform(sample_wave):
    # Test the transform method
    transformed_wave = sample_wave.transform(torch.fft.fft)

    # Check if the transformed wave is still a Wave object
    assert isinstance(transformed_wave, Wave)

    # Check if the transformed signal has the same shape as the input
    assert transformed_wave.ys.shape == sample_wave.ys.shape
