import tempfile
from pathlib import Path

import pytest
import soundfile as sf
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


class TestWaveSave:
    """Test suite for Wave.save() method."""

    @pytest.fixture
    def stereo_wave(self):
        """Create a stereo wave for testing."""
        # Create a stereo signal (2 channels)
        left = torch.sin(2 * torch.pi * 440 * torch.linspace(0, 1, 44100))
        right = torch.sin(2 * torch.pi * 880 * torch.linspace(0, 1, 44100))
        signal = torch.stack([left, right])  # [2, 44100]
        return Wave(signal, fs=44100)

    @pytest.fixture
    def mono_wave(self):
        """Create a mono wave for testing."""
        signal = torch.sin(2 * torch.pi * 440 * torch.linspace(0, 1, 44100)).unsqueeze(0)
        return Wave(signal, fs=44100)

    def test_save_wav_basic(self, mono_wave):
        """Test saving a basic WAV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_output.wav"

            # Save the wave
            mono_wave.save(output_path)

            # Verify file exists
            assert output_path.exists()

            # Load and verify
            loaded_wave = Wave.from_file(output_path)
            assert loaded_wave.fs == mono_wave.fs
            assert loaded_wave.ys.shape == mono_wave.ys.shape
            assert torch.allclose(loaded_wave.ys, mono_wave.ys, atol=1e-4)

    def test_save_wav_stereo(self, stereo_wave):
        """Test saving a stereo WAV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_stereo.wav"

            stereo_wave.save(output_path)

            assert output_path.exists()

            loaded_wave = Wave.from_file(output_path)
            assert loaded_wave.fs == stereo_wave.fs
            assert loaded_wave.channels() == 2
            assert torch.allclose(loaded_wave.ys, stereo_wave.ys, atol=1e-4)

    def test_save_flac(self, mono_wave):
        """Test saving a FLAC file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_output.flac"

            mono_wave.save(output_path, format="flac")

            assert output_path.exists()

            loaded_wave = Wave.from_file(output_path)
            assert loaded_wave.fs == mono_wave.fs
            assert torch.allclose(loaded_wave.ys, mono_wave.ys, atol=1e-4)

    def test_save_high_bit_depth_32bit_float(self, mono_wave):
        """Test saving with 32-bit float encoding."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_32bit.wav"

            mono_wave.save(output_path, encoding="PCM_F", bits_per_sample=32)

            assert output_path.exists()

            # Verify encoding via soundfile
            info = sf.info(str(output_path))
            assert info.subtype == "FLOAT"

    def test_save_high_bit_depth_64bit_float(self, mono_wave):
        """Test saving with 64-bit float encoding."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_64bit.wav"

            mono_wave.save(output_path, encoding="PCM_F", bits_per_sample=64)

            assert output_path.exists()

            # Verify encoding via soundfile
            info = sf.info(str(output_path))
            assert info.subtype == "DOUBLE"

    def test_save_high_sample_rate(self):
        """Test saving with high sample rates (96kHz, 192kHz)."""
        for sample_rate in [96000, 192000]:
            signal = torch.sin(2 * torch.pi * 1000 * torch.linspace(0, 1, sample_rate)).unsqueeze(0)
            wave = Wave(signal, fs=sample_rate)

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / f"test_{sample_rate}hz.wav"

                wave.save(output_path)

                assert output_path.exists()

                info = sf.info(str(output_path))
                assert info.samplerate == sample_rate

    def test_save_with_string_path(self, mono_wave):
        """Test saving with a string path instead of Path object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/test_string.wav"

            mono_wave.save(output_path)

            assert Path(output_path).exists()

    def test_save_different_bit_depths(self, mono_wave):
        """Test saving with different bit depths (16, 24, 32)."""
        expected_subtypes = {16: "PCM_16", 24: "PCM_24", 32: "PCM_32"}
        for bits in [16, 24, 32]:
            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / f"test_{bits}bit.wav"

                mono_wave.save(output_path, bits_per_sample=bits)

                assert output_path.exists()

                info = sf.info(str(output_path))
                assert info.subtype == expected_subtypes[bits]

    def test_save_infers_format_from_extension(self, mono_wave):
        """Test that format is inferred from file extension."""
        extensions = {
            "wav": "WAV",
            "flac": "FLAC",
        }

        for ext, expected_format in extensions.items():
            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / f"test.{ext}"

                # Don't specify format, let it be inferred
                mono_wave.save(output_path)

                assert output_path.exists()
                loaded_wave = Wave.from_file(output_path)
                assert loaded_wave.fs == mono_wave.fs

    def test_save_creates_parent_directories(self, mono_wave):
        """Test that save creates parent directories if they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "nested" / "test.wav"

            # Parent directories don't exist yet
            assert not output_path.parent.exists()

            mono_wave.save(output_path)

            assert output_path.exists()
            assert output_path.parent.exists()

    def test_save_overwrite_existing_file(self, mono_wave):
        """Test that save overwrites existing files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.wav"

            # Save first time
            mono_wave.save(output_path)
            first_mtime = output_path.stat().st_mtime

            # Save again (should overwrite)
            import time

            time.sleep(0.01)  # Ensure different timestamp
            mono_wave.save(output_path)
            second_mtime = output_path.stat().st_mtime

            assert second_mtime >= first_mtime

    def test_metadata_preservation(self, mono_wave):
        """Test that metadata is preserved when saving and loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_metadata.wav"

            # Save with specific encoding
            mono_wave.save(output_path, encoding="PCM_S", bits_per_sample=24)

            # Load and check metadata
            loaded_wave = Wave.from_file(output_path)
            assert loaded_wave.metadata is not None
            assert loaded_wave.metadata["subtype"] == "PCM_24"
            assert loaded_wave.metadata["num_channels"] == mono_wave.channels()

    def test_wave_with_metadata_initialization(self):
        """Test creating a Wave with custom metadata."""
        signal = torch.sin(2 * torch.pi * 440 * torch.linspace(0, 1, 44100)).unsqueeze(0)
        metadata = {
            "artist": "Test Artist",
            "title": "Test Song",
        }
        wave = Wave(signal, fs=44100, metadata=metadata)

        assert wave.metadata["artist"] == "Test Artist"
        assert wave.metadata["title"] == "Test Song"
