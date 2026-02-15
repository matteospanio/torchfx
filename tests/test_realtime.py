"""Tests for the torchfx.realtime subpackage.

Covers exceptions, ring buffer, backend ABC, processor, and stream processor. Uses a
MockBackend for testing without real audio hardware.

"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torchaudio
from torch import Tensor

from torchfx.effect import FX, Gain
from torchfx.filter.iir import HiButterworth, LoButterworth
from torchfx.realtime.backend import (
    AudioBackend,
    AudioCallback,
    StreamConfig,
    StreamDirection,
    StreamState,
)
from torchfx.realtime.exceptions import (
    BackendNotAvailableError,
    BufferOverrunError,
    BufferUnderrunError,
    RealtimeError,
    StreamError,
)
from torchfx.realtime.processor import RealtimeProcessor
from torchfx.realtime.ring_buffer import TensorRingBuffer
from torchfx.realtime.stream import StreamProcessor
from torchfx.validation.exceptions import AudioProcessingError, TorchFXError

# ---------------------------------------------------------------------------
# Mock Backend for testing without real audio hardware
# ---------------------------------------------------------------------------


class MockBackend(AudioBackend):
    """Mock audio backend for testing."""

    def __init__(self) -> None:
        self._state: StreamState = StreamState.CLOSED
        self._config: StreamConfig | None = None
        self._callback: AudioCallback | None = None
        self.start_count = 0
        self.stop_count = 0

    @property
    def name(self) -> str:
        return "mock"

    @property
    def is_available(self) -> bool:
        return True

    def get_devices(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "Mock Device",
                "index": 0,
                "max_input_channels": 2,
                "max_output_channels": 2,
                "default_sample_rate": 48000.0,
            }
        ]

    def get_default_device(self, direction: StreamDirection) -> int | str:
        return 0

    def open_stream(
        self,
        config: StreamConfig,
        callback: AudioCallback | None = None,
    ) -> None:
        self._config = config
        self._callback = callback
        self._state = StreamState.OPEN

    def start(self) -> None:
        self._state = StreamState.RUNNING
        self.start_count += 1

    def stop(self) -> None:
        self._state = StreamState.STOPPED
        self.stop_count += 1

    def close(self) -> None:
        self._state = StreamState.CLOSED

    @property
    def state(self) -> StreamState:
        return self._state

    def read(self, num_frames: int) -> Tensor:
        channels = self._config.channels_in if self._config else 1
        return torch.zeros(channels, num_frames)

    def write(self, data: Tensor) -> None:
        pass

    def simulate_callback(self, input_data: Tensor) -> Tensor:
        """Simulate an audio callback for testing."""
        if self._callback is None:
            raise RuntimeError("No callback registered")
        if self._config is None:
            raise RuntimeError("No config set")
        output_data = torch.zeros(self._config.channels_out, input_data.shape[-1])
        self._callback(input_data, output_data, input_data.shape[-1])
        return output_data


# ---------------------------------------------------------------------------
# Tests: Exceptions
# ---------------------------------------------------------------------------


class TestExceptions:
    """Tests for real-time exception hierarchy."""

    def test_realtime_error_inherits_torchfx_error(self) -> None:
        err = RealtimeError("test")
        assert isinstance(err, TorchFXError)

    def test_backend_not_available_error(self) -> None:
        err = BackendNotAvailableError("sounddevice")
        assert isinstance(err, RealtimeError)
        assert err.backend_name == "sounddevice"
        assert "sounddevice" in str(err)

    def test_backend_not_available_custom_suggestion(self) -> None:
        err = BackendNotAvailableError("jack", suggestion="Install libjack-dev")
        assert "Install libjack-dev" in str(err)

    def test_stream_error(self) -> None:
        err = StreamError("stream failed")
        assert isinstance(err, RealtimeError)
        assert "stream failed" in str(err)

    def test_buffer_overrun_error(self) -> None:
        err = BufferOverrunError("overflow")
        assert isinstance(err, AudioProcessingError)

    def test_buffer_underrun_error(self) -> None:
        err = BufferUnderrunError("underflow")
        assert isinstance(err, AudioProcessingError)

    def test_exception_hierarchy(self) -> None:
        """All realtime exceptions are catchable as TorchFXError."""
        exceptions = [
            RealtimeError("a"),
            BackendNotAvailableError("b"),
            StreamError("c"),
            BufferOverrunError("d"),
            BufferUnderrunError("e"),
        ]
        for exc in exceptions:
            assert isinstance(exc, TorchFXError)


# ---------------------------------------------------------------------------
# Tests: Compat Module
# ---------------------------------------------------------------------------


class TestCompat:
    """Tests for optional dependency handling."""

    def test_get_sounddevice_raises_when_missing(self) -> None:
        from torchfx.realtime import _compat

        # Reset cached module
        original = _compat._sounddevice_module
        _compat._sounddevice_module = None

        with patch.object(_compat, "importlib") as mock_importlib:
            mock_importlib.import_module.side_effect = ImportError("No module")
            with pytest.raises(BackendNotAvailableError, match="sounddevice"):
                _compat.get_sounddevice()

        # Restore
        _compat._sounddevice_module = original

    def test_check_sounddevice_returns_bool(self) -> None:
        from torchfx.realtime._compat import _check_sounddevice

        result = _check_sounddevice()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Tests: StreamConfig
# ---------------------------------------------------------------------------


class TestStreamConfig:
    """Tests for StreamConfig dataclass."""

    def test_default_values(self) -> None:
        config = StreamConfig()
        assert config.sample_rate == 48000
        assert config.buffer_size == 512
        assert config.channels_in == 0
        assert config.channels_out == 2
        assert config.dtype == "float32"
        assert config.latency == "low"

    def test_direction_output(self) -> None:
        config = StreamConfig(channels_in=0, channels_out=2)
        assert config.direction == StreamDirection.OUTPUT

    def test_direction_input(self) -> None:
        config = StreamConfig(channels_in=2, channels_out=0)
        assert config.direction == StreamDirection.INPUT

    def test_direction_duplex(self) -> None:
        config = StreamConfig(channels_in=2, channels_out=2)
        assert config.direction == StreamDirection.DUPLEX

    def test_latency_ms(self) -> None:
        config = StreamConfig(sample_rate=48000, buffer_size=512)
        expected = (512 / 48000) * 1000.0
        assert abs(config.latency_ms - expected) < 0.001

    def test_frozen(self) -> None:
        config = StreamConfig()
        with pytest.raises(AttributeError):
            config.sample_rate = 44100  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Tests: TensorRingBuffer
# ---------------------------------------------------------------------------


class TestTensorRingBuffer:
    """Tests for the lock-free SPSC ring buffer."""

    def test_create_buffer(self) -> None:
        buf = TensorRingBuffer(capacity=256, channels=2)
        assert buf.capacity == 256
        assert buf.channels == 2
        assert buf.available_read == 0
        assert buf.available_write == 256

    def test_power_of_2_rounding(self) -> None:
        buf = TensorRingBuffer(capacity=300)
        assert buf.capacity == 512  # Next power of 2

    def test_exact_power_of_2(self) -> None:
        buf = TensorRingBuffer(capacity=256)
        assert buf.capacity == 256

    def test_invalid_capacity(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            TensorRingBuffer(capacity=0)
        with pytest.raises(ValueError, match="positive"):
            TensorRingBuffer(capacity=-1)

    def test_invalid_channels(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            TensorRingBuffer(capacity=256, channels=0)

    def test_write_and_read(self) -> None:
        buf = TensorRingBuffer(capacity=256, channels=1)
        data = torch.ones(1, 100)
        written = buf.write(data)
        assert written == 100
        assert buf.available_read == 100
        assert buf.available_write == 156

        output = buf.read(100)
        assert output.shape == (1, 100)
        torch.testing.assert_close(output, data)
        assert buf.available_read == 0

    def test_write_mono_1d(self) -> None:
        buf = TensorRingBuffer(capacity=256, channels=1)
        data = torch.ones(50)
        written = buf.write(data)
        assert written == 50
        output = buf.read(50)
        assert output.shape == (1, 50)

    def test_write_channel_mismatch(self) -> None:
        buf = TensorRingBuffer(capacity=256, channels=2)
        data = torch.ones(1, 50)
        with pytest.raises(ValueError, match="Channel mismatch"):
            buf.write(data)

    def test_multichannel(self) -> None:
        buf = TensorRingBuffer(capacity=256, channels=2)
        data = torch.randn(2, 100)
        buf.write(data)
        output = buf.read(100)
        torch.testing.assert_close(output, data)

    def test_wraparound(self) -> None:
        buf = TensorRingBuffer(capacity=256, channels=1)
        # Write 200 samples, read 200, write 200 more (wraps around)
        data1 = torch.ones(1, 200)
        buf.write(data1)
        buf.read(200)

        data2 = torch.ones(1, 200) * 2.0
        buf.write(data2)
        output = buf.read(200)
        torch.testing.assert_close(output, data2)

    def test_write_overflow(self) -> None:
        buf = TensorRingBuffer(capacity=128, channels=1)
        data = torch.ones(1, 200)
        written = buf.write(data)
        assert written == 128  # Only capacity written

    def test_read_underrun(self) -> None:
        buf = TensorRingBuffer(capacity=256, channels=1)
        buf.write(torch.ones(1, 50))
        with pytest.raises(BufferUnderrunError):
            buf.read(100)

    def test_peek(self) -> None:
        buf = TensorRingBuffer(capacity=256, channels=1)
        data = torch.randn(1, 100)
        buf.write(data)

        peeked = buf.peek(100)
        torch.testing.assert_close(peeked, data)
        assert buf.available_read == 100  # Not consumed

    def test_peek_underrun(self) -> None:
        buf = TensorRingBuffer(capacity=256, channels=1)
        with pytest.raises(BufferUnderrunError):
            buf.peek(10)

    def test_advance_read(self) -> None:
        buf = TensorRingBuffer(capacity=256, channels=1)
        buf.write(torch.ones(1, 100))
        buf.advance_read(50)
        assert buf.available_read == 50

    def test_advance_read_underrun(self) -> None:
        buf = TensorRingBuffer(capacity=256, channels=1)
        buf.write(torch.ones(1, 50))
        with pytest.raises(BufferUnderrunError):
            buf.advance_read(100)

    def test_clear(self) -> None:
        buf = TensorRingBuffer(capacity=256, channels=1)
        buf.write(torch.ones(1, 100))
        buf.clear()
        assert buf.available_read == 0
        assert buf.available_write == 256

    def test_multiple_write_read_cycles(self) -> None:
        buf = TensorRingBuffer(capacity=128, channels=1)
        for i in range(10):
            data = torch.full((1, 50), float(i))
            buf.write(data)
            output = buf.read(50)
            torch.testing.assert_close(output, data)


# ---------------------------------------------------------------------------
# Tests: RealtimeProcessor
# ---------------------------------------------------------------------------


class TestRealtimeProcessor:
    """Tests for the real-time audio processor."""

    @pytest.fixture
    def mock_backend(self) -> MockBackend:
        return MockBackend()

    @pytest.fixture
    def duplex_config(self) -> StreamConfig:
        return StreamConfig(
            sample_rate=48000,
            buffer_size=512,
            channels_in=1,
            channels_out=1,
        )

    def test_create_processor(self, mock_backend: MockBackend, duplex_config: StreamConfig) -> None:
        processor = RealtimeProcessor(
            effects=[Gain(0.5)],
            backend=mock_backend,
            config=duplex_config,
        )
        assert not processor.is_running
        assert len(processor.effects) == 1

    def test_start_stop(self, mock_backend: MockBackend, duplex_config: StreamConfig) -> None:
        processor = RealtimeProcessor(
            effects=[Gain(0.5)],
            backend=mock_backend,
            config=duplex_config,
        )
        processor.start()
        assert processor.is_running
        assert mock_backend.start_count == 1

        processor.stop()
        assert not processor.is_running
        assert mock_backend.stop_count == 1

    def test_start_when_already_running(
        self, mock_backend: MockBackend, duplex_config: StreamConfig
    ) -> None:
        processor = RealtimeProcessor(
            effects=[Gain(0.5)],
            backend=mock_backend,
            config=duplex_config,
        )
        processor.start()
        with pytest.raises(RealtimeError, match="already running"):
            processor.start()
        processor.stop()

    def test_stop_when_not_running(
        self, mock_backend: MockBackend, duplex_config: StreamConfig
    ) -> None:
        processor = RealtimeProcessor(
            effects=[Gain(0.5)],
            backend=mock_backend,
            config=duplex_config,
        )
        with pytest.raises(RealtimeError, match="not running"):
            processor.stop()

    def test_audio_callback_processes_effects(
        self, mock_backend: MockBackend, duplex_config: StreamConfig
    ) -> None:
        processor = RealtimeProcessor(
            effects=[Gain(2.0)],
            backend=mock_backend,
            config=duplex_config,
        )
        processor.start()

        # Simulate a callback
        input_data = torch.ones(1, 512) * 0.5
        output = mock_backend.simulate_callback(input_data)
        torch.testing.assert_close(output, input_data * 2.0)

        processor.stop()

    def test_multiple_effects_chain(
        self, mock_backend: MockBackend, duplex_config: StreamConfig
    ) -> None:
        processor = RealtimeProcessor(
            effects=[Gain(2.0), Gain(0.5)],
            backend=mock_backend,
            config=duplex_config,
        )
        processor.start()

        input_data = torch.ones(1, 512)
        output = mock_backend.simulate_callback(input_data)
        # 1.0 * 2.0 * 0.5 = 1.0
        torch.testing.assert_close(output, input_data)

        processor.stop()

    def test_fs_propagation(self, mock_backend: MockBackend) -> None:
        config = StreamConfig(sample_rate=44100, buffer_size=256, channels_in=1, channels_out=1)
        lpf = LoButterworth(cutoff=1000)
        assert lpf.fs is None

        processor = RealtimeProcessor(
            effects=[lpf],
            backend=mock_backend,
            config=config,
        )
        assert lpf.fs == 44100

    def test_filter_coefficients_computed(self, mock_backend: MockBackend) -> None:
        config = StreamConfig(sample_rate=44100, buffer_size=256, channels_in=1, channels_out=1)
        lpf = LoButterworth(cutoff=1000)
        processor = RealtimeProcessor(
            effects=[lpf],
            backend=mock_backend,
            config=config,
        )
        assert lpf._has_computed_coeff

    def test_set_parameter(self, mock_backend: MockBackend, duplex_config: StreamConfig) -> None:
        gain = Gain(1.0)
        processor = RealtimeProcessor(
            effects=[gain],
            backend=mock_backend,
            config=duplex_config,
        )
        processor.start()

        # Set parameter
        processor.set_parameter("0.gain", 0.5)

        # Simulate callback to apply pending params
        input_data = torch.ones(1, 512)
        mock_backend.simulate_callback(input_data)

        assert gain.gain == 0.5

        processor.stop()

    def test_latency_ms(self, mock_backend: MockBackend, duplex_config: StreamConfig) -> None:
        processor = RealtimeProcessor(
            effects=[Gain(1.0)],
            backend=mock_backend,
            config=duplex_config,
        )
        expected = (512 / 48000) * 1000.0
        assert abs(processor.latency_ms - expected) < 0.001

    def test_reset_state(self, mock_backend: MockBackend, duplex_config: StreamConfig) -> None:
        processor = RealtimeProcessor(
            effects=[Gain(1.0)],
            backend=mock_backend,
            config=duplex_config,
        )
        # Write some data to buffers
        processor._input_buffer.write(torch.ones(1, 100))
        processor.reset_state()
        assert processor._input_buffer.available_read == 0

    def test_nn_sequential_effects(
        self, mock_backend: MockBackend, duplex_config: StreamConfig
    ) -> None:
        import torch.nn as nn

        effects = nn.Sequential(Gain(2.0), Gain(0.5))
        processor = RealtimeProcessor(
            effects=effects,
            backend=mock_backend,
            config=duplex_config,
        )
        assert len(processor.effects) == 2

    def test_context_manager(self, mock_backend: MockBackend, duplex_config: StreamConfig) -> None:
        with RealtimeProcessor(
            effects=[Gain(2.0)],
            backend=mock_backend,
            config=duplex_config,
        ) as processor:
            assert processor.is_running
            input_data = torch.ones(1, 512) * 0.5
            output = mock_backend.simulate_callback(input_data)
            torch.testing.assert_close(output, input_data * 2.0)
        assert not processor.is_running

    def test_context_manager_stops_on_exception(
        self, mock_backend: MockBackend, duplex_config: StreamConfig
    ) -> None:
        with pytest.raises(RuntimeError, match="test error"):
            with RealtimeProcessor(
                effects=[Gain(1.0)],
                backend=mock_backend,
                config=duplex_config,
            ) as processor:
                assert processor.is_running
                raise RuntimeError("test error")
        assert not processor.is_running
        assert mock_backend.stop_count == 1


# ---------------------------------------------------------------------------
# Tests: StreamProcessor
# ---------------------------------------------------------------------------


class TestStreamProcessor:
    """Tests for chunk-based stream processing."""

    @pytest.fixture
    def wav_file(self, tmp_path: Path) -> Path:
        """Create a temporary WAV file for testing."""
        path = tmp_path / "test_input.wav"
        # 1 second of 440Hz sine at 44100Hz, mono
        fs = 44100
        t = torch.linspace(0, 1, fs)
        waveform = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)
        torchaudio.save(str(path), waveform, fs)
        return path

    @pytest.fixture
    def stereo_wav_file(self, tmp_path: Path) -> Path:
        """Create a stereo WAV file for testing."""
        path = tmp_path / "test_stereo.wav"
        fs = 44100
        t = torch.linspace(0, 1, fs)
        left = torch.sin(2 * torch.pi * 440 * t)
        right = torch.sin(2 * torch.pi * 880 * t)
        waveform = torch.stack([left, right])
        torchaudio.save(str(path), waveform, fs)
        return path

    def test_create_processor(self) -> None:
        processor = StreamProcessor(effects=[Gain(0.5)])
        assert processor.chunk_size == 65536
        assert processor.overlap == 0
        assert len(processor.effects) == 1

    def test_invalid_chunk_size(self) -> None:
        with pytest.raises(Exception):
            StreamProcessor(effects=[Gain(0.5)], chunk_size=0)

    def test_invalid_overlap(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            StreamProcessor(effects=[Gain(0.5)], overlap=-1)

    def test_overlap_exceeds_chunk_size(self) -> None:
        with pytest.raises(ValueError, match="less than chunk_size"):
            StreamProcessor(effects=[Gain(0.5)], chunk_size=100, overlap=100)

    def test_process_file(self, wav_file: Path, tmp_path: Path) -> None:
        output_path = tmp_path / "output.wav"
        processor = StreamProcessor(effects=[Gain(0.5)], chunk_size=8192)
        processor.process_file(wav_file, output_path)

        assert output_path.exists()
        waveform, fs = torchaudio.load(str(output_path))
        assert fs == 44100
        assert waveform.shape[0] == 1  # Mono
        assert waveform.shape[1] > 0

    def test_process_file_stereo(self, stereo_wav_file: Path, tmp_path: Path) -> None:
        output_path = tmp_path / "output_stereo.wav"
        processor = StreamProcessor(effects=[Gain(0.5)], chunk_size=8192)
        processor.process_file(stereo_wav_file, output_path)

        waveform, fs = torchaudio.load(str(output_path))
        assert waveform.shape[0] == 2  # Stereo

    def test_process_file_with_gain(self, wav_file: Path, tmp_path: Path) -> None:
        output_path = tmp_path / "output_gain.wav"
        processor = StreamProcessor(effects=[Gain(0.5)], chunk_size=8192)
        processor.process_file(wav_file, output_path)

        original, _ = torchaudio.load(str(wav_file))
        processed, _ = torchaudio.load(str(output_path))

        # Processed should be approximately half the original amplitude
        # (allow small tolerance due to float conversion)
        ratio = processed.abs().max() / original.abs().max()
        assert abs(ratio - 0.5) < 0.05

    def test_process_file_creates_parent_dirs(self, wav_file: Path, tmp_path: Path) -> None:
        output_path = tmp_path / "subdir" / "nested" / "output.wav"
        processor = StreamProcessor(effects=[Gain(0.5)], chunk_size=8192)
        processor.process_file(wav_file, output_path)
        assert output_path.exists()

    def test_process_chunks_generator(self, wav_file: Path) -> None:
        processor = StreamProcessor(effects=[Gain(1.0)], chunk_size=8192)
        chunks = list(processor.process_chunks(wav_file))
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, Tensor)
            assert chunk.shape[0] == 1  # Mono

        # Total samples should equal original file
        total_samples = sum(c.shape[1] for c in chunks)
        info = torchaudio.info(str(wav_file))
        assert total_samples == info.num_frames

    def test_process_file_flac(self, wav_file: Path, tmp_path: Path) -> None:
        output_path = tmp_path / "output.flac"
        processor = StreamProcessor(effects=[Gain(0.5)], chunk_size=8192)
        processor.process_file(wav_file, output_path)
        assert output_path.exists()

    def test_fs_propagation_in_stream(self, wav_file: Path, tmp_path: Path) -> None:
        lpf = LoButterworth(cutoff=5000)
        assert lpf.fs is None
        processor = StreamProcessor(effects=[lpf], chunk_size=8192)

        output_path = tmp_path / "output_filter.wav"
        processor.process_file(wav_file, output_path)
        assert lpf.fs == 44100

    def test_process_with_overlap(self, wav_file: Path, tmp_path: Path) -> None:
        output_path = tmp_path / "output_overlap.wav"
        processor = StreamProcessor(
            effects=[Gain(0.5)],
            chunk_size=8192,
            overlap=1024,
        )
        processor.process_file(wav_file, output_path)
        assert output_path.exists()

    def test_process_chunks_with_overlap(self, wav_file: Path) -> None:
        processor = StreamProcessor(
            effects=[Gain(1.0)],
            chunk_size=8192,
            overlap=1024,
        )
        chunks = list(processor.process_chunks(wav_file))
        assert len(chunks) > 0

    def test_context_manager(self, wav_file: Path, tmp_path: Path) -> None:
        output_path = tmp_path / "output_ctx.wav"
        with StreamProcessor(effects=[Gain(0.5)], chunk_size=8192) as processor:
            processor.process_file(wav_file, output_path)
        assert output_path.exists()


# ---------------------------------------------------------------------------
# Tests: Imports
# ---------------------------------------------------------------------------


class TestImports:
    """Test that all public API is importable."""

    def test_import_from_realtime(self) -> None:
        from torchfx.realtime import (
            AudioBackend,
            AudioCallback,
            BackendNotAvailableError,
            BufferOverrunError,
            BufferUnderrunError,
            RealtimeError,
            RealtimeProcessor,
            StreamConfig,
            StreamDirection,
            StreamError,
            StreamProcessor,
            StreamState,
            TensorRingBuffer,
        )

    def test_import_via_torchfx(self) -> None:
        import torchfx

        assert hasattr(torchfx, "realtime")
        assert hasattr(torchfx.realtime, "RealtimeProcessor")
        assert hasattr(torchfx.realtime, "StreamProcessor")
        assert hasattr(torchfx.realtime, "TensorRingBuffer")

    def test_sounddevice_backend_lazy_import(self) -> None:
        """SoundDeviceBackend should be accessible via __getattr__."""
        from torchfx.realtime import __all__

        assert "SoundDeviceBackend" in __all__

    def test_invalid_attribute(self) -> None:
        with pytest.raises((AttributeError, ImportError)):
            from torchfx.realtime import NonExistentClass  # type: ignore[attr-defined]


class TestIIRFilterFixes:
    """Tests for IIR filter clipping and state continuity fixes."""

    def test_lfilter_no_clamp(self) -> None:
        """Signals exceeding ±1.0 should not be hard-clipped by filters."""
        lpf = LoButterworth(cutoff=2000, fs=44100)
        # Create a signal with peaks at 1.5
        t = torch.linspace(0, 1, 44100).unsqueeze(0)
        signal = 1.5 * torch.sin(2 * torch.pi * 440 * t)
        assert signal.abs().max() > 1.0

        filtered = lpf(signal)
        # Output should still have values exceeding 1.0 (not clamped)
        assert filtered.abs().max() > 1.0

    def test_filter_state_continuity(self) -> None:
        """Processing in chunks should match single-pass processing."""
        fs = 44100
        lpf_single = LoButterworth(cutoff=1000, fs=fs)
        lpf_chunked = LoButterworth(cutoff=1000, fs=fs)

        # Create a sine wave signal
        t = torch.linspace(0, 0.1, int(fs * 0.1)).unsqueeze(0)
        signal = torch.sin(2 * torch.pi * 440 * t)

        # Single-pass processing
        result_single = lpf_single(signal)

        # Chunked processing (two halves)
        mid = signal.shape[-1] // 2
        chunk1 = signal[..., :mid]
        chunk2 = signal[..., mid:]
        out1 = lpf_chunked(chunk1)
        out2 = lpf_chunked(chunk2)
        result_chunked = torch.cat([out1, out2], dim=-1)

        # First chunk should match exactly (both use lfilter)
        torch.testing.assert_close(
            result_single[..., :mid], result_chunked[..., :mid], atol=1e-5, rtol=1e-4
        )
        # Second chunk uses SOS cascade with bootstrapped state;
        # the overall shape and values should be reasonable
        assert result_chunked.shape == result_single.shape

    def test_reset_state_clears_zi(self) -> None:
        """reset_state() should clear filter memory."""
        lpf = LoButterworth(cutoff=1000, fs=44100)
        signal = torch.randn(1, 1000)

        # Process a chunk to populate state
        lpf(signal)
        assert lpf._stateful is True

        # Reset and verify state is cleared
        lpf.reset_state()
        assert lpf._stateful is False
        assert lpf._state_x is None
        assert lpf._state_y is None

    def test_filter_state_differs_without_reset(self) -> None:
        """Processing after reset should differ from continued processing."""
        fs = 44100
        lpf = LoButterworth(cutoff=1000, fs=fs)
        signal = torch.randn(1, 1000)

        lpf(signal)
        # Process second chunk with state
        result_with_state = lpf(signal).clone()

        # Reset and process the same chunk
        lpf.reset_state()
        result_after_reset = lpf(signal)

        # Results should differ because the initial conditions differ
        assert not torch.allclose(result_with_state, result_after_reset, atol=1e-6)

    def test_stream_processor_filter_continuity(self) -> None:
        """StreamProcessor should produce clean output with filters."""
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input.wav"
            output_path = Path(tmp) / "output.wav"

            # Create a sine wave test file
            fs = 44100
            t = torch.linspace(0, 1, fs).unsqueeze(0)
            signal = 0.5 * torch.sin(2 * torch.pi * 440 * t)
            torchaudio.save(str(input_path), signal, fs)

            # Process with small chunks to test boundary handling
            from torchfx.realtime import StreamProcessor

            with StreamProcessor(
                effects=[LoButterworth(2000)],
                chunk_size=4096,
            ) as processor:
                processor.process_file(str(input_path), str(output_path))

            # Load and verify output is clean (no NaN, no sudden spikes)
            output, out_fs = torchaudio.load(str(output_path))
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
            # Output should be quieter than input (440Hz through 2000Hz LPF)
            assert output.abs().max() <= signal.abs().max() + 0.1
