"""PipeWireBackend host-selection (#55).

Mocks sounddevice — no audio hardware.

"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from torchfx.realtime.backend import StreamConfig, StreamDirection
from torchfx.realtime.exceptions import StreamError
from torchfx.realtime.pipewire_backend import PipeWireBackend


def _backend(hostapis, *, stream=None):
    """A PipeWireBackend wired to a fake sounddevice module."""
    b = PipeWireBackend.__new__(PipeWireBackend)  # skip get_sounddevice()
    sd = MagicMock()
    sd.query_hostapis.return_value = hostapis
    sd.Stream.return_value = stream or MagicMock()
    sd.OutputStream.return_value = stream or MagicMock()
    sd.InputStream.return_value = stream or MagicMock()
    b._sd = sd
    b._stream = None
    b._config = None
    b._callback = None
    from torchfx.realtime.backend import StreamState

    b._state = StreamState.CLOSED
    return b, sd


PIPEWIRE_HOST = {"name": "PipeWire", "default_input_device": 3, "default_output_device": 4}
PULSE_HOST = {"name": "PulseAudio", "default_input_device": 5, "default_output_device": 6}
ALSA_HOST = {"name": "ALSA", "default_input_device": 0, "default_output_device": 1}


def test_available_when_pipewire_host_present():
    b, _ = _backend([ALSA_HOST, PIPEWIRE_HOST])
    assert b.is_available
    assert b.name == "pipewire"


def test_unavailable_without_pipewire_or_pulse():
    b, _ = _backend([ALSA_HOST])
    assert not b.is_available
    with pytest.raises(StreamError, match="No PipeWire"):
        b.get_default_device(StreamDirection.OUTPUT)


def test_prefers_pipewire_over_pulse():
    b, _ = _backend([PULSE_HOST, PIPEWIRE_HOST])
    assert b.get_default_device(StreamDirection.OUTPUT) == 4  # PipeWire, not Pulse


def test_falls_back_to_pulse():
    b, _ = _backend([ALSA_HOST, PULSE_HOST])
    assert b.get_default_device(StreamDirection.INPUT) == 5


def test_open_stream_routes_to_pipewire_default_device():
    b, sd = _backend([ALSA_HOST, PIPEWIRE_HOST])
    b.open_stream(StreamConfig(channels_in=0, channels_out=2))  # output-only
    # The default PipeWire output device (4) must reach PortAudio.
    assert sd.OutputStream.call_args.kwargs["device"] == 4
