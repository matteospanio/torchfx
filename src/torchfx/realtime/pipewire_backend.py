"""PipeWire audio backend for Linux.

PipeWire is the default audio server on current Linux desktops. PortAudio
(via ``sounddevice``) already exposes a PipeWire host API — and, where that is
absent, PipeWire ships a PulseAudio-compatible server that PortAudio's Pulse
host talks to. So a native, low-latency PipeWire callback path is one host
selection away from the existing, battle-tested ``SoundDeviceBackend``.

This backend therefore *is* ``SoundDeviceBackend`` with the host API pinned to
PipeWire (falling back to Pulse): it reuses the whole tested callback engine,
status mapping, and blocking read/write, and adds no dependency beyond the
``sounddevice`` already in the ``realtime`` group.

ponytail: PipeWire-via-PortAudio, not raw libpipewire. Covers every PipeWire
desktop with ~no new code. Upgrade path if sub-PortAudio-buffer latency ever
matters: a libpipewire ctypes/cffi client implementing AudioBackend directly.

Examples
--------
>>> from torchfx.realtime import PipeWireBackend, StreamConfig
>>> backend = PipeWireBackend()  # doctest: +SKIP
>>> backend.open_stream(StreamConfig(channels_out=2), callback)  # doctest: +SKIP
>>> backend.start()  # doctest: +SKIP

"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from torchfx.realtime.backend import AudioCallback, StreamConfig, StreamDirection
from torchfx.realtime.exceptions import StreamError
from torchfx.realtime.sounddevice_backend import SoundDeviceBackend


class PipeWireBackend(SoundDeviceBackend):
    """Audio backend targeting the PipeWire (or Pulse) PortAudio host.

    Raises
    ------
    BackendNotAvailableError
        If ``sounddevice`` is not installed.

    """

    @property
    def name(self) -> str:
        """Return backend name."""
        return "pipewire"

    @property
    def is_available(self) -> bool:
        """Whether a PipeWire/Pulse PortAudio host is present."""
        try:
            return self._resolve_host() is not None
        except Exception:
            return False

    def _resolve_host(self) -> tuple[int, dict[str, Any]] | None:
        """Find the PortAudio host API backed by PipeWire (else Pulse)."""
        apis = self._sd.query_hostapis()
        for preferred in ("pipewire", "pulse"):
            for idx, api in enumerate(apis):
                if preferred in api["name"].lower():
                    return idx, api
        return None

    def get_default_device(self, direction: StreamDirection) -> int | str:
        """Return the PipeWire host's default device for ``direction``."""
        host = self._resolve_host()
        if host is None:
            raise StreamError(
                "No PipeWire/PulseAudio host API found in PortAudio",
                suggestion="Ensure PipeWire is running and PortAudio has PipeWire/Pulse support",
            )
        _, api = host
        key = (
            "default_input_device"
            if direction == StreamDirection.INPUT
            else "default_output_device"
        )
        dev = api.get(key, -1)
        if dev is None or dev < 0:
            raise StreamError(f"No default PipeWire device for {direction.value}")
        return int(dev)

    def open_stream(self, config: StreamConfig, callback: AudioCallback | None = None) -> None:
        """Open a stream on the PipeWire host, filling in its default devices."""
        if config.device_in is None and config.device_out is None:
            host = self._resolve_host()
            if host is None:
                raise StreamError(
                    "No PipeWire/PulseAudio host API found in PortAudio",
                    suggestion="Ensure PipeWire is running and PortAudio has PipeWire/Pulse support",
                )
            _, api = host
            din = api.get("default_input_device", -1)
            dout = api.get("default_output_device", -1)
            config = replace(
                config,
                device_in=(
                    din
                    if config.channels_in > 0 and din is not None and din >= 0
                    else config.device_in
                ),
                device_out=(
                    dout
                    if config.channels_out > 0 and dout is not None and dout >= 0
                    else config.device_out
                ),
            )
        super().open_stream(config, callback)
