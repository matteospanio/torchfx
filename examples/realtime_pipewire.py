#!/usr/bin/env python3
"""Real-time mic -> effects -> speakers over PipeWire (Linux).

Needs: PipeWire running + `pip install torchfx[realtime]`. Ctrl-C to stop.
"""

import time

import torchfx as fx
from torchfx.realtime import PipeWireBackend, RealtimeProcessor, StreamConfig

backend = PipeWireBackend()
if not backend.is_available:
    raise SystemExit("No PipeWire/Pulse PortAudio host found — is PipeWire running?")

config = StreamConfig(sample_rate=48000, buffer_size=512, channels_in=1, channels_out=1)
proc = RealtimeProcessor(
    effects=[fx.effect.Gain(0.8), fx.effect.Reverb(room_size=0.6, mix=0.3, fs=48000)],
    backend=backend,
    config=config,
)
proc.start()
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    proc.stop()
