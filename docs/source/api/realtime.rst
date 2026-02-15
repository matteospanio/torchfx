Real-Time Processing
====================

The realtime module provides real-time audio I/O, ring buffering, and stream processing capabilities.

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

The real-time processing system consists of:

* **Audio backends** - Abstract interface for audio I/O with a PortAudio implementation via ``sounddevice``
* **Ring buffer** - Lock-free SPSC buffer for real-time tensor transfer
* **Real-time processor** - Orchestrator connecting backend to effect chain
* **Stream processor** - Chunk-based file processing for large files

Quick Start
-----------

Both processors support the context manager protocol for safe resource management.

Real-time processing (requires ``sounddevice``):

.. code-block:: python

   from torchfx.realtime import RealtimeProcessor, SoundDeviceBackend, StreamConfig
   from torchfx.effect import Gain

   config = StreamConfig(
       sample_rate=48000,
       buffer_size=512,
       channels_in=2,
       channels_out=2,
   )
   with RealtimeProcessor(
       effects=[Gain(0.5)],
       backend=SoundDeviceBackend(),
       config=config,
   ) as processor:
       input("Press Enter to stop...")
   # Stream is automatically stopped on exit

Stream processing (no extra dependencies):

.. code-block:: python

   from torchfx.realtime import StreamProcessor
   from torchfx.effect import Gain

   with StreamProcessor(effects=[Gain(0.5)], chunk_size=65536) as processor:
       processor.process_file("large_input.wav", "output.wav")

Configuration
-------------

.. autoclass:: torchfx.realtime.StreamConfig
   :members:

.. autoclass:: torchfx.realtime.StreamDirection
   :members:

.. autoclass:: torchfx.realtime.StreamState
   :members:

Audio Backend
-------------

.. autoclass:: torchfx.realtime.AudioBackend
   :members:

.. autoclass:: torchfx.realtime.sounddevice_backend.SoundDeviceBackend
   :members:

Ring Buffer
-----------

.. autoclass:: torchfx.realtime.TensorRingBuffer
   :members:

Processors
----------

Real-Time Processor
^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchfx.realtime.RealtimeProcessor
   :members:

Stream Processor
^^^^^^^^^^^^^^^^

.. autoclass:: torchfx.realtime.StreamProcessor
   :members:

Exceptions
----------

.. autoclass:: torchfx.realtime.RealtimeError

.. autoclass:: torchfx.realtime.BackendNotAvailableError

.. autoclass:: torchfx.realtime.StreamError

.. autoclass:: torchfx.realtime.BufferOverrunError

.. autoclass:: torchfx.realtime.BufferUnderrunError

Usage Examples
--------------

Real-Time Guitar Effect
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from torchfx.realtime import RealtimeProcessor, SoundDeviceBackend, StreamConfig
   from torchfx.filter.iir import LoButterworth, HiButterworth
   from torchfx.effect import Gain, Reverb

   config = StreamConfig(
       sample_rate=48000,
       buffer_size=256,
       channels_in=1,
       channels_out=1,
       latency="low",
   )

   with RealtimeProcessor(
       effects=[
           HiButterworth(80),        # Remove low rumble
           LoButterworth(8000),       # Tame high frequencies
           Gain(1.5),                 # Boost signal
           Reverb(room_size=0.3),     # Add ambience
       ],
       backend=SoundDeviceBackend(),
       config=config,
   ) as processor:
       # Tweak parameters in real time
       processor.set_parameter("2.gain", 2.0)  # Increase gain
       input("Press Enter to stop...")

Processing Large Files
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from torchfx.realtime import StreamProcessor
   from torchfx.effect import Gain, Normalize

   # Process a large file in 64K sample chunks
   with StreamProcessor(
       effects=[Gain(0.8), Normalize(peak=0.95)],
       chunk_size=65536,
   ) as processor:
       processor.process_file("large_podcast.wav", "normalized_podcast.wav")

       # Or use the generator API
       for chunk in processor.process_chunks("large_podcast.wav"):
           print(f"Processed chunk: {chunk.shape}")

Thread-Safe Parameter Updates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import time
   from torchfx.realtime import RealtimeProcessor, SoundDeviceBackend, StreamConfig
   from torchfx.filter.iir import LoButterworth

   config = StreamConfig(sample_rate=48000, buffer_size=512,
                         channels_in=1, channels_out=1)

   with RealtimeProcessor(
       effects=[LoButterworth(1000)],
       backend=SoundDeviceBackend(),
       config=config,
   ) as processor:
       # Sweep filter cutoff from 1000Hz to 5000Hz
       for cutoff in range(1000, 5001, 100):
           processor.set_parameter("0.cutoff", cutoff)
           time.sleep(0.05)
