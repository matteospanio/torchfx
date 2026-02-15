"""Example: Live performance with the TorchFX REPL.

This example demonstrates how to use the REPL's live playback mode for
real-time effect manipulation — perfect for live music, DJ sets, or
iterative sound design.

Installation
------------
Install the CLI dependencies first:

    pip install torchfx[cli]

Usage
-----
1. Launch the REPL:

    torchfx interactive

2. Load an audio file:

    torchfx> load song.wav

3. Start live playback (loops continuously):

    torchfx> live

4. Add effects in real-time — they apply immediately:

    torchfx> add reverb:decay=0.8,mix=0.4
    torchfx> add lowpass:cutoff=2000,q=0.707

5. Load presets to switch entire effect chains instantly:

    torchfx> preset load vocal-warmth

6. Remove effects on-the-fly:

    torchfx> remove 2

7. Stop live playback when done:

    torchfx> live stop

Live Performance Workflow
--------------------------
The REPL is designed for **zero-latency effect switching** during playback:

* Audio loops seamlessly
* Effect changes happen **immediately** (no restart/reload)
* Perfect for A/B testing different chains
* Great for live coding performances

Example Session
---------------
torchfx> load breakbeat.wav
✓ Loaded breakbeat.wav  (2 ch, 44100 Hz, 4.50s)

torchfx> live
▶ Live playback started  (2 ch, 44100 Hz, looping)
Change effects with 'add', 'remove', or 'preset load' — changes apply immediately!

torchfx> add lowpass:cutoff=800,q=1.0
✓ [1] Added lowpass:cutoff=800,q=1.0
# ← low-pass filter kicks in instantly

torchfx> add reverb:decay=0.9,mix=0.3
✓ [2] Added reverb:decay=0.9,mix=0.3
# ← reverb layered on top

torchfx> preset load dub-echo
✓ Loaded preset 'dub-echo' (4 effects).
# ← entire chain replaced with delay-heavy dub sound

torchfx> clear
✗ Cleared 4 effect(s).
# ← back to dry signal

torchfx> live stop
⏹ Live playback stopped.

torchfx> exit
Goodbye!

Advanced Use Cases
------------------
1. **Live DJ Sets**: Load loops, apply effects on-the-fly, switch presets
   between tracks.

2. **Sound Design**: Iterate on effect chains while listening to the result
   in real-time.

3. **Live Coding**: Write effect specs in the REPL, hear changes instantly.

4. **Performance Art**: Use the REPL as a live instrument with preset banks.

Notes
-----
* The audio file loops continuously during live playback.
* All REPL commands remain responsive — you can add/remove/list effects
  while audio is playing.
* Press Ctrl-C in the REPL prompt to stop (not during live playback — use
  'live stop' for that).
* Effect processing happens in real-time with low latency (buffer size: 2048
  samples ≈ 46ms at 44.1kHz).
"""
