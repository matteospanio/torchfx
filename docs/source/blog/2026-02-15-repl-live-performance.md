---
blogpost: true
date: Feb 15, 2026
author: Matteo Spanio
category: features
tags: repl, interactive, live-performance, real-time
---

# Interactive Audio Magic: The TorchFX REPL

Imagine tweaking a reverb decay parameter and hearing the change *instantly* while your audio loops continuously. No stopping, no reprocessing, no waiting. Just **live, real-time effect manipulation**. That's the power of the new TorchFX REPL.

## Beyond Batch Processing

While command-line audio tools excel at batch processing, they fall short for iterative sound design. You typically write a command, run it, listen to the output, adjust parameters, and repeat. This cycle breaks creative flow.

The TorchFX REPL changes the game by providing an **interactive shell** where you can:

- Load audio files
- Build effect chains incrementally
- Preview changes in real-time
- Save your work as presets
- Experiment freely without scripts

Think of it as a **live coding environment for audio DSP**, powered by PyTorch and GPU acceleration.

## Getting Started

Launch the REPL with a single command:

```bash
torchfx interactive
```

You're greeted with a prompt and tab completion:

```
TorchFX Interactive REPL
Type 'help' for commands, 'exit' to quit.

torchfx>
```

### Loading Audio

Start by loading a file:

```
torchfx> load song.wav
✓ Loaded song.wav  (2 ch, 44100 Hz, 180.50s)
```

The REPL shows you key metadata immediately — channels, sample rate, and duration.

### Building Effect Chains

Add effects one at a time:

```
torchfx> add normalize
✓ [1] Added normalize

torchfx> add reverb:decay=0.5,mix=0.2
✓ [2] Added reverb:decay=0.5,mix=0.2

torchfx> list
Effect Chain:
  1. normalize
  2. reverb:decay=0.5,mix=0.2
```

Each effect gets an index, making it easy to manage complex chains.

### Preview Your Work

Play the processed audio:

```
torchfx> play
▶ Playback complete (processed).
```

Want to compare with the original?

```
torchfx> play raw
▶ Playback complete (raw).
```

## Live Performance Mode: The Game Changer

Here's where it gets exciting. Traditional audio tools process files offline — load, process, save. The REPL introduces **live playback mode** where audio loops continuously and effect changes apply **immediately**:

```
torchfx> live
▶ Live playback started  (2 ch, 44100 Hz, looping)
Change effects with 'add', 'remove', or 'preset load' — changes apply immediately!
Use 'live stop' to end playback.
```

Now, while the audio is playing:

```
torchfx> add reverb:decay=0.8,mix=0.5
✓ [1] Added reverb:decay=0.8,mix=0.5
```

**The reverb starts INSTANTLY.** No stopping, no reprocessing. The effect integrates seamlessly into the ongoing playback.

### Hot-Swapping Effect Chains

The real power emerges when you swap entire chains mid-playback:

```
torchfx> preset load vocal-cleanup
✓ Loaded preset 'vocal-cleanup' (3 effects).
```

Your audio transitions to the new chain at the next buffer boundary (~46ms latency). It's like switching between guitar pedals — instant and glitch-free.

### Removing Effects Live

Made a mistake? Remove it in real-time:

```
torchfx> remove 2
✗ Removed [2] lowpass:cutoff=8000
```

The filter disappears from the audio stream immediately.

### Stop When You're Done

```
torchfx> live stop
⏹ Live playback stopped.
```

## Technical Deep Dive: Lock-Free Audio

How does the REPL achieve glitch-free, real-time effect switching? The secret is a **lock-free circular buffer pattern** inspired by professional audio software:

1. **Audio Thread**: Runs in real-time, reading chunks from the file and passing them through the effect chain
2. **REPL Thread**: Handles user commands, modifying the effect list
3. **Lock-Free Reads**: The audio thread reads the effect list without locks, seeing a consistent snapshot
4. **Buffer Boundaries**: Changes apply atomically at buffer boundaries (~46ms @ 2048 samples, 44.1kHz)

This architecture ensures:
- **Zero dropouts** during effect changes
- **Immediate response** (one buffer latency)
- **Thread safety** without blocking the audio path
- **Graceful degradation** (broken effects are skipped)

It's the same pattern used in `RealtimeProcessor` for live microphone input, adapted here for file-based playback.

## Complete Command Reference

### File Operations

- `load <file>` — Load an audio file
- `info` — Show loaded file metadata
- `save <file>` — Save processed audio

### Effect Chain Management

- `add <spec>` — Add effect (e.g., `add reverb:decay=0.5`)
- `remove <n>` — Remove effect at index (1-based)
- `list` — Show current effect chain
- `clear` — Clear all effects
- `effects` — List all available effect names

### Playback

- `play` — Play processed audio (blocks until complete)
- `play raw` — Play original unprocessed audio
- `live` — **Start live playback (non-blocking, loops)**
- `live stop` — **Stop live playback**

### Preset Management

- `preset save <name>` — Save chain as preset
- `preset load <name>` — Load preset into chain
- `preset list` — List saved presets

### Utility

- `help` — Show command reference
- `exit` / `quit` — Exit REPL

## Real-World Use Cases

### Sound Design Iteration

```
torchfx> load synth-pad.wav
torchfx> live

# Try different reverb settings
torchfx> add reverb:decay=0.3,mix=0.1
# Too dry...
torchfx> remove 1
torchfx> add reverb:decay=0.8,mix=0.4
# Perfect!

torchfx> preset save lush-pad
torchfx> live stop
```

### Mastering Chain Development

```
torchfx> load mix.wav
torchfx> add normalize
torchfx> add parametriceq:frequency=2000,q=1.5,gain=3
torchfx> add reverb:decay=0.4,mix=0.1
torchfx> play

# Tweak until satisfied
torchfx> preset save mastering-v1
torchfx> save mastered.wav
```

### DJ Set Preparation

```
torchfx> load track.wav
torchfx> live

# Test different combinations rapidly
torchfx> preset load filter-sweep
torchfx> preset load heavy-reverb
torchfx> preset load clean

# Find the winner
torchfx> save processed-track.wav
torchfx> live stop
```

### Educational Demos

The REPL is perfect for teaching audio DSP. Students can hear parameter changes instantly:

```
torchfx> load tone.wav
torchfx> live
torchfx> add lowpass:cutoff=1000
# Hear the filtering
torchfx> remove 1
torchfx> add lowpass:cutoff=500
# Hear the difference
```

## Advanced Features

### Tab Completion

Press Tab to complete:
- Command names
- Effect names  
- File paths
- Preset names

### Command History

Navigate previous commands with ↑/↓ arrows. History is persistent across sessions in `~/.config/torchfx/repl_history`.

### Rich Formatting

All output uses Rich markup for beautiful, colorful formatting:
- Green for success messages
- Red for errors
- Yellow for warnings
- Cyan for highlights
- Dim for hints

### Error Resilience

During live playback, if an effect throws an error, it's silently skipped to prevent audio dropouts. You'll see errors when adding effects, but playback stays robust.

## Installation

The REPL is included with the CLI package:

```bash
pip install torchfx[cli]
```

For playback (required for `play` and `live` commands):

```bash
pip install torchfx[realtime]
```

Then just run:

```bash
torchfx interactive
```

## Performance Notes

- **Latency**: ~46ms at default buffer size (2048 samples @ 44.1kHz)
- **Loop Performance**: File reads are lock-protected; effect application is lock-free
- **Effect Count**: Tested with 10+ effects in real-time without issues
- **GPU Support**: Not currently used in REPL (CPU is plenty fast for real-time with reasonable chains)

## Future Enhancements

We're exploring additions for future versions:

- **Waveform visualization** in the terminal
- **Spectrum analyzer** with ASCII/Unicode rendering
- **Parameter automation** over time
- **MIDI controller support** for live parameter changes
- **Multi-file session management**
- **Recording mode** (mic → live effects → disk)

## Philosophy: Flow State for Audio

The REPL isn't just a tool — it's designed to **enable flow states** in audio work. By removing friction between idea and sound, it lets you:

- Explore creatively without breaking concentration
- Iterate faster than ever before
- Learn DSP concepts through immediate feedback
- Perform live with confidence

## Try It Now

Experience the magic of real-time audio manipulation:

```bash
pip install torchfx[cli,realtime]
torchfx interactive
```

Load a file, type `live`, and start experimenting. You'll never go back to batch processing for creative work.

---

*Questions? Join the discussion on [GitHub](https://github.com/matteospanio/torchfx)!*
