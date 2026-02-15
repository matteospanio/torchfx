---
blogpost: true
date: Feb 15, 2026
author: Matteo Spanio
category: features
tags: cli, tools, workflow
---

# TorchFX CLI: GPU-Accelerated Audio Processing from the Terminal

One of the most requested features for TorchFX has finally arrived: a **full-featured command-line interface** that brings GPU-accelerated audio processing directly to your terminal. Whether you're a music producer, sound designer, or audio engineer, the new `torchfx` CLI tool streamlines your workflow with familiar commands and powerful new capabilities.

## From Library to Tool

While TorchFX started as a Python library for audio DSP research, we recognized that many users need quick, scriptable audio processing without writing Python code. The CLI bridges this gap, offering:

- **Sox-compatible commands** for easy migration from existing workflows
- **GPU acceleration** for batch processing hundreds of files
- **Unix pipe support** for integrating with existing audio pipelines
- **TOML configuration files** for reproducible effect chains
- **Preset management** for reusable processing templates

## Core Commands

### Process Audio Files

The `process` command is your workhorse for applying effects:

```bash
# Single file with effects
torchfx process input.wav output.wav -e normalize -e "reverb:decay=0.6,mix=0.3"

# Batch processing with GPU acceleration
torchfx process "*.wav" --output-dir ./mastered/ --device cuda -e normalize

# Unix pipes for integration
cat raw.wav | torchfx process - - -e normalize | aplay
```

The effect syntax is intuitive: `name:param1=val1,param2=val2`. With 30+ built-in effects and filters registered (from `gain` to `parametriceq`), you have a complete DSP toolkit at your fingertips.

### File Information

Quickly inspect audio metadata with rich table output:

```bash
torchfx info recording.wav
```

This displays sample rate, channels, duration, encoding, and file size in a clean, formatted table — perfect for scripts that need to validate audio properties.

### Playback & Recording

Preview your processing with direct playback:

```bash
# Play with effects applied on-the-fly
torchfx play song.wav -e normalize -e "reverb:decay=0.4"

# Record from microphone
torchfx record output.wav --duration 10 -r 48000 -C 2
```

Both commands require the optional `sounddevice` package (`pip install torchfx[realtime]`), giving you instant audio I/O without leaving the terminal.

## Sox-Compatible Commands

For users migrating from SoX or needing familiar tools, we've implemented a subset of sox commands with GPU acceleration:

### Convert

Format, sample rate, and channel conversion:

```bash
torchfx convert input.wav output.flac
torchfx convert hi-res.wav cd-quality.wav -r 44100 -b 16
torchfx convert stereo.wav mono.wav --channels 1
```

### Trim

Extract time ranges with precision:

```bash
torchfx trim input.wav clip.wav --start 1.5 --end 4.0
torchfx trim long.wav first-30s.wav --duration 30
```

### Concat

Join multiple files seamlessly:

```bash
torchfx concat part1.wav part2.wav part3.wav -o full.wav
```

### Stats

Analyze signal characteristics:

```bash
torchfx stats recording.wav
```

Get detailed statistics including peak/RMS levels in dBFS, crest factor, DC offset, dynamic range, and per-channel breakdowns — all in a beautifully formatted table.

## Configuration Files & Presets

For complex effect chains, TOML configuration files provide reproducibility:

```toml
# mastering.toml
device = "cuda"

[[effects]]
name = "normalize"
peak = 0.9

[[effects]]
name = "parametriceq"
frequency = 2000
q = 1.5
gain = 3

[[effects]]
name = "reverb"
decay = 0.5
mix = 0.15
```

Use it with any command:

```bash
torchfx process input.wav output.wav --config mastering.toml
```

### Preset Management

Save and reuse your favorite chains:

```bash
# Save a preset
torchfx preset save mastering -e normalize -e "reverb:decay=0.5,mix=0.2"

# List all presets
torchfx preset list

# Apply a preset
torchfx preset apply mastering input.wav output.wav

# View preset contents
torchfx preset show mastering
```

Presets are stored as TOML files in `~/.config/torchfx/presets/`, making them easy to share and version control.

## Watch Mode: DAW Integration

The `watch` command monitors directories for new or modified audio files and automatically applies your effect chain:

```bash
# Monitor export folder
torchfx watch ./bounces/ ./mastered/ --preset mastering

# Recursive watching with real-time updates
torchfx watch ./input/ ./output/ --recursive -e normalize -e "reverb:decay=0.4"
```

This is perfect for DAW workflows: export your tracks to a watched folder and get instant mastered versions. The watch mode runs continuously, showing rich status updates as files are processed.

## Batch Processing at Scale

One of the CLI's killer features is GPU-accelerated batch processing. When processing hundreds of files, the difference is dramatic:

```bash
# Process an entire album on GPU
torchfx --device cuda process "album/*.wav" -O ./mastered/ -e normalize -e "reverb:decay=0.5"
```

You get a beautiful progress bar with ETA, per-file success/failure reporting, and processing speeds that scale with your GPU. On modern hardware, expect >100x real-time throughput.

## Global Options

Every command supports global flags:

- `--device cuda` — GPU acceleration
- `--verbose` — DEBUG-level logging
- `--config` — Global configuration file
- `--version` — Show version

## Design Philosophy

The CLI follows Unix philosophy:

- **Do one thing well**: Each command has a clear, focused purpose
- **Composable**: Pipe commands together for complex workflows
- **Scriptable**: Predictable output formats for automation
- **Fast**: GPU acceleration where it matters

But it also brings modern conveniences:

- **Rich output**: Beautiful tables and progress bars
- **Smart defaults**: Sensible parameter values
- **Helpful errors**: Clear messages with suggestions

## Installation

Get started in seconds:

```bash
pip install torchfx[cli]
```

This installs all CLI dependencies (`typer`, `rich`, `tomli` for Python <3.11). For playback and recording:

```bash
pip install torchfx[realtime]
```

## Real-World Workflows

### Podcast Cleanup

```bash
#!/bin/bash
# podcast-cleanup.sh

torchfx process "$1" "${1%.wav}_cleaned.wav" \
  -e normalize \
  -e "highpass:cutoff=80" \
  -e "lowpass:cutoff=15000"
```

### Batch Album Mastering

```bash
torchfx --device cuda process "album/*.wav" \
  -O ./mastered/ \
  --config mastering-chain.toml
```

### DAW Export Monitor

```bash
torchfx watch ~/Music/Exports/ ~/Music/Mastered/ \
  --preset vocal-cleanup \
  --recursive
```

## What's Next?

The CLI is just the beginning. Future releases will add:

- Real-time parameter visualization
- Spectrum analyzer integration  
- A/B comparison mode
- Waveform preview in terminal

## Try It Today

The TorchFX CLI is available now in version 0.4.0. Install it, explore the commands, and let us know what you think!

```bash
pip install torchfx[cli]
torchfx --help
```

For complete documentation, see the [CLI Guide](../guides/tutorials/cli-guide.md).

---

*Have questions or feature requests? Open an issue on [GitHub](https://github.com/matteospanio/torchfx) or start a discussion!*
