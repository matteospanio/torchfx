# Command-Line Interface

TorchFX includes a full-featured CLI for audio processing directly from your
terminal. Install the CLI dependencies with:

```bash
pip install torchfx[cli]
```

All commands are available under the `torchfx` entry-point.

```bash
torchfx --help
```

```{contents}
:local:
:depth: 2
```

## Global Options

Every sub-command inherits these global flags:

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--verbose` | `-v` | off | Enable DEBUG-level logging |
| `--device` | `-d` | `cpu` | Processing device (`cpu` or `cuda`) |
| `--config` | `-c` | — | Path to a TOML configuration file |
| `--version` | `-V` | — | Print version and exit |

```bash
torchfx -d cuda process in.wav out.wav -e normalize
torchfx --version
```

---

## Effect Specifications

Many commands accept `--effect` / `-e` flags. The format is:

```
name                        # defaults
name:value                  # single positional parameter
name:k1=v1,k2=v2           # keyword parameters
name:pos,k1=v1              # mixed
```

**Examples:**

```bash
-e normalize
-e "gain:0.5"
-e "reverb:decay=0.6,mix=0.3"
-e "lowpass:cutoff=1000,q=0.707"
-e "parametriceq:frequency=2000,q=1.5,gain=6"
```

Run `torchfx interactive` and type `effects` to see the full list.

---

## Core Commands

### `process`

Apply an effect chain to one or more audio files.

```bash
# Single file
torchfx process input.wav output.wav -e normalize -e "reverb:decay=0.6"

# Batch processing (glob pattern)
torchfx process "*.wav" --output-dir ./processed/ -e gain:0.5

# From a TOML config
torchfx process input.wav output.wav --config chain.toml

# Unix pipe
cat input.wav | torchfx process - - -e normalize | aplay
```

| Option | Description |
|--------|-------------|
| `--effect` / `-e` | Effect specification (repeatable) |
| `--output-dir` / `-O` | Output directory for batch mode |
| `--config` / `-c` | TOML config with `[[effects]]` |
| `--chunk-size` | Samples per processing chunk (default 65 536) |
| `--overlap` | Overlap samples between chunks |
| `--format` / `-f` | Pipe format: `wav` or `raw` |
| `--rate` / `-r` | Sample rate for raw pipe input |
| `--channels` | Channel count for raw pipe input |

### `info`

Display metadata for an audio file.

```bash
torchfx info recording.wav
```

### `play`

Play an audio file through the default output device (requires `sounddevice`).

```bash
torchfx play song.wav
torchfx play recording.wav -e normalize -e "reverb:decay=0.4"
```

### `record`

Record audio from a microphone (requires `sounddevice`).

```bash
torchfx record output.wav --duration 10
torchfx record output.wav -t 5 -r 48000 -C 2
torchfx record output.wav -t 10 -e normalize -e "reverb:decay=0.3"
```

---

## Sox-Compatible Commands

These commands mirror common SoX workflows, with GPU acceleration under the
hood.

### `convert`

Convert between formats, sample rates, and channel layouts.

```bash
torchfx convert input.wav output.flac
torchfx convert song.wav mono.wav --channels 1 --rate 16000
torchfx convert hi-res.wav cd.wav -r 44100 -b 16
```

| Option | Description |
|--------|-------------|
| `--rate` / `-r` | Target sample rate (Hz) |
| `--channels` / `-c` | Target channel count |
| `--bits` / `-b` | Bit depth (8, 16, 24, 32) |

### `trim`

Extract a time range from an audio file.

```bash
torchfx trim input.wav clip.wav --start 1.5 --end 4.0
torchfx trim input.wav clip.wav -s 10 -d 5
```

| Option | Description |
|--------|-------------|
| `--start` / `-s` | Start time in seconds (default 0) |
| `--end` / `-e` | End time in seconds |
| `--duration` / `-d` | Duration in seconds (alternative to `--end`) |

### `concat`

Concatenate multiple audio files into one (must share sample rate and channels).

```bash
torchfx concat part1.wav part2.wav part3.wav -o full.wav
```

### `stats`

Display signal statistics: peak/RMS levels, crest factor, DC offset, dynamic
range — with per-channel breakdown.

```bash
torchfx stats recording.wav
```

---

## Preset Management

Presets let you save and reuse effect chains. They are stored as TOML files in
`~/.config/torchfx/presets/`.

### `preset save`

```bash
torchfx preset save mastering -e normalize -e "reverb:decay=0.4,mix=0.2"
torchfx preset save loud --from chain.toml
torchfx preset save mastering -e normalize --force   # overwrite
```

### `preset list`

```bash
torchfx preset list
```

### `preset show`

```bash
torchfx preset show mastering
```

### `preset apply`

```bash
torchfx preset apply mastering input.wav output.wav
```

### `preset delete`

```bash
torchfx preset delete mastering
```

---

## Interactive REPL

The `interactive` command launches a REPL powered by `prompt_toolkit` with
tab completion and persistent history.

```bash
torchfx interactive
```

Inside the REPL:

```
torchfx> load song.wav
✓ Loaded song.wav  (2 ch, 44100 Hz, 180.50s)

torchfx> add normalize
✓ [1] Added normalize

torchfx> add reverb:decay=0.5,mix=0.2
✓ [2] Added reverb:decay=0.5,mix=0.2

torchfx> list
Effect Chain:
  1. normalize
  2. reverb:decay=0.5,mix=0.2

torchfx> play
▶ Playback complete (processed).

torchfx> play raw
▶ Playback complete (raw).

torchfx> save output.wav
✓ Saved → output.wav

torchfx> preset save my-chain
✓ Preset 'my-chain' saved → ~/.config/torchfx/presets/my-chain.toml

torchfx> clear
✗ Cleared 2 effect(s).

torchfx> preset load my-chain
✓ Loaded preset 'my-chain' (2 effects).

torchfx> exit
Goodbye!
```

### Live Performance Mode

The REPL includes a **live playback mode** for real-time effect changes — perfect
for live music performance or sound design experimentation.

The live mode uses a **lock-free circular buffer pattern** where:
- Audio playback runs in a real-time callback thread
- Effect chain modifications happen on the main REPL thread
- Each audio buffer sees a consistent snapshot of the effect chain
- Changes apply at the next buffer boundary (~46ms latency at 44.1kHz)

```
torchfx> live
▶ Live playback started  (2 ch, 44100 Hz, looping)
Change effects with 'add', 'remove', or 'preset load' — changes apply immediately!
Use 'live stop' to end playback.

torchfx> add reverb:decay=0.8,mix=0.5
✓ [1] Added reverb:decay=0.8,mix=0.5
# ← reverb starts immediately

torchfx> preset load vocal-cleanup
✓ Loaded preset 'vocal-cleanup' (3 effects).
# ← entire chain switches instantly

torchfx> remove 2
✗ Removed [2] lowpass:cutoff=8000
# ← filter removed in real-time

torchfx> live stop
⏹ Live playback stopped.
```

**Key features:**
- Audio loops continuously
- Effect changes apply **immediately** at the next buffer boundary
- Add/remove effects, load presets, clear the chain — all in real-time
- Lock-free reads ensure glitch-free playback even during effect swaps
- Perfect for live performance, DJ sets, or iterative sound design

### REPL Commands

| Command | Description |
|---------|-------------|
| `load <file>` | Load an audio file |
| `add <spec>` | Add an effect to the chain |
| `remove <n>` | Remove effect at index *n* (1-based) |
| `list` | Show the current effect chain |
| `effects` | List all available effect names |
| `info` | Show loaded file metadata |
| `play` | Play processed audio (blocks until complete) |
| `play raw` | Play unprocessed audio |
| `live` | **Start live playback** (non-blocking, loops) |
| `live stop` | **Stop live playback** |
| `save <file>` | Save processed audio to file |
| `preset save <name>` | Save chain as a preset |
| `preset load <name>` | Load a preset into the chain |
| `preset list` | List saved presets |
| `clear` | Clear the effect chain |
| `help` | Show help |
| `exit` / `quit` | Exit the REPL |

---

## Watch Mode

Monitor a directory for new or modified audio files and automatically apply
effects. Ideal for DAW integration where you export to a watched folder.

```bash
torchfx watch ./input/ ./output/ -e normalize -e "reverb:decay=0.4"
torchfx watch ./bounces/ ./mastered/ --config master.toml
torchfx watch ./raw/ ./processed/ --preset vocal-cleanup --recursive
```

| Option | Description |
|--------|-------------|
| `--effect` / `-e` | Effect specification (repeatable) |
| `--config` / `-c` | TOML config file |
| `--preset` / `-p` | Named preset to apply |
| `--recursive` / `-r` | Watch subdirectories |
| `--existing` | Process existing files on startup |

Press `Ctrl-C` to stop watching.

---

## TOML Configuration

Instead of passing `--effect` flags, you can define chains in TOML files:

```toml
# chain.toml
device = "cuda"
chunk_size = 131072

[[effects]]
name = "normalize"
peak = 0.9

[[effects]]
name = "reverb"
decay = 0.5
mix = 0.2

[[effects]]
name = "lowpass"
cutoff = 8000
q = 0.707
```

Use with any command that accepts `--config`:

```bash
torchfx process input.wav output.wav --config chain.toml
torchfx watch ./input/ ./output/ --config chain.toml
```

Or save as a preset:

```bash
torchfx preset save mastering --from chain.toml
```
