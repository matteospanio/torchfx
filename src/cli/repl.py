"""Interactive REPL for building and previewing effect chains.

Uses ``prompt_toolkit`` for tab completion, syntax highlighting, and
persistent command history.  The REPL lets users:

* Load audio files.
* Add / remove / reorder effects interactively.
* Preview playback of the current chain (requires ``sounddevice``).
* Save the chain to a file or preset.
* A/B compare: toggle the chain on/off during playback.
* **Live playback**: stream audio with real-time effect changes.

"""

from __future__ import annotations

import shlex
import threading
from pathlib import Path

from prompt_toolkit.completion import WordCompleter

from cli.parsing import list_effects, parse_effect_string
from torchfx.effect import FX


def _configure_effect(fx: FX, fs: int) -> None:
    """Configure an effect/filter with sample rate and coefficients."""
    from torchfx.filter.__base import AbstractFilter

    if hasattr(fx, "fs") and fx.fs is None:
        fx.fs = fs
    if isinstance(fx, AbstractFilter) and not fx._has_computed_coeff:
        fx.compute_coefficients()


# ---------------------------------------------------------------------------
# REPL state
# ---------------------------------------------------------------------------


class _ReplState:
    """Mutable state for the REPL session."""

    def __init__(self) -> None:
        self.effects: list[FX] = []
        self.effect_specs: list[str] = []  # parallel list of original spec strings
        self.wave: object | None = None
        self.file_path: str | None = None
        self.sample_rate: int = 44100
        self.live_processor: object | None = None  # non-None sentinel = running
        self.live_stream: object | None = None  # sd.OutputStream
        self.live_thread: threading.Thread | None = None
        self.live_position: int = 0  # current playback position in samples
        self.live_lock = threading.Lock()

    def clear(self) -> None:
        self.effects.clear()
        self.effect_specs.clear()


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

_HELP_TEXT = """\
[bold cyan]TorchFX Interactive REPL[/bold cyan]

  [green]load[/green] <file>                     Load an audio file
  [green]add[/green] <effect-spec>               Add an effect (e.g. reverb:decay=0.5)
  [green]remove[/green] <index>                  Remove effect at index (1-based)
  [green]list[/green]                            Show current effect chain
  [green]effects[/green]                         List all available effects
  [green]info[/green]                            Show loaded file info
  [green]play[/green]                            Play processed audio (blocks)
  [green]play raw[/green]                        Play original (unprocessed) audio
  [green]live[/green] [buffer|buffer=4096 blocks=16] Start live playback (non-blocking, loop)
  [green]live stop[/green]                       Stop live playback
  [green]save[/green] <file>                     Save processed audio to file
  [green]preset save[/green] <name>              Save current chain as preset
  [green]preset load[/green] <name>              Load a preset
  [green]preset list[/green]                     List saved presets
  [green]clear[/green]                           Clear the effect chain
  [green]help[/green]                            Show this message
  [green]exit[/green] / [green]quit[/green]                     Exit the REPL
"""


def _cmd_load(state: _ReplState, args: list[str]) -> str:
    """Load an audio file."""
    if not args:
        return "[red]Usage: load <file>[/red]"

    from torchfx.wave import Wave

    path = Path(args[0]).expanduser()
    if not path.exists():
        return f"[red]File not found: {path}[/red]"

    wave = Wave.from_file(path)
    state.wave = wave
    state.file_path = str(path)
    state.sample_rate = wave.fs
    dur = wave.duration("sec")
    return f"[green]✓ Loaded {path.name}[/green]  ({wave.channels()} ch, {wave.fs} Hz, {dur:.2f}s)"


def _cmd_add(state: _ReplState, args: list[str]) -> str:
    """Add an effect to the chain."""
    if not args:
        return "[red]Usage: add <effect-spec>[/red]"
    spec = args[0]
    try:
        fx = parse_effect_string(spec)
    except ValueError as exc:
        return f"[red]{exc}[/red]"
    if state.wave is not None:
        _configure_effect(fx, state.sample_rate)
    state.effects.append(fx)
    state.effect_specs.append(spec)
    idx = len(state.effects)
    return f"[green]✓ [{idx}] Added {spec}[/green]"


def _cmd_remove(state: _ReplState, args: list[str]) -> str:
    """Remove an effect by 1-based index."""
    if not args:
        return "[red]Usage: remove <index>[/red]"
    try:
        idx = int(args[0]) - 1
    except ValueError:
        return "[red]Index must be an integer.[/red]"
    if idx < 0 or idx >= len(state.effects):
        return f"[red]Invalid index. Chain has {len(state.effects)} effect(s).[/red]"
    removed = state.effect_specs.pop(idx)
    state.effects.pop(idx)
    return f"[yellow]✗ Removed [{idx + 1}] {removed}[/yellow]"


def _cmd_list(state: _ReplState, _args: list[str]) -> str:
    """Display the current effect chain."""
    if not state.effects:
        return "[dim]Chain is empty. Use [green]add <effect>[/green] to get started.[/dim]"
    lines = ["[bold]Effect Chain:[/bold]"]
    for i, spec in enumerate(state.effect_specs, start=1):
        lines.append(f"  [cyan]{i}.[/cyan] {spec}")
    return "\n".join(lines)


def _cmd_effects(_state: _ReplState, _args: list[str]) -> str:
    """List all available effects."""
    names = list_effects()
    return "[bold]Available effects:[/bold]\n  " + ", ".join(names)


def _cmd_info(state: _ReplState, _args: list[str]) -> str:
    """Show info about the loaded file."""
    if state.wave is None:
        return "[dim]No file loaded. Use [green]load <file>[/green] first.[/dim]"

    from torchfx.wave import Wave

    wave: Wave = state.wave  # type: ignore[assignment]
    return (
        f"[bold]File:[/bold] {state.file_path}\n"
        f"[bold]Channels:[/bold] {wave.channels()}\n"
        f"[bold]Sample rate:[/bold] {wave.fs} Hz\n"
        f"[bold]Duration:[/bold] {wave.duration('sec'):.2f}s\n"
        f"[bold]Samples:[/bold] {wave.ys.shape[1]:,}\n"
        f"[bold]Effects:[/bold] {len(state.effects)}"
    )


def _cmd_play(state: _ReplState, args: list[str]) -> str:
    """Play audio (processed unless 'raw' is specified)."""
    if state.wave is None:
        return "[dim]No file loaded.[/dim]"

    import numpy as np

    from torchfx.realtime._compat import get_sounddevice
    from torchfx.wave import Wave

    sd = get_sounddevice()
    wave: Wave = state.wave  # type: ignore[assignment]

    # Apply effects (unless 'raw' requested)
    if args and args[0] == "raw":
        output = wave
    else:
        output = wave
        for fx in state.effects:
            output = output | fx

    audio = output.ys.cpu().numpy().T.astype(np.float32)
    mode = "raw" if (args and args[0] == "raw") else "processed"
    try:
        sd.stop()
        sd.play(audio, samplerate=output.fs)
        sd.wait()
    except KeyboardInterrupt:
        sd.stop()
        return "[yellow]⏹ Playback stopped.[/yellow]"
    finally:
        sd.stop()
    return f"[green]▶ Playback complete ({mode}).[/green]"


def _cmd_save(state: _ReplState, args: list[str]) -> str:
    """Save processed audio to a file."""
    if state.wave is None:
        return "[dim]No file loaded.[/dim]"
    if not args:
        return "[red]Usage: save <output-file>[/red]"

    from torchfx.wave import Wave

    wave: Wave = state.wave  # type: ignore[assignment]
    output = wave
    for fx in state.effects:
        output = output | fx

    out_path = Path(args[0]).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(out_path)
    return f"[green]✓ Saved → {out_path}[/green]"


def _cmd_preset(state: _ReplState, args: list[str]) -> str:
    """Handle preset sub-commands (save/load/list)."""
    if not args:
        return "[red]Usage: preset save|load|list [name][/red]"

    sub = args[0].lower()

    if sub == "list":
        from cli.commands.preset import PRESETS_DIR, _ensure_presets_dir

        _ensure_presets_dir()
        presets = sorted(PRESETS_DIR.glob("*.toml"))
        if not presets:
            return "[dim]No presets found.[/dim]"
        lines = ["[bold]Presets:[/bold]"]
        for p in presets:
            lines.append(f"  [cyan]•[/cyan] {p.stem}")
        return "\n".join(lines)

    if sub == "save":
        if len(args) < 2:
            return "[red]Usage: preset save <name>[/red]"
        if not state.effect_specs:
            return "[red]Chain is empty — nothing to save.[/red]"

        from cli.commands.preset import _ensure_presets_dir, _preset_path

        _ensure_presets_dir()
        name = args[1]
        path = _preset_path(name)

        lines_out: list[str] = []
        for spec in state.effect_specs:
            effect_name, _, params_str = spec.partition(":")
            lines_out.append("[[effects]]")
            lines_out.append(f'name = "{effect_name.strip().lower()}"')
            if params_str.strip():
                for token in params_str.split(","):
                    token = token.strip()
                    if "=" in token:
                        key, _, val = token.partition("=")
                        try:
                            float(val.strip())
                            lines_out.append(f"{key.strip()} = {val.strip()}")
                        except ValueError:
                            lines_out.append(f'{key.strip()} = "{val.strip()}"')
            lines_out.append("")

        path.write_text("\n".join(lines_out))
        return f"[green]✓ Preset '{name}' saved → {path}[/green]"

    if sub == "load":
        if len(args) < 2:
            return "[red]Usage: preset load <name>[/red]"

        from cli.commands.preset import _preset_path
        from cli.parsing import load_effects_from_config

        name = args[1]
        path = _preset_path(name)
        if not path.exists():
            return f"[red]Preset '{name}' not found.[/red]"

        state.effects = load_effects_from_config(path)
        if state.wave is not None:
            for fx in state.effects:
                _configure_effect(fx, state.sample_rate)
        # Rebuild spec strings from the toml for display
        state.effect_specs = [type(fx).__name__.lower() for fx in state.effects]
        return f"[green]✓ Loaded preset '{name}' ({len(state.effects)} effects).[/green]"

    return f"[red]Unknown preset sub-command: {sub}[/red]"


def _cmd_clear(state: _ReplState, _args: list[str]) -> str:
    """Clear the effect chain."""
    n = len(state.effects)
    state.clear()
    return f"[yellow]✗ Cleared {n} effect(s).[/yellow]"


def _cmd_live(state: _ReplState, args: list[str]) -> str:
    """Start or stop live playback with real-time effect changes.

    Architecture
    ------------
    A **producer thread** reads chunks from the source file, applies the
    current effect chain (snapshot), and writes pre-computed float32 numpy
    audio into a lock-protected ring buffer.  The **cffi audio callback**
    only copies from that buffer — no tensor ops, no allocations — which
    avoids ``MemoryError`` in the real-time PortAudio thread.

    Effect changes (add/remove/preset load) are picked up by the producer
    on the next iteration (~one buffer latency ≈ 46 ms @ 2048/44.1 kHz).

    """
    if args and args[0] == "stop":
        if state.live_processor is None:
            return "[dim]Live playback not running.[/dim]"

        state.live_processor = None  # signal both threads to stop

        if state.live_thread:
            state.live_thread.join(timeout=3.0)
            state.live_thread = None

        # Close the stream gracefully to avoid SIGSEGV
        if state.live_stream is not None:
            try:
                state.live_stream.stop()  # type: ignore[attr-defined]
                state.live_stream.close()  # type: ignore[attr-defined]
            except Exception:
                pass
            state.live_stream = None

        return "[yellow]⏹ Live playback stopped.[/yellow]"

    # --- start live playback -------------------------------------------------
    if state.wave is None:
        return "[dim]No file loaded.[/dim]"
    if state.live_processor is not None:
        return "[yellow]Live playback already running. Use 'live stop' first.[/yellow]"

    import numpy as np

    from torchfx.realtime._compat import get_sounddevice
    from torchfx.wave import Wave

    sd = get_sounddevice()
    wave: Wave = state.wave  # type: ignore[assignment]
    n_channels = wave.channels()
    blocksize = 2048
    num_blocks = 8

    # Parse optional buffer settings: "live 4096", "live buffer=4096 blocks=16"
    for token in args:
        if token.isdigit():
            blocksize = int(token)
            continue
        if "=" in token:
            key, _, val = token.partition("=")
            key = key.strip().lower()
            val = val.strip()
            if val.isdigit():
                if key in {"buffer", "block", "blocksize"}:
                    blocksize = int(val)
                elif key in {"blocks", "ring", "ringblocks"}:
                    num_blocks = int(val)

    if blocksize <= 0 or num_blocks <= 1:
        return "[red]Invalid live buffer settings. Use positive values (e.g. live 4096 blocks=16).[/red]"

    # Pre-allocated ring buffer (multiple blocks so the producer can stay
    # ahead of the callback).
    _NUM_BLOCKS = num_blocks
    ring_buf = np.zeros((_NUM_BLOCKS * blocksize, n_channels), dtype=np.float32)
    ring_size = ring_buf.shape[0]

    # Shared indices — written by one side each, so no torn reads on
    # aligned ints.  ``write_pos`` is advanced by the producer thread,
    # ``read_pos`` by the audio callback.
    ring_write = [0]  # producer writes here (list for mutability in closures)
    ring_read = [0]  # callback reads here

    state.live_position = 0
    state.live_processor = object()  # non-None sentinel → "running"

    # -- producer thread ------------------------------------------------------
    def _producer() -> None:
        """Fill the ring buffer with effect-processed audio."""
        import time

        while state.live_processor is not None:
            # How many frames can we write?
            wp = ring_write[0]
            rp = ring_read[0]
            available = ring_size - ((wp - rp) % ring_size or ring_size)
            if available == 0:
                available = ring_size - ((wp - rp) % ring_size)

            # Only write in blocksize increments so the callback always
            # gets complete, aligned blocks.
            if (wp - rp) % ring_size >= ring_size - blocksize and wp != rp:
                # Buffer is nearly full — wait for the callback to consume.
                time.sleep(blocksize / wave.fs * 0.5)
                continue

            # Read a block from the source file (loops)
            with state.live_lock:
                start = state.live_position
                end = start + blocksize
                total_samples = wave.ys.shape[1]

                if start >= total_samples:
                    state.live_position = 0
                    start = 0
                    end = blocksize

                if end > total_samples:
                    # Wrap: take tail + head
                    tail = wave.ys[:, start:total_samples].clone()
                    head = wave.ys[:, 0 : end - total_samples].clone()
                    import torch

                    chunk = torch.cat([tail, head], dim=1)
                    state.live_position = end - total_samples
                else:
                    chunk = wave.ys[:, start:end].clone()
                    state.live_position = end

            # Apply effects — snapshot read is safe (list may be swapped
            # between iterations but each iteration sees a consistent ref).
            import contextlib

            effects = list(state.effects)  # shallow copy of current list
            for fx in effects:
                _configure_effect(fx, wave.fs)
                with contextlib.suppress(Exception):
                    # skip broken effects
                    chunk = fx(chunk)

            # Convert to numpy (still in producer thread, NOT in callback)
            audio = chunk.cpu().numpy().T.astype(np.float32)

            # Handle shape: ensure (blocksize, channels)
            if audio.shape[0] < blocksize:
                pad = np.zeros((blocksize - audio.shape[0], n_channels), dtype=np.float32)
                audio = np.vstack([audio, pad])
            elif audio.shape[0] > blocksize:
                audio = audio[:blocksize]

            # Write into ring buffer
            wp = ring_write[0]
            offset = wp % ring_size
            space_to_end = ring_size - offset
            if space_to_end >= blocksize:
                ring_buf[offset : offset + blocksize] = audio
            else:
                ring_buf[offset : offset + space_to_end] = audio[:space_to_end]
                ring_buf[: blocksize - space_to_end] = audio[space_to_end:]

            ring_write[0] = wp + blocksize

            # Throttle so we don't spin-lock
            time.sleep(blocksize / wave.fs * 0.25)

    # -- audio callback (cffi) ------------------------------------------------
    def _live_callback(outdata: object, frames: int, _time: object, _status: object) -> None:
        """Copy pre-computed audio from ring buffer — zero allocations."""
        out = outdata
        if state.live_processor is None:
            out[:] = 0  # type: ignore[index]
            raise sd.CallbackStop()

        wp = ring_write[0]
        rp = ring_read[0]
        buffered = wp - rp

        if buffered < frames:
            # Under-run: output silence (producer hasn't caught up yet)
            out[:] = 0  # type: ignore[index]
            return

        offset = rp % ring_size
        space_to_end = ring_size - offset
        if space_to_end >= frames:
            out[:] = ring_buf[offset : offset + frames]  # type: ignore[index]
        else:
            out[:space_to_end] = ring_buf[offset : offset + space_to_end]  # type: ignore[index]
            out[space_to_end:frames] = ring_buf[: frames - space_to_end]  # type: ignore[index]

        ring_read[0] = rp + frames

    # -- kick off -------------------------------------------------------------
    try:
        stream = sd.OutputStream(
            samplerate=wave.fs,
            channels=n_channels,
            callback=_live_callback,
            blocksize=blocksize,
        )

        # Start producer first so the ring buffer has data before the
        # callback fires.
        producer = threading.Thread(target=_producer, daemon=True)
        producer.start()
        state.live_thread = producer

        stream.start()
        state.live_stream = stream

        return (
            f"[green]▶ Live playback started[/green]  "
            f"({n_channels} ch, {wave.fs} Hz, looping)\n"
            f"[dim]Change effects with 'add', 'remove', or 'preset load' — "
            f"changes apply immediately!\n"
            f"Use 'live stop' to end playback.[/dim]"
        )
    except Exception as exc:
        state.live_processor = None
        return f"[red]Error starting live playback: {exc}[/red]"


# Dispatch table
_COMMANDS: dict[str, object] = {
    "load": _cmd_load,
    "add": _cmd_add,
    "remove": _cmd_remove,
    "list": _cmd_list,
    "effects": _cmd_effects,
    "info": _cmd_info,
    "play": _cmd_play,
    "save": _cmd_save,
    "preset": _cmd_preset,
    "clear": _cmd_clear,
    "live": _cmd_live,
}


# ---------------------------------------------------------------------------
# prompt_toolkit completer
# ---------------------------------------------------------------------------


def _build_completer() -> WordCompleter:
    """Build a prompt_toolkit completer for the REPL."""
    top_commands = list(_COMMANDS) + ["help", "exit", "quit"]
    effect_names = list_effects()
    all_words = top_commands + effect_names
    return WordCompleter(all_words, ignore_case=True)


# ---------------------------------------------------------------------------
# REPL entry-point
# ---------------------------------------------------------------------------


def interactive_cmd() -> None:
    """Launch the interactive REPL for building effect chains.

    \b
    The REPL provides tab-completion for commands and effect names,
    persistent command history, and Rich-formatted output.

    \b
    Examples
    --------
      torchfx interactive

    """
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from rich.console import Console

    console = Console()
    state = _ReplState()

    history_path = Path.home() / ".config" / "torchfx" / "repl_history"
    history_path.parent.mkdir(parents=True, exist_ok=True)

    session: PromptSession[str] = PromptSession(
        history=FileHistory(str(history_path)),
        completer=_build_completer(),
    )

    console.print("[bold cyan]TorchFX Interactive REPL[/bold cyan]")
    console.print("[dim]Type 'help' for commands, 'exit' to quit.[/dim]\n")

    try:
        while True:
            try:
                raw = session.prompt("torchfx> ")
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Goodbye![/dim]")
                break

            raw = raw.strip()
            if not raw:
                continue

            parts = shlex.split(raw)
            cmd_name = parts[0].lower()
            cmd_args = parts[1:]

            if cmd_name in ("exit", "quit"):
                console.print("[dim]Goodbye![/dim]")
                break

            if cmd_name == "help":
                console.print(_HELP_TEXT)
                continue

            handler = _COMMANDS.get(cmd_name)
            if handler is None:
                console.print(f"[red]Unknown command: {cmd_name}[/red]")
                console.print("[dim]Type 'help' for available commands.[/dim]")
                continue

            try:
                result = handler(state, cmd_args)  # type: ignore[operator]
                if result:
                    console.print(result)
            except Exception as exc:
                console.print(f"[red]Error: {exc}[/red]")
    finally:
        # Clean up live playback if still running
        if state.live_processor is not None:
            state.live_processor = None
            if state.live_thread:
                state.live_thread.join(timeout=2.0)
                state.live_thread = None
            if state.live_stream is not None:
                try:
                    state.live_stream.stop()  # type: ignore[attr-defined]
                    state.live_stream.close()  # type: ignore[attr-defined]
                except Exception:
                    pass
                state.live_stream = None
            console.print("[dim]Stopping live playback...[/dim]")
