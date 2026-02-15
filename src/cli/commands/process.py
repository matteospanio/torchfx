"""``torchfx process`` — apply effects to one or many audio files.

Supports:
* Single-file processing (``torchfx process in.wav out.wav --effect reverb``)
* Batch glob processing (``torchfx process "*.wav" --output-dir ./out/ --effect gain:0.5``)
* Unix pipes (``cat in.wav | torchfx process - - --effect normalize | aplay``)
* GPU acceleration (``--device cuda``)
* Effect chains from TOML config files (``--config chain.toml``)

"""

from __future__ import annotations

import glob
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from cli.parsing import (
    list_effects,
    load_config_defaults,
    load_effects_from_config,
    parse_effect_list,
)

if TYPE_CHECKING:
    from torchfx.effect import FX

# Pre-compute for help text (avoids B008 in function default)
_EXAMPLE_EFFECTS = ", ".join(list_effects()[:8])


def _resolve_effects(
    effect_specs: list[str],
    config_path: str | None,
) -> list[FX]:
    """Build a flat effect list from CLI ``--effect`` flags and/or config file."""
    effects: list[FX] = []

    # 1. Effects from config file
    if config_path:
        effects.extend(load_effects_from_config(config_path))

    # 2. Effects from CLI flags (appended *after* config effects)
    if effect_specs:
        effects.extend(parse_effect_list(effect_specs))

    if not effects:
        typer.echo("Error: no effects specified. Use --effect or --config.", err=True)
        raise typer.Exit(code=1)

    return effects


def _process_single(
    input_path: Path,
    output_path: Path,
    effects: list[FX],
    device: str,
    chunk_size: int,
    overlap: int,
) -> None:
    """Process a single file through *effects*."""
    from torchfx.realtime import StreamProcessor

    processor = StreamProcessor(
        effects=effects,
        chunk_size=chunk_size,
        overlap=overlap,
        device=device,
    )
    processor.process_file(input_path, output_path)


def _process_pipe(
    effects: list[FX],
    device: str,
    fmt: str,
    rate: int,
    channels: int,
) -> None:
    """Read WAV/raw from stdin, apply effects, write to stdout."""
    import io

    import torch
    import torchaudio

    stdin_bytes = sys.stdin.buffer.read()
    buf = io.BytesIO(stdin_bytes)

    if fmt == "wav":
        waveform, sr = torchaudio.load(buf, format="wav")
    else:
        import numpy as np

        raw = np.frombuffer(stdin_bytes, dtype=np.float32)
        waveform = torch.from_numpy(raw).reshape(channels, -1)
        sr = rate

    if device != "cpu":
        waveform = waveform.to(device)

    for fx in effects:
        if hasattr(fx, "fs") and fx.fs is None:
            fx.fs = sr
        waveform = fx(waveform)

    if device != "cpu":
        waveform = waveform.cpu()

    out_buf = io.BytesIO()
    torchaudio.save(out_buf, waveform, sr, format="wav")
    sys.stdout.buffer.write(out_buf.getvalue())


# ---------------------------------------------------------------------------
# Typer command
# ---------------------------------------------------------------------------


def process_cmd(
    input_path: str = typer.Argument(
        ...,
        help="Input audio file path, glob pattern, or '-' for stdin.",
    ),
    output_path: str | None = typer.Argument(  # noqa: UP007
        None,
        help="Output file path or '-' for stdout.  Omit when using --output-dir.",
    ),
    effect: list[str] = typer.Option(  # noqa: B008
        [],
        "--effect",
        "-e",
        help=(
            "Effect specification.  Repeatable.  "
            "Format: name[:param1=val1,param2=val2].  "
            f"Available: {_EXAMPLE_EFFECTS}…"
        ),
    ),
    output_dir: str | None = typer.Option(  # noqa: UP007
        None,
        "--output-dir",
        "-O",
        help="Output directory for batch processing.",
    ),
    config: str | None = typer.Option(  # noqa: UP007
        None,
        "--config",
        "-c",
        help="TOML config file defining an effect chain.",
    ),
    chunk_size: int = typer.Option(
        65536,
        "--chunk-size",
        help="Samples per processing chunk.",
    ),
    overlap: int = typer.Option(
        0,
        "--overlap",
        help="Overlap samples between chunks.",
    ),
    pipe_format: str = typer.Option(
        "wav",
        "--format",
        "-f",
        help="Pipe audio format [dim](wav|raw)[/].",
    ),
    pipe_rate: int = typer.Option(
        44100,
        "--rate",
        "-r",
        help="Sample rate for raw pipe input.",
    ),
    pipe_channels: int = typer.Option(
        1,
        "--channels",
        help="Number of channels for raw pipe input.",
    ),
) -> None:
    """Apply an effect chain to audio files.

    \b
    Examples
    --------
      torchfx process in.wav out.wav -e normalize -e "reverb:decay=0.6"
      torchfx process "*.wav" -O ./processed/ -e gain:0.5
      cat in.wav | torchfx process - - -e normalize | aplay
      torchfx process in.wav out.wav --config chain.toml

    """
    from cli.app import get_state

    state = get_state()
    device = str(state.get("device", "cpu"))
    global_config = state.get("config")

    # Merge config sources (--config on process takes precedence)
    cfg_path = config or (global_config if isinstance(global_config, str) else None)
    cfg_defaults: dict[str, object] = {}
    if cfg_path:
        cfg_defaults = load_config_defaults(cfg_path)
        device = str(cfg_defaults.get("device", device))
        _cs = cfg_defaults.get("chunk_size", chunk_size)
        chunk_size = int(_cs) if isinstance(_cs, int | float | str) else chunk_size
        _ov = cfg_defaults.get("overlap", overlap)
        overlap = int(_ov) if isinstance(_ov, int | float | str) else overlap

    effects = _resolve_effects(effect, cfg_path)

    # ── Pipe mode ─────────────────────────────────────────────
    if input_path == "-":
        _process_pipe(effects, device, pipe_format, pipe_rate, pipe_channels)
        return

    # ── Batch mode (glob) ─────────────────────────────────────
    input_files = sorted(glob.glob(input_path))
    if not input_files:
        typer.echo(f"Error: no files match '{input_path}'.", err=True)
        raise typer.Exit(code=1)

    if len(input_files) > 1 or output_dir:
        _batch_process(input_files, output_dir, effects, device, chunk_size, overlap)
        return

    # ── Single-file mode ──────────────────────────────────────
    resolved_input = Path(input_files[0])

    if output_path is None or output_path == "-":
        if output_path == "-":
            # Write to stdout
            _single_to_stdout(resolved_input, effects, device, chunk_size, overlap)
        else:
            typer.echo("Error: output path is required for single-file mode.", err=True)
            raise typer.Exit(code=1)
        return

    resolved_output = Path(output_path)
    _process_single(resolved_input, resolved_output, effects, device, chunk_size, overlap)
    typer.echo(f"✓ {resolved_input} → {resolved_output}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _single_to_stdout(
    input_path: Path,
    effects: list[FX],
    device: str,
    _chunk_size: int,
    _overlap: int,
) -> None:
    """Process single file and write WAV bytes to stdout."""
    import io

    import torchaudio

    from torchfx.wave import Wave

    wave = Wave.from_file(input_path)
    if device != "cpu":
        wave = wave.to(device)  # type: ignore[arg-type]

    for fx in effects:
        wave = wave | fx

    if device != "cpu":
        wave = wave.to("cpu")

    out_buf = io.BytesIO()
    torchaudio.save(out_buf, wave.ys, wave.fs, format="wav")
    sys.stdout.buffer.write(out_buf.getvalue())


def _batch_process(
    input_files: list[str],
    output_dir: str | None,
    effects: list[FX],
    device: str,
    chunk_size: int,
    overlap: int,
) -> None:
    """Batch-process a list of files with a Rich progress bar."""
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TextColumn,
        TimeRemainingColumn,
    )

    from torchfx.realtime import StreamProcessor

    if output_dir is None:
        typer.echo(
            "Error: --output-dir is required for batch processing (multiple input files).",
            err=True,
        )
        raise typer.Exit(code=1)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    processor = StreamProcessor(
        effects=effects,
        chunk_size=chunk_size,
        overlap=overlap,
        device=device,
    )

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Processing", total=len(input_files))
        for fpath in input_files:
            src = Path(fpath)
            dst = out / src.name
            try:
                processor.process_file(src, dst)
                progress.console.print(f"  [green]✓[/green] {src.name}")
            except Exception as exc:
                progress.console.print(f"  [red]✗[/red] {src.name}: {exc}")
            progress.advance(task)

    typer.echo(f"Batch complete — {len(input_files)} file(s) → {out}/")
