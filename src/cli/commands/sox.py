"""Sox-compatible commands: ``convert``, ``trim``, ``concat``, ``stats``.

These commands mirror a subset of the SoX (Sound eXchange) CLI to provide
familiar workflows for users transitioning from SoX, while leveraging
TorchFX's GPU acceleration under the hood.
"""

from __future__ import annotations

from pathlib import Path

import typer

# ── convert ──────────────────────────────────────────────────────────


def convert_cmd(
    input_file: str = typer.Argument(..., help="Input audio file."),
    output_file: str = typer.Argument(
        ..., help="Output audio file (format inferred from extension)."
    ),
    sample_rate: int | None = typer.Option(  # noqa: UP007
        None,
        "--rate",
        "-r",
        help="Target sample rate in Hz (resamples if different).",
    ),
    channels: int | None = typer.Option(  # noqa: UP007
        None,
        "--channels",
        "-c",
        help="Target number of channels (1=mono, 2=stereo).",
    ),
    bits: int | None = typer.Option(  # noqa: UP007
        None,
        "--bits",
        "-b",
        help="Target bits per sample (8, 16, 24, 32).",
    ),
) -> None:
    """Convert audio between formats, sample rates, and channel layouts.

    \b
    Examples
    --------
      torchfx convert input.wav output.flac
      torchfx convert song.wav mono.wav --channels 1 --rate 16000
      torchfx convert hi-res.wav cd-quality.wav -r 44100 -b 16

    """
    import torch
    from torchaudio import functional as F

    from torchfx.wave import Wave

    src = Path(input_file)
    if not src.exists():
        typer.echo(f"Error: file not found: {src}", err=True)
        raise typer.Exit(code=1)

    wave = Wave.from_file(src)

    # Resample
    if sample_rate is not None and sample_rate != wave.fs:
        wave.ys = F.resample(wave.ys, wave.fs, sample_rate)
        wave.fs = sample_rate

    # Channel conversion
    if channels is not None and channels != wave.channels():
        if channels == 1 and wave.channels() > 1:
            # Mix down to mono
            wave.ys = wave.ys.mean(dim=0, keepdim=True)
        elif channels == 2 and wave.channels() == 1:
            # Duplicate mono to stereo
            wave.ys = torch.cat([wave.ys, wave.ys], dim=0)
        elif channels > wave.channels():
            # Pad with silence
            pad = torch.zeros(channels - wave.channels(), wave.ys.shape[1])
            wave.ys = torch.cat([wave.ys, pad], dim=0)
        else:
            # Truncate channels
            wave.ys = wave.ys[:channels]

    # Save with optional bit depth
    kwargs: dict[str, object] = {}
    if bits is not None:
        kwargs["bits_per_sample"] = bits

    wave.save(output_file, **kwargs)  # type: ignore[arg-type]
    typer.echo(f"✓ {src} → {output_file}")


# ── trim ─────────────────────────────────────────────────────────────


def trim_cmd(
    input_file: str = typer.Argument(..., help="Input audio file."),
    output_file: str = typer.Argument(..., help="Output audio file."),
    start: float = typer.Option(
        0.0,
        "--start",
        "-s",
        help="Start time in seconds.",
    ),
    end: float | None = typer.Option(  # noqa: UP007
        None,
        "--end",
        "-e",
        help="End time in seconds (defaults to end of file).",
    ),
    duration: float | None = typer.Option(  # noqa: UP007
        None,
        "--duration",
        "-d",
        help="Duration in seconds (alternative to --end).",
    ),
) -> None:
    """Extract a time range from an audio file.

    \b
    Examples
    --------
      torchfx trim input.wav clip.wav --start 1.5 --end 4.0
      torchfx trim input.wav clip.wav -s 10 -d 5
      torchfx trim input.wav first10s.wav --duration 10

    """
    from torchfx.wave import Wave

    src = Path(input_file)
    if not src.exists():
        typer.echo(f"Error: file not found: {src}", err=True)
        raise typer.Exit(code=1)

    wave = Wave.from_file(src)

    start_sample = int(start * wave.fs)
    if duration is not None:
        end_sample = start_sample + int(duration * wave.fs)
    elif end is not None:
        end_sample = int(end * wave.fs)
    else:
        end_sample = wave.ys.shape[1]

    # Clamp
    end_sample = min(end_sample, wave.ys.shape[1])
    start_sample = max(0, start_sample)

    if start_sample >= end_sample:
        typer.echo("Error: start must be before end.", err=True)
        raise typer.Exit(code=1)

    wave.ys = wave.ys[:, start_sample:end_sample]
    wave.save(output_file)

    trim_dur = (end_sample - start_sample) / wave.fs
    typer.echo(f"✓ Trimmed {trim_dur:.2f}s → {output_file}")


# ── concat ───────────────────────────────────────────────────────────


def concat_cmd(
    files: list[str] = typer.Argument(  # noqa: B008
        ..., help="Audio files to concatenate (at least 2)."
    ),
    output: str = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output file path.",
    ),
) -> None:
    """Concatenate multiple audio files into one.

    All files must share the same sample rate and number of channels.

    \b
    Examples
    --------
      torchfx concat part1.wav part2.wav part3.wav -o full.wav

    """
    import torch

    from torchfx.wave import Wave

    if len(files) < 2:
        typer.echo("Error: at least 2 files are required for concatenation.", err=True)
        raise typer.Exit(code=1)

    waves: list[Wave] = []
    for f in files:
        p = Path(f)
        if not p.exists():
            typer.echo(f"Error: file not found: {p}", err=True)
            raise typer.Exit(code=1)
        waves.append(Wave.from_file(p))

    # Validate compatibility
    fs = waves[0].fs
    ch = waves[0].channels()
    for i, w in enumerate(waves[1:], start=1):
        if w.fs != fs:
            typer.echo(
                f"Error: sample rate mismatch — {files[0]} is {fs} Hz, {files[i]} is {w.fs} Hz.",
                err=True,
            )
            raise typer.Exit(code=1)
        if w.channels() != ch:
            typer.echo(
                f"Error: channel count mismatch — {files[0]} has {ch} ch, "
                f"{files[i]} has {w.channels()} ch.",
                err=True,
            )
            raise typer.Exit(code=1)

    combined = torch.cat([w.ys for w in waves], dim=1)
    result = Wave(combined, fs=fs)
    result.save(output)

    total_dur = result.duration("sec")
    typer.echo(f"✓ Concatenated {len(files)} files ({total_dur:.2f}s) → {output}")


# ── stats ────────────────────────────────────────────────────────────


def stats_cmd(
    file: str = typer.Argument(..., help="Audio file to analyse."),
) -> None:
    """Display signal statistics for an audio file.

    \b
    Reported metrics:
      Peak level (dBFS), RMS level (dBFS), crest factor, DC offset,
      min/max sample values, and dynamic range.

    \b
    Examples
    --------
      torchfx stats recording.wav

    """
    import math

    from rich.console import Console
    from rich.table import Table

    from torchfx.wave import Wave

    src = Path(file)
    if not src.exists():
        typer.echo(f"Error: file not found: {src}", err=True)
        raise typer.Exit(code=1)

    wave = Wave.from_file(src)
    data = wave.ys.float()

    peak = data.abs().max().item()
    rms = data.pow(2).mean().sqrt().item()
    dc_offset = data.mean().item()
    min_val = data.min().item()
    max_val = data.max().item()

    peak_db = 20 * math.log10(peak) if peak > 0 else float("-inf")
    rms_db = 20 * math.log10(rms) if rms > 0 else float("-inf")
    crest = peak / rms if rms > 0 else float("inf")
    crest_db = 20 * math.log10(crest) if crest > 0 and crest != float("inf") else float("inf")
    dynamic_range = peak_db - rms_db if peak_db != float("-inf") else 0.0

    table = Table(
        title=f"[bold]{Path(file).name}[/bold]  —  Signal Statistics",
        show_header=False,
        border_style="dim",
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    for ch in range(wave.channels()):
        ch_data = data[ch]
        ch_peak = ch_data.abs().max().item()
        ch_rms = ch_data.pow(2).mean().sqrt().item()
        ch_peak_db = 20 * math.log10(ch_peak) if ch_peak > 0 else float("-inf")
        ch_rms_db = 20 * math.log10(ch_rms) if ch_rms > 0 else float("-inf")
        table.add_row(f"Channel {ch}", "")
        table.add_row("  Peak", f"{ch_peak_db:+.2f} dBFS  ({ch_peak:.6f})")
        table.add_row("  RMS", f"{ch_rms_db:+.2f} dBFS  ({ch_rms:.6f})")

    table.add_section()
    table.add_row("Overall Peak", f"{peak_db:+.2f} dBFS  ({peak:.6f})")
    table.add_row("Overall RMS", f"{rms_db:+.2f} dBFS  ({rms:.6f})")
    table.add_row("Crest Factor", f"{crest_db:.2f} dB  ({crest:.2f}×)")
    table.add_row("DC Offset", f"{dc_offset:+.6f}")
    table.add_row("Min Sample", f"{min_val:+.6f}")
    table.add_row("Max Sample", f"{max_val:+.6f}")
    table.add_row("Dynamic Range", f"{dynamic_range:.2f} dB")

    console = Console()
    console.print(table)
