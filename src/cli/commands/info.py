"""``torchfx info`` — display audio file metadata."""

from __future__ import annotations

from pathlib import Path

import typer


def info_cmd(
    file: str = typer.Argument(..., help="Path to the audio file."),
) -> None:
    """Display sample rate, channels, duration, encoding and file size.

    \b
    Examples
    --------
      torchfx info recording.wav
      torchfx info podcast.flac

    """
    import soundfile as sf  # type: ignore[import-untyped]
    from rich.console import Console
    from rich.table import Table

    path = Path(file)
    if not path.exists():
        typer.echo(f"Error: file not found: {path}", err=True)
        raise typer.Exit(code=1)

    try:
        meta = sf.info(str(path))
    except Exception as exc:
        typer.echo(f"Error reading file: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    duration_sec = meta.frames / meta.samplerate
    minutes = int(duration_sec // 60)
    seconds = duration_sec % 60
    duration_str = f"{minutes}:{seconds:05.2f}" if minutes else f"{seconds:.2f}s"

    size_bytes = path.stat().st_size
    if size_bytes >= 1_048_576:
        size_str = f"{size_bytes / 1_048_576:.2f} MB"
    elif size_bytes >= 1024:
        size_str = f"{size_bytes / 1024:.1f} KB"
    else:
        size_str = f"{size_bytes} B"

    table = Table(title=f"[bold]{path.name}[/bold]", show_header=False, border_style="dim")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("File", str(path))
    table.add_row("Format", meta.format)
    table.add_row("Sample Rate", f"{meta.samplerate:,} Hz")
    table.add_row("Channels", str(meta.channels))
    table.add_row("Duration", duration_str)
    table.add_row("Frames", f"{meta.frames:,}")
    table.add_row("Subtype", meta.subtype)
    table.add_row("File Size", size_str)

    console = Console()
    console.print(table)
