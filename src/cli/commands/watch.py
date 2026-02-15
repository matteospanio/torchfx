"""``torchfx watch`` — monitor a directory and auto-apply effects.

Uses the ``watchdog`` library to watch for new or modified audio files in
a source directory, automatically applies the configured effect chain,
and writes results to an output directory.  Useful for DAW integration
where an export folder is monitored for fresh bounces.

"""

from __future__ import annotations

import time
from pathlib import Path

import typer
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

_AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif"}


def watch_cmd(
    input_dir: str = typer.Argument(..., help="Directory to watch for audio files."),
    output_dir: str = typer.Argument(..., help="Output directory for processed files."),
    effect: list[str] = typer.Option(  # noqa: B008
        [],
        "--effect",
        "-e",
        help="Effect specification (repeatable).",
    ),
    config: str | None = typer.Option(  # noqa: UP007
        None,
        "--config",
        "-c",
        help="TOML config file defining the effect chain.",
    ),
    preset: str | None = typer.Option(  # noqa: UP007
        None,
        "--preset",
        "-p",
        help="Named preset to apply.",
    ),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Watch subdirectories."),
    existing: bool = typer.Option(
        False,
        "--existing",
        help="Process files already present in the directory on startup.",
    ),
) -> None:
    """Watch a directory and auto-process new/modified audio files.

    \b
    Examples
    --------
      torchfx watch ./input/ ./output/ -e normalize -e "reverb:decay=0.4"
      torchfx watch ./bounces/ ./mastered/ --config master.toml
      torchfx watch ./raw/ ./processed/ --preset vocal-cleanup --recursive

    """
    from rich.console import Console

    from torchfx.effect import FX

    console = Console()

    # ── Resolve effects ────────────────────────────────────────
    effects: list[FX] = []

    if preset:
        from cli.commands.preset import _preset_path
        from cli.parsing import load_effects_from_config

        pp = _preset_path(preset)
        if not pp.exists():
            typer.echo(f"Error: preset '{preset}' not found.", err=True)
            raise typer.Exit(code=1)
        effects.extend(load_effects_from_config(pp))

    if config:
        from cli.parsing import load_effects_from_config as load_cfg

        effects.extend(load_cfg(config))

    if effect:
        from cli.parsing import parse_effect_list

        effects.extend(parse_effect_list(effect))

    if not effects:
        typer.echo("Error: no effects specified.  Use --effect, --config, or --preset.", err=True)
        raise typer.Exit(code=1)

    # ── Directories ────────────────────────────────────────────
    src = Path(input_dir)
    dst = Path(output_dir)
    if not src.is_dir():
        typer.echo(f"Error: not a directory: {src}", err=True)
        raise typer.Exit(code=1)
    dst.mkdir(parents=True, exist_ok=True)

    # ── Process helper ─────────────────────────────────────────
    def _process(path: Path) -> None:
        if path.suffix.lower() not in _AUDIO_EXTENSIONS:
            return
        out_path = dst / path.relative_to(src)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from torchfx.realtime import StreamProcessor

            processor = StreamProcessor(effects=list(effects))
            processor.process_file(path, out_path)
            console.print(f"  [green]✓[/green] {path.name} → {out_path}")
        except Exception as exc:
            console.print(f"  [red]✗[/red] {path.name}: {exc}")

    # ── Process existing files ─────────────────────────────────
    if existing:
        pattern = "**/*" if recursive else "*"
        for f in sorted(src.glob(pattern)):
            if f.is_file():
                _process(f)

    # ── Watchdog handler ───────────────────────────────────────
    class _Handler(FileSystemEventHandler):
        def on_created(self, event: FileSystemEvent) -> None:
            if not event.is_directory:
                _process(Path(str(event.src_path)))

        def on_modified(self, event: FileSystemEvent) -> None:
            if not event.is_directory:
                _process(Path(str(event.src_path)))

    observer = Observer()
    observer.schedule(_Handler(), str(src), recursive=recursive)
    observer.start()

    console.print(
        f"[bold cyan]👁 Watching[/bold cyan] {src}  →  {dst}\n"
        f"[dim]Effects: {len(effects)}  |  Recursive: {recursive}  |  "
        f"Press Ctrl-C to stop.[/dim]\n"
    )

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        console.print("\n[dim]⏹ Watch stopped.[/dim]")
    observer.join()
