"""Preset management — save, load, list, and delete effect-chain presets.

Presets are stored as TOML files under ``~/.config/torchfx/presets/``.
Each preset file uses the same ``[[effects]]`` format as configuration files
so they can also be passed via ``--config``.

"""

from __future__ import annotations

import shutil
from pathlib import Path

import typer

PRESETS_DIR = Path.home() / ".config" / "torchfx" / "presets"


def _ensure_presets_dir() -> Path:
    """Create the presets directory if it doesn't exist and return it."""
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    return PRESETS_DIR


def _preset_path(name: str) -> Path:
    """Return the file path for a named preset."""
    if not name.endswith(".toml"):
        name = name + ".toml"
    return PRESETS_DIR / name


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def preset_list_cmd() -> None:
    """List all saved presets.

    \b
    Examples
    --------
      torchfx preset list

    """
    from rich.console import Console
    from rich.table import Table

    _ensure_presets_dir()
    presets = sorted(PRESETS_DIR.glob("*.toml"))

    if not presets:
        typer.echo("No presets found.  Save one with: torchfx preset save <name> -e ...")
        return

    table = Table(title="[bold]Saved Presets[/bold]", border_style="dim")
    table.add_column("Name", style="cyan")
    table.add_column("File", style="dim")
    table.add_column("Size", style="white", justify="right")

    for p in presets:
        size = p.stat().st_size
        table.add_row(p.stem, str(p), f"{size} B")

    Console().print(table)


def preset_save_cmd(
    name: str = typer.Argument(..., help="Preset name (e.g. 'mastering', 'vocal-cleanup')."),
    effect: list[str] = typer.Option(  # noqa: B008
        [],
        "--effect",
        "-e",
        help="Effect specification (repeatable).  Same format as 'process --effect'.",
    ),
    source: str | None = typer.Option(  # noqa: UP007
        None,
        "--from",
        help="Copy effects from an existing TOML config file.",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing preset."),
) -> None:
    """Save an effect chain as a reusable preset.

    \b
    Examples
    --------
      torchfx preset save mastering -e normalize -e "reverb:decay=0.4,mix=0.2"
      torchfx preset save loud --from my_chain.toml

    """
    _ensure_presets_dir()
    path = _preset_path(name)

    if path.exists() and not force:
        typer.echo(
            f"Error: preset '{name}' already exists.  Use --force to overwrite.",
            err=True,
        )
        raise typer.Exit(code=1)

    # Source: copy from existing file
    if source is not None:
        src = Path(source)
        if not src.exists():
            typer.echo(f"Error: source file not found: {src}", err=True)
            raise typer.Exit(code=1)
        shutil.copy2(src, path)
        typer.echo(f"✓ Preset '{name}' saved (copied from {src})")
        return

    # Source: build from --effect flags
    if not effect:
        typer.echo("Error: provide at least one --effect or --from.", err=True)
        raise typer.Exit(code=1)

    # Validate that all effects parse correctly
    from cli.parsing import parse_effect_list

    parse_effect_list(effect)  # raises ValueError on bad specs

    # Write TOML
    lines: list[str] = []
    for spec in effect:
        effect_name, _, params_str = spec.partition(":")
        lines.append("[[effects]]")
        lines.append(f'name = "{effect_name.strip().lower()}"')
        if params_str.strip():
            for token in params_str.split(","):
                token = token.strip()
                if "=" in token:
                    key, _, val = token.partition("=")
                    # Try to write numeric values without quotes
                    try:
                        float(val.strip())
                        lines.append(f"{key.strip()} = {val.strip()}")
                    except ValueError:
                        lines.append(f'{key.strip()} = "{val.strip()}"')
                else:
                    # Positional — skip in TOML (user should use key=val)
                    pass
        lines.append("")

    path.write_text("\n".join(lines))
    typer.echo(f"✓ Preset '{name}' saved → {path}")


def preset_show_cmd(
    name: str = typer.Argument(..., help="Preset name to display."),
) -> None:
    """Show the contents of a preset.

    \b
    Examples
    --------
      torchfx preset show mastering

    """
    path = _preset_path(name)
    if not path.exists():
        typer.echo(f"Error: preset '{name}' not found.", err=True)
        raise typer.Exit(code=1)

    from rich.console import Console
    from rich.syntax import Syntax

    Console().print(Syntax(path.read_text(), "toml", theme="monokai"))


def preset_delete_cmd(
    name: str = typer.Argument(..., help="Preset name to delete."),
) -> None:
    """Delete a saved preset.

    \b
    Examples
    --------
      torchfx preset delete mastering

    """
    path = _preset_path(name)
    if not path.exists():
        typer.echo(f"Error: preset '{name}' not found.", err=True)
        raise typer.Exit(code=1)

    path.unlink()
    typer.echo(f"✓ Preset '{name}' deleted.")


def preset_apply_cmd(
    name: str = typer.Argument(..., help="Preset name to apply."),
    input_file: str = typer.Argument(..., help="Input audio file."),
    output_file: str = typer.Argument(..., help="Output audio file."),
) -> None:
    """Apply a saved preset to an audio file.

    \b
    Examples
    --------
      torchfx preset apply mastering input.wav output.wav

    """
    path = _preset_path(name)
    if not path.exists():
        typer.echo(f"Error: preset '{name}' not found.", err=True)
        raise typer.Exit(code=1)

    from cli.parsing import load_effects_from_config
    from torchfx.realtime import StreamProcessor

    effects = load_effects_from_config(path)
    processor = StreamProcessor(effects=effects)
    processor.process_file(input_file, output_file)
    typer.echo(f"✓ Applied '{name}' → {output_file}")
