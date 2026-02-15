"""Root Typer application with global options and subcommand registration."""

from __future__ import annotations

import typer

from cli.commands import info, play, preset, process, record, sox, watch
from cli.repl import interactive_cmd

# ---------------------------------------------------------------------------
# Root application
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="torchfx",
    help="TorchFX — GPU-accelerated audio processing from the command line.",
    no_args_is_help=True,
    rich_markup_mode="rich",
    pretty_exceptions_enable=True,
)


# ---------------------------------------------------------------------------
# Global callback (runs before every subcommand)
# ---------------------------------------------------------------------------


@app.callback()
def _global_callback(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose (DEBUG) logging."),
    device: str = typer.Option(
        "cpu", "--device", "-d", help="Processing device [dim](cpu|cuda)[/]."
    ),
    config: str | None = typer.Option(  # noqa: UP007
        None,
        "--config",
        "-c",
        help="Path to a TOML configuration file.",
    ),
) -> None:
    """Configure global settings before any subcommand runs."""
    import torchfx.logging as tfx_log

    if verbose:
        tfx_log.enable_debug_logging()
    else:
        tfx_log.enable_logging()

    # Store globals on the app so subcommands can access them.
    _state["device"] = device
    _state["config"] = config


# Shared mutable state for the CLI session.
_state: dict[str, object] = {
    "device": "cpu",
    "config": None,
}


def get_state() -> dict[str, object]:
    """Return the global CLI state dictionary."""
    return _state


# ---------------------------------------------------------------------------
# Register subcommands
# ---------------------------------------------------------------------------

app.command(name="process", help="Apply effects to audio files.")(process.process_cmd)
app.command(name="info", help="Display audio file metadata.")(info.info_cmd)
app.command(name="play", help="Play an audio file through speakers.")(play.play_cmd)
app.command(name="record", help="Record audio from a microphone.")(record.record_cmd)

# Phase 2: sox-compatible commands
app.command(name="convert", help="Convert audio between formats/rates/channels.")(sox.convert_cmd)
app.command(name="trim", help="Extract a time range from an audio file.")(sox.trim_cmd)
app.command(name="concat", help="Concatenate multiple audio files.")(sox.concat_cmd)
app.command(name="stats", help="Display signal statistics for an audio file.")(sox.stats_cmd)

# Phase 2: preset management (sub-app)
preset_app = typer.Typer(
    name="preset",
    help="Manage reusable effect-chain presets.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
preset_app.command(name="list", help="List all saved presets.")(preset.preset_list_cmd)
preset_app.command(name="save", help="Save an effect chain as a preset.")(preset.preset_save_cmd)
preset_app.command(name="show", help="Display a preset's contents.")(preset.preset_show_cmd)
preset_app.command(name="delete", help="Delete a saved preset.")(preset.preset_delete_cmd)
preset_app.command(name="apply", help="Apply a preset to an audio file.")(preset.preset_apply_cmd)
app.add_typer(preset_app, name="preset")

# Phase 2: interactive REPL and watch mode
app.command(name="interactive", help="Launch the interactive REPL.")(interactive_cmd)
app.command(name="watch", help="Watch a directory and auto-process files.")(watch.watch_cmd)


# ---------------------------------------------------------------------------
# Version callback
# ---------------------------------------------------------------------------


def _version_callback(value: bool) -> None:
    if value:
        from importlib.metadata import version

        try:
            ver = version("torchfx")
        except Exception:
            ver = "unknown"
        typer.echo(f"torchfx {ver}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def _main(
    version: bool = typer.Option(  # noqa: ARG001
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """TorchFX — GPU-accelerated audio DSP from the command line."""
