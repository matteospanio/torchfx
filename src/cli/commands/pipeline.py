"""``torchfx pipe`` — SoX-style positional effect pipeline on a file.

Mirrors SoX's ``sox in out effect1 args effect2 args`` muscle memory, but with modern
long ``--flags`` and explicit Hz units::

    torchfx pipe in.wav out.wav lowpass --cutoff 800 reverb --mix 0.4

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import typer

from cli.parsing import parse_pipeline

if TYPE_CHECKING:
    from torchfx.effect import FX


def pipe_cmd(
    input_file: str = typer.Argument(..., help="Input audio file."),
    output_file: str = typer.Argument(..., help="Output audio file."),
    effects: list[str] = typer.Argument(  # noqa: B008
        ...,
        help='Positional effect pipeline, e.g. "lowpass --cutoff 800 reverb --mix 0.4".',
    ),
) -> None:
    """Apply a SoX-style positional effect pipeline to a file.

    \b
    Example
    -------
      torchfx pipe in.wav out.wav lowpass --cutoff 800 reverb --mix 0.4

    """
    from cli.app import get_state
    from torchfx.realtime import StreamProcessor

    device = str(get_state().get("device", "cpu"))
    try:
        chain = parse_pipeline(effects)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    effects_list = cast("list[FX]", list(chain.children()))
    processor = StreamProcessor(effects=effects_list, device=device)
    processor.process_file(Path(input_file), Path(output_file))
    typer.echo(f"✓ {input_file} → {output_file}")
