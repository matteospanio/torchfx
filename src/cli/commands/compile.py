"""``torchfx compile`` — freeze an effect pipeline into a portable ``.fxg`` artifact.

The artifact stores each node either as a precomputed ``[K, 6]`` SOS matrix (filters,
designed once at compile time) or as a re-instantiable effect spec (reverb, normalize,
…). Loading it skips coefficient design entirely — handy for deployment / repeated runs.

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import typer

from cli.parsing import EFFECT_REGISTRY, parse_pipeline_specs

if TYPE_CHECKING:
    from torchfx.chain import FilterChain
    from torchfx.effect import FX

#: Bump when the on-disk artifact layout changes.
ARTIFACT_VERSION = 1


def save_compiled(pipeline: str, fs: int, path: str | Path) -> None:
    """Compile *pipeline* at *fs* and write a frozen ``.fxg`` artifact to *path*."""
    import torch

    nodes: list[dict[str, Any]] = []
    for name, kwargs in parse_pipeline_specs(pipeline):
        cls, _ = EFFECT_REGISTRY[name]
        try:
            fx = cls(**kwargs)
        except TypeError as exc:
            raise ValueError(f"Invalid parameters for effect '{name}': {exc}") from exc
        fx.compile(fs)  # eager coefficient design
        sos = getattr(fx, "_sos", None)
        if sos is not None:
            nodes.append({"kind": "sos", "sos": sos.detach().cpu()})
        else:
            nodes.append({"kind": "effect", "name": name, "kwargs": kwargs})

    torch.save({"version": ARTIFACT_VERSION, "fs": fs, "nodes": nodes}, str(path))


def load_compiled(path: str | Path) -> FilterChain:
    """Load a ``.fxg`` artifact into a frozen :class:`~torchfx.chain.FilterChain`."""
    import torch

    from torchfx.chain import FilterChain
    from torchfx.filter import SOSFilter

    data = torch.load(str(path), weights_only=True)
    if data.get("version") != ARTIFACT_VERSION:
        raise ValueError(
            f"Unsupported .fxg version {data.get('version')!r} (expected {ARTIFACT_VERSION})."
        )
    fs = int(data["fs"])
    nodes: list[FX] = []
    for nd in data["nodes"]:
        if nd["kind"] == "sos":
            nodes.append(SOSFilter(nd["sos"], fs=fs))
        else:
            cls, _ = EFFECT_REGISTRY[nd["name"]]
            fx = cls(**nd["kwargs"])
            cast(Any, fx).fs = fs
            nodes.append(fx)
    return FilterChain(*nodes).freeze()


def compile_cmd(
    pipeline: str = typer.Argument(
        ...,
        help='Effect pipeline, e.g. "lowpass --cutoff 800 | reverb --mix 0.4".',
    ),
    output: str = typer.Option(..., "--output", "-o", help="Output .fxg artifact path."),
    fs: int = typer.Option(48000, "--fs", help="Sampling rate to design coefficients for."),
) -> None:
    """Freeze an effect pipeline into a portable ``.fxg`` artifact.

    \b
    Example
    -------
      torchfx compile "lowpass --cutoff 800 | reverb --mix 0.4" --fs 48000 -o chain.fxg

    """
    try:
        save_compiled(pipeline, fs, output)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"✓ compiled pipeline → {output} (fs={fs})")
