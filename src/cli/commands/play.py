"""``torchfx play`` — stream an audio file through speakers.

Requires the optional ``sounddevice`` package (install group ``realtime``).

"""

from __future__ import annotations

from pathlib import Path

import typer


def play_cmd(
    file: str = typer.Argument(..., help="Path to the audio file to play."),
    effect: list[str] = typer.Option(  # noqa: B008
        [],
        "--effect",
        "-e",
        help="Effect to apply before playback (repeatable).",
    ),
    device_id: int | None = typer.Option(  # noqa: UP007
        None,
        "--output-device",
        help="PortAudio output device index.",
    ),
) -> None:
    """Play an audio file through the default audio output.

    Optionally apply effects before playback.

    \b
    Examples
    --------
      torchfx play song.wav
      torchfx play recording.wav -e normalize -e "reverb:decay=0.4"

    """
    import numpy as np

    from torchfx.realtime._compat import get_sounddevice
    from torchfx.wave import Wave

    sd = get_sounddevice()

    path = Path(file)
    if not path.exists():
        typer.echo(f"Error: file not found: {path}", err=True)
        raise typer.Exit(code=1)

    wave = Wave.from_file(path)

    # Apply optional effects
    if effect:
        from cli.parsing import parse_effect_list

        for fx in parse_effect_list(effect):
            wave = wave | fx

    # Ensure CPU and float32 numpy array in (samples, channels) layout
    audio = wave.ys.cpu().numpy().T.astype(np.float32)
    sample_rate = wave.fs

    typer.echo(
        f"▶ Playing {path.name}  "
        f"({wave.channels()} ch, {sample_rate} Hz, "
        f"{wave.duration('sec'):.1f}s)"
    )

    try:
        sd.play(audio, samplerate=sample_rate, device=device_id)
        sd.wait()
    except KeyboardInterrupt:
        sd.stop()
        typer.echo("\n⏹ Playback stopped.")
