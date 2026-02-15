"""``torchfx record`` — capture audio from a microphone.

Requires the optional ``sounddevice`` package (install group ``realtime``).

"""

from __future__ import annotations

from pathlib import Path

import typer


def record_cmd(
    output: str = typer.Argument(..., help="Output file path (e.g. recording.wav)."),
    duration: float = typer.Option(
        ...,
        "--duration",
        "-t",
        help="Recording duration in seconds.",
    ),
    sample_rate: int = typer.Option(
        44100,
        "--sample-rate",
        "-r",
        help="Sample rate in Hz.",
    ),
    channels: int = typer.Option(
        1,
        "--channels",
        "-C",
        help="Number of input channels.",
    ),
    device_id: int | None = typer.Option(  # noqa: UP007
        None,
        "--input-device",
        help="PortAudio input device index.",
    ),
    effect: list[str] = typer.Option(  # noqa: B008
        [],
        "--effect",
        "-e",
        help="Effect to apply after recording (repeatable).",
    ),
) -> None:
    """Record audio from the default input device.

    \b
    Examples
    --------
      torchfx record out.wav --duration 10
      torchfx record out.wav -t 5 -r 48000 -C 2
      torchfx record out.wav -t 10 -e normalize -e "reverb:decay=0.3"

    """
    import torch

    from torchfx.realtime._compat import get_sounddevice
    from torchfx.wave import Wave

    sd = get_sounddevice()

    total_frames = int(duration * sample_rate)

    typer.echo(
        f"● Recording {duration:.1f}s  "
        f"({channels} ch, {sample_rate} Hz) — press Ctrl-C to stop early."
    )

    try:
        audio = sd.rec(
            frames=total_frames,
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
            device=device_id,
        )
        sd.wait()
    except KeyboardInterrupt:
        sd.stop()
        typer.echo("\n⏹ Recording stopped early.")
        audio = sd.rec(0)

    # Convert (samples, channels) → (channels, samples)
    tensor = torch.from_numpy(audio.T.copy())
    wave = Wave(tensor, fs=sample_rate)

    # Apply optional post-processing effects
    if effect:
        from cli.parsing import parse_effect_list

        for fx in parse_effect_list(effect):
            wave = wave | fx

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wave.save(out_path)

    recorded_sec = wave.duration("sec")
    typer.echo(f"✓ Saved {recorded_sec:.1f}s to {out_path}")
