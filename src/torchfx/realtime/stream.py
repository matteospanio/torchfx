"""Chunk-based stream processing for large audio files.

This module provides the ``StreamProcessor`` class for processing audio
files in chunks without loading the entire file into memory. This is
essential for processing files that are too large to fit in memory or
for streaming applications.

Classes
-------
StreamProcessor
    Chunk-based file processor for large audio files.

Examples
--------
>>> from torchfx.realtime import StreamProcessor
>>> from torchfx.effect import Gain
>>> processor = StreamProcessor(effects=[Gain(0.5)], chunk_size=65536)
>>> # processor.process_file("large_input.wav", "output.wav")

"""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterable, Sequence
from pathlib import Path
from typing import cast

import soundfile as sf  # type: ignore[import-untyped]
import torch
import torchaudio
from torch import Tensor, nn

from torchfx.effect import FX
from torchfx.filter.__base import AbstractFilter
from torchfx.logging import get_logger
from torchfx.validation import validate_positive

_logger = get_logger("realtime.stream")


class StreamProcessor:
    """Process audio files in chunks without loading the entire file.

    Reads audio in configurable chunk sizes, applies an effect chain
    to each chunk, and writes output progressively. Supports an
    overlap parameter for effects that need context beyond chunk
    boundaries.

    Parameters
    ----------
    effects : Sequence[FX] | nn.Sequential
        Chain of effects to apply in order.
    chunk_size : int
        Number of samples per processing chunk. Default is 65536.
    overlap : int
        Number of overlap samples between chunks. Default is 0.
    device : str
        Processing device (``"cpu"`` or ``"cuda"``). Default is ``"cpu"``.

    Examples
    --------
    >>> from torchfx.realtime import StreamProcessor
    >>> from torchfx.effect import Gain
    >>> processor = StreamProcessor(effects=[Gain(0.5)])

    """

    def __init__(
        self,
        effects: Sequence[FX] | nn.Sequential,
        chunk_size: int = 65536,
        overlap: int = 0,
        device: str = "cpu",
    ) -> None:
        validate_positive(chunk_size, "chunk_size")
        if overlap < 0:
            raise ValueError(f"Overlap must be non-negative, got {overlap}")
        if overlap >= chunk_size:
            raise ValueError(f"Overlap ({overlap}) must be less than chunk_size ({chunk_size})")

        self._effects: list[FX] = self._normalize_effects(effects)
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._device = device

    def __enter__(self) -> StreamProcessor:
        """Return self for use as context manager.

        The context manager form is a convenience for scoping the
        processor lifetime. No special start/stop is needed.

        Examples
        --------
        >>> with StreamProcessor(effects=[Gain(0.5)]) as processor:
        ...     processor.process_file("in.wav", "out.wav")

        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Clean up on context exit."""
        pass

    @staticmethod
    def _normalize_effects(effects: Sequence[FX] | nn.Sequential) -> list[FX]:
        modules: Iterable[FX] = (
            cast(Iterable[FX], effects) if isinstance(effects, nn.Sequential) else effects
        )

        normalized: list[FX] = []
        for effect in modules:
            if not isinstance(effect, FX):
                raise TypeError("All effects must inherit from FX when used in StreamProcessor")
            normalized.append(effect)

        return normalized

    def _configure_effects(self, fs: int) -> None:
        """Set sample rate and compute coefficients for all effects.

        Resets coefficients if the sample rate has changed since
        the last configuration.

        Parameters
        ----------
        fs : int
            Sample rate in Hz.

        Raises
        ------
        ValueError
            If a filter's cutoff frequency exceeds the Nyquist frequency.

        """
        nyquist = fs / 2.0
        for effect in self._effects:
            if hasattr(effect, "fs"):
                current_fs = effect.fs
                if current_fs != fs:
                    effect.fs = fs  # type: ignore
                    # Force coefficient recomputation with new sample rate
                    if isinstance(effect, AbstractFilter):
                        effect.compute_coefficients()
                        reset_state = getattr(effect, "reset_state", None)
                        if callable(reset_state):
                            reset_state()
            # Validate cutoff before computing coefficients
            if isinstance(effect, AbstractFilter) and hasattr(effect, "cutoff"):
                cutoff = effect.cutoff
                if isinstance(cutoff, int | float) and cutoff >= nyquist:
                    raise ValueError(
                        f"{type(effect).__name__} cutoff ({cutoff} Hz) must be "
                        f"below the Nyquist frequency ({nyquist} Hz) for sample rate "
                        f"{fs} Hz. Reduce the cutoff or use a higher sample rate file."
                    )
            if isinstance(effect, AbstractFilter) and not effect._has_computed_coeff:
                effect.compute_coefficients()

    @torch.no_grad()
    def process_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        format: str | None = None,  # noqa: A002
        subtype: str | None = None,
    ) -> None:
        """Process an audio file chunk by chunk.

        Reads the input file in chunks, applies the effect chain to each
        chunk, and writes the result to the output file.

        Parameters
        ----------
        input_path : str | Path
            Path to the input audio file.
        output_path : str | Path
            Path to the output audio file.
        format : str | None
            Output format (e.g., ``"WAV"``, ``"FLAC"``). Inferred from
            extension if None.
        subtype : str | None
            Output subtype (e.g., ``"PCM_16"``, ``"FLOAT"``). Uses
            default for format if None.

        Examples
        --------
        >>> from torchfx.realtime import StreamProcessor
        >>> from torchfx.effect import Gain
        >>> processor = StreamProcessor(effects=[Gain(0.5)])
        >>> # processor.process_file("input.wav", "output.wav")

        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get file metadata without loading (torchaudio.info removed in 2.10+)
        info = sf.info(str(input_path))
        fs = info.samplerate
        num_frames = info.frames
        channels = info.channels

        _logger.info(
            "Processing %s: %d frames, %d channels, %dHz",
            input_path,
            num_frames,
            channels,
            fs,
        )

        self._configure_effects(fs)

        # Determine output format
        if format is None:
            ext = output_path.suffix.lower()
            format_map = {".wav": "WAV", ".flac": "FLAC", ".ogg": "OGG"}
            format = format_map.get(ext, "WAV")  # noqa: A001

        if subtype is None:
            subtype = "FLOAT" if format == "WAV" else None

        # Process and write chunks
        with sf.SoundFile(
            str(output_path),
            mode="w",
            samplerate=fs,
            channels=channels,
            format=format,
            subtype=subtype,
        ) as out_file:
            hop_size = self._chunk_size - self._overlap
            offset = 0

            while offset < num_frames:
                # Read chunk (with overlap)
                read_size = min(self._chunk_size, num_frames - offset)
                waveform, sample_rate = torchaudio.load(
                    str(input_path),
                    frame_offset=offset,
                    num_frames=read_size,
                )

                # Move to processing device
                if self._device != "cpu":
                    waveform = waveform.to(self._device)

                # Apply effect chain
                for effect in self._effects:
                    if not isinstance(effect, nn.Module):
                        raise TypeError("Effects must inherit from torch.nn.Module")
                    call_effect = cast(Callable[[Tensor], Tensor], effect)
                    waveform = call_effect(waveform)

                # Move back to CPU for writing
                if self._device != "cpu":
                    waveform = waveform.cpu()

                # For overlap mode, only write the non-overlapping part
                if self._overlap > 0 and offset > 0:
                    write_data = waveform[:, self._overlap :]
                else:
                    write_data = waveform

                # Write: convert (channels, frames) -> (frames, channels)
                out_file.write(write_data.numpy().T)

                offset += hop_size
                _logger.debug("Processed %d / %d frames", min(offset, num_frames), num_frames)

        _logger.info("Output written to %s", output_path)

    @torch.no_grad()
    def process_chunks(
        self,
        input_path: str | Path,
    ) -> Generator[Tensor, None, None]:
        """Yield processed chunks as tensors.

        Generator API for streaming to another process, network,
        or real-time playback.

        Parameters
        ----------
        input_path : str | Path
            Path to the input audio file.

        Yields
        ------
        Tensor
            Processed audio chunks of shape ``(channels, chunk_size)``.

        Examples
        --------
        >>> from torchfx.realtime import StreamProcessor
        >>> from torchfx.effect import Gain
        >>> processor = StreamProcessor(effects=[Gain(0.5)])
        >>> # for chunk in processor.process_chunks("input.wav"):
        >>> #     print(chunk.shape)

        """
        input_path = Path(input_path)

        info = sf.info(str(input_path))
        fs = info.samplerate
        num_frames = info.frames

        self._configure_effects(fs)

        hop_size = self._chunk_size - self._overlap
        offset = 0

        while offset < num_frames:
            read_size = min(self._chunk_size, num_frames - offset)
            waveform, _sample_rate = torchaudio.load(
                str(input_path),
                frame_offset=offset,
                num_frames=read_size,
            )

            if self._device != "cpu":
                waveform = waveform.to(self._device)

            for effect in self._effects:
                if not isinstance(effect, nn.Module):
                    raise TypeError("Effects must inherit from torch.nn.Module")
                call_effect = cast(Callable[[Tensor], Tensor], effect)
                waveform = call_effect(waveform)

            if self._device != "cpu":
                waveform = waveform.cpu()

            # For overlap mode, only yield the non-overlapping part
            if self._overlap > 0 and offset > 0:
                yield waveform[:, self._overlap :]
            else:
                yield waveform

            offset += hop_size

    @property
    def chunk_size(self) -> int:
        """Processing chunk size in samples."""
        return self._chunk_size

    @property
    def overlap(self) -> int:
        """Overlap between chunks in samples."""
        return self._overlap

    @property
    def effects(self) -> list[FX]:
        """The current effect chain."""
        return self._effects
