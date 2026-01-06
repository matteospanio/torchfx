"""Module containing the Wave class.

Digital signal is usually represented as a 2D tensor, where the first dimension is the
channel and the second dimension is the time. Of course the signal is discrete, so the
time is represented as a sequence of samples. The sampling frequency is the number of
samples per second. The sampling frequency is usually represented as `fs`.
The Wave class is a wrapper around the 2D tensor that represents the signal. It
provides methods to manipulate the signal, such as applying filters, transforming the
signal, and saving the signal to a file.

"""

import typing as tp
from collections.abc import Callable
from pathlib import Path

import torch
import torchaudio
from numpy.typing import ArrayLike
from torch import Tensor, nn
from typing_extensions import Self

from torchfx.effect import FX
from torchfx.filter.__base import AbstractFilter
from torchfx.typing import BitRate, Device, Millisecond, Second


class Wave:
    """A discrete time waveform.

    Attributes
    ----------
    ys : Tensor
        The signal.
    fs : int
        The sampling frequency.
    metadata : dict, optional
        Optional metadata dictionary for the audio file.

    """

    ys: Tensor
    fs: int
    __device: Device  # private field
    metadata: dict[str, tp.Any] | None

    def __init__(
        self,
        ys: ArrayLike,
        fs: int,
        device: Device = "cpu",
        metadata: dict[str, tp.Any] | None = None,
    ) -> None:
        self.fs = fs
        self.ys = Tensor(ys)
        self.metadata = metadata or {}
        self.to(device)

    @property
    def device(self) -> Device:
        """Print the device where is located this object, if there's an assignment move
        the object to that device.

        See Also
        --------
        Wave.to

        """  # noqa: D205
        return self.__device

    @device.setter
    def device(self, device: Device) -> None:
        self.to(device)

    def to(self, device: Device) -> Self:
        """Move the wave object to a specific device (`cpu` or `cuda`).

        Parameters
        ----------
        device : {"cpu", "cuda"}
            The device to move the wave object to.

        Returns
        -------
        Wave
            The wave object.

        Examples
        --------
        >>> wave = Wave.from_file("path/to/file.wav")
        >>> wave.to("cuda")

        """
        self.__device = device
        self.ys = self.ys.to(device)
        return self

    def transform(self, func: Callable[..., Tensor], *args, **kwargs) -> "Wave":  # type: ignore
        """Apply a functional transformation to the signal.

        Parameters
        ----------
        func : Callable[..., Tensor]
            The function to apply to the signal.

        Returns
        -------
        Wave
            A new wave object with the transformed signal.

        Examples
        --------
        >>> wave = Wave.from_file("path/to/file.wav")
        >>> wave.transform(torch.fft.fft)

        """
        return Wave(func(self.ys, *args, **kwargs), self.fs)

    @classmethod
    def from_file(cls, path: str | Path, *args, **kwargs) -> "Wave":  # type: ignore
        """Instantiate a wave from an audio file.

        Parameters
        ----------
        path : str or Path
            The path to the audio file.
        *args
            Additional arguments to pass to `torchaudio.load`.
        **kwargs
            Additional keyword arguments to pass to `torchaudio.load`.

        Returns
        -------
        Wave
            The wave object with audio data and metadata.

        Examples
        --------
        >>> wave = Wave.from_file("path/to/file.wav")
        >>> print(wave.metadata)  # Access file metadata

        """
        data, fs = torchaudio.load(path, *args, **kwargs)

        # Extract metadata from the file
        try:
            info = torchaudio.info(path)
            metadata = {
                "num_frames": info.num_frames,
                "num_channels": info.num_channels,
                "bits_per_sample": info.bits_per_sample,
                "encoding": info.encoding,
            }
        except Exception:
            # If metadata extraction fails, continue without it
            metadata = {}

        return cls(data, fs, metadata=metadata)

    def save(
        self,
        path: str | Path,
        format: str | None = None,  # noqa: A002
        encoding: str | None = None,
        bits_per_sample: BitRate | None = None,
        **kwargs: tp.Any,
    ) -> None:
        """Save the wave to an audio file.

        Parameters
        ----------
        path : str or Path
            The path where to save the audio file.
        format : str, optional
            Override the audio format. If not specified, the format is inferred
            from the file extension. Valid values include: "wav", "flac", "ogg".
        encoding : str, optional
            Changes the encoding for supported formats (wav, flac).
            Valid values: "PCM_S" (signed int), "PCM_U" (unsigned int),
            "PCM_F" (float), "ULAW", "ALAW".
        bits_per_sample : int, optional
            Changes the bit depth for supported formats.
            Valid values: 8, 16, 24, 32, 64.
        **kwargs
            Additional keyword arguments to pass to `torchaudio.save`.

        Returns
        -------
        None

        Examples
        --------
        Save as WAV file:

        >>> wave = Wave.from_file("input.wav")
        >>> wave.save("output.wav")

        Save as FLAC with specific encoding:

        >>> wave.save("output.flac", encoding="PCM_S", bits_per_sample=24)

        Save with high bit depth:

        >>> wave.save("output.wav", encoding="PCM_F", bits_per_sample=32)

        Notes
        -----
        - The method automatically creates parent directories if they don't exist.
        - The audio data is moved to CPU before saving.
        - Supported formats depend on the available torchaudio backend.

        See Also
        --------
        from_file : Load a wave from an audio file.
        torchaudio.save : Underlying function used for saving.

        """
        # Convert path to Path object for easier manipulation
        output_path = Path(path)

        # Create parent directories if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Move tensor to CPU for saving (torchaudio requires CPU tensors)
        audio_data = self.ys.cpu()

        # Save using torchaudio
        torchaudio.save(
            uri=str(output_path),
            src=audio_data,
            sample_rate=self.fs,
            format=format,
            encoding=encoding,
            bits_per_sample=bits_per_sample,
            **kwargs,
        )

    def __or__(self, f: nn.Module) -> "Wave":
        """Apply a module to the wave through the pipeline operator: `|`.

        Parameters
        ----------
        f : nn.Module
            The module to apply to the wave.

        Returns
        -------
        Wave
            The wave object.

        Examples
        --------
        >>> wave = Wave.from_file("path/to/file.wav")
        >>> wave | iir.HiButterworth(1000) | iir.LoButterworth(2000)
        >>> wave | nn.Sequential(iir.HiButterworth(1000), iir.LoButterworth(2000))

        Notes
        -----
        The module must have a `forward` method that takes a tensor as input and
        returns a tensor as output. The module must also have a `fs` attribute that
        specifies the sampling frequency of the filter. If the module is a sequential
        module, the `fs` attribute will be set for each module in the sequence.
        The `fs` attribute of the wave object will be set to the `fs` attribute of the
        module.
        """
        if not isinstance(f, nn.Module):
            raise TypeError(f"Expected nn.Module, but got {type(f).__name__} instead.")

        if isinstance(f, FX):
            self.__update_config(f)

        elif isinstance(f, (nn.Sequential | nn.ModuleList)):
            for a in f:
                if isinstance(a, FX):
                    self.__update_config(a)

        return self.transform(f.forward)

    def __update_config(self, f: FX) -> None:
        """Update the configuration of the filter with the wave's sampling frequency."""
        if hasattr(f, "fs") and f.fs is None:
            f.fs = self.fs

        if isinstance(f, AbstractFilter) and not f._has_computed_coeff:
            f.compute_coefficients()

    def __len__(self) -> int:
        """Return the length, in samples, of the wave."""
        return self.ys.shape[1]

    def channels(self) -> int:
        """Return the number of channels of the wave.

        Returns
        -------
        int
            The number of channels of the wave.

        """
        return self.ys.shape[0]

    def get_channel(self, index: int) -> "Wave":
        """Return a specific channel of the wave.

        Parameters
        ----------
        index : int
            The index of the channel to return.

        Returns
        -------
        Wave
            The wave object with only the specified channel.

        Examples
        --------
        >>> wave = Wave.from_file("path/to/file.wav")
        >>> wave.get_channel(0)

        """
        return Wave(self.ys[index], self.fs)

    def duration(self, unit: tp.Literal["sec", "ms"]) -> Second | Millisecond:
        """Return the length of the wave in seconds or milliseconds.

        Parameters
        ----------
        unit : {"sec", "ms"}
            The unit of time to return the duration in.

        Returns
        -------
        float
            The duration of the wave in the specified unit of time.

        Examples
        --------
        >>> wave = Wave.from_file("path/to/file.wav")
        >>> wave.duration("sec")
        3.0
        >>> wave.duration("ms")
        3000.0

        """
        return len(self) / self.fs * (1000 if unit == "ms" else 1)

    @classmethod
    def merge(cls, waves: tp.Sequence["Wave"], split_channels: bool = False) -> "Wave":
        """Merge multiple waves into a single wave.

        Parameters
        ----------
        waves : Sequence[Wave]
            The waves to merge.
        split_channels : bool, optional
            If False, the channels of the waves will be merged into a single channel.
            If True, the channels will be merged into multiple channels. Default is False.

        Returns
        -------
        Wave
            The merged wave object.

        Examples
        --------
        >>> wave1 = Wave.from_file("path/to/file1.wav")
        >>> wave2 = Wave.from_file("path/to/file2.wav")
        >>> merged_wave = wave1.merge([wave2])

        """

        if not waves:
            raise ValueError("No waves to merge. Provide at least one wave.")

        fs = waves[0].fs
        for w in waves:
            if w.fs != fs:
                raise ValueError(
                    f"Sampling frequency mismatch: {w.fs} != {fs}. "
                    "All waves must have the same sampling frequency."
                )

        if split_channels:
            ys = torch.cat([w.ys for w in waves], dim=0)
        else:
            # Get the maximum length and number of channels
            max_length = max(len(w) for w in waves)
            num_channels = waves[0].ys.shape[0]

            ys = torch.zeros(
                (num_channels, max_length),
                dtype=waves[0].ys.dtype,
                device=waves[0].device,
            )
            for w in waves:
                # Add each wave, handling different lengths
                ys[:, : len(w)] += w.ys

        return Wave(ys, fs)
