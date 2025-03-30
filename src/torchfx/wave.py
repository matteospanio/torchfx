"""Module containing the Wave class."""

import typing as tp
from collections.abc import Callable
from pathlib import Path

import torchaudio
from numpy.typing import ArrayLike
from torch import Tensor, nn
from typing_extensions import Self

from torchfx.filter import iir
from torchfx.typing import Device


class Wave:
    """A discrete time waveform.

    Attributes
    ----------
    ys : Tensor
        The signal.
    fs : int
        The sampling frequency.

    """

    ys: Tensor
    fs: int
    __device: Device  # private field

    def __init__(self, ys: ArrayLike, fs: int, device: Device = "cpu") -> None:
        self.fs = fs
        self.ys = Tensor(ys)
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
        self.ys.to(device)
        return self

    def transform(self, func: Callable[..., Tensor], *args, **kwargs) -> "Wave":
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
    def from_file(cls, path: str | Path, *args, **kwargs) -> "Wave":
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
            The wave object.

        Examples
        --------
        >>> wave = Wave.from_file("path/to/file.wav")

        """
        data, fs = torchaudio.load(path, *args, **kwargs)
        return cls(data, fs)

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
        >>> wave | iir.HighPass(1000) | iir.LowPass(2000)
        >>> wave | nn.Sequential(iir.HighPass(1000), iir.LowPass(2000))

        Notes
        -----
        The module must have a `forward` method that takes a tensor as input and
        returns a tensor as output. The module must also have a `fs` attribute that
        specifies the sampling frequency of the filter. If the module is a sequential
        module, the `fs` attribute will be set for each module in the sequence.
        The `fs` attribute of the wave object will be set to the `fs` attribute of the
        module.
        """
        if isinstance(f, iir.IIR):
            f.fs = self.fs
        if isinstance(f, nn.Sequential):
            for a in f:
                if isinstance(a, iir.IIR):
                    a.fs = self.fs
        return self.transform(f.forward)

    def __len__(self) -> int:
        """Return the length, in samples, of the wave."""
        return len(self.ys)

    def channels(self) -> int:
        """Return the number of channels of the wave."""
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

    def duration(self, unit: tp.Literal["sec", "ms"]) -> float:
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
