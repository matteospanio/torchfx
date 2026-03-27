"""Wave class for discrete-time audio signal processing.

This module provides the Wave class, the core data container for audio signals in
torchfx. The Wave class encapsulates a PyTorch tensor representing multi-channel audio
samples along with its sampling frequency, enabling PyTorch-compatible audio processing
with support for GPU acceleration and functional transformations.

Tensor Shape Convention
-----------------------
Audio signals are represented as 2D tensors with shape (channels, samples):
- Mono audio: (1, N) where N is the sample count
- Stereo audio: (2, N)
- Multi-channel: (C, N) where C is the channel count

This convention is consistent with torchaudio and PyTorch audio processing standards.

Typical Usage Pattern
---------------------
The Wave class supports a functional pipeline approach for audio processing:

    >>> import torch
    >>> from torchfx import Wave
    >>> from torchfx.filter import iir
    >>>
    >>> # Load audio file
    >>> wave = Wave.from_file("input.wav")
    >>>
    >>> # Move to GPU if available
    >>> wave = wave.to("cuda" if torch.cuda.is_available() else "cpu")
    >>>
    >>> # Apply effects using pipeline operator
    >>> result = (wave
    ...     | iir.LoButterworth(100, order=2)
    ...     | iir.HiButterworth(2000, order=2))
    >>>
    >>> # Save result
    >>> result.save("output.wav")

See Also
--------
torchfx.effect.FX : Base class for audio effects and filters
torchfx.filter.AbstractFilter : Base class for digital filters

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
    """A discrete-time waveform representing multi-channel audio signals.

    The Wave class is the fundamental data structure in torchfx, wrapping a PyTorch
    tensor containing audio samples along with its sampling frequency. It provides
    device management (CPU/CUDA), file I/O integration with torchaudio, functional
    transformations, and a pipeline operator for chaining effects and filters.

    Attributes
    ----------
    ys : Tensor
        Audio signal tensor with shape (channels, samples). Each row represents a
        channel, and each column represents a time sample.
    fs : int
        Sampling frequency in Hz (samples per second). Used by filters and effects
        to compute time-domain parameters.
    metadata : dict
        Optional metadata dictionary containing audio file information such as
        num_frames, num_channels, bits_per_sample, and encoding.
    device : Device
        Current device location ("cpu" or "cuda") where the tensor resides.
        Read-only property; use the setter or to() method to move devices.

    Examples
    --------
    Create a Wave from an array:

    >>> import torch
    >>> from torchfx import Wave
    >>> samples = torch.randn(1, 44100)  # 1 second of mono audio at 44.1kHz
    >>> wave = Wave(samples, fs=44100)

    Load from an audio file:

    >>> wave = Wave.from_file("audio.wav")
    >>> print(f"Channels: {wave.channels()}, Duration: {wave.duration('sec')}s")

    Process on GPU with pipeline operator:

    >>> from torchfx.filter import iir
    >>> wave = Wave.from_file("input.wav").to("cuda")
    >>> result = wave | iir.LoButterworth(1000, order=4)

    Multi-channel processing:

    >>> stereo = Wave.from_file("stereo.wav")
    >>> left = stereo.get_channel(0)
    >>> right = stereo.get_channel(1)
    >>> # Process channels independently
    >>> processed_left = left | some_effect
    >>> processed_right = right | other_effect
    >>> result = Wave.merge([processed_left, processed_right], split_channels=True)

    Notes
    -----
    The Wave class follows an immutability pattern for most operations. Methods like
    transform(), __or__(), and get_channel() return new Wave objects rather than
    modifying in place. Only device management methods (to(), device setter) modify
    the Wave in place while returning self for method chaining.

    The pipe operator (|) automatically configures FX modules by setting the sampling
    frequency if not already set and computing filter coefficients before first use.
    This eliminates boilerplate and prevents common configuration errors.

    See Also
    --------
    from_file : Load a Wave from an audio file
    merge : Combine multiple Wave objects into one
    transform : Apply functional transformations to the signal
    to : Move the Wave to a different device

    """

    ys: Tensor
    fs: int
    __device: Device  # private field
    metadata: dict[str, tp.Any]

    def __init__(
        self,
        ys: ArrayLike,
        fs: int,
        device: Device = "cpu",
        metadata: dict[str, tp.Any] = {},  # noqa: B006
    ) -> None:
        """Initialize a Wave object from array-like audio data.

        Parameters
        ----------
        ys : ArrayLike
            Audio signal data as an array-like object (NumPy array, list, or PyTorch
            tensor) with shape (channels, samples). Will be converted to a PyTorch
            Tensor if not already.
        fs : int
            Sampling frequency in Hz (samples per second).
        device : {"cpu", "cuda"}, optional
            Target device for the audio tensor. Default is "cpu".
        metadata : dict, optional
            Optional metadata dictionary for storing audio file information.
            Default is an empty dict.

        Examples
        --------
        Create from a NumPy array:

        >>> import numpy as np
        >>> from torchfx import Wave
        >>> audio_data = np.random.randn(2, 44100)  # 1 second stereo at 44.1kHz
        >>> wave = Wave(audio_data, fs=44100)

        Create on GPU:

        >>> import torch
        >>> audio_data = torch.randn(1, 48000)  # 1 second mono at 48kHz
        >>> wave = Wave(audio_data, fs=48000, device="cuda")

        Create with metadata:

        >>> wave = Wave(audio_data, fs=44100, metadata={"source": "microphone"})

        Notes
        -----
        The constructor automatically converts the input data to a PyTorch Tensor
        and immediately moves it to the specified device.

        See Also
        --------
        from_file : Alternative constructor that loads audio from a file
        to : Method to move the Wave to a different device after creation

        """
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
        """Move the Wave object to a specific device (CPU or CUDA).

        This method transfers the internal audio tensor to the specified device and
        returns self to enable method chaining. The device transfer uses PyTorch's
        standard tensor movement mechanism.

        Parameters
        ----------
        device : {"cpu", "cuda"}
            Target device to move the Wave object to.

        Returns
        -------
        Wave
            Returns self for method chaining support.

        Examples
        --------
        Move to GPU:

        >>> wave = Wave.from_file("audio.wav")
        >>> wave.to("cuda")

        Conditional device selection:

        >>> import torch
        >>> device = "cuda" if torch.cuda.is_available() else "cpu"
        >>> wave = Wave.from_file("audio.wav").to(device)

        Method chaining pattern:

        >>> from torchfx.filter import iir
        >>> result = (Wave.from_file("input.wav")
        ...     .to("cuda")
        ...     | iir.LoButterworth(1000))

        Process on GPU, then move back to CPU:

        >>> wave = Wave.from_file("audio.wav")
        >>> result = (wave.to("cuda") | some_filter).to("cpu")

        Notes
        -----
        Unlike most Wave methods, to() modifies the Wave in place but returns self
        to support fluent interface patterns. Effects and filters applied via the
        pipeline operator will automatically inherit the Wave's device location.

        See Also
        --------
        device : Property for getting or setting the current device
        save : Method that automatically moves to CPU before saving

        """
        self.__device = device
        self.ys = self.ys.to(device)
        return self

    def transform(self, func: Callable[..., Tensor], *args, **kwargs) -> "Wave":  # type: ignore
        """Apply a functional transformation to the audio signal tensor.

        This method applies an arbitrary function to the audio tensor, creating a new
        Wave object with the transformed signal while preserving the sampling frequency.
        The original Wave object remains unchanged, following an immutability pattern.

        Parameters
        ----------
        func : Callable[..., Tensor]
            Function that takes a tensor as its first argument and returns a tensor.
            Can be any PyTorch or torchaudio function, or a custom callable.
        *args
            Additional positional arguments passed to func after the tensor.
        **kwargs
            Additional keyword arguments passed to func.

        Returns
        -------
        Wave
            New Wave object with the transformed signal and the same sampling frequency.

        Examples
        --------
        Apply FFT transformation:

        >>> import torch
        >>> wave = Wave.from_file("audio.wav")
        >>> freq_domain = wave.transform(torch.fft.fft)

        Apply normalization:

        >>> def normalize_peak(tensor):
        ...     return tensor / tensor.abs().max()
        >>> normalized = wave.transform(normalize_peak)

        Apply torchaudio transforms with parameters:

        >>> import torchaudio.transforms as T
        >>> # Resample to 16kHz (requires passing sample rate)
        >>> resampled = wave.transform(
        ...     T.Resample(wave.fs, 16000).forward
        ... )

        Apply custom function with arguments:

        >>> def add_noise(tensor, noise_level=0.01):
        ...     noise = torch.randn_like(tensor) * noise_level
        ...     return tensor + noise
        >>> noisy = wave.transform(add_noise, noise_level=0.05)

        Chain transformations:

        >>> wave = Wave.from_file("audio.wav")
        >>> result = (wave
        ...     .transform(lambda x: x / x.abs().max())  # Normalize
        ...     .transform(torch.fft.fft)                # FFT
        ...     .transform(torch.fft.ifft)               # IFFT
        ...     .transform(torch.real))                  # Extract real part

        Notes
        -----
        The transform method creates a new Wave object rather than modifying in place,
        supporting functional programming patterns. The original Wave is unchanged.

        The function must accept a tensor as its first argument and return a tensor.
        The returned tensor shape should maintain the (channels, samples) convention,
        though the number of samples may change.

        See Also
        --------
        __or__ : Pipeline operator for applying nn.Module effects and filters
        to : Move the Wave to a different device before transformation

        """
        return Wave(func(self.ys, *args, **kwargs), self.fs)

    @classmethod
    def from_file(cls, path: str | Path, *args, **kwargs) -> "Wave":  # type: ignore
        """Load a Wave object from an audio file.

        This classmethod uses torchaudio.load to read audio files, automatically
        detecting the format and extracting metadata. Supported formats include WAV,
        MP3, FLAC, OGG, and others depending on the available torchaudio backend.

        Parameters
        ----------
        path : str or Path
            Path to the audio file to load. Can be a string or pathlib.Path object.
        *args
            Additional positional arguments passed to torchaudio.load.
        **kwargs
            Additional keyword arguments passed to torchaudio.load. Common options
            include frame_offset, num_frames, normalize, channels_first, and format.

        Returns
        -------
        Wave
            New Wave object containing the loaded audio data, sampling frequency,
            and extracted metadata.

        Examples
        --------
        Load a WAV file:

        >>> wave = Wave.from_file("audio.wav")
        >>> print(f"Loaded {wave.channels()} channels at {wave.fs}Hz")

        Load a specific portion of a file:

        >>> # Load 1 second starting at 2 seconds
        >>> wave = Wave.from_file("long_audio.wav", frame_offset=88200, num_frames=44100)

        Load and check metadata:

        >>> wave = Wave.from_file("audio.flac")
        >>> print(wave.metadata)
        {'num_frames': 220500, 'num_channels': 2, 'bits_per_sample': 16, ...}

        Load different formats:

        >>> wav_wave = Wave.from_file("audio.wav")
        >>> mp3_wave = Wave.from_file("audio.mp3")
        >>> flac_wave = Wave.from_file("audio.flac")

        Load with normalization:

        >>> # Normalize to [-1, 1] range
        >>> wave = Wave.from_file("audio.wav", normalize=True)

        Notes
        -----
        The method automatically extracts metadata including num_frames, num_channels,
        bits_per_sample, and encoding when available. If metadata extraction fails,
        an empty metadata dictionary is used instead.

        The loaded audio tensor will be on CPU by default. Use the to() method or
        device parameter in subsequent processing to move to GPU:

        >>> wave = Wave.from_file("audio.wav").to("cuda")

        Format support depends on the torchaudio backend (SoX or FFmpeg). Check
        torchaudio documentation for your installation's supported formats.

        See Also
        --------
        __init__ : Direct constructor for creating Wave from array data
        save : Save a Wave object to an audio file
        torchaudio.load : Underlying function used for loading

        """
        data, fs = torchaudio.load(path, *args, **kwargs)

        # Extract metadata from the file using soundfile (torchaudio.info
        # was removed in torchaudio 2.10+).
        try:
            import soundfile as _sf  # type: ignore[import-untyped]

            info = _sf.info(str(path))
            metadata = {
                "num_frames": info.frames,
                "num_channels": info.channels,
                "subtype": info.subtype,
                "format": info.format,
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
    ) -> None:
        """Save the wave to an audio file.

        Parameters
        ----------
        path : str or Path
            The path where to save the audio file.
        format : str, optional
            Override the audio format. If not specified, the format is inferred
            from the file extension. Valid values include: "wav", "flac".
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
        import soundfile as _sf

        output_path = Path(path)

        # Create parent directories if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Move tensor to CPU for saving
        audio_data = self.ys.cpu()

        # Determine format from extension if not provided
        if format is None:
            ext = output_path.suffix.lower()
            format_map = {".wav": "WAV", ".flac": "FLAC", ".ogg": "OGG"}
            format = format_map.get(ext, "WAV")  # noqa: A001

        # Map encoding/bits_per_sample to soundfile subtype
        subtype: str | None = None
        if encoding is not None and bits_per_sample is not None:
            subtype = f"{encoding.replace('PCM_', 'PCM_')}{bits_per_sample}"
            # Normalise common patterns: "PCM_S16" -> "PCM_16", "PCM_F32" -> "FLOAT"
            if encoding == "PCM_S":
                subtype = f"PCM_{bits_per_sample}"
            elif encoding == "PCM_U":
                subtype = "PCM_U8" if bits_per_sample == 8 else f"PCM_{bits_per_sample}"
            elif encoding == "PCM_F":
                subtype = "FLOAT" if bits_per_sample == 32 else "DOUBLE"
        elif bits_per_sample is not None:
            subtype = f"PCM_{bits_per_sample}"
        elif encoding is not None and encoding == "PCM_F":
            subtype = "FLOAT"

        # Save using soundfile (channels, samples) -> (samples, channels)
        _sf.write(
            str(output_path),
            audio_data.numpy().T,
            self.fs,
            format=format,
            subtype=subtype,
        )

    def __or__(self, f: nn.Module) -> "Wave":
        """Apply effects and filters using the pipeline operator (|).

        The pipe operator provides a clean, readable syntax for chaining audio effects
        and filters. It automatically configures FX modules by setting their sampling
        frequency and computing filter coefficients before application, eliminating
        boilerplate and preventing configuration errors.

        Parameters
        ----------
        f : nn.Module
            PyTorch module to apply to the wave. Typically an FX subclass (effect or
            filter) or nn.Sequential containing multiple effects.

        Returns
        -------
        Wave
            New Wave object with the effect/filter applied.

        Raises
        ------
        TypeError
            If f is not an instance of nn.Module.

        Examples
        --------
        Apply a single filter:

        >>> from torchfx import Wave
        >>> from torchfx.filter import iir
        >>> wave = Wave.from_file("audio.wav")
        >>> filtered = wave | iir.LoButterworth(cutoff=1000, order=4)

        Chain multiple filters:

        >>> result = (wave
        ...     | iir.HiButterworth(100, order=2)
        ...     | iir.LoButterworth(5000, order=2))

        Use with nn.Sequential:

        >>> import torch.nn as nn
        >>> from torchfx.filter import iir
        >>> chain = nn.Sequential(
        ...     iir.HiButterworth(100),
        ...     iir.LoButterworth(5000)
        ... )
        >>> result = wave | chain

        Combine with other operations:

        >>> from torchfx.effect import transform as T
        >>> result = (Wave.from_file("input.wav")
        ...     .to("cuda")
        ...     | iir.LoButterworth(1000)
        ...     | T.Vol(0.5)
        ...     .to("cpu"))

        Parallel filter combination (using + operator from FX):

        >>> from torchfx.filter import iir
        >>> # Apply two filters in parallel and sum results
        >>> result = wave | (iir.HiButterworth(2000) + iir.HiChebyshev1(2000))

        Notes
        -----
        The pipeline operator performs automatic configuration in the following order:

        1. **Type Validation**: Ensures f is an nn.Module instance
        2. **FX Configuration**: For FX instances:
           - Sets f.fs to wave.fs if f.fs is None
           - Computes filter coefficients for AbstractFilter instances
        3. **Sequential Handling**: For nn.Sequential or nn.ModuleList:
           - Recursively configures each contained FX module
        4. **Application**: Calls f.forward() via transform() method

        This automatic configuration means you can write:

        >>> wave | iir.LoButterworth(1000)

        Instead of the more verbose:

        >>> filter = iir.LoButterworth(1000)
        >>> filter.fs = wave.fs
        >>> filter.compute_coefficients()
        >>> wave.transform(filter.forward)

        The module's forward method must accept a tensor and return a tensor. The
        returned tensor shape should maintain the (channels, samples) convention.

        See Also
        --------
        transform : Apply arbitrary functions to the signal
        to : Move Wave to different device before applying effects
        FX : Base class for effects and filters
        AbstractFilter : Base class for digital filters

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
        """Return the number of audio channels in the wave.

        Returns
        -------
        int
            Number of channels. Returns 1 for mono, 2 for stereo, or higher values
            for multi-channel audio.

        Examples
        --------
        Check channel count:

        >>> wave = Wave.from_file("audio.wav")
        >>> print(f"This file has {wave.channels()} channel(s)")

        Process based on channel count:

        >>> wave = Wave.from_file("audio.wav")
        >>> if wave.channels() == 1:
        ...     print("Mono audio")
        ... elif wave.channels() == 2:
        ...     print("Stereo audio")
        ... else:
        ...     print(f"Multi-channel audio with {wave.channels()} channels")

        Extract all channels:

        >>> wave = Wave.from_file("multichannel.wav")
        >>> channel_list = [wave.get_channel(i) for i in range(wave.channels())]

        Notes
        -----
        This method returns the first dimension of the ys tensor shape. The tensor
        follows the convention (channels, samples), so channels() returns ys.shape[0].

        See Also
        --------
        get_channel : Extract a specific channel as a new Wave object
        merge : Combine multiple Wave objects with different merge strategies

        """
        return self.ys.shape[0]

    def get_channel(self, index: int) -> "Wave":
        """Extract a specific channel as a new Wave object.

        This method creates a new Wave containing only the specified channel from
        the original multi-channel audio. The new Wave has the same sampling frequency
        and can be processed independently.

        Parameters
        ----------
        index : int
            Zero-based index of the channel to extract. For stereo audio, use 0 for
            left channel and 1 for right channel.

        Returns
        -------
        Wave
            New Wave object containing only the specified channel with shape (1, samples).

        Examples
        --------
        Extract left and right channels from stereo:

        >>> stereo = Wave.from_file("stereo.wav")
        >>> left_channel = stereo.get_channel(0)
        >>> right_channel = stereo.get_channel(1)

        Process channels independently:

        >>> from torchfx.filter import iir
        >>> stereo = Wave.from_file("stereo.wav")
        >>> # Apply different filters to each channel
        >>> left = stereo.get_channel(0) | iir.LoButterworth(1000)
        >>> right = stereo.get_channel(1) | iir.HiButterworth(1000)
        >>> # Merge back to stereo
        >>> result = Wave.merge([left, right], split_channels=True)

        Extract all channels as a list:

        >>> wave = Wave.from_file("multichannel.wav")
        >>> channels = [wave.get_channel(i) for i in range(wave.channels())]

        Process mono from stereo by averaging:

        >>> stereo = Wave.from_file("stereo.wav")
        >>> left = stereo.get_channel(0)
        >>> right = stereo.get_channel(1)
        >>> # Mix to mono (this uses merge to sum channels)
        >>> mono = Wave.merge([left, right], split_channels=False)

        Notes
        -----
        The returned Wave object is independent of the original. Modifications to
        the returned Wave do not affect the original multi-channel Wave.

        The index must be within the valid range [0, channels()-1]. Python's standard
        indexing rules apply, so negative indices are supported (e.g., -1 for the
        last channel).

        See Also
        --------
        channels : Get the total number of channels
        merge : Combine multiple Wave objects back into multi-channel audio

        """
        return Wave(self.ys[index], self.fs)

    def duration(self, unit: tp.Literal["sec", "ms"]) -> Second | Millisecond:
        """Calculate the duration of the audio signal.

        Computes the time length of the audio based on the number of samples and
        the sampling frequency. The duration can be returned in either seconds or
        milliseconds.

        Parameters
        ----------
        unit : {"sec", "ms"}
            Unit for the returned duration. Use "sec" for seconds or "ms" for
            milliseconds.

        Returns
        -------
        float
            Duration in the specified time unit. The return type is annotated as
            Second (float) when unit="sec" or Millisecond (float) when unit="ms".

        Examples
        --------
        Get duration in seconds:

        >>> wave = Wave.from_file("audio.wav")
        >>> duration_sec = wave.duration("sec")
        >>> print(f"Duration: {duration_sec:.2f} seconds")

        Get duration in milliseconds:

        >>> wave = Wave.from_file("audio.wav")
        >>> duration_ms = wave.duration("ms")
        >>> print(f"Duration: {duration_ms:.0f} ms")

        Compare durations:

        >>> wave1 = Wave.from_file("short.wav")
        >>> wave2 = Wave.from_file("long.wav")
        >>> if wave1.duration("sec") < wave2.duration("sec"):
        ...     print("wave1 is shorter")

        Calculate processing time estimate:

        >>> wave = Wave.from_file("audio.wav")
        >>> duration = wave.duration("sec")
        >>> # Estimate processing time (example: 10x realtime)
        >>> estimated_time = duration * 10
        >>> print(f"Estimated processing: {estimated_time:.2f} seconds")

        Use in validation:

        >>> wave = Wave.from_file("audio.wav")
        >>> max_duration_sec = 60.0
        >>> if wave.duration("sec") > max_duration_sec:
        ...     print("Audio file is too long")

        Notes
        -----
        The duration is calculated using the formula:

            duration = (number_of_samples / sampling_frequency) * multiplier

        Where:
        - number_of_samples = len(self) = self.ys.shape[1]
        - sampling_frequency = self.fs
        - multiplier = 1000 for milliseconds, 1 for seconds

        For example, a Wave with 44100 samples at 44100 Hz has a duration of:
        - 1.0 second (44100 / 44100 * 1)
        - 1000.0 milliseconds (44100 / 44100 * 1000)

        The duration is independent of the number of channels; it represents the
        time length of the audio signal.

        See Also
        --------
        __len__ : Get the number of samples in the wave
        channels : Get the number of channels

        """
        return len(self) / self.fs * (1000 if unit == "ms" else 1)

    @classmethod
    def merge(cls, waves: tp.Sequence["Wave"], split_channels: bool = False) -> "Wave":
        """Combine multiple Wave objects into a single Wave.

        This classmethod provides two merging strategies: mixing (summing waveforms
        element-wise) or channel concatenation (preserving each wave as separate
        channels). All waves must have the same sampling frequency.

        Parameters
        ----------
        waves : Sequence[Wave]
            Sequence of Wave objects to merge. Must contain at least one wave.
        split_channels : bool, optional
            Determines the merge strategy:
            - False (default): Mix waves by summing them element-wise
            - True: Concatenate waves along the channel dimension

        Returns
        -------
        Wave
            New Wave object containing the merged audio with the same sampling
            frequency as the input waves.

        Raises
        ------
        ValueError
            If no waves are provided or if waves have different sampling frequencies.

        Examples
        --------
        Mix two mono waves into one (sum strategy):

        >>> wave1 = Wave.from_file("voice.wav")
        >>> wave2 = Wave.from_file("music.wav")
        >>> mixed = Wave.merge([wave1, wave2], split_channels=False)
        >>> # Result: mono wave with voice and music mixed together

        Combine two mono waves into stereo (concatenate strategy):

        >>> left = Wave.from_file("left.wav")
        >>> right = Wave.from_file("right.wav")
        >>> stereo = Wave.merge([left, right], split_channels=True)
        >>> # Result: stereo wave with left and right channels

        Merge after independent processing:

        >>> from torchfx.filter import iir
        >>> stereo = Wave.from_file("stereo.wav")
        >>> left = stereo.get_channel(0) | iir.LoButterworth(1000)
        >>> right = stereo.get_channel(1) | iir.HiButterworth(1000)
        >>> result = Wave.merge([left, right], split_channels=True)

        Mix multiple sources with different effects:

        >>> wave1 = Wave.from_file("track1.wav") | effect1
        >>> wave2 = Wave.from_file("track2.wav") | effect2
        >>> wave3 = Wave.from_file("track3.wav") | effect3
        >>> final_mix = Wave.merge([wave1, wave2, wave3], split_channels=False)

        Create multi-channel output from mono sources:

        >>> channels = [Wave.from_file(f"channel_{i}.wav") for i in range(8)]
        >>> multichannel = Wave.merge(channels, split_channels=True)
        >>> print(f"Created {multichannel.channels()} channel audio")

        Notes
        -----
        **Merge Strategy Comparison:**

        When split_channels=False (mixing):
        - Waves are summed element-wise
        - If waves have different lengths, shorter ones are zero-padded
        - Result has same number of channels as input waves
        - Use for: audio mixing, layering multiple sounds

        When split_channels=True (concatenation):
        - Waves are concatenated along channel dimension
        - Each input wave becomes a separate channel in output
        - Result channel count = sum of all input channel counts
        - Use for: creating stereo/multichannel from mono sources

        **Length Handling:**

        When merging waves of different lengths:
        - The output length is the maximum length among all input waves
        - Shorter waves are zero-padded to match the longest wave

        **Validation:**

        All input waves must have identical sampling frequencies. This is enforced
        because merging waves with different sampling rates would be technically
        invalid.

        **Device Compatibility:**

        All waves should be on the same device. The merged Wave will be on the
        device of the first input wave.

        See Also
        --------
        get_channel : Extract individual channels from multi-channel audio
        channels : Get the number of channels in a Wave
        __init__ : Constructor for creating Wave from array data

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
