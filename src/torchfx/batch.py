"""Batched multi-signal processing in a single kernel launch.

Processing many short signals one at a time underuses the GPU: a stereo file is two
threads on a device with thousands of cores, and each file is its own kernel launch.
:func:`batch_process` instead pads the signals to a common length, concatenates them
along the channel dimension into one ``(sum_channels, max_samples)`` tensor, and runs the
effect **once** — a single launch over all channels at high occupancy. The native filter
kernels treat every channel independently and are causal, so the trailing zero-pad never
affects a signal's valid region; the result is numerically identical to processing each
signal separately.

Examples
--------
>>> import torch
>>> import torchfx as fx
>>> waves = [fx.Wave(torch.randn(1, 44100), 44100) for _ in range(64)]
>>> out = fx.batch_process(waves, fx.filter.LoButterworth(4000, order=8))
>>> len(out)
64

"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from torchfx.wave import Wave

__all__ = ["batch_process"]


def _reject_channel_coupled(effect: nn.Module) -> None:
    """Raise if ``effect`` (or any submodule) aggregates across channels.

    Batched processing concatenates independent signals on the channel axis, so an
    effect that couples channels — a global-peak/RMS/percentile ``Normalize`` —
    would mix statistics across signals and silently produce wrong output.
    Per-channel strategies are safe and pass.

    """
    from torchfx.effect import Normalize, PerChannelNormalizationStrategy

    for module in effect.modules():
        if isinstance(module, Normalize) and not isinstance(
            module.strategy, PerChannelNormalizationStrategy
        ):
            raise ValueError(
                f"batch_process cannot apply a channel-coupled effect: "
                f"{type(module).__name__} with {type(module.strategy).__name__} "
                "aggregates across all channels, so batched signals would leak into "
                "each other. Use PerChannelNormalizationStrategy or apply the effect "
                "per signal."
            )


def batch_process(waves: Sequence[Wave], effect: nn.Module) -> list[Wave]:
    """Apply ``effect`` to many :class:`~torchfx.Wave` signals in one batched launch.

    Parameters
    ----------
    waves : Sequence[Wave]
        Signals to process. All must share the same sampling rate and device. Lengths
        and channel counts may differ.
    effect : nn.Module
        The effect or filter chain to apply (e.g. a filter, ``f1 | f2``, or any
        :class:`~torchfx.FX`). It is applied to every signal with the shared ``fs``.

    Returns
    -------
    list[Wave]
        One output ``Wave`` per input, in the same order, each trimmed back to its
        original length and channel count (and carrying its original metadata).

    Raises
    ------
    ValueError
        If ``waves`` is empty, or the signals do not share a single ``fs`` / device.

    Notes
    -----
    Equivalent to ``[w | effect for w in waves]`` but issues a **single** native kernel
    dispatch over the concatenated channels instead of one per signal, which fills the
    GPU (and amortises Python dispatch + OpenMP setup on CPU) when many signals are
    processed together — the CLI batch / watch workloads.

    This holds only for effects that process each channel **independently** and causally
    — filters, ``Gain``, the dynamics effects, per-channel ``Normalize``. Effects that
    aggregate across channels or the whole signal (a global-peak ``Normalize``, mixing)
    would see all the batched signals at once and are **not** batch-safe; apply those per
    signal.

    Examples
    --------
    >>> import torch
    >>> import torchfx as fx
    >>> waves = [fx.Wave(torch.randn(2, 24000), 48000) for _ in range(8)]
    >>> out = fx.batch_process(waves, fx.effect.Gain(0.5))
    >>> [w.ys.shape for w in out][0]
    torch.Size([2, 24000])

    """
    if len(waves) == 0:
        raise ValueError("batch_process requires at least one Wave.")

    _reject_channel_coupled(effect)

    fs = waves[0].fs
    device = waves[0].ys.device
    dtype = waves[0].ys.dtype
    for i, w in enumerate(waves):
        if w.fs != fs:
            raise ValueError(
                f"All waves must share one sampling rate; wave[0].fs={fs} but "
                f"wave[{i}].fs={w.fs}. Resample first or group by fs."
            )
        if w.ys.device != device:
            raise ValueError(
                f"All waves must be on the same device; wave[0] on {device} but "
                f"wave[{i}] on {w.ys.device}."
            )

    lengths = [w.ys.shape[1] for w in waves]
    channels = [w.ys.shape[0] for w in waves]
    max_len = max(lengths)

    # Pad each signal's tail with zeros to the common length, then stack on channels.
    padded = [
        (
            w.ys
            if w.ys.shape[1] == max_len
            else torch.nn.functional.pad(w.ys, (0, max_len - w.ys.shape[1]))
        )
        for w in waves
    ]
    stacked = torch.cat(padded, dim=0).to(dtype=dtype)

    # One pipeline evaluation over all channels => a single native dispatch.
    processed = (Wave(stacked, fs, device=device) | effect).ys

    results: list[Wave] = []
    offset = 0
    for c, length, w in zip(channels, lengths, waves, strict=True):
        segment = processed[offset : offset + c, :length]
        results.append(Wave(segment, fs, device=device, metadata=w.metadata))
        offset += c
    return results
