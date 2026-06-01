"""CUDA Graph capture for fixed-shape streaming filter chains (Roadmap Epic C1).

When a stateful SOS cascade is streamed over fixed-size chunks (e.g. a GPU
``StreamProcessor``), every chunk issues the *same* sequence of CUDA kernels — a
``K``-section cascade is ~``4*K`` launches per chunk. For short chunks that launch
overhead dominates: the parallel scan measures ~135 us of pure launch/dispatch
overhead on an RTX 3070 regardless of chunk length. Capturing the per-chunk forward
into a :class:`torch.cuda.CUDAGraph` once and replaying it collapses that to a single
graph launch.

This works because the native SOS kernel updates its DF1 state *in place*
(``sos_forward_cuda``), so the captured graph carries streaming state across replays
through stable buffer addresses — no per-chunk re-issue, no per-chunk allocation.

Example
-------
>>> import torch
>>> from torchfx.filter import HiButterworth, LoButterworth
>>> from torchfx.filter.fused import FusedSOSCascade
>>> from torchfx.realtime.cuda_graph import CudaGraphRunner
>>> chain = FusedSOSCascade(
...     HiButterworth(80, order=2, fs=48000), LoButterworth(8000, order=4, fs=48000)
... )
>>> example = torch.randn(2, 1024, device="cuda")           # doctest: +SKIP
>>> runner = CudaGraphRunner(chain, example)                # doctest: +SKIP
>>> for chunk in chunks:                                    # doctest: +SKIP
...     y = runner.run(chunk).clone()

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import torch.nn as nn


class CudaGraphRunner:
    """Capture a fixed-shape GPU forward into a CUDA graph and replay it.

    Parameters
    ----------
    module : nn.Module
        A GPU filter/cascade whose ``forward`` updates its state in place (the
        native SOS path does). The captured graph reuses the module's state
        buffers, so streaming continuity is preserved across :meth:`run` calls.
    example : Tensor
        A representative input chunk on CUDA. Its shape, dtype, and device fix the
        graph; every :meth:`run` input must match.
    warmup : int, default 3
        Eager iterations on a side stream before capture, so coefficient caches and
        scratch allocations are established (required for stable capture).

    Notes
    -----
    The captured filter must have static coefficients — the SOS taps are baked into
    the graph at capture time. Call :meth:`reset_state` to restart streaming from a
    zero state (e.g. between files). The returned tensor from :meth:`run` is the
    module's *shared* output buffer; clone it before the next :meth:`run`.

    """

    def __init__(self, module: nn.Module, example: torch.Tensor, warmup: int = 3) -> None:
        if not example.is_cuda:
            raise ValueError("CudaGraphRunner requires a CUDA example input.")
        if warmup < 1:
            raise ValueError("warmup must be >= 1 so allocations exist before capture.")

        self.module = module
        self._static_in = example.clone()

        # Warm up on a side stream so the coefficient cache, state buffers, and
        # scratch are allocated before the capture stream records anything.
        side = torch.cuda.Stream()  # type: ignore[no-untyped-call]
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                self.module(self._static_in)
        torch.cuda.current_stream().wait_stream(side)

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._static_out: torch.Tensor = self.module(self._static_in)

    def reset_state(self) -> None:
        """Zero the captured streaming state in place to restart a new stream.

        Does not call ``module.reset_state()`` — that may drop coefficients or
        rebind the state buffers the graph was captured against; instead the
        existing state buffers are zeroed in place.

        Caveat: graph *continuation* is exact, but the very first chunk after a
        reset can show a small initial transient versus a never-run eager filter,
        because the capture warmup is baked into the recorded kernels. For exact
        fresh-stream behaviour, warm up the runner with input representative of the
        stream's start (e.g. silence). The transient decays within a few samples.

        """
        for name in ("_state_x", "_state_y"):
            buf = getattr(self.module, name, None)
            if isinstance(buf, torch.Tensor):
                buf.zero_()

    def run(self, chunk: torch.Tensor) -> torch.Tensor:
        """Process one fixed-shape chunk via graph replay.

        Returns the module's shared output buffer (clone it before the next call).

        """
        if chunk.shape != self._static_in.shape:
            raise ValueError(
                f"chunk shape {tuple(chunk.shape)} does not match captured shape "
                f"{tuple(self._static_in.shape)}"
            )
        self._static_in.copy_(chunk)
        self._graph.replay()
        return self._static_out
