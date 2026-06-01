"""Fused SOS cascade for merging multiple IIR filters into a single kernel call."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from torchfx.filter.iir import IIR, _sos_cascade_forward

if TYPE_CHECKING:
    from torch import Tensor

    from torchfx.filter.biquad import Biquad
    from torchfx.typing import Device


class FusedSOSCascade(nn.Module):
    """A fused second-order-sections cascade combining multiple IIR filters.

    Merges the SOS matrices of several IIR filters into a single ``[K_total, 6]``
    tensor and processes them in one native ``sos_forward`` call, eliminating
    per-filter Python dispatch overhead.

    Parameters
    ----------
    filters : IIR
        One or more IIR filter instances to fuse.  Coefficients are computed
        eagerly (``compute_coefficients`` is called at construction time).

    Examples
    --------
    >>> chain = HiButterworth(1000, order=2, fs=44100) | LoButterworth(5000, order=2, fs=44100)
    >>> fused = FusedSOSCascade.from_chain(chain)
    >>> y = fused(x)

    """

    def __init__(self, *filters: IIR | Biquad, gain: float = 1.0) -> None:
        super().__init__()

        if not filters:
            raise ValueError("FusedSOSCascade requires at least one IIR filter")

        sos_parts: list[Tensor] = []
        fs_val: int | None = None

        for f in filters:
            if not hasattr(f, "_sos"):
                raise TypeError(f"Expected filter with SOS coefficients, got {type(f).__name__}")

            # Ensure coefficients are computed.
            if f._sos is None:
                if f.fs is None:
                    raise ValueError(
                        f"Filter {type(f).__name__} has no sampling frequency set. "
                        "Set fs before fusing."
                    )
                f.compute_coefficients()

            assert f._sos is not None
            sos_parts.append(f._sos)

            # Validate consistent sampling frequency.
            if f.fs is not None:
                if fs_val is None:
                    fs_val = f.fs
                elif f.fs != fs_val:
                    raise ValueError(
                        f"Cannot fuse filters with different sample rates: {fs_val} vs {f.fs}"
                    )

        # Concatenate all SOS sections: [K_total, 6]. torch.cat copies, so the
        # source filters' coefficients are never mutated by the gain fold below.
        self._sos: Tensor = torch.cat(sos_parts, dim=0).to(dtype=torch.float64)
        # Fold a static scalar gain (e.g. a `Gain` between fused filters) into the
        # LAST section's numerator. A scalar commutes through the linear cascade, so
        # this is exact; folding into the last section scales only the final output,
        # avoiding any intermediate over/underflow the gain might otherwise cause.
        if gain != 1.0:
            self._sos[-1, :3] *= gain
        self._num_sections: int = self._sos.shape[0]
        self.fs: int | None = fs_val

        # Cached device-matched copy — avoids per-forward .to() calls.
        self._sos_device_cache: Tensor | None = None

        # State for stateful processing (initialized lazily).
        self._state_x: Tensor | None = None
        self._state_y: Tensor | None = None
        self._stateful: bool = False

    @classmethod
    def from_chain(cls, chain: nn.Sequential | nn.Module) -> FusedSOSCascade:
        """Create a fused cascade from an ``nn.Sequential`` or pipe chain.

        Walks the chain and collects all children that have SOS coefficients (IIR
        filters and biquad filters).

        """
        from torchfx.filter.biquad import Biquad

        if isinstance(chain, nn.Sequential):
            filters = [m for m in chain if isinstance(m, (IIR, Biquad))]
        elif isinstance(chain, (IIR, Biquad)):
            filters = [chain]
        else:
            raise TypeError(f"Expected nn.Sequential or IIR/Biquad, got {type(chain).__name__}")

        if not filters:
            raise ValueError("No IIR/Biquad filters found in chain to fuse")

        return cls(*filters)

    def move_coeff(self, device: Device) -> None:
        """Move the SOS matrix to the specified device."""
        self._sos = self._sos.to(device=device, dtype=torch.float64)

    def reset_state(self) -> None:
        """Clear accumulated state."""
        self._state_x = None
        self._state_y = None
        self._stateful = False
        self._sos_device_cache = None

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """Apply the fused SOS cascade.

        On the first call, bootstraps state and enters stateful mode. Subsequent calls
        carry state across chunks.

        """
        result, self._sos_device_cache, self._state_x, self._state_y = _sos_cascade_forward(
            x, self._sos, self._sos_device_cache, self._state_x, self._state_y
        )
        self._stateful = True
        return result
