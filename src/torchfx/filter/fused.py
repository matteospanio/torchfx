"""Fused SOS cascade for merging multiple IIR filters into a single kernel call."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from torchfx.filter.iir import IIR

if TYPE_CHECKING:
    from torch import Tensor

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

    def __init__(self, *filters: IIR) -> None:
        super().__init__()

        if not filters:
            raise ValueError("FusedSOSCascade requires at least one IIR filter")

        sos_parts: list[Tensor] = []
        fs_val: int | None = None

        for f in filters:
            if not isinstance(f, IIR):
                raise TypeError(f"Expected IIR filter, got {type(f).__name__}")

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

        # Concatenate all SOS sections: [K_total, 6]
        self._sos: Tensor = torch.cat(sos_parts, dim=0).to(dtype=torch.float64)
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

        Walks the chain and collects all IIR filter children.

        """
        if isinstance(chain, nn.Sequential):
            filters = [m for m in chain if isinstance(m, IIR)]
        elif isinstance(chain, IIR):
            filters = [chain]
        else:
            raise TypeError(f"Expected nn.Sequential or IIR, got {type(chain).__name__}")

        if not filters:
            raise ValueError("No IIR filters found in chain to fuse")

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
        orig_shape = x.shape
        out_dtype = x.dtype
        device = x.device

        # Normalize to [C, T]
        if x.ndim == 1:
            x = x.unsqueeze(0)
        elif x.ndim == 3:
            B, C, T = x.shape
            x = x.reshape(B * C, T)

        C, T = x.shape
        K = self._num_sections

        # Use cached device-matched SOS to avoid per-forward .to() calls.
        cache = self._sos_device_cache
        if cache is None or cache.device != device:
            cache = self._sos.to(device=device, dtype=torch.float64)
            self._sos_device_cache = cache
        sos = cache

        # Initialize or resize state.
        if self._state_x is None or self._state_x.shape[1] != C:
            self._state_x = torch.zeros(K, C, 2, device=device, dtype=torch.float64)
            self._state_y = torch.zeros(K, C, 2, device=device, dtype=torch.float64)
        elif self._state_x.device != device:
            self._state_x = self._state_x.to(device=device)
            assert self._state_y is not None
            self._state_y = self._state_y.to(device=device)

        assert self._state_y is not None

        # Try native kernel (single call for the entire cascade).
        # Pass canonical CPU SOS to avoid per-call CUDA→CPU copies.
        from torchfx._ops import parallel_iir_forward

        native_result = parallel_iir_forward(
            x,
            sos,
            self._state_x,
            self._state_y,
            sos_cpu=self._sos,
        )
        if native_result is not None:
            out, self._state_x, self._state_y = native_result
            result = out.to(dtype=out_dtype)
            self._stateful = True
            if len(orig_shape) == 1:
                return result.squeeze(0)
            elif len(orig_shape) == 3:
                return result.reshape(orig_shape)
            return result

        # Single-pass DF1 fallback using JIT-compiled biquad loop.
        from torchfx.filter.iir import _biquad_df1_fallback

        section_input = x.to(dtype=torch.float64)

        for s in range(K):
            b0 = sos[s, 0]
            b1 = sos[s, 1]
            b2 = sos[s, 2]
            a1 = sos[s, 4]
            a2 = sos[s, 5]

            sx = self._state_x[s]
            sy = self._state_y[s]

            out = _biquad_df1_fallback(section_input, b0, b1, b2, a1, a2, sx, sy)

            # In-place state update to avoid per-section allocation.
            if T >= 2:
                self._state_x[s, :, 0].copy_(section_input[:, -1])
                self._state_x[s, :, 1].copy_(section_input[:, -2])
                self._state_y[s, :, 0].copy_(out[:, -1])
                self._state_y[s, :, 1].copy_(out[:, -2])

            section_input = out

        self._stateful = True
        result = section_input.to(dtype=out_dtype)

        if len(orig_shape) == 1:
            return result.squeeze(0)
        elif len(orig_shape) == 3:
            return result.reshape(orig_shape)
        return result
