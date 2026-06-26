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

        for f in filters:
            if not hasattr(f, "_sos"):
                raise TypeError(f"Expected filter with SOS coefficients, got {type(f).__name__}")

        # Keep the source filters (plain list — not registered as submodules) so
        # the cascade can re-derive its SOS matrix when the sampling rate changes.
        self._filters: list[IIR | Biquad] = list(filters)
        self._gain: float = gain

        fs_val: int | None = None
        for f in filters:
            # Ensure coefficients are computed.
            if f._sos is None:
                if f.fs is None:
                    raise ValueError(
                        f"Filter {type(f).__name__} has no sampling frequency set. "
                        "Set fs before fusing."
                    )
                f.compute_coefficients()
                f._coeff_fs = f.fs
                f._coeff_fingerprint = f._design_fingerprint()

            # Validate consistent sampling frequency.
            if f.fs is not None:
                if fs_val is None:
                    fs_val = f.fs
                elif f.fs != fs_val:
                    raise ValueError(
                        f"Cannot fuse filters with different sample rates: {fs_val} vs {f.fs}"
                    )

        self._sos: Tensor = self._build_sos()
        self._num_sections: int = self._sos.shape[0]
        self.fs: int | None = fs_val
        # Sampling rate the fused SOS matrix was built for (see forward).
        self._coeff_fs: int | None = fs_val

        # Cached device-matched copy — avoids per-forward .to() calls.
        self._sos_device_cache: Tensor | None = None

        # State for stateful processing (initialized lazily).
        self._state_x: Tensor | None = None
        self._state_y: Tensor | None = None
        self._stateful: bool = False

    def _build_sos(self) -> Tensor:
        """Concatenate the source filters' SOS rows and fold the static gain.

        ``torch.cat`` copies, so the source filters' coefficients are never mutated
        by the gain fold below.

        """
        sos_parts: list[Tensor] = []
        for f in self._filters:
            assert f._sos is not None
            sos_parts.append(f._sos)
        sos = torch.cat(sos_parts, dim=0).to(dtype=torch.float64)
        # Fold a static scalar gain (e.g. a `Gain` between fused filters) into the
        # LAST section's numerator. A scalar commutes through the linear cascade, so
        # this is exact; folding into the last section scales only the final output,
        # avoiding any intermediate over/underflow the gain might otherwise cause.
        if self._gain != 1.0:
            sos[-1, :3] *= self._gain
        return sos

    def _recompute(self) -> None:
        """Re-derive the fused SOS matrix for the current ``fs``.

        Propagates ``fs`` to the source filters, recomputes their coefficients, and
        rebuilds the concatenated matrix. Accumulated DF1 state and the device cache
        no longer match the new coefficients, so both are dropped.

        """
        assert self.fs is not None
        for f in self._filters:
            f.fs = self.fs
            f.compute_coefficients()
            f._coeff_fs = f.fs
            f._coeff_fingerprint = f._design_fingerprint()
            f._sos_device_cache = None
            f._state_x = None
            f._state_y = None
        self._sos = self._build_sos()
        self._num_sections = self._sos.shape[0]
        self._coeff_fs = self.fs
        self._sos_device_cache = None
        self._state_x = None
        self._state_y = None

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
        # Recompute when the sampling rate changed after construction (e.g. a
        # RealtimeProcessor assigning `effect.fs = config.sample_rate`). Without
        # this guard the cascade would silently keep coefficients designed for
        # the old rate — same contract as IIR.forward / Biquad.forward.
        if self.fs is not None and self.fs != self._coeff_fs:
            self._recompute()

        result, self._sos_device_cache, self._state_x, self._state_y = _sos_cascade_forward(
            x, self._sos, self._sos_device_cache, self._state_x, self._state_y
        )
        self._stateful = True
        return result


def build_fused_plan(pipeline: list[nn.Module]) -> list[nn.Module]:
    """Group a pipeline into the executable plan, fusing contiguous SOS stages.

    Contiguous ``IIR``/``Biquad`` runs are merged into a single ``FusedSOSCascade``.
    A static linear ``Gain`` (``clamp=False``) does **not** break a run: its scalar
    is folded into the fused cascade's numerator (a scalar commutes through a linear
    filter), so ``IIR | Gain | IIR`` becomes one cascade instead of three stages. A
    dynamic ``Normalize`` or a clamping ``Gain`` is non-linear and is kept as its own
    stage.

    Shared by ``Wave._build_plan`` (offline materialization) and ``FX.compile()``
    so both lower a chain to the same fused plan.

    """
    from torchfx.effect import Gain
    from torchfx.filter.biquad import Biquad

    plan: list[nn.Module] = []
    iir_run: list[IIR | Biquad] = []
    pending_gain = 1.0  # product of foldable gains to apply to the current run

    def flush() -> None:
        nonlocal iir_run, pending_gain
        if iir_run:
            if len(iir_run) >= 2 or pending_gain != 1.0:
                plan.append(FusedSOSCascade(*iir_run, gain=pending_gain))
            else:
                plan.append(iir_run[0])
        elif pending_gain != 1.0:
            # A foldable gain with no SOS filter to fold into — keep it as an op.
            plan.append(Gain(pending_gain))
        iir_run = []
        pending_gain = 1.0

    for module in pipeline:
        if isinstance(module, (IIR, Biquad)):
            iir_run.append(module)
        else:
            factor = module._linear_gain() if isinstance(module, Gain) else None
            if factor is not None:
                pending_gain *= factor  # fold into the current/next fused run
            else:
                flush()
                plan.append(module)
    flush()
    return plan
