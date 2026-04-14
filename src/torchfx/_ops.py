"""Native C++/CUDA extension dispatch for torchfx.

The extension is compiled at build time via scikit-build-core / CMake.
Import ``torchfx_ext`` provides the pre-built C++ module with:

- ``biquad_forward``  — single biquad section (DF1, CUDA or CPU)
- ``sos_forward``     — K-section SOS cascade (CUDA or CPU)
- ``delay_line_forward`` — fused delay with feedback & mix (CUDA or CPU)

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from torchfx import torchfx_ext as _ext  # type: ignore[attr-defined]

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)

# Minimum signal length for parallel scan to be worthwhile.
# Below this, the sequential C++ kernel is faster.
PARALLEL_SCAN_THRESHOLD = 2048


def is_native_available() -> bool:
    """Check whether the native C++/CUDA extension is available.

    Always returns ``True`` since the extension is compiled at install time.

    """
    return True


def biquad_forward(
    x: Tensor,
    b: Tensor,
    a: Tensor,
    state_x: Tensor | None,
    state_y: Tensor | None,
    *,
    a1_f64: float | None = None,
    a2_f64: float | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Dispatch biquad filter to native kernel.

    Returns ``(y, new_state_x, new_state_y)``.

    Parameters
    ----------
    a1_f64, a2_f64 : float, optional
        Pre-extracted feedback coefficients as Python floats.  When supplied,
        avoids a ``float()`` call per forward — which on CUDA triggers a
        GPU→CPU synchronisation.

    """
    # Ensure state tensors exist
    C = x.shape[0] if x.ndim >= 2 else 1
    device = x.device
    dtype = torch.float64

    if state_x is None:
        state_x = torch.zeros(C, 2, device=device, dtype=dtype)
    if state_y is None:
        state_y = torch.zeros(C, 2, device=device, dtype=dtype)

    # Use pre-extracted floats when available to avoid GPU→CPU sync.
    if a1_f64 is None or a2_f64 is None:
        a_f64 = a if a.dtype == dtype else a.to(dtype=dtype)
        a1_f64 = float(a_f64[1])
        a2_f64 = float(a_f64[2])

    x_f64 = x if x.dtype == dtype else x.to(dtype=dtype)
    b_f64 = b if b.dtype == dtype else b.to(dtype=dtype)
    sx = (
        state_x
        if (state_x.device == device and state_x.dtype == dtype)
        else state_x.to(device=device, dtype=dtype)
    )
    sy = (
        state_y
        if (state_y.device == device and state_y.dtype == dtype)
        else state_y.to(device=device, dtype=dtype)
    )

    result: tuple[Tensor, Tensor, Tensor] = _ext.biquad_forward(
        x_f64,
        b_f64,
        a1_f64,
        a2_f64,
        sx,
        sy,
    )
    return result


def parallel_iir_forward(
    x: Tensor,
    sos: Tensor,
    state_x: Tensor | None,
    state_y: Tensor | None,
    *,
    sos_cpu: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Dispatch SOS cascade to native kernel.

    Returns ``(y, new_state_x, new_state_y)``.

    Parameters
    ----------
    sos_cpu : Tensor, optional
        Pre-computed CPU copy of the SOS matrix (float64).  When supplied,
        avoids a per-call CUDA→CPU transfer that otherwise triggers a device
        synchronisation.

    """
    C = x.shape[0] if x.ndim >= 2 else 1
    K = sos.shape[0]
    device = x.device
    dtype = torch.float64

    if state_x is None:
        state_x = torch.zeros(K, C, 2, device=device, dtype=dtype)
    if state_y is None:
        state_y = torch.zeros(K, C, 2, device=device, dtype=dtype)

    x_f64 = x if x.dtype == dtype else x.to(dtype=dtype)
    sos_device = (
        sos if (sos.device == device and sos.dtype == dtype) else sos.to(device=device, dtype=dtype)
    )
    sx = (
        state_x
        if (state_x.device == device and state_x.dtype == dtype)
        else state_x.to(device=device, dtype=dtype)
    )
    sy = (
        state_y
        if (state_y.device == device and state_y.dtype == dtype)
        else state_y.to(device=device, dtype=dtype)
    )

    # Use pre-computed CPU copy when available to avoid per-call
    # CUDA→CPU transfer.  Fall back to computing it here.
    if sos_cpu is None:
        sos_cpu = sos.detach().to(dtype=dtype, device="cpu") if sos.is_cuda else sos_device

    result: tuple[Tensor, Tensor, Tensor] = _ext.sos_forward(
        x_f64,
        sos_device,
        sos_cpu,
        sx,
        sy,
    )
    return result


def delay_line_forward(
    x: Tensor,
    delay_samples: int,
    decay: float,
    mix: float,
) -> Tensor:
    """Dispatch delay line to native kernel (CUDA or CPU).

    Returns the processed tensor.

    """
    result: Tensor = _ext.delay_line_forward(x, delay_samples, decay, mix)
    return result
