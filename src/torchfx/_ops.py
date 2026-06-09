"""Native C++/CUDA extension dispatch for torchfx.

The precompiled ``torchfx_ext`` module exposes the kernels backing every
stateful filter in TorchFX:

- ``biquad_forward``     — single biquad section (DF1, CUDA or CPU)
- ``sos_forward``        — K-section SOS cascade (CUDA or CPU)
- ``delay_line_forward`` — fused delay with feedback and wet/dry mix
  (CUDA or CPU)

The kernels are dispatched based on the input tensor's device. The threshold
``PARALLEL_SCAN_THRESHOLD`` (default 2048 samples) selects between the
sequential C++ kernel and the CUDA Blelloch parallel-scan kernel for SOS
cascades.

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from torchfx import torchfx_ext as _ext  # type: ignore[attr-defined]

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)

# Dispatch boundary: signals with T <= threshold use the sequential CUDA kernel;
# longer signals use the work-efficient parallel scan. The crossover is dtype-
# dependent (measured on an RTX 3070 via benchmarks/bench_threshold_sweep.py): the
# parallel scan is ~flat at its launch overhead (~135 us) regardless of dtype,
# while the sequential kernel grows ~2x faster in FP64 than FP32 — so FP64 hits the
# crossover sooner. float32 sequential wins up to ~2560 samples; float64 to ~1024.
# A single 2048 default would leave FP64 ~57% slower at T~2048, so the default is
# dtype-aware (use _default_threshold).
PARALLEL_SCAN_THRESHOLD = 2048  # float32 default
PARALLEL_SCAN_THRESHOLD_FP64 = 1024  # float64 default (FP64 scan overhead is hit sooner)


def _default_threshold(dtype: torch.dtype) -> int:
    """Dtype-aware sequential-vs-parallel-scan default boundary (see above)."""
    return PARALLEL_SCAN_THRESHOLD_FP64 if dtype == torch.float64 else PARALLEL_SCAN_THRESHOLD


def _select_native_dtype(x: Tensor) -> torch.dtype:
    """Select native kernel dtype for an input tensor.

    Both the CPU and CUDA native IIR/biquad kernels are templated on float32 and
    float64. The native execution dtype follows the input: float64 in → float64,
    float32 in → float32. This lets float32 inputs run the FP32 GPU path (a large
    win on consumer cards with a 1:32 FP32:FP64 ratio) instead of being upcast.

    Half-precision inputs are rejected rather than silently upcast: the IIR
    feedback recurrence is not numerically safe in float16/bfloat16.

    Raises
    ------
    TypeError
        If ``x`` is not floating point, or uses ``float16``/``bfloat16``.

    """
    if not x.is_floating_point():
        raise TypeError("Input tensor must use a floating-point dtype.")
    if x.dtype in (torch.float16, torch.bfloat16):
        raise TypeError(
            f"Half-precision input ({x.dtype}) is not supported by the native filter "
            "kernels: the IIR feedback recurrence is not numerically safe in "
            "float16/bfloat16. Cast to float32 or float64 first (e.g. x.float())."
        )
    return torch.float64 if x.dtype == torch.float64 else torch.float32


def is_native_available() -> bool:
    """Check whether the native C++/CUDA extension is available.

    Always returns ``True`` in supported installs: the extension is compiled
    at install time and bundled into the wheel. A ``False`` value would
    indicate the import of ``torchfx_ext`` failed, which means the install
    is broken.

    Examples
    --------
    >>> from torchfx import is_native_available
    >>> is_native_available()
    True

    .. versionadded:: 0.5.0

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
    threshold: int | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Dispatch biquad filter to native kernel.

    Returns ``(y, new_state_x, new_state_y)``.

    Parameters
    ----------
    a1_f64, a2_f64 : float, optional
        Pre-extracted feedback coefficients as Python floats.  When supplied,
        avoids a ``float()`` call per forward — which on CUDA triggers a
        GPU→CPU synchronisation.
    threshold : int, optional
        Sequential-vs-parallel-scan boundary for the CUDA path (signals with
        ``T <= threshold`` use the sequential kernel). Defaults to
        :data:`PARALLEL_SCAN_THRESHOLD`. Pass ``0`` to force the parallel scan or
        a large value to force the sequential kernel — used by the dispatch
        crossover ablation. Ignored on CPU.

    """
    # Ensure state tensors exist
    C = x.shape[0] if x.ndim >= 2 else 1
    device = x.device
    dtype = _select_native_dtype(x)
    if threshold is None:
        threshold = _default_threshold(dtype)

    if state_x is None:
        state_x = torch.zeros(C, 2, device=device, dtype=dtype)
    if state_y is None:
        state_y = torch.zeros(C, 2, device=device, dtype=dtype)

    # Use pre-extracted floats when available to avoid GPU→CPU sync.
    if a1_f64 is None or a2_f64 is None:
        a_native = a if a.dtype == dtype else a.to(dtype=dtype)
        a1_f64 = float(a_native[1])
        a2_f64 = float(a_native[2])

    x_native = x if x.dtype == dtype else x.to(dtype=dtype)
    b_native = b if b.dtype == dtype else b.to(dtype=dtype)
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
        x_native,
        b_native,
        a1_f64,
        a2_f64,
        sx,
        sy,
        threshold,
    )
    return result


def parallel_iir_forward(
    x: Tensor,
    sos: Tensor,
    state_x: Tensor | None,
    state_y: Tensor | None,
    *,
    sos_cpu: Tensor | None = None,
    threshold: int | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Dispatch SOS cascade to native kernel.

    Returns ``(y, new_state_x, new_state_y)``.

    Parameters
    ----------
    sos_cpu : Tensor, optional
        Pre-computed CPU copy of the SOS matrix on the native execution dtype
        (float32 or float64). When supplied,
        avoids a per-call CUDA→CPU transfer that otherwise triggers a device
        synchronisation.
    threshold : int, optional
        Sequential-vs-parallel-scan boundary for the CUDA path (per section).
        Defaults to :data:`PARALLEL_SCAN_THRESHOLD`. Pass ``0`` to force the
        parallel scan or a large value to force the sequential kernel — used by
        the dispatch crossover ablation. Ignored on CPU.

    """
    C = x.shape[0] if x.ndim >= 2 else 1
    K = sos.shape[0]
    device = x.device
    dtype = _select_native_dtype(x)
    if threshold is None:
        threshold = _default_threshold(dtype)

    if state_x is None:
        state_x = torch.zeros(K, C, 2, device=device, dtype=dtype)
    if state_y is None:
        state_y = torch.zeros(K, C, 2, device=device, dtype=dtype)

    x_native = x if x.dtype == dtype else x.to(dtype=dtype)
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
    elif sos_cpu.device.type != "cpu" or sos_cpu.dtype != dtype:
        sos_cpu = sos_cpu.to(dtype=dtype, device="cpu")

    result: tuple[Tensor, Tensor, Tensor] = _ext.sos_forward(
        x_native,
        sos_device,
        sos_cpu,
        sx,
        sy,
        threshold,
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
    if x.ndim < 1:
        raise ValueError("Input tensor must have at least 1 dimension.")
    if delay_samples < 0:
        raise ValueError(f"delay_samples must be non-negative, got {delay_samples}.")
    if not x.is_floating_point():
        raise TypeError("Input tensor must use a floating-point dtype.")

    original_shape = x.shape
    if x.ndim == 1:
        x_2d = x.unsqueeze(0)
    elif x.ndim == 2:
        x_2d = x
    else:
        x_2d = x.reshape(-1, x.size(-1))

    # Keep native kernels on their supported dtypes.
    native_dtype = torch.float64 if x_2d.dtype == torch.float64 else torch.float32
    x_native = x_2d if x_2d.dtype == native_dtype else x_2d.to(dtype=native_dtype)

    result_native: Tensor = _ext.delay_line_forward(x_native, delay_samples, decay, mix)
    result_2d = result_native if result_native.dtype == x.dtype else result_native.to(dtype=x.dtype)

    if len(original_shape) == 1:
        return result_2d.squeeze(0)
    if len(original_shape) == 2:
        return result_2d
    return result_2d.reshape(original_shape)


def compressor_forward(
    x: Tensor,
    threshold_db: float,
    inv_ratio: float,
    knee_db: float,
    makeup_db: float,
    attack_coeff: float,
    release_coeff: float,
    rms_coeff: float,
    detector: int,
) -> Tensor:
    """Dispatch the compressor to the native kernel (CUDA or CPU).

    Ballistics coefficients are precomputed by the caller. ``detector`` is 0 (peak)
    or 1 (rms); ``inv_ratio`` is ``1 / ratio`` (0 for an infinite-ratio limiter, so
    the kernel never does ``inf`` arithmetic). Returns the processed tensor with the
    input shape and dtype preserved.

    """
    if x.ndim < 1:
        raise ValueError("Input tensor must have at least 1 dimension.")
    if not x.is_floating_point():
        raise TypeError("Input tensor must use a floating-point dtype.")

    original_shape = x.shape
    if x.ndim == 1:
        x_2d = x.unsqueeze(0)
    elif x.ndim == 2:
        x_2d = x
    else:
        x_2d = x.reshape(-1, x.size(-1))

    # Keep native kernels on their supported dtypes (float16/bfloat16 -> float32).
    native_dtype = torch.float64 if x_2d.dtype == torch.float64 else torch.float32
    x_native = x_2d if x_2d.dtype == native_dtype else x_2d.to(dtype=native_dtype)

    result_native: Tensor = _ext.compressor_forward(
        x_native,
        threshold_db,
        inv_ratio,
        knee_db,
        makeup_db,
        attack_coeff,
        release_coeff,
        rms_coeff,
        detector,
    )
    result_2d = result_native if result_native.dtype == x.dtype else result_native.to(dtype=x.dtype)

    if len(original_shape) == 1:
        return result_2d.squeeze(0)
    if len(original_shape) == 2:
        return result_2d
    return result_2d.reshape(original_shape)


def expander_forward(
    x: Tensor,
    threshold_db: float,
    slope: float,
    knee_db: float,
    floor_db: float,
    attack_coeff: float,
    release_coeff: float,
    rms_coeff: float,
    detector: int,
) -> Tensor:
    """Dispatch the downward expander / gate to the native kernel (CUDA or CPU).

    Ballistics coefficients are precomputed by the caller. ``detector`` is 0 (peak)
    or 1 (rms); ``slope`` is ``ratio - 1`` (a large finite value stands in for an
    infinite-ratio gate, so the kernel never does ``inf`` arithmetic); ``floor_db`` is
    the deepest attenuation in dB. Returns the processed tensor with the input shape
    and dtype preserved.

    """
    if x.ndim < 1:
        raise ValueError("Input tensor must have at least 1 dimension.")
    if not x.is_floating_point():
        raise TypeError("Input tensor must use a floating-point dtype.")

    original_shape = x.shape
    if x.ndim == 1:
        x_2d = x.unsqueeze(0)
    elif x.ndim == 2:
        x_2d = x
    else:
        x_2d = x.reshape(-1, x.size(-1))

    # Keep native kernels on their supported dtypes (float16/bfloat16 -> float32).
    native_dtype = torch.float64 if x_2d.dtype == torch.float64 else torch.float32
    x_native = x_2d if x_2d.dtype == native_dtype else x_2d.to(dtype=native_dtype)

    result_native: Tensor = _ext.expander_forward(
        x_native,
        threshold_db,
        slope,
        knee_db,
        floor_db,
        attack_coeff,
        release_coeff,
        rms_coeff,
        detector,
    )
    result_2d = result_native if result_native.dtype == x.dtype else result_native.to(dtype=x.dtype)

    if len(original_shape) == 1:
        return result_2d.squeeze(0)
    if len(original_shape) == 2:
        return result_2d
    return result_2d.reshape(original_shape)


def limiter_forward(
    x: Tensor,
    threshold_lin: float,
    attack_coeff: float,
    release_coeff: float,
    lookahead_samples: int,
) -> Tensor:
    """Dispatch the look-ahead brick-wall limiter to the native kernel (CUDA or CPU).

    The look-ahead windowed peak ``peak_env[n] = max(|x[n .. n+L]|)`` is computed here
    with a vectorised forward max-pool (so the gain is reduced *before* a peak arrives);
    the native kernel then runs only the sequential gain recurrence (attack/release
    smoothing plus a per-sample brick-wall clamp). ``threshold_lin`` is the linear
    ceiling, ``lookahead_samples`` is ``L``. Returns the processed tensor with the input
    shape and dtype preserved.

    """
    if x.ndim < 1:
        raise ValueError("Input tensor must have at least 1 dimension.")
    if not x.is_floating_point():
        raise TypeError("Input tensor must use a floating-point dtype.")

    original_shape = x.shape
    if x.ndim == 1:
        x_2d = x.unsqueeze(0)
    elif x.ndim == 2:
        x_2d = x
    else:
        x_2d = x.reshape(-1, x.size(-1))

    native_dtype = torch.float64 if x_2d.dtype == torch.float64 else torch.float32
    x_native = x_2d if x_2d.dtype == native_dtype else x_2d.to(dtype=native_dtype)

    # Forward look-ahead windowed peak: max of |x| over [n, n+L]. Zero-pad the tail so the
    # window shrinks gracefully at the end (zeros never raise the max).
    abs_x = x_native.abs()
    lookahead = max(0, int(lookahead_samples))
    if lookahead > 0:
        padded = torch.nn.functional.pad(abs_x, (0, lookahead))
        peak_env = torch.nn.functional.max_pool1d(
            padded.unsqueeze(0), kernel_size=lookahead + 1, stride=1
        ).squeeze(0)
    else:
        peak_env = abs_x

    result_native: Tensor = _ext.limiter_forward(
        x_native, peak_env.contiguous(), threshold_lin, attack_coeff, release_coeff
    )
    result_2d = result_native if result_native.dtype == x.dtype else result_native.to(dtype=x.dtype)

    if len(original_shape) == 1:
        return result_2d.squeeze(0)
    if len(original_shape) == 2:
        return result_2d
    return result_2d.reshape(original_shape)
