"""CUDA extension dispatch layer for torchfx.

This module handles JIT compilation of the C++/CUDA extension and provides
dispatch functions that route to native kernels when available, falling back
to ``None`` so callers can use the existing pure-PyTorch implementation.

The extension is compiled on first use via ``torch.utils.cpp_extension.load()``
and cached in ``~/.cache/torch_extensions/``. Compilation requires a CUDA
toolkit and (optionally) ninja for faster builds.

"""

from __future__ import annotations

import logging
import os
import warnings
from types import ModuleType
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)

_ext = None
_ext_load_attempted = False

_CSRC_DIR = os.path.join(os.path.dirname(__file__), "_csrc")
_INCLUDE_DIR = os.path.join(_CSRC_DIR, "include")

# Minimum signal length for parallel scan to be worthwhile.
# Below this, the sequential C++ kernel is faster.
PARALLEL_SCAN_THRESHOLD = 2048


def _load_extension() -> ModuleType | None:
    """JIT-compile the CUDA/C++ extension on first use.

    Returns the compiled module, or ``None`` if compilation fails.

    """
    global _ext, _ext_load_attempted

    if _ext_load_attempted:
        return _ext
    _ext_load_attempted = True

    cpp_sources = [os.path.join(_CSRC_DIR, "binding.cpp")]
    cpu_sources = [os.path.join(_CSRC_DIR, "cpu", "iir_cpu.cpp")]

    all_sources = cpp_sources + cpu_sources
    extra_cflags: list[str] = []
    extra_cuda_cflags: list[str] = []

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        cuda_dir = os.path.join(_CSRC_DIR, "cuda")
        cuda_sources = [
            os.path.join(cuda_dir, "parallel_scan.cu"),
            os.path.join(cuda_dir, "biquad_forward.cu"),
            os.path.join(cuda_dir, "delay_forward.cu"),
        ]
        all_sources += cuda_sources
        extra_cflags.append("-DWITH_CUDA")
        extra_cuda_cflags.append("-DWITH_CUDA")

    # Verify all source files exist before attempting compilation
    for src in all_sources:
        if not os.path.isfile(src):
            logger.warning("Missing source file %s; skipping extension compilation", src)
            return None

    try:
        from torch.utils.cpp_extension import load

        _ext = load(
            name="torchfx_ext",
            sources=all_sources,
            extra_include_paths=[_INCLUDE_DIR],
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            verbose=False,
        )
        logger.info("torchfx native extension compiled successfully (CUDA=%s)", use_cuda)
    except Exception as exc:
        logger.debug(
            "Failed to compile torchfx native extension; using PyTorch fallback", exc_info=True
        )
        # Include the first ~500 chars of the error so the user can diagnose
        # compilation failures without enabling debug logging.
        err_snippet = str(exc)[:500]
        warnings.warn(
            "torchfx: native C++/CUDA extension failed to compile. "
            "Falling back to pure-PyTorch implementation which is significantly slower.\n"
            f"Error: {err_snippet}",
            stacklevel=2,
        )
        _ext = None

    return _ext


def is_native_available() -> bool:
    """Check whether the native C++/CUDA extension is available.

    Triggers JIT compilation on first call if not already attempted.

    """
    return _load_extension() is not None


def biquad_forward(
    x: Tensor,
    b: Tensor,
    a: Tensor,
    state_x: Tensor | None,
    state_y: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor] | None:
    """Dispatch biquad filter to native kernel.

    Returns ``(y, new_state_x, new_state_y)`` or ``None`` if native
    extension is unavailable.

    """
    ext = _load_extension()
    if ext is None:
        return None

    # Ensure state tensors exist
    C = x.shape[0] if x.ndim >= 2 else 1
    device = x.device
    dtype = torch.float64

    if state_x is None:
        state_x = torch.zeros(C, 2, device=device, dtype=dtype)
    if state_y is None:
        state_y = torch.zeros(C, 2, device=device, dtype=dtype)

    try:
        # Extract a1, a2 as Python floats before any device transfer to avoid
        # GPU→CPU sync when coefficients are on CUDA.
        a_f64 = a if a.dtype == dtype else a.to(dtype=dtype)
        a1 = float(a_f64[1])
        a2 = float(a_f64[2])

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

        result: tuple[Tensor, Tensor, Tensor] = ext.biquad_forward(
            x_f64,
            b_f64,
            a1,
            a2,
            sx,
            sy,
        )
        return result
    except Exception:
        logger.debug("Native biquad_forward failed; falling back to PyTorch", exc_info=True)
        return None


def parallel_iir_forward(
    x: Tensor,
    sos: Tensor,
    state_x: Tensor | None,
    state_y: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor] | None:
    """Dispatch SOS cascade to native kernel.

    Returns ``(y, new_state_x, new_state_y)`` or ``None`` if native
    extension is unavailable.

    """
    ext = _load_extension()
    if ext is None:
        return None

    C = x.shape[0] if x.ndim >= 2 else 1
    K = sos.shape[0]
    device = x.device
    dtype = torch.float64

    if state_x is None:
        state_x = torch.zeros(K, C, 2, device=device, dtype=dtype)
    if state_y is None:
        state_y = torch.zeros(K, C, 2, device=device, dtype=dtype)

    try:
        x_f64 = x if x.dtype == dtype else x.to(dtype=dtype)
        sos_device = (
            sos
            if (sos.device == device and sos.dtype == dtype)
            else sos.to(device=device, dtype=dtype)
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

        # Pre-compute a CPU copy of the SOS matrix so the CUDA kernel can
        # extract scalar coefficients without a GPU→CPU sync.
        sos_cpu = sos.detach().to(dtype=dtype, device="cpu") if sos.is_cuda else sos_device

        result: tuple[Tensor, Tensor, Tensor] = ext.sos_forward(
            x_f64,
            sos_device,
            sos_cpu,
            sx,
            sy,
        )
        return result
    except Exception:
        logger.debug("Native sos_forward failed; falling back to PyTorch", exc_info=True)
        return None


def delay_line_forward(
    x: Tensor,
    delay_samples: int,
    decay: float,
    mix: float,
) -> Tensor | None:
    """Dispatch delay line to native CUDA kernel.

    Returns the processed tensor or ``None`` if native extension is
    unavailable or input is not on CUDA.

    """
    if not x.is_cuda:
        return None

    ext = _load_extension()
    if ext is None:
        return None

    try:
        result: Tensor = ext.delay_line_forward(x, delay_samples, decay, mix)
        return result
    except Exception:
        logger.debug("Native delay_line_forward failed; falling back to PyTorch", exc_info=True)
        return None
