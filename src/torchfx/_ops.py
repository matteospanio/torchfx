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

    if not torch.cuda.is_available():
        logger.debug("CUDA not available; skipping native extension compilation")
        return None

    cpp_sources = [os.path.join(_CSRC_DIR, "binding.cpp")]
    cpu_sources = [os.path.join(_CSRC_DIR, "cpu", "iir_cpu.cpp")]

    cuda_dir = os.path.join(_CSRC_DIR, "cuda")
    cuda_sources = [
        os.path.join(cuda_dir, "parallel_scan.cu"),
        os.path.join(cuda_dir, "biquad_forward.cu"),
        os.path.join(cuda_dir, "delay_forward.cu"),
    ]

    all_sources = cpp_sources + cpu_sources + cuda_sources

    # Verify all source files exist before attempting compilation
    for src in all_sources:
        if not os.path.isfile(src):
            logger.warning("Missing source file %s; skipping extension compilation", src)
            return None

    try:
        from torch.utils.cpp_extension import load

        _ext = load(
            name="torchfx_cuda",
            sources=all_sources,
            extra_include_paths=[_INCLUDE_DIR],
            verbose=False,
        )
        logger.info("torchfx CUDA extension compiled successfully")
    except Exception:
        logger.debug(
            "Failed to compile torchfx CUDA extension; using PyTorch fallback", exc_info=True
        )
        _ext = None

    return _ext


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
        result: tuple[Tensor, Tensor, Tensor] = ext.biquad_forward(
            x.to(dtype=dtype),
            b.to(dtype=dtype),
            a.to(dtype=dtype),
            state_x.to(device=device, dtype=dtype),
            state_y.to(device=device, dtype=dtype),
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
        result: tuple[Tensor, Tensor, Tensor] = ext.sos_forward(
            x.to(dtype=dtype),
            sos.to(device=device, dtype=dtype),
            state_x.to(device=device, dtype=dtype),
            state_y.to(device=device, dtype=dtype),
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
