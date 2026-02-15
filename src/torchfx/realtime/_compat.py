"""Optional dependency handling for real-time audio backends.

This module provides lazy import utilities for optional dependencies like sounddevice,
ensuring the core library works without them.

"""

from __future__ import annotations

import importlib
from typing import Any

_sounddevice_module: Any = None


def _check_sounddevice() -> bool:
    """Check if sounddevice is available without importing at module level.

    Returns
    -------
    bool
        True if sounddevice is importable.

    """
    global _sounddevice_module
    if _sounddevice_module is not None:
        return True
    try:
        _sounddevice_module = importlib.import_module("sounddevice")
        return True
    except (ImportError, OSError):
        return False


def get_sounddevice() -> Any:
    """Get the sounddevice module, raising a clear error if not installed.

    Returns
    -------
    module
        The sounddevice module.

    Raises
    ------
    BackendNotAvailableError
        If sounddevice is not installed.

    """
    if not _check_sounddevice():
        from torchfx.realtime.exceptions import BackendNotAvailableError

        raise BackendNotAvailableError(
            backend_name="sounddevice",
            suggestion="Install with: pip install sounddevice",
        )
    return _sounddevice_module
