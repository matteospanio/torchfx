"""Shared sample-rate configuration helpers.

The logic here used to live privately inside ``Wave.__update_config``. It is
extracted so the explicit ``FX.compile()`` path (and anything else that needs to
resolve a chain's sampling rate ahead of ``forward()``) can reuse exactly the
same fs-injection + eager coefficient-design behaviour the pipe operator uses.

"""

from __future__ import annotations

import typing as tp

import torch.nn as nn


def _configure_one(f: nn.Module, fs: int) -> None:
    """Inject ``fs`` into a single ``FX`` and (re)compute its coefficients."""
    # Imported lazily to avoid an import cycle (effect/filter import this module).
    from torchfx.effect import FX
    from torchfx.filter.__base import AbstractFilter

    if not isinstance(f, FX):
        return

    fs_changed = False
    current_fs = getattr(f, "fs", None)
    if current_fs != fs:
        tp.cast(tp.Any, f).fs = fs
        fs_changed = True

    if fs_changed:
        reset_state = getattr(f, "reset_state", None)
        if callable(reset_state):
            reset_state()

    if isinstance(f, AbstractFilter) and (fs_changed or not f._has_computed_coeff):
        f.compute_coefficients()
        # Record the fs and design-parameter snapshot the coefficients were
        # designed for so a subsequent direct forward() does not needlessly
        # recompute (and so a genuine fs/parameter change is still detected
        # on the direct-call path).
        f._coeff_fs = getattr(f, "fs", None)
        f._coeff_fingerprint = f._design_fingerprint()


def apply_fs(module: nn.Module, fs: int) -> None:
    """Configure ``fs`` on every ``FX`` in ``module`` (itself + all submodules).

    Walks ``module.modules()`` so arbitrarily nested ``Sequential`` / ``ModuleList``
    / ``ParallelFilterCombination`` structures all get their sampling rate set and
    their coefficients eagerly designed.

    """
    for m in module.modules():
        _configure_one(m, fs)


def freeze_fx(module: nn.Module, fs: int | None = None) -> None:
    """Back :meth:`torchfx.effect.FX.freeze`: design (if ``fs`` given) then lock.

    Marks every ``AbstractFilter`` submodule as frozen so its forward guard skips
    the per-call fs/fingerprint recompute check. Coefficients are still designed
    lazily once on first forward if they were never computed.

    """
    from torchfx.filter.__base import AbstractFilter

    if fs is not None:
        apply_fs(module, fs)
    for m in module.modules():
        if isinstance(m, AbstractFilter):
            m._frozen = True
