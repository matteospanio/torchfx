"""Project-wide pytest configuration.

This conftest intentionally lives at the repository root so that both the
unit-test suite under ``tests/`` and any doctests collected from ``src/``
share the same fixtures and ignore rules.

It does two things:

1. Populates the doctest namespace with names commonly used in docstring
   examples (``torch``, ``np``, ``fx``, ``Wave``, plus a synthetic ``wave``
   built from a short tone). This lets docstrings show realistic snippets
   without each example having to repeat the same import + setup boilerplate.
2. Excludes modules whose docstring examples deliberately reference real
   audio files on disk. Those examples are user-facing tutorials, not
   self-contained doctests. The exclusion list is meant to shrink over time
   as docstrings get rewritten to be doctest-safe.
"""

from __future__ import annotations

import pytest

# Modules that are excluded from --doctest-modules collection because their
# docstrings showcase usage against real audio files (Wave.from_file("..."))
# or otherwise depend on resources the test environment does not provide.
# Add new entries sparingly; prefer making the docstring doctest-safe.
collect_ignore_glob = [
    "src/torchfx/wave.py",
    "src/torchfx/effect.py",
    "src/torchfx/filter/__base.py",
    "src/torchfx/filter/iir.py",
    "src/torchfx/filter/fir.py",
    "src/torchfx/filter/biquad.py",
    "src/torchfx/filter/filterbank.py",
    "src/torchfx/filter/fused.py",
    "src/torchfx/realtime/*.py",
    "src/cli/**/*.py",
    "benchmarks/**/*.py",
]


@pytest.fixture(autouse=True)
def _doctest_namespace(doctest_namespace: dict[str, object]) -> None:
    """Inject names used by docstring examples into the doctest namespace.

    Provides:

    - ``torch``, ``np`` — imported once, available everywhere.
    - ``fx`` — alias for the ``torchfx`` package.
    - ``Wave`` — the public class, ready to use.
    - ``wave`` — a deterministic 1-second 440 Hz mono sine at 44.1 kHz, so
      examples like ``>>> wave | filter`` work without on-disk audio.
    """
    import numpy as np
    import torch

    import torchfx as fx
    from torchfx import Wave

    torch.manual_seed(0)
    t = torch.linspace(0.0, 1.0, 44100)
    ys = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)
    sample_wave = Wave(ys, fs=44100)

    doctest_namespace["torch"] = torch
    doctest_namespace["np"] = np
    doctest_namespace["fx"] = fx
    doctest_namespace["Wave"] = Wave
    doctest_namespace["wave"] = sample_wave
