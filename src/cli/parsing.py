"""Effect-chain parser and TOML configuration loader.

Parses ``--effect`` CLI strings and TOML config files into instantiated
``FX`` / ``AbstractFilter`` objects that can be fed to ``StreamProcessor``.

String format
-------------
``name``                     — effect with default parameters
``name:value``               — single positional parameter
``name:k1=v1,k2=v2``        — keyword parameters
``name:pos,k1=v1``           — mixed positional + keyword

Examples::

    gain:0.5
    reverb:decay=0.6,mix=0.3
    normalize
    lowpass:cutoff=1000,order=4
    parametriceq:frequency=2000,q=1.5,gain=6,gain_scale=db

TOML configuration
------------------
.. code-block:: toml

    [[effects]]
    name = "reverb"
    decay = 0.6
    mix = 0.3

    [[effects]]
    name = "normalize"
    peak = 0.8

"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from torchfx.effect import FX, Delay, Gain, Normalize, Reverb
from torchfx.filter.biquad import (
    BiquadAllPass,
    BiquadBPF,
    BiquadBPFPeak,
    BiquadHPF,
    BiquadLPF,
    BiquadNotch,
)
from torchfx.filter.iir import (
    AllPass,
    Butterworth,
    Chebyshev1,
    Chebyshev2,
    Elliptic,
    HiButterworth,
    HiChebyshev1,
    HiChebyshev2,
    HiElliptic,
    HiShelving,
    LoButterworth,
    LoChebyshev1,
    LoChebyshev2,
    LoElliptic,
    LoShelving,
    Notch,
    ParametricEQ,
    Peaking,
)

# ---------------------------------------------------------------------------
# Registry: CLI name → (class, positional_param_names)
# ---------------------------------------------------------------------------

#: Maps a lowercase CLI name to a tuple of (class, ordered positional params).
#: The positional list controls which params can be given without ``key=``.
EFFECT_REGISTRY: dict[str, tuple[type[FX], list[str]]] = {
    # ── effects ──────────────────────────────────────────────
    "gain": (Gain, ["gain"]),
    "normalize": (Normalize, ["peak"]),
    "reverb": (Reverb, ["delay", "decay", "mix"]),
    "delay": (Delay, ["delay_samples", "feedback", "mix"]),
    # ── biquad filters ──────────────────────────────────────
    "lowpass": (BiquadLPF, ["cutoff", "q"]),
    "highpass": (BiquadHPF, ["cutoff", "q"]),
    "bandpass": (BiquadBPF, ["cutoff", "q"]),
    "bandpasspeak": (BiquadBPFPeak, ["cutoff", "q"]),
    "notch": (BiquadNotch, ["cutoff", "q"]),
    "allpass": (BiquadAllPass, ["cutoff", "q"]),
    # ── IIR shortcuts ──────────────────────────────────────
    "lobutterworth": (LoButterworth, ["cutoff", "order"]),
    "hibutterworth": (HiButterworth, ["cutoff", "order"]),
    "butterworth": (Butterworth, ["btype", "cutoff", "order"]),
    "lochebyshev1": (LoChebyshev1, ["cutoff", "order"]),
    "hichebyshev1": (HiChebyshev1, ["cutoff", "order"]),
    "chebyshev1": (Chebyshev1, ["btype", "cutoff", "order"]),
    "lochebyshev2": (LoChebyshev2, ["cutoff", "order"]),
    "hichebyshev2": (HiChebyshev2, ["cutoff", "order"]),
    "chebyshev2": (Chebyshev2, ["btype", "cutoff", "order"]),
    "loelliptic": (LoElliptic, ["cutoff", "order"]),
    "hielliptic": (HiElliptic, ["cutoff", "order"]),
    "elliptic": (Elliptic, ["btype", "cutoff", "order"]),
    "loshelving": (LoShelving, ["cutoff", "q", "gain"]),
    "hishelving": (HiShelving, ["cutoff", "q", "gain"]),
    "parametriceq": (ParametricEQ, ["frequency", "q", "gain"]),
    "peaking": (Peaking, ["cutoff", "q"]),
    "iirnotch": (Notch, ["cutoff", "q"]),
    "iirallpass": (AllPass, ["cutoff", "q"]),
}


# ---------------------------------------------------------------------------
# Value coercion
# ---------------------------------------------------------------------------


def _coerce_value(raw: str) -> int | float | bool | str:
    """Attempt to interpret *raw* as int, float, or bool; fall back to str."""
    if raw.lower() in ("true", "yes"):
        return True
    if raw.lower() in ("false", "no"):
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


# ---------------------------------------------------------------------------
# String parser
# ---------------------------------------------------------------------------


def parse_effect_string(spec: str) -> FX:
    """Parse a single ``--effect`` CLI string into an instantiated ``FX``.

    Parameters
    ----------
    spec : str
        Effect specification string.  See module docstring for format.

    Returns
    -------
    FX
        Instantiated effect or filter ready for use.

    Raises
    ------
    ValueError
        If *spec* refers to an unknown effect name or the parameters are
        invalid.

    """
    name, _, params_str = spec.partition(":")
    name = name.strip().lower()

    if name not in EFFECT_REGISTRY:
        available = ", ".join(sorted(EFFECT_REGISTRY))
        raise ValueError(f"Unknown effect '{name}'. Available effects: {available}")

    cls, positional_names = EFFECT_REGISTRY[name]

    if not params_str.strip():
        return cls()  # default parameters

    kwargs: dict[str, Any] = {}
    positional_idx = 0

    for token in params_str.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" in token:
            key, _, val = token.partition("=")
            kwargs[key.strip()] = _coerce_value(val.strip())
        else:
            # Positional parameter
            if positional_idx >= len(positional_names):
                raise ValueError(
                    f"Too many positional parameters for '{name}'. "
                    f"Expected at most {len(positional_names)}: {positional_names}"
                )
            kwargs[positional_names[positional_idx]] = _coerce_value(token)
            positional_idx += 1

    return cls(**kwargs)


def parse_effect_list(specs: list[str]) -> list[FX]:
    """Parse a list of ``--effect`` strings into a list of ``FX`` instances."""
    return [parse_effect_string(s) for s in specs]


# ---------------------------------------------------------------------------
# TOML configuration loader
# ---------------------------------------------------------------------------


def _load_toml(path: str | Path) -> dict[str, Any]:
    """Load a TOML file using stdlib ``tomllib`` (3.11+) or ``tomli``."""
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        try:
            import tomli as tomllib
        except ImportError as exc:
            raise ImportError(
                "Python <3.11 requires the 'tomli' package for TOML support. "
                "Install it with: pip install tomli"
            ) from exc

    with open(path, "rb") as f:
        return tomllib.load(f)


def load_effects_from_config(path: str | Path) -> list[FX]:
    """Load an effect chain from a TOML configuration file.

    Parameters
    ----------
    path : str | Path
        Path to the ``.toml`` configuration file.

    Returns
    -------
    list[FX]
        Ordered list of instantiated effects.

    Raises
    ------
    ValueError
        If the config contains an unknown effect name.
    FileNotFoundError
        If *path* does not exist.

    Examples
    --------
    Given a file ``chain.toml``::

        [[effects]]
        name = "reverb"
        decay = 0.6
        mix = 0.3

        [[effects]]
        name = "normalize"
        peak = 0.8

    >>> effects = load_effects_from_config("chain.toml")  # doctest: +SKIP

    """
    data = _load_toml(path)

    effects_list: list[dict[str, Any]] = data.get("effects", [])
    if not effects_list:
        raise ValueError(f"No [[effects]] entries found in {path}")

    result: list[FX] = []
    for entry in effects_list:
        name = entry.pop("name", None)
        if name is None:
            raise ValueError("Each [[effects]] entry must have a 'name' field.")
        name = name.lower()
        if name not in EFFECT_REGISTRY:
            available = ", ".join(sorted(EFFECT_REGISTRY))
            raise ValueError(f"Unknown effect '{name}' in config. Available: {available}")
        cls, _ = EFFECT_REGISTRY[name]
        result.append(cls(**entry))

    return result


def load_config_defaults(path: str | Path) -> dict[str, Any]:
    """Load non-effect global defaults from a TOML configuration file.

    Recognised top-level keys: ``device``, ``verbose``, ``chunk_size``,
    ``overlap``, ``output_dir``, ``format``.

    Parameters
    ----------
    path : str | Path
        Path to the ``.toml`` configuration file.

    Returns
    -------
    dict[str, Any]
        Mapping of option names to their configured values.

    """
    data = _load_toml(path)
    # Return everything that is not the ``[[effects]]`` table.
    return {k: v for k, v in data.items() if k != "effects"}


def list_effects() -> list[str]:
    """Return sorted list of available effect/filter names for tab-completion."""
    return sorted(EFFECT_REGISTRY)
