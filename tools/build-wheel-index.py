"""Build a PEP 503 simple-repository index from a directory of CUDA wheels.

The CUDA wheels emitted by ``.github/workflows/wheels-cuda.yml`` carry a PEP
440 local-version segment encoding the CUDA toolkit they were built against
(``0.5.3+cu124``, ``0.5.3+cu128``, ...). This script groups the input wheels
by that segment and writes a per-CUDA static index that ``pip`` can use as
either an ``--index-url`` or ``--extra-index-url``::

    <output>/cu124/
        index.html              # lists projects (just torchfx)
        torchfx/
            index.html          # lists every wheel for this CUDA tag
            torchfx-0.5.3+cu124-cp310-cp310-manylinux_2_28_x86_64.whl
            torchfx-0.5.3+cu124-cp311-cp311-manylinux_2_28_x86_64.whl
            ...
    <output>/cu128/
        ...

The same output tree is then deployed under ``wheels/`` on the ``gh-pages``
branch so that, e.g.::

    pip install torchfx \
        --index-url https://matteospanio.github.io/torchfx/wheels/cu124/ \
        --extra-index-url https://pypi.org/simple

resolves the ``cu124`` build, falling back to PyPI for runtime dependencies.

Usage
-----
    python tools/build-wheel-index.py <input-dir> <output-dir>
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

# Wheel filename grammar (PEP 427): {distribution}-{version}(-{build})?-
#   {python}-{abi}-{platform}.whl
WHEEL_RE = re.compile(
    r"^(?P<distribution>[A-Za-z0-9_]+)-(?P<version>[^-]+)-.+\.whl$"
)


def _parts(wheel_name: str) -> tuple[str, str]:
    """Return (distribution, version) parsed from a wheel filename."""
    m = WHEEL_RE.match(wheel_name)
    if not m:
        raise ValueError(f"unrecognised wheel name: {wheel_name}")
    return m.group("distribution"), m.group("version")


def _local_segment(version: str) -> str:
    """Return the PEP 440 local-version segment, or "" if absent."""
    if "+" not in version:
        return ""
    return version.split("+", 1)[1]


def _write_html(path: Path, links: list[tuple[str, str]]) -> None:
    """Write a minimal PEP 503 simple-repository page."""
    body = "\n".join(f'    <a href="{href}">{text}</a><br>' for href, text in links)
    path.write_text(
        "<!DOCTYPE html>\n<html><body>\n" + body + "\n</body></html>\n"
    )


def build_index(input_dir: Path, output_dir: Path) -> None:
    wheels = sorted(input_dir.glob("*.whl"))
    if not wheels:
        sys.exit(f"no .whl files found under {input_dir}")

    grouped: dict[tuple[str, str], list[Path]] = {}
    skipped: list[str] = []
    for w in wheels:
        try:
            dist, version = _parts(w.name)
        except ValueError:
            skipped.append(w.name)
            continue
        cuda = _local_segment(version)
        if not cuda:
            skipped.append(w.name)  # no +cuXXX -> not a GPU wheel
            continue
        grouped.setdefault((cuda, dist), []).append(w)

    if skipped:
        print("skipping (no CUDA local-version segment):", file=sys.stderr)
        for name in skipped:
            print(f"  {name}", file=sys.stderr)

    for (cuda_tag, dist), files in grouped.items():
        proj_dir = output_dir / cuda_tag / dist
        proj_dir.mkdir(parents=True, exist_ok=True)
        for f in sorted(files):
            shutil.copy2(f, proj_dir / f.name)
        _write_html(
            proj_dir / "index.html",
            [(f.name, f.name) for f in sorted(files)],
        )

    # Top-level index per CUDA tag listing the projects served from it.
    for cuda_tag in sorted({c for (c, _) in grouped}):
        projects = sorted({d for (c, d) in grouped if c == cuda_tag})
        _write_html(
            output_dir / cuda_tag / "index.html",
            [(f"{p}/", p) for p in projects],
        )

    print(
        f"wrote index for {len(grouped)} (cuda, project) groups "
        f"under {output_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="directory containing .whl files")
    parser.add_argument(
        "output_dir", type=Path, help="root directory to write the index into"
    )
    args = parser.parse_args()
    build_index(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
