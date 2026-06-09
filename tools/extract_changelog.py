"""Extract one version's section from the CHANGELOG, for GitHub release notes.

The release workflow (``.github/workflows/release.yml``) runs this on the version it is
tagging and feeds the output to ``gh release create --notes-file``. Given a ``CHANGELOG``
with `Keep a Changelog`-style headers::

    ## [0.7.0] - 2026-06-09
    ### Added
    - ...

    ## [0.6.0] - 2026-06-04
    ...

``extract_changelog.py 0.7.0`` prints everything under ``## [0.7.0]`` up to (but not
including) the next ``## `` header.

Usage::

    python tools/extract_changelog.py 0.7.0 [--changelog CHANGELOG]

"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def extract(changelog: str, version: str) -> str:
    """Return the release-notes body for ``version`` (raises ``KeyError`` if absent)."""
    header = re.compile(r"^## \[" + re.escape(version) + r"\]")
    lines = changelog.splitlines()
    start = next((i for i, line in enumerate(lines) if header.match(line)), None)
    if start is None:
        raise KeyError(f"No CHANGELOG section for version {version!r}.")
    body: list[str] = []
    for line in lines[start + 1 :]:
        if line.startswith("## "):
            break
        body.append(line)
    return "\n".join(body).strip("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="version to extract, e.g. 0.7.0 (no leading 'v')")
    parser.add_argument("--changelog", default="CHANGELOG", help="path to the changelog file")
    args = parser.parse_args(argv)

    text = Path(args.changelog).read_text(encoding="utf-8")
    try:
        print(extract(text, args.version))
    except KeyError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
