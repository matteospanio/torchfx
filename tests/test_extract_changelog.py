"""Tests for the CHANGELOG release-notes extractor (issue #29)."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "extract_changelog.py"


def _load_extract():
    spec = importlib.util.spec_from_file_location("extract_changelog", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.extract


extract = _load_extract()

SAMPLE = """# Changelog

## [Unreleased]

### Added

- pending thing

## [0.7.0] - 2026-06-09

### Added

- shiny feature

### Fixed

- a bug

## [0.6.0] - 2026-06-04

### Added

- older feature
"""


def test_extract_middle_version():
    out = extract(SAMPLE, "0.7.0")
    assert "shiny feature" in out
    assert "a bug" in out
    assert "older feature" not in out  # stops at the next "## " header
    assert "pending thing" not in out  # does not bleed into the previous section


def test_extract_last_version_to_eof():
    assert "older feature" in extract(SAMPLE, "0.6.0")


def test_extract_unreleased():
    assert "pending thing" in extract(SAMPLE, "Unreleased")


def test_missing_version_raises():
    with pytest.raises(KeyError):
        extract(SAMPLE, "9.9.9")


def test_cli_found(tmp_path):
    cl = tmp_path / "CHANGELOG"
    cl.write_text(SAMPLE)
    res = subprocess.run(
        [sys.executable, str(SCRIPT), "0.7.0", "--changelog", str(cl)],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "shiny feature" in res.stdout


def test_cli_missing_exits_nonzero(tmp_path):
    cl = tmp_path / "CHANGELOG"
    cl.write_text(SAMPLE)
    res = subprocess.run(
        [sys.executable, str(SCRIPT), "9.9.9", "--changelog", str(cl)],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 1


def test_real_changelog_section_non_empty():
    text = (ROOT / "CHANGELOG").read_text(encoding="utf-8")
    assert extract(text, "0.6.0").strip()
