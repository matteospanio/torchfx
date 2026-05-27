"""Aggregate pytest-benchmark JSON output with percentile statistics.

``pytest-benchmark`` reports min/max/mean/stddev/median by default but
does not emit p95 / p99 / p99.9, which is what the IS² 2026 paper needs
for the realtime story (where the tail of the distribution matters
more than the mean).

This tool reads one or more pytest-benchmark JSON files and emits a
flat row per benchmark with:

* min, p50, p95, p99, p99.9, max in milliseconds
* IQR, stddev, mean
* the ``extra_info`` carried by each benchmark (so e.g. the realtime
  ``budget_p99`` / ``xrun_count`` fields are propagated)
* per-run identification (machine, commit, python, torch, etc.) lifted
  from pytest-benchmark's ``machine_info`` and ``commit_info`` blocks

Output formats: ``json``, ``csv``, or ``table`` (default).

Usage
-----

.. code-block:: bash

    uv run python tools/aggregate_benchmarks.py \\
        IS22026/results/*.json --format table

    uv run python tools/aggregate_benchmarks.py \\
        IS22026/results/local-cpu.json \\
        --format csv \\
        --out IS22026/results/local-cpu-percentiles.csv

"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Row:
    file: str
    group: str
    name: str
    fullname: str
    rounds: int
    iterations: int
    # All timings in milliseconds.
    min_ms: float
    p50_ms: float
    mean_ms: float
    stddev_ms: float
    iqr_ms: float
    p95_ms: float
    p99_ms: float
    p999_ms: float
    max_ms: float
    extra: dict[str, Any] = field(default_factory=dict)


def _percentile(sorted_values: list[float], p: float) -> float:
    """Linear-interpolation percentile (R type 7 — same as numpy default)."""
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    k = (n - 1) * p
    f = int(k)
    c = min(f + 1, n - 1)
    if f == c:
        return sorted_values[f]
    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return d0 + d1


def _summarise(name: str, fullname: str, group: str, bench: dict[str, Any], src: str) -> Row:
    stats = bench["stats"]
    raw_s = stats.get("data") or []
    # pytest-benchmark stores per-round times in seconds.
    raw_ms = sorted(t * 1000.0 for t in raw_s)
    iterations = int(stats.get("iterations", 1) or 1)
    rounds = int(stats.get("rounds", len(raw_ms)) or len(raw_ms))

    if raw_ms:
        min_ms = raw_ms[0]
        max_ms = raw_ms[-1]
        mean_ms = sum(raw_ms) / len(raw_ms)
        p25 = _percentile(raw_ms, 0.25)
        p75 = _percentile(raw_ms, 0.75)
        p50 = _percentile(raw_ms, 0.50)
        p95 = _percentile(raw_ms, 0.95)
        p99 = _percentile(raw_ms, 0.99)
        p999 = _percentile(raw_ms, 0.999)
        iqr_ms = p75 - p25
        if len(raw_ms) > 1:
            var = sum((t - mean_ms) ** 2 for t in raw_ms) / (len(raw_ms) - 1)
            stddev_ms = var**0.5
        else:
            stddev_ms = 0.0
    else:
        # Fall back to whatever stats pytest-benchmark wrote.
        min_ms = float(stats.get("min", 0.0)) * 1000.0
        max_ms = float(stats.get("max", 0.0)) * 1000.0
        mean_ms = float(stats.get("mean", 0.0)) * 1000.0
        p50 = float(stats.get("median", 0.0)) * 1000.0
        iqr_ms = float(stats.get("iqr", 0.0)) * 1000.0
        stddev_ms = float(stats.get("stddev", 0.0)) * 1000.0
        p95 = p99 = p999 = mean_ms

    return Row(
        file=src,
        group=group,
        name=name,
        fullname=fullname,
        rounds=rounds,
        iterations=iterations,
        min_ms=min_ms,
        p50_ms=p50,
        mean_ms=mean_ms,
        stddev_ms=stddev_ms,
        iqr_ms=iqr_ms,
        p95_ms=p95,
        p99_ms=p99,
        p999_ms=p999,
        max_ms=max_ms,
        extra=bench.get("extra_info") or {},
    )


def load_rows(paths: list[Path]) -> list[Row]:
    rows: list[Row] = []
    for p in paths:
        with open(p) as fh:
            data = json.load(fh)
        if not isinstance(data, dict) or "benchmarks" not in data:
            # Not a pytest-benchmark JSON (e.g. the kernel-launch
            # counter writes a flat list). Skip silently.
            print(f"[skip] {p}: not a pytest-benchmark JSON", file=sys.stderr)
            continue
        for bench in data["benchmarks"]:
            rows.append(
                _summarise(
                    name=bench["name"],
                    fullname=bench["fullname"],
                    group=bench.get("group", "") or "",
                    bench=bench,
                    src=str(p),
                )
            )
    return rows


def emit_table(rows: list[Row]) -> None:
    if not rows:
        print("(no benchmarks found)")
        return

    headers = [
        "group",
        "name",
        "rounds",
        "min",
        "p50",
        "p95",
        "p99",
        "p99.9",
        "max",
        "iqr",
    ]
    widths = [len(h) for h in headers]
    table: list[list[str]] = []
    for r in rows:
        row = [
            r.group[:20],
            r.name[:40],
            str(r.rounds),
            f"{r.min_ms:.3f}",
            f"{r.p50_ms:.3f}",
            f"{r.p95_ms:.3f}",
            f"{r.p99_ms:.3f}",
            f"{r.p999_ms:.3f}",
            f"{r.max_ms:.3f}",
            f"{r.iqr_ms:.3f}",
        ]
        widths = [max(w, len(c)) for w, c in zip(widths, row, strict=True)]
        table.append(row)

    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for trow in table:
        print(fmt.format(*trow))
    print()
    print("All values in milliseconds. Sorted by input order.")


def emit_csv(rows: list[Row], out: Path | None) -> None:
    fields = [
        "file",
        "group",
        "name",
        "fullname",
        "rounds",
        "iterations",
        "min_ms",
        "p50_ms",
        "mean_ms",
        "stddev_ms",
        "iqr_ms",
        "p95_ms",
        "p99_ms",
        "p999_ms",
        "max_ms",
    ]
    # Collect union of extra-info keys across all rows so each row has a stable column set.
    extra_keys: list[str] = sorted({k for r in rows for k in r.extra})
    all_fields = fields + [f"extra.{k}" for k in extra_keys]

    def _write(fh: Any) -> None:
        w = csv.writer(fh)
        w.writerow(all_fields)
        for r in rows:
            d = asdict(r)
            row = [d[k] for k in fields]
            row += [r.extra.get(k, "") for k in extra_keys]
            w.writerow(row)

    if out is None:
        _write(sys.stdout)
    else:
        with open(out, "w", newline="") as fh:
            _write(fh)


def emit_json(rows: list[Row], out: Path | None) -> None:
    payload = [asdict(r) for r in rows]
    text = json.dumps(payload, indent=2)
    if out is None:
        print(text)
    else:
        with open(out, "w") as fh:
            fh.write(text)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+", type=Path, help="pytest-benchmark JSON files")
    p.add_argument(
        "--format",
        choices=["table", "csv", "json"],
        default="table",
        help="Output format (default: table)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path (default: stdout). Only used for csv/json formats.",
    )
    args = p.parse_args()

    rows = load_rows(args.inputs)
    if args.format == "table":
        emit_table(rows)
    elif args.format == "csv":
        emit_csv(rows, args.out)
    else:
        emit_json(rows, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
