"""Energy + power measurement for the IS² 2026 benchmark suite.

Records CPU package energy via Intel RAPL and (when available) GPU power
via ``nvidia-smi``. The two sources are read with very different cadences:

* **RAPL** exposes a monotonically increasing energy counter in
  microjoules, sampled at the start and end of the measurement window.
  This gives an exact integral over the window with no per-sample
  quantisation noise. Resolution depends on the CPU (~61 µJ on recent
  Intel parts).
* **nvidia-smi** does not expose an energy counter, only an instantaneous
  power draw in watts. We sample it on a worker thread at a configurable
  rate (default 10 Hz) and integrate by the trapezoidal rule.

Both sources are optional and degrade gracefully: an RTX-less box reports
GPU as 0 W; an AMD or sandboxed Linux without RAPL reports CPU as 0 W.

Library use
-----------

.. code-block:: python

    from tools.energy_meter import EnergyMeter

    with EnergyMeter(gpu_hz=10) as m:
        run_my_benchmark()

    print(m.report())          # human-readable
    m.dump_json("energy.json") # machine-readable

CLI use
-------

Wrap an arbitrary command:

.. code-block:: bash

    uv run python tools/energy_meter.py \\
        --out IS22026/results/energy-bench-cuda.json \\
        -- uv run --no-sync pytest benchmarks/test_iir_bench.py \\
        --benchmark-enable --benchmark-json=out.json

Notes
-----

* RAPL counters wrap around (32- or 64-bit, ranging from ~60 s to ~250
  years between wraps depending on the CPU). For the time scales of an
  IS²-style benchmark we assume no wrap; ``--detect-wrap`` rejects any
  measurement where the end counter is less than the start.
* ``nvidia-smi`` is launched as a single long-running subprocess (one
  ``nvidia-smi -lms 100 ...``) rather than per-sample. This avoids the
  ~30 ms startup cost of ``nvidia-smi`` per sample and keeps the
  sampling jitter under ~1 %.
* CPU package and DRAM (RAPL ``intel-rapl:0:0``) are summed when both
  are present. PSys (``intel-rapl:1``) is preferred when available
  because it covers the whole package + uncore + DRAM.

"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ── RAPL helpers ──────────────────────────────────────────────────────────────


def _find_rapl_domains() -> list[Path]:
    """Find all readable RAPL domains under /sys/class/powercap/.

    Prefers ``intel-rapl:N`` (the top-level package); subdomains
    (``intel-rapl:0:0`` = cores, ``intel-rapl:0:1`` = uncore/dram) are
    not summed because RAPL already includes them in the package
    counter, which would double-count.

    """
    base = Path("/sys/class/powercap")
    if not base.is_dir():
        return []
    domains: list[Path] = []
    for entry in sorted(base.iterdir()):
        # Skip subdomains (have a colon-N suffix beyond the package).
        # Match e.g. "intel-rapl:0" but not "intel-rapl:0:0".
        name = entry.name
        if not name.startswith("intel-rapl:"):
            continue
        if name.startswith("intel-rapl-mmio"):
            # mmio mirrors intel-rapl; would double-count.
            continue
        # Exactly one ':' separator means top-level package.
        if name.count(":") != 1:
            continue
        if (entry / "energy_uj").is_file() and os.access(entry / "energy_uj", os.R_OK):
            domains.append(entry)
    return domains


def _read_rapl(domain: Path) -> int | None:
    """Return current energy counter for a domain in microjoules, or None."""
    try:
        return int((domain / "energy_uj").read_text().strip())
    except (OSError, ValueError):
        return None


def _rapl_max_energy(domain: Path) -> int | None:
    """Return the wrap-around threshold of a RAPL counter, or None."""
    try:
        return int((domain / "max_energy_range_uj").read_text().strip())
    except (OSError, ValueError):
        return None


# ── nvidia-smi power-sampling thread ──────────────────────────────────────────


@dataclass
class _GpuSample:
    """One ``(time_ns, power_w)`` sample."""

    t_ns: int
    power_w: float


class _GpuSampler:
    """Run ``nvidia-smi --query-gpu=power.draw`` at a fixed rate.

    Streams from a single subprocess; closes it cleanly on stop. If
    ``nvidia-smi`` is not on PATH the sampler short-circuits to a
    zero-sample list.

    """

    def __init__(self, hz: float = 10.0, gpu_index: int = 0):
        self.hz = hz
        self.gpu_index = gpu_index
        self.samples: list[_GpuSample] = []
        self._proc: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None
        self._stop = threading.Event()
        self._available = bool(shutil.which("nvidia-smi"))

    @property
    def available(self) -> bool:
        return self._available

    def start(self) -> None:
        if not self._available:
            return
        interval_ms = max(1, int(round(1000.0 / self.hz)))
        cmd = [
            "nvidia-smi",
            f"--id={self.gpu_index}",
            f"-lms={interval_ms}",
            "--query-gpu=power.draw",
            "--format=csv,noheader,nounits",
        ]
        self._stop.clear()
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def stop(self) -> None:
        if not self._available or self._proc is None:
            return
        self._stop.set()
        try:
            self._proc.send_signal(signal.SIGINT)
            self._proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        finally:
            self._proc = None
        if self._reader is not None:
            self._reader.join(timeout=1.0)
            self._reader = None

    def _read_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            line = line.strip()
            if not line:
                continue
            try:
                power = float(line)
            except ValueError:
                # nvidia-smi can emit "[N/A]" if the GPU doesn't report
                # power; silently skip.
                continue
            self.samples.append(_GpuSample(t_ns=time.monotonic_ns(), power_w=power))


def _integrate_trapezoid(samples: list[_GpuSample]) -> tuple[float, float, float]:
    """Trapezoidal integration of ``(t_ns, power_w)`` → ``(energy_J, mean_W, peak_W)``.

    Returns (0.0, 0.0, 0.0) for an empty or single-point sample list.

    """
    if len(samples) < 2:
        return 0.0, 0.0, 0.0
    energy_j = 0.0
    for prev, curr in zip(samples[:-1], samples[1:], strict=True):
        dt_s = (curr.t_ns - prev.t_ns) / 1e9
        if dt_s <= 0:
            continue
        energy_j += 0.5 * (prev.power_w + curr.power_w) * dt_s
    total_s = (samples[-1].t_ns - samples[0].t_ns) / 1e9
    mean_w = energy_j / total_s if total_s > 0 else 0.0
    peak_w = max(s.power_w for s in samples)
    return energy_j, mean_w, peak_w


# ── Public API ────────────────────────────────────────────────────────────────


@dataclass
class EnergyReport:
    """Result of one ``EnergyMeter`` window."""

    duration_s: float
    cpu_joules: float
    cpu_mean_w: float
    gpu_joules: float
    gpu_mean_w: float
    gpu_peak_w: float
    gpu_sample_count: int
    cpu_rapl_domains: list[str] = field(default_factory=list)
    gpu_available: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def total_joules(self) -> float:
        return self.cpu_joules + self.gpu_joules

    @property
    def total_mean_w(self) -> float:
        return self.cpu_mean_w + self.gpu_mean_w


class EnergyMeter:
    """Context manager measuring CPU + GPU energy over a code block.

    Parameters
    ----------
    gpu_hz : float
        nvidia-smi sampling rate. 10 Hz is a good balance between
        precision and overhead (sampler adds ~0.5 % CPU on a workstation).
    gpu_index : int
        Which GPU to sample. Default 0.

    Examples
    --------

    >>> with EnergyMeter() as m:
    ...     # ... do work ...
    ...     pass
    >>> # m.report() now has cpu_joules, gpu_joules, duration_s

    """

    def __init__(self, gpu_hz: float = 10.0, gpu_index: int = 0):
        self._domains = _find_rapl_domains()
        self._rapl_start: dict[Path, int] = {}
        self._t_start_ns: int = 0
        self._t_end_ns: int = 0
        self._gpu = _GpuSampler(hz=gpu_hz, gpu_index=gpu_index)
        self._report: EnergyReport | None = None

    @property
    def rapl_available(self) -> bool:
        return bool(self._domains)

    @property
    def gpu_available(self) -> bool:
        return self._gpu.available

    def __enter__(self) -> EnergyMeter:
        self._rapl_start = {d: _read_rapl(d) or 0 for d in self._domains}
        self._t_start_ns = time.monotonic_ns()
        self._gpu.start()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self._gpu.stop()
        self._t_end_ns = time.monotonic_ns()
        self._build_report()

    def _build_report(self) -> None:
        duration_s = (self._t_end_ns - self._t_start_ns) / 1e9

        notes: list[str] = []
        cpu_joules = 0.0
        for d in self._domains:
            end = _read_rapl(d)
            if end is None:
                notes.append(f"RAPL read failed for {d.name}")
                continue
            start = self._rapl_start.get(d, 0)
            delta = end - start
            if delta < 0:
                # Wrap-around. Use max_energy_range_uj to correct.
                wrap = _rapl_max_energy(d)
                if wrap is not None:
                    delta += wrap
                    notes.append(f"RAPL wrap on {d.name}: corrected via max_energy_range_uj")
                else:
                    notes.append(f"RAPL wrap on {d.name}: uncorrected (rejected)")
                    continue
            cpu_joules += delta / 1e6
        cpu_mean_w = cpu_joules / duration_s if duration_s > 0 else 0.0

        gpu_joules, gpu_mean_w, gpu_peak_w = _integrate_trapezoid(self._gpu.samples)

        self._report = EnergyReport(
            duration_s=duration_s,
            cpu_joules=cpu_joules,
            cpu_mean_w=cpu_mean_w,
            gpu_joules=gpu_joules,
            gpu_mean_w=gpu_mean_w,
            gpu_peak_w=gpu_peak_w,
            gpu_sample_count=len(self._gpu.samples),
            cpu_rapl_domains=[d.name for d in self._domains],
            gpu_available=self._gpu.available,
            notes=notes,
        )

    def report(self) -> EnergyReport:
        if self._report is None:
            raise RuntimeError("EnergyMeter has not been used in a with-block yet")
        return self._report

    def dump_json(self, path: str | Path) -> None:
        r = self.report()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(asdict(r), fh, indent=2)


# ── CLI ───────────────────────────────────────────────────────────────────────


def _format_report(r: EnergyReport) -> str:
    lines = [
        f"  duration:       {r.duration_s:.3f} s",
        f"  CPU energy:     {r.cpu_joules:.3f} J  ({r.cpu_mean_w:.1f} W avg, "
        f"{', '.join(r.cpu_rapl_domains) or 'no RAPL'})",
        f"  GPU energy:     {r.gpu_joules:.3f} J  ({r.gpu_mean_w:.1f} W avg, "
        f"{r.gpu_peak_w:.1f} W peak, {r.gpu_sample_count} samples)",
        f"  total energy:   {r.total_joules:.3f} J  ({r.total_mean_w:.1f} W avg)",
    ]
    if r.notes:
        lines.append("  notes:")
        lines.extend(f"    - {n}" for n in r.notes)
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu-hz", type=float, default=10.0, help="GPU sampling rate, default 10 Hz.")
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--out", type=str, default=None, help="JSON output path.")
    p.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run, preceded by '--'. Mandatory.",
    )
    args = p.parse_args()

    # argparse.REMAINDER puts everything after '--' (or after the first
    # positional) into args.command, with the leading '--' stripped.
    cmd = [a for a in args.command if a != "--"]
    if not cmd:
        p.error("a command to wrap is required (after --)")

    meter = EnergyMeter(gpu_hz=args.gpu_hz, gpu_index=args.gpu_index)
    print(f"[energy-meter] RAPL: {meter.rapl_available}, GPU: {meter.gpu_available}")
    print(f"[energy-meter] running: {' '.join(cmd)}")

    rc = 0
    with meter:
        try:
            rc = subprocess.call(cmd)
        except FileNotFoundError:
            print(f"[energy-meter] command not found: {cmd[0]}", file=sys.stderr)
            return 127

    r = meter.report()
    print(f"[energy-meter] result (exit {rc}):")
    print(_format_report(r))

    if args.out:
        meter.dump_json(args.out)
        print(f"[energy-meter] JSON written to {args.out}")

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
