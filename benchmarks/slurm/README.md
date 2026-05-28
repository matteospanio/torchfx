# SLURM cluster runs

TorchFX CUDA build/test, benchmark, and profiler jobs for the DEI
cluster. See [`../../SLURM.md`](../../SLURM.md) for cluster layout,
SSH alias, project path, queue etiquette, and GPU inventory.

## Scripts

| Script | Purpose | GPU GRES | Typical runtime |
|---|---|---|---|
| [`run_cuda_tests.sbatch`](run_cuda_tests.sbatch) | CUDA build sanity + GPU-focused pytest subset | `l40s` (default) | ~10–30 min |
| [`run_profiles.sbatch`](run_profiles.sbatch) | torch.profiler traces + CUDA memory snapshot | `l40s` (default) | ~5–15 min |
| [`bench_l40s.sbatch`](bench_l40s.sbatch) | Full bench sweep on L40S (`gpu7`) | `l40s` | ~45–60 min |
| [`bench_a40.sbatch`](bench_a40.sbatch) | Full bench sweep on A40 (`gpu1`, `gpu4`–`gpu6`) | `a40` | ~45–60 min |
| [`bench_rtx.sbatch`](bench_rtx.sbatch) | Full bench sweep on RTX 3090 (`gpu2`–`gpu4`) | `rtx` | ~45–60 min |
| [`../run_benchmarks.slurm`](../run_benchmarks.slurm) | Legacy: only the original bench files (no comparators, no realtime, no fusion/threshold tools) | `l40s` (default) | ~30–60 min |

The three `bench_*.sbatch` scripts call `make -C benchmarks all-cuda`
plus the matched-node CPU baselines, so a single job produces:

- `benchmarks/results/cuda-<host>.json`
- `benchmarks/results/comparators-<host>.json`
- `benchmarks/results/launches-cuda-<host>.json`
- `benchmarks/results/threshold-cuda-<host>.json`
- `benchmarks/results/cpu-<host>.json` (matched CPU baseline)
- `benchmarks/results/realtime-<host>.json`
- `benchmarks/results/launches-cpu-<host>.json`
- `benchmarks/results/summary.{csv,txt}`

## Submit a job

```bash
# Connect via the documented alias (see SLURM.md):
ssh dei

cd /nfsd/voce/machine_learning/experiments/torchfx
git pull --ff-only

# Pick the GPU you want; submit the corresponding script.
sbatch benchmarks/slurm/bench_l40s.sbatch
# or
sbatch benchmarks/slurm/bench_a40.sbatch
# or
sbatch benchmarks/slurm/bench_rtx.sbatch
```

The single partition `allgroups` is used for all jobs. Override
resources (e.g. raise wall time, change GPU) with sbatch flags:

```bash
sbatch --gres=gpu:a40:1 --cpus-per-task=12 --mem=64G \
    benchmarks/slurm/bench_a40.sbatch
```

Track:

```bash
squeue --me
tail -f benchmarks/results/slurm-bench-l40s-<jobid>.out
```

## Pull artifacts back

From a local workstation (see also [`../../SLURM.md`](../../SLURM.md)):

```bash
rsync -av --exclude='*.out' --exclude='*.err' \
   dei:/nfsd/voce/machine_learning/experiments/torchfx/benchmarks/results/ \
   benchmarks/results/
```

## Compare benchmark runs

```bash
# Compare a baseline JSON against a new run:
uv run pytest-benchmark compare \
   benchmarks/results/<baseline>.json \
   benchmarks/results/<new-run>.json
```

For full per-row p50/p95/p99 across many JSONs, use the aggregator:

```bash
uv run python tools/aggregate_benchmarks.py \
    benchmarks/results/*.json --format table
```

## Profile outputs

- Open `*.json` Chrome traces in <https://ui.perfetto.dev>
- Visualize CUDA memory snapshots:

```bash
python -m torch.cuda.memory_viz benchmarks/results/<run>/memory_snapshot.pkl
```

## Scenarios

Scenarios are defined in [`../profiles/scenarios.py`](../profiles/scenarios.py).
Any new scenario there is available for both CPU and CUDA profile
scripts.

## Running benchmarks without SLURM

The full sweep is also runnable on any machine via the Makefile (no
SLURM, no Bash glue):

```bash
make -C benchmarks all          # CPU + realtime + comparators + launches + aggregate
make -C benchmarks all-cuda     # adds bench-cuda + launches-cuda + threshold-cuda
make -C benchmarks help         # list every target
```

Override the output dir, host tag, or pytest flags:

```bash
make -C benchmarks all RESULTS=/scratch/torchfx HOST=alienware
ENERGY=1 make -C benchmarks bench-cuda    # wrap with tools/energy_meter.py
```
