# SLURM cluster runs

TorchFX CUDA build/test, benchmark, and profiler jobs for the cluster.

## Scripts

| Script | Purpose | Typical runtime |
|---|---|---|
| [`run_cuda_tests.sbatch`](run_cuda_tests.sbatch) | CUDA build sanity + GPU-focused pytest subset (`test_cuda_kernels`, `test_cuda_fallback`, `test_ops_dispatch`) | ~10–30 min |
| [`../run_benchmarks.slurm`](../run_benchmarks.slurm) | pytest-benchmark sweep (hotpath, pipeline, api, biquad, iir, fir, fftconv, design), JSON per file | ~30–90 min |
| [`run_profiles.sbatch`](run_profiles.sbatch) | torch.profiler traces for all scenarios + CUDA memory snapshot | ~5–15 min |

All scripts write outputs to `benchmarks/results/<stamp>_<jobid>/`.

## Submit jobs

```bash
# 1) CUDA build + tests
sbatch benchmarks/slurm/run_cuda_tests.sbatch

# 2) Benchmarks
sbatch benchmarks/run_benchmarks.slurm

# 3) Profiles
sbatch benchmarks/slurm/run_profiles.sbatch
```

Need a different queue/GPU?

```bash
sbatch -p medium --gres=gpu:l40s:1 benchmarks/run_benchmarks.slurm
```

## Pull artifacts back

From local workstation:

```bash
rsync -av --exclude='*.out' --exclude='*.err' \
   <user>@<cluster>:~/torchfx/benchmarks/results/ \
   benchmarks/results/
```

## Compare benchmark runs

```bash
uv run pytest-benchmark compare \
   benchmarks/results/<baseline>.json \
   benchmarks/results/<new-run>/*.json
```

Examples:

```bash
# Compare against historical local CPU baseline
uv run pytest-benchmark compare \
   benchmarks/results/local-cpu-baseline.json \
   benchmarks/results/<new-run>/*.json

# Compare two cluster benchmark directories
uv run pytest-benchmark compare \
   benchmarks/results/<baseline-run>/*.json \
   benchmarks/results/<new-run>/*.json
```

## Profile outputs

- Open `*.json` traces in <https://ui.perfetto.dev>
- Visualize memory snapshot:

```bash
python -m torch.cuda.memory_viz benchmarks/results/<run>/memory_snapshot.pkl
```

## Scenarios

Scenarios are defined in [`../profiles/scenarios.py`](../profiles/scenarios.py).
Any new scenario there is available for both CPU and CUDA profile scripts.
