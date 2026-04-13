# SLURM cluster runs

TorchFX benchmark and profile jobs for the GPU cluster. The local workstation
has no GPU, so every CUDA measurement in the campaign comes from here.

## Scripts

| Script | Purpose | Typical runtime |
|---|---|---|
| [`../run_benchmarks.slurm`](../run_benchmarks.slurm) | pytest-benchmark sweep across all filter families, writes one JSON per bench file | ~20–40 min |
| [`run_profiles.sbatch`](run_profiles.sbatch) | `torch.profiler` for each scenario in [`../profiles/scenarios.py`](../profiles/scenarios.py), plus a cold-import / JIT-compile baseline and a `torch.cuda.memory._snapshot()` pickle | ~5–10 min |

Both scripts land their output under `benchmarks/results/<stamp>_<jobid>/`
(created by the job). Pull those directories back to the repo for local
comparison with `pytest-benchmark compare`.

## Submitting

```bash
# Benchmarks
sbatch benchmarks/run_benchmarks.slurm

# Profiles (traces + cold-import + memory snapshot)
sbatch benchmarks/slurm/run_profiles.sbatch
```

## Pulling artifacts back

From the local workstation (replace `<user>@<cluster>` with your real host):

```bash
rsync -av --exclude='*.out' --exclude='*.err' \
    <user>@<cluster>:~/torchfx/benchmarks/results/ \
    benchmarks/results/
```

Then:

```bash
# Compare against the local CPU baseline
uv run pytest-benchmark compare \
    benchmarks/results/local-cpu-baseline.json \
    benchmarks/results/<stamp>_<jobid>/*.json

# Chrome traces: open in https://ui.perfetto.dev (drag + drop the JSON)
# Memory snapshot: python -m torch.cuda.memory_viz <snapshot.pkl>
```

## What the profile job measures

1. **Cold import + JIT compile** — `torchfx._ops._load_extension()` timing.
   This is the headline number for the deferred JIT→AOT migration (see the
   master plan's "Deferred" section): if the value here is large, AOT will
   pay off; if it is small, AOT is lower priority.

2. **torch.profiler traces** — one Chrome trace per scenario
   (`ProfilerActivity.CPU + ProfilerActivity.CUDA`, with shapes + stacks +
   memory). Used to identify the top CUDA kernels and spot `aten::to` /
   allocator churn.

3. **Memory snapshot** — a single
   `torch.cuda.memory._snapshot()` pickle taken while running the
   `offline_batch_chain` scenario (`B=32, C=2, T=480k`). Visualize with:

   ```bash
   python -m torch.cuda.memory_viz memory_snapshot.pkl
   ```

## Scenarios

The scenarios are defined in Python so the same code runs locally on CPU and
remotely on CUDA. Add new ones in
[`../profiles/scenarios.py`](../profiles/scenarios.py) and they will be picked
up automatically by both `run_cpu_profile.py` and `run_cuda_profile.py`.
