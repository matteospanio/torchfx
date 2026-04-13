#!/bin/bash
#SBATCH --job-name=tfx-profile
#SBATCH --output=benchmarks/results/profile-%j.out
#SBATCH --error=benchmarks/results/profile-%j.err
#SBATCH --gres=gpu:rtx
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=01:00:00

# TorchFX CUDA profile run.
#
# Produces:
#   - a Chrome trace per scenario (open in https://ui.perfetto.dev)
#   - a top-ops text table per scenario (sorted by CUDA time)
#   - a torch.cuda memory snapshot pickle for the batch scenario
#
# Submit with:   sbatch benchmarks/slurm/run_profiles.sbatch
# Pull back:     rsync -av <cluster>:torchfx/benchmarks/results/<jobid>/ \
#                          benchmarks/results/<jobid>/

set -euo pipefail

module load GCC
module load CUDA

RESULTS_DIR="benchmarks/results/profile_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"

echo "=== TorchFX CUDA profile run ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Date:      $(date -Iseconds)"
echo "Results:   $RESULTS_DIR"
echo ""

# Log environment info
{
    echo "hostname: $(hostname)"
    echo "date: $(date -Iseconds)"
    echo "slurm_job_id: $SLURM_JOB_ID"
    uv run python -c "
import torch
print(f'torch: {torch.__version__}')
print(f'cuda_available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'gpu: {torch.cuda.get_device_name(0)}')
    print(f'gpu_memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print(f'cuda_version: {torch.version.cuda}')
"
} > "$RESULTS_DIR/environment.txt"

cat "$RESULTS_DIR/environment.txt"
echo ""

# --- Cold-import / JIT-compile timing (AOT-migration baseline, deferred phase) ---
echo ">>> Cold-import + native extension compile timing"
uv run python -c "
import time

t0 = time.perf_counter()
import torchfx
t1 = time.perf_counter()
from torchfx._ops import is_native_available
t2 = time.perf_counter()
available = is_native_available()  # triggers JIT compile on first call
t3 = time.perf_counter()

print(f'import_torchfx_s: {t1 - t0:.3f}')
print(f'import_ops_s: {t2 - t1:.3f}')
print(f'native_ext_load_s: {t3 - t2:.3f}')
print(f'native_available: {available}')
" > "$RESULTS_DIR/cold_import.txt"
cat "$RESULTS_DIR/cold_import.txt"
echo ""

# --- Run torch.profiler across all scenarios ---
echo ">>> Running torch.profiler scenarios"
PYTHONPATH=. uv run python benchmarks/profiles/run_cuda_profile.py \
    --device cuda \
    --scenario all \
    --out "$RESULTS_DIR" 2>&1 | tee "$RESULTS_DIR/profile.log"

echo ""
echo "=== Profile run complete ==="
ls -lh "$RESULTS_DIR"
