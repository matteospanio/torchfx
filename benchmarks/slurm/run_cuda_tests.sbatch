#!/bin/bash
#SBATCH --job-name=tfx-cuda-tests
#SBATCH --partition=short
#SBATCH --output=benchmarks/results/cuda-tests-%j.out
#SBATCH --error=benchmarks/results/cuda-tests-%j.err
#SBATCH --gres=gpu:tesla:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

module load GCC >/dev/null 2>&1 || true
module load CUDA >/dev/null 2>&1 || true

RESULTS_DIR="benchmarks/results/cuda-tests_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"

echo "=== TorchFX CUDA build + test sanity ==="
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    $(hostname)"
echo "Date:    $(date -Iseconds)"
echo "Results: ${RESULTS_DIR}"
echo ""

echo ">>> Syncing dependencies (clean resolver/cache)"
uv sync --all-groups --no-cache

echo ">>> Environment"
uv run python - <<'PY' | tee "$RESULTS_DIR/environment.txt"
import torch
import torchfx
from torchfx._ops import is_native_available

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"gpu_memory_gb={props.total_memory / 1e9:.1f}")
print(f"torchfx_native_available={is_native_available()}")
PY

echo ">>> Running CUDA-focused tests"
uv run pytest \
    tests/test_cuda_kernels.py \
    tests/test_cuda_fallback.py \
    tests/test_ops_dispatch.py \
    -q 2>&1 | tee "$RESULTS_DIR/pytest_cuda.log"

echo ""
echo "=== CUDA sanity complete ==="
ls -lh "$RESULTS_DIR"
