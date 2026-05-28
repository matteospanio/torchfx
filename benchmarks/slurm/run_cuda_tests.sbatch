#!/bin/bash
#SBATCH --job-name=tfx-cuda-tests
#SBATCH --partition=allgroups
#SBATCH --output=benchmarks/results/cuda-tests-%j.out
#SBATCH --error=benchmarks/results/cuda-tests-%j.err
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#
# Override the GPU type with sbatch when other queues are less busy:
#     sbatch --gres=gpu:a40:1 benchmarks/slurm/run_cuda_tests.sbatch
#     sbatch --gres=gpu:rtx:1 benchmarks/slurm/run_cuda_tests.sbatch

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# DEI cluster: nvcc lives at /usr/local/cuda/bin (CUDA 13.2) and is not on
# PATH by default. CUDA_HOME drives CMake's find_package(CUDAToolkit) when
# scikit-build-core builds the .cu kernels. UV_LINK_MODE=copy silences the
# hardlink-cross-filesystem warning between /nfsd and the build cache.
export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDACXX="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
export UV_LINK_MODE=copy

module load GCC >/dev/null 2>&1 || true
module load CUDA >/dev/null 2>&1 || true

if ! command -v nvcc >/dev/null 2>&1; then
    echo "ERROR: nvcc not found. CUDA module/toolkit is not active in this job environment."
    exit 2
fi
echo "nvcc: $(nvcc --version | tail -n 1)"

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

echo ">>> Rebuilding local torchfx package in job environment"
uv pip install --python .venv/bin/python --no-deps --force-reinstall --no-cache .

echo ">>> CUDA native dispatch preflight"
uv run python - <<'PY'
import torch
from torchfx._ops import parallel_iir_forward

if not torch.cuda.is_available():
    raise SystemExit("CUDA not available in this job.")

x = torch.randn(1, 16, device="cuda", dtype=torch.float64)
sos = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]], device="cuda", dtype=torch.float64)
sx = torch.zeros(1, 1, 2, device="cuda", dtype=torch.float64)
sy = torch.zeros(1, 1, 2, device="cuda", dtype=torch.float64)

try:
    parallel_iir_forward(x, sos, sx, sy, sos_cpu=sos.cpu())
except RuntimeError as exc:
    raise SystemExit(f"CUDA native dispatch check failed: {exc}")

print("cuda_native_dispatch_ok=True")
PY

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
