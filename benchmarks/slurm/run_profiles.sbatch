#!/bin/bash
#SBATCH --job-name=tfx-profile
#SBATCH --partition=allgroups
#SBATCH --output=benchmarks/results/profile-%j.out
#SBATCH --error=benchmarks/results/profile-%j.err
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#
# Override the GPU type with sbatch:
#     sbatch --gres=gpu:a40:1 benchmarks/slurm/run_profiles.sbatch
#     sbatch --gres=gpu:rtx:1 benchmarks/slurm/run_profiles.sbatch

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

# --- Import/native-extension timing ---
echo ">>> Import + native-extension availability timing"
uv run python -c "
import time

t0 = time.perf_counter()
import torchfx
t1 = time.perf_counter()
from torchfx._ops import is_native_available
t2 = time.perf_counter()
available = is_native_available()
t3 = time.perf_counter()

print(f'import_torchfx_s: {t1 - t0:.3f}')
print(f'import_ops_s: {t2 - t1:.3f}')
print(f'native_available_check_s: {t3 - t2:.3f}')
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
