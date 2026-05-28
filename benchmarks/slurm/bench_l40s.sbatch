#!/bin/bash
#SBATCH --job-name=tfx-bench-l40s
#SBATCH --partition=allgroups
#SBATCH --output=benchmarks/results/slurm-bench-l40s-%j.out
#SBATCH --error=benchmarks/results/slurm-bench-l40s-%j.err
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#
# TorchFX benchmark sweep on an L40S node (gpu7).
#
# Runs every CUDA-relevant make target in benchmarks/Makefile, plus the
# CPU baselines for the same node so the numbers are directly
# comparable. Output goes to benchmarks/results/*-<hostname>.json.
#
# Submit from the project root:
#     sbatch benchmarks/slurm/bench_l40s.sbatch
#
# Pull results back from the local workstation:
#     rsync -av --include='*.json' --include='*/' --exclude='*' \
#         dei:/nfsd/voce/machine_learning/experiments/torchfx/benchmarks/results/ \
#         benchmarks/results/
#
# See benchmarks/slurm/README.md for the broader workflow.

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

echo ">>> nvcc: $(nvcc --version | tail -n 1)"
echo ">>> SLURM_JOB_ID=$SLURM_JOB_ID hostname=$(hostname)"

# ---- Dependency sync ---------------------------------------------------------
echo ">>> Syncing dependencies (clean cache)"
uv sync --all-groups --no-cache

echo ">>> Rebuilding torchfx with CUDA extension"
uv pip install --python .venv/bin/python --no-deps --force-reinstall --no-cache .

# ---- CUDA preflight ----------------------------------------------------------
echo ">>> CUDA native dispatch preflight"
uv run --no-sync python - <<'PY'
import torch
from torchfx._ops import parallel_iir_forward

if not torch.cuda.is_available():
    raise SystemExit("CUDA not available in this job.")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA: {torch.version.cuda}, torch: {torch.__version__}")

x = torch.randn(1, 16, device="cuda", dtype=torch.float64)
sos = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]], device="cuda", dtype=torch.float64)
sx = torch.zeros(1, 1, 2, device="cuda", dtype=torch.float64)
sy = torch.zeros(1, 1, 2, device="cuda", dtype=torch.float64)
y, _, _ = parallel_iir_forward(x, sos, sx, sy)
assert y.shape == x.shape
print("[ok] parallel_iir_forward dispatched cleanly")
PY

# ---- Run benchmarks via the Makefile ----------------------------------------
# All targets write to benchmarks/results/ tagged with the current hostname.

echo ">>> bench-cuda"
make -C benchmarks bench-cuda

echo ">>> bench-comparators (picks up GPU automatically)"
make -C benchmarks bench-comparators

echo ">>> launches-cuda"
make -C benchmarks launches-cuda

echo ">>> threshold-cuda (PARALLEL_SCAN_THRESHOLD ablation)"
make -C benchmarks threshold-cuda

echo ">>> bench-cpu (matched-node CPU baseline)"
make -C benchmarks bench-cpu

echo ">>> bench-realtime"
make -C benchmarks bench-realtime

echo ">>> launches (CPU dispatch counts)"
make -C benchmarks launches

echo ">>> aggregate"
make -C benchmarks aggregate

echo ""
echo "=== done. Results: ==="
ls -lh benchmarks/results/ | tail -30
