#!/bin/bash
#SBATCH --job-name=tfx-bench-a40
#SBATCH --partition=allgroups
#SBATCH --output=benchmarks/results/slurm-bench-a40-%j.out
#SBATCH --error=benchmarks/results/slurm-bench-a40-%j.err
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#
# TorchFX benchmark sweep on an A40 node (gpu1, gpu4, gpu5, gpu6).
#
# A40s have 48 GB vRAM and the Ampere architecture (sm_86) — a useful
# datacenter-class FP32-friendly counterpoint to the FP64-heavy L40S
# kernel results. Same workflow as bench_l40s.sbatch; see that file for
# detailed comments.

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
# Keep uv's cache + build temp off the (quota-limited) home filesystem.
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/.uvcache}"
mkdir -p "$UV_CACHE_DIR"

module load GCC >/dev/null 2>&1 || true
module load CUDA >/dev/null 2>&1 || true

if ! command -v nvcc >/dev/null 2>&1; then
    echo "ERROR: nvcc not found."
    exit 2
fi

echo ">>> nvcc: $(nvcc --version | tail -n 1)"
echo ">>> SLURM_JOB_ID=$SLURM_JOB_ID hostname=$(hostname)"

# --reinstall-package forces a rebuild of the editable extension; plain `uv sync`
# (even with --no-cache) treats an installed torchfx as satisfied and skips the
# rebuild after a `git pull`, leaving a STALE .so that runs the OLD kernels.
uv sync --all-groups --reinstall-package torchfx

uv run --no-sync python - <<'PY'
import torch
from torchfx._ops import parallel_iir_forward

if not torch.cuda.is_available():
    raise SystemExit("CUDA not available in this job.")
print(f"GPU: {torch.cuda.get_device_name(0)}")
x = torch.randn(1, 16, device="cuda", dtype=torch.float64)
sos = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]], device="cuda", dtype=torch.float64)
sx = torch.zeros(1, 1, 2, device="cuda", dtype=torch.float64)
sy = torch.zeros(1, 1, 2, device="cuda", dtype=torch.float64)
y, _, _ = parallel_iir_forward(x, sos, sx, sy)
print("[ok] parallel_iir_forward dispatched cleanly")
PY

make -C benchmarks bench-cuda
make -C benchmarks bench-comparators
make -C benchmarks launches-cuda
make -C benchmarks threshold-cuda
make -C benchmarks bench-cpu
make -C benchmarks bench-realtime
make -C benchmarks launches
make -C benchmarks aggregate

echo ""
echo "=== done. Results: ==="
ls -lh benchmarks/results/ | tail -30
