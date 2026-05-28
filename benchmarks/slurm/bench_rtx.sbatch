#!/bin/bash
#SBATCH --job-name=tfx-bench-rtx
#SBATCH --partition=allgroups
#SBATCH --output=benchmarks/results/slurm-bench-rtx-%j.out
#SBATCH --error=benchmarks/results/slurm-bench-rtx-%j.err
#SBATCH --gres=gpu:rtx:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#
# TorchFX benchmark sweep on an RTX 3090 node (gpu2, gpu3, gpu4).
#
# RTX 3090s are consumer-class Ampere parts (24 GB vRAM, 1:32 FP32:FP64
# throughput ratio). They give the same FP64 disadvantage story as the
# local Alienware RTX 3070 but at a higher core count and bandwidth —
# useful for confirming the consumer-GPU FP64 bottleneck is structural,
# not just a 3070 artefact.

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
    echo "ERROR: nvcc not found."
    exit 2
fi

echo ">>> nvcc: $(nvcc --version | tail -n 1)"
echo ">>> SLURM_JOB_ID=$SLURM_JOB_ID hostname=$(hostname)"

uv sync --all-groups --no-cache
uv pip install --python .venv/bin/python --no-deps --force-reinstall --no-cache .

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
