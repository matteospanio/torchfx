#!/bin/bash
#SBATCH --job-name=tfx-soak-fused
#SBATCH --partition=allgroups
#SBATCH --output=benchmarks/results/soak-fused-%j.out
#SBATCH --error=benchmarks/results/soak-fused-%j.err
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#
# Soak test for the single-pass decoupled-look-back SOS scan (TORCHFX_FUSED_SCAN=1).
# Validates the kernel's forward progress and numerical correctness on a GPU
# architecture other than the RTX 3070 it was developed on, so the fused path can be
# made the default with confidence. The decoupled look-back's deadlock-free forward
# progress depends on block scheduling, which differs by SM count / architecture --
# this is exactly what we want to confirm on L40S (sm_89) and A40 (sm_86).
#
# Submit for each GPU type (override --gres):
#     sbatch benchmarks/slurm/soak_fused_scan.sbatch                  # L40S (gpu7)
#     sbatch --gres=gpu:a40:1 benchmarks/slurm/soak_fused_scan.sbatch # A40  (gpu1/5/6)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDACXX="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
export UV_LINK_MODE=copy
# Keep uv's cache + build temp off the (quota-limited) home filesystem.
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/.uvcache}"
mkdir -p "$UV_CACHE_DIR"
module load GCC >/dev/null 2>&1 || true
module load CUDA >/dev/null 2>&1 || true
command -v nvcc >/dev/null 2>&1 || { echo "ERROR: nvcc not found"; exit 2; }

echo "=== TorchFX single-pass-scan soak ==="
echo "Job ${SLURM_JOB_ID} on $(hostname) @ $(date -Iseconds)"
echo "git: $(git rev-parse --short HEAD) ($(git branch --show-current 2>/dev/null))"
nvcc --version | tail -1

echo ">>> build (scikit-build-core compiles the .cu kernels for this node's arch)"
# IMPORTANT: --reinstall-package forces a rebuild of the editable extension. Plain
# `uv sync` (even with --no-cache) treats an already-installed torchfx as satisfied
# and does NOT recompile after a `git pull`, leaving a stale .so — which silently
# tests/benchmarks the OLD kernels (e.g. an old sos_forward signature -> TypeError).
uv sync --all-groups --reinstall-package torchfx

uv run --no-sync python - <<'PY'
import torch
print("GPU:", torch.cuda.get_device_name(0))
p = torch.cuda.get_device_properties(0)
print(f"compute capability sm_{p.major}{p.minor}, {p.multi_processor_count} SMs, {p.total_memory/1e9:.1f} GB")
PY

echo ""
echo ">>> [1/5] forced-fused equivalence + forward-progress test file"
TORCHFX_FUSED_SCAN=1 uv run --no-sync python -m pytest tests/test_fused_scan_equivalence.py -q

echo ""
echo ">>> [2/5] FULL SUITE via the fused path (the soak)"
TORCHFX_FUSED_SCAN=1 uv run --no-sync python -m pytest tests/ -q

echo ""
echo ">>> [3/5] default (oracle) full suite -- no-regression check"
uv run --no-sync python -m pytest tests/ -q

echo ""
echo ">>> [4/5] launch counts + wall-time: oracle vs single-pass"
echo "--- oracle (3-phase) ---"
uv run --no-sync python tools/count_kernel_launches.py --device cuda --depths 5 10 20 50 --duration 5 --iters 20
echo "--- single-pass (TORCHFX_FUSED_SCAN=1) ---"
TORCHFX_FUSED_SCAN=1 uv run --no-sync python tools/count_kernel_launches.py --device cuda --depths 5 10 20 50 --duration 5 --iters 20

echo ""
echo ">>> [5/5] deep forward-progress stress (up to ~2048 tiles, 8 channels)"
TORCHFX_FUSED_SCAN=1 uv run --no-sync python - <<'PY'
import numpy as np
import torch
from scipy.signal import butter, sosfilt
from torchfx._ops import parallel_iir_forward

sos_np = np.concatenate([butter(2, c, output="sos") for c in (0.1, 0.2, 0.3, 0.4)], axis=0)
for t in (262_144, 1_048_576):  # ~512 and ~2048 tiles of 512 samples
    for dt in (torch.float32, torch.float64):
        g = torch.Generator().manual_seed(t)
        x = torch.randn(8, t, generator=g, dtype=dt)
        sos = torch.tensor(sos_np, dtype=dt, device="cuda")
        y, _, _ = parallel_iir_forward(x.cuda(), sos, None, None, sos_cpu=sos.cpu(), threshold=0)
        torch.cuda.synchronize()
        assert torch.isfinite(y).all(), (t, dt)
        ref = torch.tensor(sosfilt(sos_np, x.double().numpy(), axis=-1), dtype=dt)
        md = (y.cpu() - ref).abs().max().item()
        tol = 5e-3 if dt == torch.float32 else 1e-7
        ok = md < tol
        print(f"  T={t:>9} {str(dt):>14} tiles~{t//512:<4} max|y-scipy|={md:.2e} {'OK' if ok else 'FAIL'}")
        assert ok, (t, dt, md)
print("deep forward-progress stress: PASS")
PY

echo ""
echo "=== soak complete: PASS ==="
