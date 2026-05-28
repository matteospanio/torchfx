#!/bin/bash
#SBATCH --job-name=tfx-bench
#SBATCH --partition=allgroups
#SBATCH --output=benchmarks/results/slurm-bench-%j.out
#SBATCH --error=benchmarks/results/slurm-bench-%j.err
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#
# Legacy benchmark sweep. Submits pytest-benchmark directly across the
# original suite of bench files (no Makefile orchestration, no
# torchaudio comparator, no realtime suite, no fusion/threshold tools).
# Kept for continuity with pre-IS²-2026 results.
#
# For the new full sweep (CPU + CUDA + comparators + realtime + fusion
# + threshold + aggregate), use one of:
#     sbatch benchmarks/slurm/bench_l40s.sbatch
#     sbatch benchmarks/slurm/bench_a40.sbatch
#     sbatch benchmarks/slurm/bench_rtx.sbatch
# or run ``make -C benchmarks all-cuda`` directly inside an
# allgroups-partition allocation.
#
# Override the GPU type with sbatch:
#     sbatch --gres=gpu:a40:1 benchmarks/run_benchmarks.slurm
#     sbatch --gres=gpu:rtx:1 benchmarks/run_benchmarks.slurm

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

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

# ── Setup ────────────────────────────────────────────────────────────────────
RESULTS_DIR="benchmarks/results/$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"

echo "=== TorchFX Benchmark Run ==="
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
try:
    from numba import cuda as nc
    print(f'numba_cuda: {nc.is_available()}')
except ImportError:
    print('numba_cuda: not installed')
"
} > "$RESULTS_DIR/environment.txt"

cat "$RESULTS_DIR/environment.txt"
echo ""

BENCH_OPTS="--benchmark-enable --benchmark-name=short -v --tb=short"
K_EXPR="${K_EXPR:-not numba_cuda}"

run_bench_file () {
    local bench_file="$1"
    local stem="$2"

    echo ">>> Running ${bench_file}..."
    uv run pytest "benchmarks/${bench_file}" \
        ${BENCH_OPTS} \
        -k "${K_EXPR}" \
        --benchmark-json="${RESULTS_DIR}/${stem}.json" \
        2>&1 | tee "${RESULTS_DIR}/${stem}.log"
}

# ── Run benchmarks ───────────────────────────────────────────────────────────

run_bench_file "test_hotpath_bench.py" "hotpath_bench"
run_bench_file "test_pipeline_bench.py" "pipeline_bench"
run_bench_file "test_api_bench.py" "api_bench"
run_bench_file "test_biquad_bench.py" "biquad_bench"
run_bench_file "test_iir_bench.py" "iir_bench"
run_bench_file "test_fir_bench.py" "fir_bench"
run_bench_file "test_fftconv_bench.py" "fftconv_bench"
run_bench_file "test_design_benchmarks.py" "design_bench"

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "=== Benchmark run complete ==="
echo "Results saved to: $RESULTS_DIR"
ls -lh "$RESULTS_DIR"
