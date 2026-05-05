#!/bin/bash
#SBATCH --job-name=tfx-bench
#SBATCH --partition=short
#SBATCH --output=benchmarks/results/slurm-bench-%j.out
#SBATCH --error=benchmarks/results/slurm-bench-%j.err
#SBATCH --gres=gpu:tesla:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

module load GCC >/dev/null 2>&1 || true
module load CUDA >/dev/null 2>&1 || true

echo ">>> Syncing dependencies (clean resolver/cache)"
uv sync --all-groups --no-cache

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
